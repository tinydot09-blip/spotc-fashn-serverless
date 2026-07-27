"""TryOn Pipeline."""

import logging
import os
from dataclasses import dataclass
from typing import List, Literal, Optional

import cv2
import numpy as np
import torch
from fashn_human_parser import (
    CATEGORY_TO_BODY_COVERAGE,
    FashnHumanParser,
)
from PIL import Image
from tqdm.auto import tqdm

from .dwpose import DWposeDetector, draw_pose
from .preprocessing import (
    BODY_COVERAGE_TO_FASHN_LABELS,
    FASHN_LABELS_TO_IDS,
    AspectPreserveResize,
    ResizePad,
    create_clothing_agnostic_image,
    create_garment_image,
)
from .tryon_mmdit import TryOnModel
from .utils import (
    get_dummy_dw_keypoints,
    get_rf_schedule,
    load_checkpoint,
    normalize_uint8_to_neg1_1,
    numpy_to_torch,
    setup_logger,
    tensor_to_pil,
)


@dataclass
class PipelineOutput:
    """Pipeline output container."""

    images: List[Image.Image]


class TryOnPipeline:
    """
    TryOn inference pipeline.

    Args:
        weights_dir: Directory containing model weights.
        device: Device to run on.
        logger: Optional logger instance.
    """

    CATEGORY_TO_LABEL = {
        "tops": 1,
        "bottoms": 2,
        "one-pieces": 3,
    }

    def __init__(
        self,
        weights_dir: str,
        device: Optional[str] = None,
        logger: Optional[logging.Logger] = None,
    ):
        self.weights_dir = os.path.abspath(weights_dir)

        self.logger = logger or setup_logger(
            "TryOnPipeline",
            level=logging.INFO,
        )

        self.device = torch.device(
            device
            if device
            else (
                "cuda"
                if torch.cuda.is_available()
                else "cpu"
            )
        )

        self.logger.info(
            f"Using device: {self.device}"
        )

        self.inference_dtype = torch.float32

        if (
            self.device.type == "cuda"
            and torch.cuda.is_bf16_supported()
        ):
            self.inference_dtype = torch.bfloat16

        self.logger.info(
            f"Using dtype: {self.inference_dtype}"
        )

        self._validate_weights()
        self._setup_tryon_model()
        self._setup_pose_model()
        self._setup_hp_model()

        h, w = self.tryon_model.input_shape
        max_dim = max(h, w)

        self.pre_resize = AspectPreserveResize(
            target_size=(max_dim, max_dim),
            mode="fit",
            backend="pil",
        )

        self.resize_pad_fn = ResizePad(
            (w, h),
            backend="opencv",
        )

    def _validate_weights(self):
        """Check that required weight files exist."""

        tryon_path = os.path.join(
            self.weights_dir,
            "model.safetensors",
        )

        dwpose_dir = os.path.join(
            self.weights_dir,
            "dwpose",
        )

        yolox_path = os.path.join(
            dwpose_dir,
            "yolox_l.onnx",
        )

        dwpose_path = os.path.join(
            dwpose_dir,
            "dw-ll_ucoco_384.onnx",
        )

        missing = []

        if not os.path.exists(tryon_path):
            missing.append(tryon_path)

        if not os.path.exists(yolox_path):
            missing.append(yolox_path)

        if not os.path.exists(dwpose_path):
            missing.append(dwpose_path)

        if missing:
            raise FileNotFoundError(
                "Missing model weights:\n"
                + "\n".join(
                    f"  - {path}"
                    for path in missing
                )
                + (
                    "\n\nPlease run:\n"
                    "python scripts/download_weights.py "
                    f"--weights-dir {self.weights_dir}"
                )
            )

    def _setup_tryon_model(self):
        """Load the TryOn model."""

        model_path = os.path.join(
            self.weights_dir,
            "model.safetensors",
        )

        self.logger.info(
            f"Loading TryOnModel from {model_path}"
        )

        self.tryon_model = TryOnModel()

        state_dict = load_checkpoint(
            model_path,
            device=str(self.device),
        )

        self.tryon_model.load_state_dict(
            state_dict
        )

        self.tryon_model.to(
            self.device,
            dtype=self.inference_dtype,
        ).eval()

        self.logger.info(
            "TryOnModel loaded"
        )

    def _setup_pose_model(self):
        """Load DWPose model."""

        dwpose_dir = os.path.join(
            self.weights_dir,
            "dwpose",
        )

        self.logger.info(
            f"Loading DWPose from {dwpose_dir}"
        )

        if self.device.type == "cuda":
            device_index = (
                self.device.index
                if self.device.index is not None
                else 0
            )

            dwpose_device = (
                f"cuda:{device_index}"
            )
        else:
            dwpose_device = "cpu"

        self.pose_model = DWposeDetector(
            checkpoints_dir=dwpose_dir,
            device=dwpose_device,
        )

        self.logger.info(
            "DWPose loaded"
        )

    def _setup_hp_model(self):
        """Load human parsing model."""

        self.logger.info(
            "Loading FashnHumanParser"
        )

        hp_device = (
            "cuda"
            if self.device.type == "cuda"
            else "cpu"
        )

        self.hp_model = FashnHumanParser(
            device=hp_device
        )

        self.logger.info(
            "FashnHumanParser loaded"
        )

    @torch.inference_mode()
    def _sample(
        self,
        *,
        ca_images: torch.Tensor,
        garment_images: torch.Tensor,
        person_poses: torch.Tensor,
        garment_poses: torch.Tensor,
        garment_categories: torch.Tensor,
        num_timesteps: int = 30,
        time_shift_mu: float = 1.5,
        guidance_scale: float = 1.5,
        skip_cfg_last_n_steps: int = 1,
        use_tqdm: bool = True,
    ) -> List[Image.Image]:
        """Euler sampling with CFG."""

        device = ca_images.device
        dtype = ca_images.dtype
        batch_size = ca_images.shape[0]

        c = self.tryon_model.channels_in
        h, w = self.tryon_model.input_shape

        images = torch.randn(
            (
                batch_size,
                c,
                h,
                w,
            ),
            dtype=dtype,
            device=device,
        )

        timesteps = get_rf_schedule(
            num_steps=num_timesteps,
            mu=time_shift_mu,
        )

        model_kwargs = {
            "person_poses": person_poses,
            "garment_poses": garment_poses,
            "ca_images": ca_images,
            "garment_images": garment_images,
            "garment_categories": (
                garment_categories
            ),
        }

        progress = tqdm(
            zip(
                timesteps[:-1],
                timesteps[1:],
            ),
            desc="Sampling",
            total=len(timesteps) - 1,
            disable=not use_tqdm,
        )

        for step_idx, (
            t_curr,
            t_prev,
        ) in enumerate(progress):
            dt = t_prev - t_curr

            t_vec = torch.full(
                (batch_size,),
                t_curr,
                dtype=dtype,
                device=device,
            )

            pred = (
                self.tryon_model
                .forward_for_cfg(
                    images,
                    t_vec,
                    **model_kwargs,
                )
            )

            v_c = pred["v_c"]
            v_u = pred["v_u"]

            if (
                skip_cfg_last_n_steps > 0
                and step_idx
                >= (
                    num_timesteps
                    - skip_cfg_last_n_steps
                )
            ):
                v_guided = v_c
            else:
                v_guided = (
                    v_u
                    + guidance_scale
                    * (v_c - v_u)
                )

            images = (
                images
                + dt * v_guided
            )

        images = (
            images
            .to(dtype=torch.float)
            .clamp_(-1.0, 1.0)
        )

        return [
            tensor_to_pil(
                image,
                unnormalize=True,
            )
            for image in images
        ]

    @torch.inference_mode()
    def __call__(
        self,
        person_image: Image.Image,
        garment_image: Image.Image,
        category: Literal[
            "tops",
            "bottoms",
            "one-pieces",
        ],
        garment_photo_type: Literal[
            "model",
            "flat-lay",
        ] = "model",
        num_samples: int = 1,
        num_timesteps: int = 30,
        guidance_scale: float = 1.5,
        skip_cfg_last_n_steps: int = 1,
        seed: int = 42,
        segmentation_free: bool = True,
    ) -> PipelineOutput:
        """Run virtual try-on inference."""

        torch.manual_seed(seed)

        if self.device.type == "cuda":
            torch.cuda.manual_seed_all(seed)

        np.random.seed(seed)

        person_image = self.pre_resize(
            person_image,
            allow_upsampling=False,
        )

        garment_image = self.pre_resize(
            garment_image,
            allow_upsampling=False,
        )

        person_image_np = np.asarray(
            person_image.convert("RGB"),
            dtype=np.uint8,
        )

        garment_image_np = np.asarray(
            garment_image.convert("RGB"),
            dtype=np.uint8,
        )

        person_image_np = (
            np.ascontiguousarray(
                person_image_np
            )
        )

        garment_image_np = (
            np.ascontiguousarray(
                garment_image_np
            )
        )

        # DWPose expects BGR OpenCV images.
        person_bgr = cv2.cvtColor(
            person_image_np,
            cv2.COLOR_RGB2BGR,
        )

        person_bgr = np.ascontiguousarray(
            person_bgr,
            dtype=np.uint8,
        )

        person_pose = self.pose_model(
            person_bgr
        )

        if garment_photo_type == "flat-lay":
            garment_pose = (
                get_dummy_dw_keypoints()
            )
        else:
            garment_bgr = cv2.cvtColor(
                garment_image_np,
                cv2.COLOR_RGB2BGR,
            )

            garment_bgr = (
                np.ascontiguousarray(
                    garment_bgr,
                    dtype=np.uint8,
                )
            )

            garment_pose = self.pose_model(
                garment_bgr
            )

        person_pose_img = draw_pose(
            person_pose,
            person_image_np.shape[0],
            person_image_np.shape[1],
            grayscale=True,
        )

        garment_pose_img = draw_pose(
            garment_pose,
            garment_image_np.shape[0],
            garment_image_np.shape[1],
            grayscale=True,
        )

        person_seg_pred = (
            self.hp_model.predict(
                person_image_np
            )
        )

        garment_seg_pred = (
            self.hp_model.predict(
                garment_image_np
            )
        )

        body_coverage = (
            CATEGORY_TO_BODY_COVERAGE.get(
                category
            )
        )

        if body_coverage is None:
            raise ValueError(
                f"Unsupported category: {category}"
            )

        labels_to_segment = (
            BODY_COVERAGE_TO_FASHN_LABELS.get(
                body_coverage
            )
        )

        if labels_to_segment is None:
            raise ValueError(
                "No segmentation labels found "
                f"for category: {category}"
            )

        labels_to_segment_indices = [
            FASHN_LABELS_TO_IDS[label]
            for label in labels_to_segment
        ]

        ca_image = (
            create_clothing_agnostic_image(
                img_np=person_image_np.copy(),
                seg_pred=person_seg_pred.copy(),
                labels_to_segment_indices=(
                    labels_to_segment_indices.copy()
                ),
                body_coverage=body_coverage,
                disable_masking=(
                    segmentation_free
                ),
                logger=self.logger,
            )
        )

        garment_image_processed = (
            create_garment_image(
                img_np=garment_image_np.copy(),
                seg_pred=garment_seg_pred.copy(),
                labels_to_segment_indices=(
                    labels_to_segment_indices.copy()
                ),
                disable_masking=(
                    garment_photo_type
                    == "flat-lay"
                ),
            )
        )

        ca_image = self.resize_pad_fn(
            ca_image,
            mem_padding=True,
        )

        garment_image_processed = (
            self.resize_pad_fn(
                garment_image_processed
            )
        )

        person_pose_img = (
            self.resize_pad_fn(
                person_pose_img,
                interpolation=(
                    cv2.INTER_NEAREST_EXACT
                ),
            )
        )

        garment_pose_img = (
            self.resize_pad_fn(
                garment_pose_img,
                interpolation=(
                    cv2.INTER_NEAREST_EXACT
                ),
            )
        )

        def prepare_tensor(
            image: np.ndarray,
        ) -> torch.Tensor:
            image = np.ascontiguousarray(
                image
            )

            tensor = numpy_to_torch(
                image
            ).unsqueeze(0)

            tensor = (
                normalize_uint8_to_neg1_1(
                    tensor
                )
            )

            tensor = (
                tensor
                .to(self.device)
                .repeat(
                    num_samples,
                    1,
                    1,
                    1,
                )
            )

            return tensor

        ca_tensor = prepare_tensor(
            ca_image
        )

        garment_tensor = prepare_tensor(
            garment_image_processed
        )

        person_pose_tensor = prepare_tensor(
            person_pose_img
        )

        garment_pose_tensor = prepare_tensor(
            garment_pose_img
        )

        garment_categories = (
            torch.tensor(
                self.CATEGORY_TO_LABEL[
                    category
                ],
                dtype=torch.long,
            )
            .unsqueeze(0)
            .repeat(num_samples)
            .to(self.device)
        )

        ca_tensor = ca_tensor.to(
            dtype=self.inference_dtype
        )

        garment_tensor = (
            garment_tensor.to(
                dtype=self.inference_dtype
            )
        )

        person_pose_tensor = (
            person_pose_tensor.to(
                dtype=self.inference_dtype
            )
        )

        garment_pose_tensor = (
            garment_pose_tensor.to(
                dtype=self.inference_dtype
            )
        )

        self.logger.info(
            "Running inference with "
            f"{num_timesteps} timesteps..."
        )

        images = self._sample(
            ca_images=ca_tensor,
            garment_images=garment_tensor,
            person_poses=person_pose_tensor,
            garment_poses=garment_pose_tensor,
            garment_categories=(
                garment_categories
            ),
            num_timesteps=num_timesteps,
            guidance_scale=guidance_scale,
            skip_cfg_last_n_steps=(
                skip_cfg_last_n_steps
            ),
            use_tqdm=False,
        )

        images = [
            self.resize_pad_fn.unpad(image)
            for image in images
        ]

        self.logger.info(
            f"Generated {len(images)} images"
        )

        return PipelineOutput(
            images=images
        )
