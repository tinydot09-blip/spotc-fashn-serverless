import numpy as np
import torch

from .utils import (
    draw_bodypose,
    draw_bodypose_gray,
    draw_facepose,
    draw_facepose_gray,
    draw_handpose,
    draw_handpose_gray,
)
from .wholebody import Wholebody

__all__ = ["DWposeDetector", "draw_pose"]

# Minimum confidence threshold for keypoint visibility
KEYPOINT_VISIBILITY_THRESHOLD = 0.3


def draw_pose(
    pose,
    H,
    W,
    canvas_value: int = 0,
    grayscale: bool = False,
):
    bodies = pose["bodies"]
    candidate = bodies["candidate"]
    subset = bodies["subset"]

    if grayscale:
        draw_bodypose_fn = draw_bodypose_gray
        draw_handpose_fn = draw_handpose_gray
        draw_facepose_fn = draw_facepose_gray

        canvas = np.full(
            (H, W),
            canvas_value,
            dtype=np.uint8,
        )
    else:
        draw_bodypose_fn = draw_bodypose
        draw_handpose_fn = draw_handpose
        draw_facepose_fn = draw_facepose

        canvas_value = (
            int(canvas_value / 0.6)
            if canvas_value > 0
            else 0
        )

        canvas = np.full(
            (H, W, 3),
            canvas_value,
            dtype=np.uint8,
        )

    canvas = draw_bodypose_fn(
        canvas,
        candidate,
        subset,
    )

    if "hands" in pose:
        canvas = draw_handpose_fn(
            canvas,
            pose.get("hands"),
        )

    if "faces" in pose:
        canvas = draw_facepose_fn(
            canvas,
            pose.get("faces"),
        )

    return canvas


class DWposeDetector:
    def __init__(
        self,
        checkpoints_dir,
        device="cuda:0",
    ):
        self.pose_estimation = Wholebody(
            checkpoints_dir=checkpoints_dir,
            device=device,
        )

    def _find_best_candidate(
        self,
        subset,
        candidate,
        score_threshold=KEYPOINT_VISIBILITY_THRESHOLD,
    ):
        # Apply score threshold to subset keypoints
        valid_keypoints = (
            subset[:, 1:14] > score_threshold
        )

        # Calculate scores for each candidate,
        # only counting valid keypoints
        headless_scores = np.sum(
            subset[:, 1:14] * valid_keypoints,
            axis=1,
        )

        # Extract keypoints for each candidate,
        # excluding the head
        headless_keypoints = candidate[:, 1:14]

        def compute_area(keypoints):
            # Filter keypoints using the first candidate's
            # visibility mask, matching the original logic.
            valid_kp = keypoints[
                valid_keypoints[0]
            ]

            valid_x = valid_kp[:, 0][
                valid_kp[:, 0] > 0
            ]

            valid_y = valid_kp[:, 1][
                valid_kp[:, 1] > 0
            ]

            if not len(valid_x) or not len(valid_y):
                return 0

            return (
                (np.max(valid_x) - np.min(valid_x))
                * (np.max(valid_y) - np.min(valid_y))
            )

        areas = [
            compute_area(keypoints)
            for keypoints in headless_keypoints
        ]

        with np.errstate(
            divide="ignore",
            invalid="ignore",
        ):
            scores_times_areas = (
                headless_scores * np.array(areas)
            )

        scores_times_areas[
            np.isnan(scores_times_areas)
            | np.isinf(scores_times_areas)
        ] = 0

        if np.all(scores_times_areas == 0):
            best_candidate_idx = np.argmax(
                headless_scores
            )
        else:
            best_candidate_idx = np.nanargmax(
                scores_times_areas
            )

        return (
            candidate[
                best_candidate_idx:
                best_candidate_idx + 1
            ],
            subset[
                best_candidate_idx:
                best_candidate_idx + 1
            ],
        )

    @torch.inference_mode()
    def __call__(
        self,
        oriImg,
        single: bool = True,
    ) -> dict:
        # Convert Torch tensor, PIL image or another
        # compatible image object into a NumPy array.
        if isinstance(oriImg, torch.Tensor):
            oriImg = (
                oriImg
                .detach()
                .cpu()
                .numpy()
            )
        elif not isinstance(oriImg, np.ndarray):
            oriImg = np.asarray(oriImg)

        if oriImg.ndim not in (2, 3):
            raise ValueError(
                "DWPose input must be a 2D or 3D image array. "
                f"Received shape: {oriImg.shape}"
            )

        # Convert grayscale to RGB.
        if oriImg.ndim == 2:
            oriImg = np.stack(
                [oriImg, oriImg, oriImg],
                axis=-1,
            )

        # Convert CHW format to HWC format.
        if (
            oriImg.ndim == 3
            and oriImg.shape[0] in (1, 3, 4)
            and oriImg.shape[-1] not in (1, 3, 4)
        ):
            oriImg = np.transpose(
                oriImg,
                (1, 2, 0),
            )

        # Expand a single image channel to RGB.
        if (
            oriImg.ndim == 3
            and oriImg.shape[2] == 1
        ):
            oriImg = np.repeat(
                oriImg,
                3,
                axis=2,
            )

        # Remove alpha channel.
        if (
            oriImg.ndim == 3
            and oriImg.shape[2] == 4
        ):
            oriImg = oriImg[:, :, :3]

        if (
            oriImg.ndim != 3
            or oriImg.shape[2] != 3
        ):
            raise ValueError(
                "DWPose input must have 3 RGB channels. "
                f"Received shape: {oriImg.shape}"
            )

        # Convert floating-point images into uint8.
        if np.issubdtype(
            oriImg.dtype,
            np.floating,
        ):
            if oriImg.size == 0:
                raise ValueError(
                    "DWPose received an empty image."
                )

            image_min = float(np.nanmin(oriImg))
            image_max = float(np.nanmax(oriImg))

            # Handle images in the range -1 to 1.
            if (
                image_min >= -1.0
                and image_max <= 1.0
                and image_min < 0.0
            ):
                oriImg = (
                    (oriImg + 1.0) * 127.5
                )

            # Handle images in the range 0 to 1.
            elif (
                image_min >= 0.0
                and image_max <= 1.0
            ):
                oriImg = oriImg * 255.0

            oriImg = np.nan_to_num(
                oriImg,
                nan=0.0,
                posinf=255.0,
                neginf=0.0,
            )

            oriImg = np.clip(
                oriImg,
                0,
                255,
            ).astype(np.uint8)
        else:
            oriImg = np.clip(
                oriImg,
                0,
                255,
            ).astype(
                np.uint8,
                copy=False,
            )

        # OpenCV and ONNX require contiguous NumPy memory.
        oriImg = np.ascontiguousarray(oriImg)

        if oriImg.size == 0:
            raise ValueError(
                "DWPose received an empty image."
            )

        H, W, C = oriImg.shape

        candidate, subset = self.pose_estimation(
            oriImg
        )

        nums, keys, locs = candidate.shape

        if nums == 0:
            raise RuntimeError(
                "DWPose could not detect a person "
                "in the uploaded image."
            )

        if single and nums > 1:
            candidate, subset = (
                self._find_best_candidate(
                    subset,
                    candidate,
                )
            )

            nums = 1

        candidate[..., 0] /= float(W)
        candidate[..., 1] /= float(H)

        body = candidate[:, :18].copy()
        body = body.reshape(
            nums * 18,
            locs,
        )

        score = subset[:, :18].copy()

        for i in range(len(score)):
            for j in range(len(score[i])):
                if (
                    score[i][j]
                    > KEYPOINT_VISIBILITY_THRESHOLD
                ):
                    score[i][j] = int(
                        18 * i + j
                    )
                else:
                    score[i][j] = -1

        un_visible = (
            subset
            < KEYPOINT_VISIBILITY_THRESHOLD
        )

        candidate[un_visible] = -1

        faces = candidate[:, 24:92]
        hands = candidate[:, 92:113]

        hands = np.vstack(
            [
                hands,
                candidate[:, 113:],
            ]
        )

        bodies = {
            "candidate": body,
            "subset": score,
        }

        pose = {
            "bodies": bodies,
            "hands": hands,
            "faces": faces,
        }

        return pose
