import base64
import io
import time
import traceback
from typing import Any

import runpod
from PIL import Image, ImageOps


# The FASHN model is loaded only when the first real generation arrives.
# Later jobs on the same warm worker reuse the already-loaded model.
_generate_tryon = None


def get_generate_tryon():
    global _generate_tryon

    if _generate_tryon is None:
        print(
            "Loading SPOTC FASHN generation core...",
            flush=True,
        )

        from core import generate_tryon

        _generate_tryon = generate_tryon

        print(
            "SPOTC FASHN generation core is ready.",
            flush=True,
        )

    return _generate_tryon


def remove_data_url_prefix(value: str) -> str:
    value = value.strip()

    if value.startswith("data:") and "," in value:
        return value.split(",", 1)[1]

    return value


def decode_base64_image(
    encoded_image: str,
    image_name: str,
) -> Image.Image:
    if not isinstance(encoded_image, str):
        raise ValueError(
            f"{image_name} must be a base64 string."
        )

    encoded_image = remove_data_url_prefix(
        encoded_image
    )

    if not encoded_image:
        raise ValueError(
            f"{image_name} is empty."
        )

    try:
        image_bytes = base64.b64decode(
            encoded_image,
            validate=True,
        )
    except Exception as error:
        raise ValueError(
            f"{image_name} is not valid base64 data."
        ) from error

    if not image_bytes:
        raise ValueError(
            f"{image_name} contains no image data."
        )

    try:
        image = Image.open(
            io.BytesIO(image_bytes)
        )

        image.load()
        image = ImageOps.exif_transpose(image)
        image = image.convert("RGB")

        return image

    except Exception as error:
        raise ValueError(
            f"{image_name} could not be opened as an image: "
            f"{error}"
        ) from error


def encode_image_to_base64(
    image: Image.Image,
) -> str:
    output = io.BytesIO()

    image.convert("RGB").save(
        output,
        format="JPEG",
        quality=92,
        optimize=True,
    )

    return base64.b64encode(
        output.getvalue()
    ).decode("utf-8")


def read_boolean(
    value: Any,
    default: bool = False,
) -> bool:
    if value is None:
        return default

    if isinstance(value, bool):
        return value

    if isinstance(value, str):
        return value.strip().lower() in {
            "true",
            "1",
            "yes",
            "on",
        }

    return bool(value)


def handler(job):
    started_at = time.time()

    try:
        job_input = job.get("input") or {}

        if not isinstance(job_input, dict):
            raise ValueError(
                "The input field must be a JSON object."
            )

        person_base64 = job_input.get(
            "person_image_base64"
        )

        garment_base64 = job_input.get(
            "garment_image_base64"
        )

        # Validate before loading the large FASHN model.
        if not person_base64:
            raise ValueError(
                "person_image_base64 is required."
            )

        if not garment_base64:
            raise ValueError(
                "garment_image_base64 is required."
            )

        person_image = decode_base64_image(
            person_base64,
            "person_image_base64",
        )

        garment_image = decode_base64_image(
            garment_base64,
            "garment_image_base64",
        )

        category = job_input.get(
            "category",
            "tops",
        )

        garment_photo_type = job_input.get(
            "garment_photo_type",
            "model",
        )

        quality = job_input.get(
            "quality",
            "Balanced",
        )

        tryon_mode = job_input.get(
            "tryon_mode",
            "Natural / Maskless",
        )

        seed_mode = job_input.get(
            "seed_mode",
            "Random",
        )

        clean_flatlay = read_boolean(
            job_input.get("clean_flatlay"),
            default=False,
        )

        print("=" * 68, flush=True)
        print("SPOTC SERVERLESS JOB RECEIVED", flush=True)
        print(f"Category: {category}", flush=True)
        print(
            f"Garment photo type: "
            f"{garment_photo_type}",
            flush=True,
        )
        print(f"Quality: {quality}", flush=True)
        print(f"Try-on mode: {tryon_mode}", flush=True)
        print(f"Seed mode: {seed_mode}", flush=True)
        print(
            f"Clean flat-lay: {clean_flatlay}",
            flush=True,
        )
        print("=" * 68, flush=True)

        generate_tryon = get_generate_tryon()

        gallery_results, _, status = generate_tryon(
            person_image=person_image,
            garment_image=garment_image,
            category=category,
            garment_photo_type=garment_photo_type,
            quality=quality,
            tryon_mode=tryon_mode,
            seed_mode=seed_mode,
            clean_flatlay=clean_flatlay,
        )

        output_images = []

        for index, gallery_item in enumerate(
            gallery_results,
            start=1,
        ):
            if (
                isinstance(gallery_item, tuple)
                and len(gallery_item) >= 1
            ):
                generated_image = gallery_item[0]

                caption = (
                    gallery_item[1]
                    if len(gallery_item) > 1
                    else f"Result {index}"
                )
            else:
                generated_image = gallery_item
                caption = f"Result {index}"

            if not isinstance(
                generated_image,
                Image.Image,
            ):
                raise RuntimeError(
                    "FASHN returned an invalid image result."
                )

            output_images.append(
                {
                    "index": index,
                    "mime_type": "image/jpeg",
                    "image_base64": (
                        encode_image_to_base64(
                            generated_image
                        )
                    ),
                    "caption": str(caption),
                    "width": generated_image.width,
                    "height": generated_image.height,
                }
            )

        total_seconds = time.time() - started_at

        return {
            "success": True,
            "status": "COMPLETED",
            "message": status,
            "image_count": len(output_images),
            "images": output_images,
            "total_seconds": round(
                total_seconds,
                3,
            ),
        }

    except Exception as error:
        traceback.print_exc()

        total_seconds = time.time() - started_at

        return {
            "success": False,
            "status": "FAILED",
            "error": str(error),
            "error_type": type(error).__name__,
            "total_seconds": round(
                total_seconds,
                3,
            ),
        }


print(
    "SPOTC RunPod Serverless handler is ready.",
    flush=True,
)

runpod.serverless.start(
    {
        "handler": handler,
    }
)
