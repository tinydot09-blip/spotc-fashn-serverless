import base64
import io
import traceback
from pathlib import Path
from typing import Any

import runpod
from PIL import Image

from app import generate_tryon


def decode_image(value: str, field_name: str) -> Image.Image:
    if not value:
        raise ValueError(f"{field_name} is required")

    # Accept both plain base64 and data URLs.
    if "," in value and value.strip().lower().startswith("data:"):
        value = value.split(",", 1)[1]

    try:
        image_bytes = base64.b64decode(value, validate=True)
    except Exception as error:
        raise ValueError(f"{field_name} is not valid base64") from error

    try:
        return Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as error:
        raise ValueError(f"{field_name} is not a valid image") from error


def handler(job: dict[str, Any]) -> dict[str, Any]:
    try:
        job_input = job.get("input") or {}

        person = decode_image(
            job_input.get("person_image_base64", ""),
            "person_image_base64",
        )

        garment = decode_image(
            job_input.get("garment_image_base64", ""),
            "garment_image_base64",
        )

        category = job_input.get("category", "tops")
        garment_photo_type = job_input.get("garment_photo_type", "model")
        quality = job_input.get("quality", "Balanced")
        tryon_mode = job_input.get("tryon_mode", "Natural / Maskless")
        seed_mode = job_input.get("seed_mode", "Random")
        clean_flatlay = bool(job_input.get("clean_flatlay", False))

        _, zip_path, status = generate_tryon(
            person,
            garment,
            category,
            garment_photo_type,
            quality,
            tryon_mode,
            seed_mode,
            clean_flatlay,
        )

        output_path = Path(zip_path)

        if not output_path.exists():
            raise RuntimeError(
                f"Generated output was not found: {output_path}"
            )

        output_base64 = base64.b64encode(
            output_path.read_bytes()
        ).decode("utf-8")

        return {
            "success": True,
            "status": status,
            "filename": output_path.name,
            "content_type": "application/zip",
            "output_base64": output_base64,
        }

    except Exception as error:
        traceback.print_exc()

        return {
            "success": False,
            "error": str(error),
        }


runpod.serverless.start({"handler": handler})
