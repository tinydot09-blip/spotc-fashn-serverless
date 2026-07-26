import base64
import io
import sys
import traceback
from pathlib import Path
from typing import Any

import runpod
from PIL import Image


def decode_image(value: str, field_name: str) -> Image.Image:
    if not value:
        raise ValueError(f"{field_name} is required")

    value = value.strip()

    if value.lower().startswith("data:") and "," in value:
        value = value.split(",", 1)[1]

    try:
        image_bytes = base64.b64decode(value, validate=True)
    except Exception as error:
        raise ValueError(
            f"{field_name} is not valid base64"
        ) from error

    try:
        return Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as error:
        raise ValueError(
            f"{field_name} is not a valid image"
        ) from error


def handler(job: dict[str, Any]) -> dict[str, Any]:
    try:
        print("SPOTC SERVERLESS: Job received", flush=True)

        job_input = job.get("input") or {}

        # Import only after the worker is running.
        print("SPOTC SERVERLESS: Importing app.generate_tryon", flush=True)

        try:
            from app import generate_tryon
        except Exception as import_error:
            print(
                "SPOTC SERVERLESS: Failed to import app.py",
                file=sys.stderr,
                flush=True,
            )
            traceback.print_exc()

            return {
                "success": False,
                "stage": "app_import",
                "error": str(import_error),
                "traceback": traceback.format_exc(),
                "working_directory": str(Path.cwd()),
                "app_exists": Path("/app/app.py").exists(),
                "weights_in_app": Path("/app/weights").exists(),
            }

        print("SPOTC SERVERLESS: app.py imported", flush=True)

        person = decode_image(
            job_input.get("person_image_base64", ""),
            "person_image_base64",
        )

        garment = decode_image(
            job_input.get("garment_image_base64", ""),
            "garment_image_base64",
        )

        category = job_input.get("category", "tops")
        garment_photo_type = job_input.get(
            "garment_photo_type",
            "model",
        )
        quality = job_input.get("quality", "Balanced")
        tryon_mode = job_input.get(
            "tryon_mode",
            "Natural / Maskless",
        )
        seed_mode = job_input.get("seed_mode", "Random")
        clean_flatlay = bool(
            job_input.get("clean_flatlay", False)
        )

        print("SPOTC SERVERLESS: Starting generation", flush=True)

        result = generate_tryon(
            person,
            garment,
            category,
            garment_photo_type,
            quality,
            tryon_mode,
            seed_mode,
            clean_flatlay,
        )

        if not isinstance(result, (list, tuple)) or len(result) < 3:
            raise RuntimeError(
                "generate_tryon returned an unexpected result"
            )

        _, output_value, status = result

        output_path = Path(str(output_value))

        if not output_path.exists():
            raise RuntimeError(
                f"Generated output was not found: {output_path}"
            )

        output_base64 = base64.b64encode(
            output_path.read_bytes()
        ).decode("utf-8")

        print("SPOTC SERVERLESS: Generation completed", flush=True)

        return {
            "success": True,
            "status": str(status),
            "filename": output_path.name,
            "content_type": "application/zip",
            "output_base64": output_base64,
        }

    except Exception as error:
        print(
            "SPOTC SERVERLESS: Request failed",
            file=sys.stderr,
            flush=True,
        )
        traceback.print_exc()

        return {
            "success": False,
            "stage": "request",
            "error": str(error),
            "traceback": traceback.format_exc(),
        }


if __name__ == "__main__":
    print("SPOTC SERVERLESS WORKER STARTING", flush=True)
    runpod.serverless.start({"handler": handler})
