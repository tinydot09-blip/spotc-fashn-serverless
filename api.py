from pathlib import Path
import shutil
import tempfile
import traceback

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from PIL import Image

from app import generate_tryon


api = FastAPI(
    title="SPOTC FASHN VTON API",
    version="1.0.0",
)


@api.get("/health")
def health():
    return {
        "status": "ok",
        "service": "SPOTC FASHN VTON",
    }


@api.post("/generate")
async def generate(
    person_image: UploadFile = File(...),
    garment_image: UploadFile = File(...),
    category: str = Form("tops"),
    garment_photo_type: str = Form("model"),
    quality: str = Form("Balanced"),
    tryon_mode: str = Form("Natural / Maskless"),
    seed_mode: str = Form("Random"),
    clean_flatlay: bool = Form(False),
):
    temp_dir = Path(tempfile.mkdtemp(prefix="spotc_vton_"))
    person_path = temp_dir / "person.png"
    garment_path = temp_dir / "garment.png"

    try:
        with person_path.open("wb") as file:
            shutil.copyfileobj(person_image.file, file)

        with garment_path.open("wb") as file:
            shutil.copyfileobj(garment_image.file, file)

        with Image.open(person_path) as img:
            person = img.convert("RGB")

        with Image.open(garment_path) as img:
            garment = img.convert("RGB")

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

        zip_file = Path(zip_path)

        if not zip_file.exists():
            raise RuntimeError(
                f"Generated ZIP file was not found: {zip_file}"
            )

        return FileResponse(
            path=str(zip_file),
            media_type="application/zip",
            filename=zip_file.name,
            headers={"X-SPOTC-Status": status},
        )

    except Exception as error:
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=str(error),
        ) from error

    finally:
        try:
            await person_image.close()
        except Exception:
            pass

        try:
            await garment_image.close()
        except Exception:
            pass

        shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "api:api",
        host="0.0.0.0",
        port=7865,
        reload=False,
    )
