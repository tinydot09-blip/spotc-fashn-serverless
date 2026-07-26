import json
import os
import secrets
import time
import traceback
import zipfile
from datetime import datetime
from pathlib import Path

from PIL import Image, ImageOps
from fashn_vton import TryOnPipeline

# ============================================================
# PATHS AND LIMITS
# ============================================================

print("=" * 68, flush=True)
print("SPOTC FASHN VTON CORE STARTING", flush=True)
print("=" * 68, flush=True)

# Works in both:
# - RunPod Serverless: /app
# - RunPod Pod/local project directory
PROJECT_DIR = Path(__file__).resolve().parent

# Optional environment override:
# FASHN_WEIGHTS_DIR=/custom/path/weights
WEIGHTS_DIR = Path(
    os.environ.get(
        "FASHN_WEIGHTS_DIR",
        str(PROJECT_DIR / "weights"),
    )
)

# Serverless workers can safely write temporary output here.
OUTPUT_DIR = Path(
    os.environ.get(
        "FASHN_OUTPUT_DIR",
        "/tmp/spotc-fashn-outputs",
    )
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MIN_PERSON_WIDTH = 512
MIN_PERSON_HEIGHT = 512
MIN_GARMENT_WIDTH = 400
MIN_GARMENT_HEIGHT = 400
MAX_INPUT_SIDE = 2048
MAX_SEED = 2_147_483_000

print(f"Project directory: {PROJECT_DIR}", flush=True)
print(f"Weights directory: {WEIGHTS_DIR}", flush=True)
print(f"Output directory: {OUTPUT_DIR}", flush=True)

if not WEIGHTS_DIR.exists():
    raise RuntimeError(
        f"FASHN weights directory was not found: {WEIGHTS_DIR}. "
        "Confirm that the weights folder exists inside the Docker image."
    )


# ============================================================
# OPTIONAL BACKGROUND REMOVAL
# ============================================================

try:
    from rembg import remove as remove_background

    REMBG_AVAILABLE = True
    print("Flat-lay cleanup is available.", flush=True)
except Exception as error:
    remove_background = None
    REMBG_AVAILABLE = False

    print(
        f"Flat-lay cleanup is unavailable: {error}",
        flush=True,
    )


# ============================================================
# LOAD PIPELINE ONCE PER WORKER
# ============================================================

print("Loading FASHN VTON model pipeline...", flush=True)

pipeline_load_started = time.time()

try:
    pipeline = TryOnPipeline(
        weights_dir=str(WEIGHTS_DIR),
    )
except Exception:
    print("FASHN VTON pipeline failed to load.", flush=True)
    traceback.print_exc()
    raise

print(
    "FASHN VTON pipeline loaded successfully in "
    f"{time.time() - pipeline_load_started:.1f} seconds.",
    flush=True,
)


# ============================================================
# QUALITY SETTINGS
# ============================================================

QUALITY_PROFILES = {
    "Fast": {
        "timesteps": 20,
        "guidance": 1.5,
        "samples": 1,
    },
    "Balanced": {
        "timesteps": 30,
        "guidance": 1.7,
        "samples": 1,
    },
    "High Quality": {
        "timesteps": 40,
        "guidance": 2.0,
        "samples": 4,
    },
    "Premium Quality": {
        "timesteps": 50,
        "guidance": 2.0,
        "samples": 4,
    },
}


# ============================================================
# GARMENT PRESETS
# ============================================================

GARMENT_PRESETS = {
    "Men's Shirt / Polo - Model Photo": (
        "tops",
        "model",
        "Natural / Maskless",
        "Random",
    ),
    "Men's Shirt / Polo - Flat Lay": (
        "tops",
        "flat-lay",
        "Structured / Parsed",
        "Random",
    ),
    "T-Shirt / Casual Top": (
        "tops",
        "model",
        "Natural / Maskless",
        "Random",
    ),
    "Women's Top / Blouse": (
        "tops",
        "model",
        "Natural / Maskless",
        "Random",
    ),
    "Kurti / Salwar Suit": (
        "one-pieces",
        "model",
        "Structured / Parsed",
        "Random",
    ),
    "Dress / Gown": (
        "one-pieces",
        "model",
        "Structured / Parsed",
        "Random",
    ),
    "Lehenga / Long Skirt Outfit": (
        "one-pieces",
        "model",
        "Structured / Parsed",
        "Random",
    ),
    "Saree - Model Photo": (
        "one-pieces",
        "model",
        "Natural / Maskless",
        "Random",
    ),
    "Pants / Jeans / Shorts": (
        "bottoms",
        "model",
        "Structured / Parsed",
        "Random",
    ),
    "Pants / Jeans - Flat Lay": (
        "bottoms",
        "flat-lay",
        "Structured / Parsed",
        "Random",
    ),
}

DEFAULT_PRESET = "Men's Shirt / Polo - Model Photo"
DEFAULT_VALUES = GARMENT_PRESETS[DEFAULT_PRESET]


# ============================================================
# VALIDATION
# ============================================================

VALID_CATEGORIES = {
    "tops",
    "bottoms",
    "one-pieces",
}

VALID_GARMENT_PHOTO_TYPES = {
    "model",
    "flat-lay",
}

VALID_TRYON_MODES = {
    "Natural / Maskless",
    "Structured / Parsed",
}

VALID_SEED_MODES = {
    "Fixed 42",
    "Random",
}


def validate_generation_options(
    category,
    garment_photo_type,
    quality,
    tryon_mode,
    seed_mode,
):
    if category not in VALID_CATEGORIES:
        raise ValueError(
            "Invalid category. Use tops, bottoms or one-pieces."
        )

    if garment_photo_type not in VALID_GARMENT_PHOTO_TYPES:
        raise ValueError(
            "Invalid garment_photo_type. Use model or flat-lay."
        )

    if quality not in QUALITY_PROFILES:
        raise ValueError(
            "Invalid quality. Use Fast, Balanced, "
            "High Quality or Premium Quality."
        )

    if tryon_mode not in VALID_TRYON_MODES:
        raise ValueError(
            "Invalid tryon_mode. Use Natural / Maskless "
            "or Structured / Parsed."
        )

    if seed_mode not in VALID_SEED_MODES:
        raise ValueError(
            "Invalid seed_mode. Use Fixed 42 or Random."
        )


# ============================================================
# IMAGE PREPARATION
# ============================================================

def resize_large_image(image: Image.Image) -> Image.Image:
    width, height = image.size
    largest_side = max(width, height)

    if largest_side <= MAX_INPUT_SIDE:
        return image

    scale = MAX_INPUT_SIDE / largest_side

    new_width = max(1, round(width * scale))
    new_height = max(1, round(height * scale))

    print(
        f"Resizing image from {width}x{height} "
        f"to {new_width}x{new_height}.",
        flush=True,
    )

    return image.resize(
        (new_width, new_height),
        Image.Resampling.LANCZOS,
    )


def prepare_image(
    image,
    image_name,
    minimum_width,
    minimum_height,
):
    if image is None:
        raise ValueError(
            f"Upload the {image_name.lower()}."
        )

    if not isinstance(image, Image.Image):
        raise ValueError(
            f"{image_name} must be a valid Pillow image."
        )

    try:
        prepared = ImageOps.exif_transpose(image)
        prepared = prepared.convert("RGB")
    except Exception as error:
        raise ValueError(
            f"Could not read the {image_name.lower()}: {error}"
        ) from error

    width, height = prepared.size

    if width < minimum_width or height < minimum_height:
        raise ValueError(
            f"{image_name} is too small: {width} × {height}. "
            f"Minimum size is "
            f"{minimum_width} × {minimum_height}."
        )

    aspect_ratio = width / height

    if aspect_ratio < 0.25 or aspect_ratio > 4.0:
        raise ValueError(
            f"{image_name} has an unsuitable shape: "
            f"{width} × {height}."
        )

    return resize_large_image(prepared)


def clean_flatlay_image(
    image: Image.Image,
) -> Image.Image:
    if not REMBG_AVAILABLE or remove_background is None:
        raise RuntimeError(
            "Flat-lay cleanup cannot run because rembg "
            "is not installed or failed to load."
        )

    print(
        "Removing flat-lay garment background...",
        flush=True,
    )

    try:
        rgba = remove_background(
            image.convert("RGBA")
        )
    except Exception as error:
        raise RuntimeError(
            f"Flat-lay background removal failed: {error}"
        ) from error

    alpha = rgba.getchannel("A")
    bounding_box = alpha.getbbox()

    if bounding_box is None:
        raise RuntimeError(
            "The garment could not be detected during cleanup."
        )

    cropped = rgba.crop(bounding_box)

    padding = max(
        30,
        round(
            max(cropped.width, cropped.height) * 0.08
        ),
    )

    canvas_width = cropped.width + padding * 2
    canvas_height = cropped.height + padding * 2

    canvas = Image.new(
        "RGBA",
        (canvas_width, canvas_height),
        (255, 255, 255, 255),
    )

    canvas.alpha_composite(
        cropped,
        (padding, padding),
    )

    cleaned = canvas.convert("RGB")

    print(
        "Flat-lay garment cleaned to "
        f"{cleaned.width}x{cleaned.height}.",
        flush=True,
    )

    return resize_large_image(cleaned)


# ============================================================
# OUTPUT FILES
# ============================================================

def create_generation_folder() -> Path:
    timestamp = datetime.utcnow().strftime(
        "%Y%m%d_%H%M%S_%f"
    )

    folder = OUTPUT_DIR / f"generation_{timestamp}"

    folder.mkdir(
        parents=True,
        exist_ok=False,
    )

    return folder


def save_generation(
    images,
    generation_folder,
    metadata,
):
    saved_paths = []

    for index, image in enumerate(
        images,
        start=1,
    ):
        image_path = (
            generation_folder / f"result_{index}.png"
        )

        image.save(
            image_path,
            format="PNG",
            optimize=True,
        )

        saved_paths.append(image_path)

    metadata_path = generation_folder / "metadata.json"

    metadata_path.write_text(
        json.dumps(
            metadata,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    zip_path = generation_folder.with_suffix(".zip")

    with zipfile.ZipFile(
        zip_path,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
    ) as archive:
        for file_path in generation_folder.iterdir():
            archive.write(
                file_path,
                arcname=file_path.name,
            )

    return saved_paths, zip_path


# ============================================================
# GENERATION
# ============================================================

def generate_tryon(
    person_image,
    garment_image,
    category="tops",
    garment_photo_type="model",
    quality="Balanced",
    tryon_mode="Natural / Maskless",
    seed_mode="Random",
    clean_flatlay=False,
):
    validate_generation_options(
        category=category,
        garment_photo_type=garment_photo_type,
        quality=quality,
        tryon_mode=tryon_mode,
        seed_mode=seed_mode,
    )

    print("Preparing uploaded images...", flush=True)

    person_image = prepare_image(
        person_image,
        "Person image",
        MIN_PERSON_WIDTH,
        MIN_PERSON_HEIGHT,
    )

    garment_image = prepare_image(
        garment_image,
        "Garment image",
        MIN_GARMENT_WIDTH,
        MIN_GARMENT_HEIGHT,
    )

    if clean_flatlay:
        if garment_photo_type != "flat-lay":
            raise ValueError(
                "Flat-lay cleanup can only be used when "
                "garment_photo_type is flat-lay."
            )

        garment_image = clean_flatlay_image(
            garment_image
        )

    profile = QUALITY_PROFILES[quality]

    selected_seed = (
        42
        if seed_mode == "Fixed 42"
        else secrets.randbelow(MAX_SEED)
    )

    segmentation_free = (
        tryon_mode == "Natural / Maskless"
    )

    print("=" * 68, flush=True)
    print("SPOTC FASHN VTON GENERATION", flush=True)
    print(f"Quality: {quality}", flush=True)
    print(f"Category: {category}", flush=True)
    print(
        f"Garment photo type: {garment_photo_type}",
        flush=True,
    )
    print(f"Try-on mode: {tryon_mode}", flush=True)
    print(f"Seed: {selected_seed}", flush=True)
    print(f"Samples: {profile['samples']}", flush=True)
    print(
        f"Timesteps: {profile['timesteps']}",
        flush=True,
    )
    print(
        f"Guidance: {profile['guidance']}",
        flush=True,
    )
    print(
        f"Flat-lay cleanup: {clean_flatlay}",
        flush=True,
    )
    print("=" * 68, flush=True)

    generation_started = time.time()

    try:
        result = pipeline(
            person_image=person_image,
            garment_image=garment_image,
            category=category,
            garment_photo_type=garment_photo_type,
            num_samples=profile["samples"],
            num_timesteps=profile["timesteps"],
            guidance_scale=profile["guidance"],
            skip_cfg_last_n_steps=1,
            seed=selected_seed,
            segmentation_free=segmentation_free,
        )
    except Exception as error:
        traceback.print_exc()

        error_message = str(error)
        error_lower = error_message.lower()

        if (
            "out of memory" in error_lower
            or "cuda out of memory" in error_lower
        ):
            raise RuntimeError(
                "GPU memory is full. Restart the worker and "
                "use Balanced mode."
            ) from error

        raise RuntimeError(
            f"Try-on generation failed: {error_message}"
        ) from error

    generation_seconds = (
        time.time() - generation_started
    )

    print(
        f"Generation completed in "
        f"{generation_seconds:.1f} seconds.",
        flush=True,
    )

    generated_images = list(result.images or [])

    if not generated_images:
        raise RuntimeError(
            "FASHN VTON returned no generated images."
        )

    generation_folder = create_generation_folder()

    metadata = {
        "created_at": datetime.utcnow().isoformat() + "Z",
        "quality": quality,
        "category": category,
        "garment_photo_type": garment_photo_type,
        "tryon_mode": tryon_mode,
        "seed_mode": seed_mode,
        "base_seed": selected_seed,
        "sample_count": len(generated_images),
        "timesteps": profile["timesteps"],
        "guidance_scale": profile["guidance"],
        "segmentation_free": segmentation_free,
        "flatlay_cleanup": bool(clean_flatlay),
        "person_input_size": list(person_image.size),
        "garment_input_size": list(garment_image.size),
        "generation_seconds": round(
            generation_seconds,
            3,
        ),
    }

    saved_paths, zip_path = save_generation(
        generated_images,
        generation_folder,
        metadata,
    )

    gallery_results = []

    for index, image in enumerate(
        generated_images,
        start=1,
    ):
        gallery_results.append(
            (
                image,
                (
                    f"Result {index} · "
                    f"Seed {selected_seed} · "
                    f"{quality}"
                ),
            )
        )

    status = (
        f"Generated {len(generated_images)} image(s). "
        f"Seed: {selected_seed}. "
        f"Time: {generation_seconds:.1f} seconds."
    )

    print(status, flush=True)

    return (
        gallery_results,
        str(zip_path),
        status,
    )


# ============================================================
# UI HELPERS USED BY app.py
# ============================================================

def apply_garment_preset(preset):
    return GARMENT_PRESETS.get(
        preset,
        DEFAULT_VALUES,
    )


def clear_inputs():
    return (
        None,
        None,
        DEFAULT_PRESET,
        DEFAULT_VALUES[0],
        DEFAULT_VALUES[1],
        "Balanced",
        DEFAULT_VALUES[2],
        DEFAULT_VALUES[3],
        False,
        [],
        None,
        "Ready.",
    )
