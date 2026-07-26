import sys
import time

print("=" * 68, flush=True)
print("SPOTC FASHN VTON STARTING", flush=True)
print(f"Python executable: {sys.executable}", flush=True)
print(f"Python version: {sys.version}", flush=True)
print("=" * 68, flush=True)

print("[1/7] Importing standard Python libraries...", flush=True)

import json
import secrets
import traceback
import zipfile
from datetime import datetime
from pathlib import Path

print("[2/7] Importing Gradio and Pillow...", flush=True)

import gradio as gr
from PIL import Image, ImageOps

print("[3/7] Importing FASHN VTON pipeline and PyTorch modules...", flush=True)

fashn_import_started = time.time()

from fashn_vton import TryOnPipeline

print(
    f"[3/7] FASHN VTON imports completed in "
    f"{time.time() - fashn_import_started:.1f} seconds.",
    flush=True,
)


# ============================================================
# PATHS AND LIMITS
# ============================================================

print("[4/7] Preparing application directories...", flush=True)

BASE_DIR = Path("/workspace/AIStudio/fashn-vton-1.5")
WEIGHTS_DIR = BASE_DIR / "weights"
OUTPUT_DIR = BASE_DIR / "outputs"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MIN_PERSON_WIDTH = 512
MIN_PERSON_HEIGHT = 512
MIN_GARMENT_WIDTH = 400
MIN_GARMENT_HEIGHT = 400
MAX_INPUT_SIDE = 2048
MAX_SEED = 2_147_483_000

print(f"Base directory: {BASE_DIR}", flush=True)
print(f"Weights directory: {WEIGHTS_DIR}", flush=True)
print(f"Output directory: {OUTPUT_DIR}", flush=True)

if not WEIGHTS_DIR.exists():
    raise RuntimeError(
        f"Weights directory was not found: {WEIGHTS_DIR}"
    )


# ============================================================
# OPTIONAL BACKGROUND REMOVAL
# ============================================================

print("[5/7] Checking optional flat-lay cleanup...", flush=True)

try:
    from rembg import remove as remove_background

    REMBG_AVAILABLE = True
    print(
        "Flat-lay background cleanup is available.",
        flush=True,
    )
except Exception as error:
    remove_background = None
    REMBG_AVAILABLE = False
    print(
        f"Flat-lay background cleanup unavailable: {error}",
        flush=True,
    )


# ============================================================
# LOAD PIPELINE
# ============================================================

print("[6/7] Loading FASHN VTON model pipeline...", flush=True)
print("This stage loads the model weights into GPU memory.", flush=True)

pipeline_load_started = time.time()

try:
    pipeline = TryOnPipeline(
        weights_dir=str(WEIGHTS_DIR),
    )
except Exception:
    print(
        "FASHN VTON pipeline failed to load.",
        flush=True,
    )
    traceback.print_exc()
    raise

print(
    f"FASHN VTON pipeline loaded successfully in "
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
        f"Resizing input from {width}x{height} "
        f"to {new_width}x{new_height}",
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
        raise gr.Error(
            f"Upload the {image_name.lower()}."
        )

    try:
        image = ImageOps.exif_transpose(image)
        image = image.convert("RGB")
    except Exception as error:
        raise gr.Error(
            f"Could not read the {image_name.lower()}: {error}"
        ) from error

    width, height = image.size

    if width < minimum_width or height < minimum_height:
        raise gr.Error(
            f"{image_name} is too small: {width} × {height}. "
            f"Minimum recommended size is "
            f"{minimum_width} × {minimum_height}."
        )

    aspect_ratio = width / height

    if aspect_ratio < 0.25 or aspect_ratio > 4.0:
        raise gr.Error(
            f"{image_name} has an unsuitable shape: "
            f"{width} × {height}."
        )

    return resize_large_image(image)


def clean_flatlay_image(image: Image.Image) -> Image.Image:
    if not REMBG_AVAILABLE:
        raise gr.Error(
            "Flat-lay cleanup could not start because rembg "
            "is unavailable."
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
        raise gr.Error(
            f"Flat-lay background removal failed: {error}"
        ) from error

    alpha = rgba.getchannel("A")
    bounding_box = alpha.getbbox()

    if bounding_box is None:
        raise gr.Error(
            "The garment could not be detected during cleanup."
        )

    cropped = rgba.crop(bounding_box)

    padding = max(
        30,
        round(max(cropped.width, cropped.height) * 0.08),
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
        f"Flat-lay garment cleaned and cropped to "
        f"{cleaned.width}x{cleaned.height}",
        flush=True,
    )

    return resize_large_image(cleaned)


# ============================================================
# FILE OUTPUT
# ============================================================

def create_generation_folder() -> Path:
    timestamp = datetime.now().strftime(
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

    metadata_path = (
        generation_folder / "metadata.json"
    )

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
    category,
    garment_photo_type,
    quality,
    tryon_mode,
    seed_mode,
    clean_flatlay,
):
    print(
        "Preparing uploaded images...",
        flush=True,
    )

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
            raise gr.Error(
                "Flat-lay cleanup can only be used when "
                "Garment Photo Type is flat-lay."
            )

        garment_image = clean_flatlay_image(
            garment_image
        )

    profile = QUALITY_PROFILES.get(
        quality,
        QUALITY_PROFILES["Balanced"],
    )

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
    print(
        f"Try-on mode: {tryon_mode}",
        flush=True,
    )
    print(f"Seed: {selected_seed}", flush=True)
    print(
        f"Samples: {profile['samples']}",
        flush=True,
    )
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
            raise gr.Error(
                "GPU memory is full. Restart the app and use "
                "Balanced mode, or reduce the number of samples."
            ) from error

        raise gr.Error(
            f"Try-on generation failed: {error_message}"
        ) from error

    print(
        f"Generation completed in "
        f"{time.time() - generation_started:.1f} seconds.",
        flush=True,
    )

    generated_images = list(result.images or [])

    if not generated_images:
        raise gr.Error(
            "FASHN VTON returned no generated images."
        )

    generation_folder = create_generation_folder()

    metadata = {
        "created_at": datetime.now().isoformat(),
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
        "flatlay_cleanup": clean_flatlay,
        "person_input_size": list(person_image.size),
        "garment_input_size": list(garment_image.size),
    }

    _, zip_path = save_generation(
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
        f"Saved in: {generation_folder}"
    )

    print(status, flush=True)

    return (
        gallery_results,
        str(zip_path),
        status,
    )


# ============================================================
# UI HELPERS
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


# ============================================================
# GRADIO UI
# ============================================================

print("[7/7] Building Gradio interface...", flush=True)

with gr.Blocks(
    title="SPOTC FASHN VTON AI Studio",
) as demo:
    gr.Markdown(
        """
# SPOTC FASHN VTON AI Studio

Upload a clear person image and garment image.

- **Balanced:** one economical result
- **High Quality:** four candidates
- **Premium Quality:** four candidates with additional inference
- Use **Flat-Lay Cleanup** only for garment-only flat-lay photographs
"""
    )

    garment_preset = gr.Dropdown(
        choices=list(GARMENT_PRESETS.keys()),
        value=DEFAULT_PRESET,
        label="Garment Preset",
    )

    with gr.Row():
        person = gr.Image(
            type="pil",
            label="Upload Person Image",
            height=470,
        )

        garment = gr.Image(
            type="pil",
            label="Upload Garment Image",
            height=470,
        )

    with gr.Row():
        category = gr.Dropdown(
            choices=[
                "tops",
                "bottoms",
                "one-pieces",
            ],
            value=DEFAULT_VALUES[0],
            label="Garment Category",
        )

        garment_photo_type = gr.Dropdown(
            choices=[
                "model",
                "flat-lay",
            ],
            value=DEFAULT_VALUES[1],
            label="Garment Photo Type",
        )

        quality = gr.Dropdown(
            choices=[
                "Fast",
                "Balanced",
                "High Quality",
                "Premium Quality",
            ],
            value="Balanced",
            label="Quality",
        )

    with gr.Row():
        tryon_mode = gr.Dropdown(
            choices=[
                "Natural / Maskless",
                "Structured / Parsed",
            ],
            value=DEFAULT_VALUES[2],
            label="Try-On Mode",
        )

        seed_mode = gr.Dropdown(
            choices=[
                "Fixed 42",
                "Random",
            ],
            value=DEFAULT_VALUES[3],
            label="Seed Mode",
        )

    clean_flatlay = gr.Checkbox(
        value=False,
        label="Clean Flat-Lay Background",
        info=(
            "Enable only for garment-only flat-lay photos. "
            "Do not use for garments worn by a model."
        ),
    )

    with gr.Row():
        generate_button = gr.Button(
            "Generate Try-On",
            variant="primary",
        )

        clear_button = gr.Button(
            "Clear",
        )

    status = gr.Textbox(
        value="Ready.",
        label="Status",
        interactive=False,
    )

    output = gr.Gallery(
        label="Generated Try-On Results",
        columns=2,
        rows=2,
        height=720,
        object_fit="contain",
        preview=True,
    )

    download_zip = gr.File(
        label="Download All Results",
    )

    garment_preset.change(
        fn=apply_garment_preset,
        inputs=garment_preset,
        outputs=[
            category,
            garment_photo_type,
            tryon_mode,
            seed_mode,
        ],
    )

    generate_button.click(
        fn=generate_tryon,
        inputs=[
            person,
            garment,
            category,
            garment_photo_type,
            quality,
            tryon_mode,
            seed_mode,
            clean_flatlay,
        ],
        outputs=[
            output,
            download_zip,
            status,
        ],
        api_name="generate",
    )

    clear_button.click(
        fn=clear_inputs,
        inputs=[],
        outputs=[
            person,
            garment,
            garment_preset,
            category,
            garment_photo_type,
            quality,
            tryon_mode,
            seed_mode,
            clean_flatlay,
            output,
            download_zip,
            status,
        ],
    )


# ============================================================
# START SERVER
# ============================================================

if __name__ == "__main__":
    print("=" * 68, flush=True)
    print("STARTING SPOTC TRY-ON HTTP SERVER", flush=True)
    print("Host: 0.0.0.0", flush=True)
    print("Port: 7865", flush=True)
    print("Public Gradio tunnel: disabled", flush=True)
    print("=" * 68, flush=True)

    try:
        demo.queue(
            default_concurrency_limit=1,
            max_size=2,
        ).launch(
            server_name="0.0.0.0",
            server_port=7865,
            share=False,
            show_error=True,
            prevent_thread_lock=False,
        )
    except Exception:
        print(
            "Gradio server failed to start.",
            flush=True,
        )
        traceback.print_exc()
        raise