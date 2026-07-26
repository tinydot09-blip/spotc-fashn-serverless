import sys
import traceback

import gradio as gr

from core import (
    DEFAULT_PRESET,
    DEFAULT_VALUES,
    GARMENT_PRESETS,
    apply_garment_preset,
    clear_inputs,
    generate_tryon,
)


print("=" * 68, flush=True)
print("SPOTC FASHN VTON GRADIO APP STARTING", flush=True)
print(f"Python executable: {sys.executable}", flush=True)
print(f"Python version: {sys.version}", flush=True)
print("=" * 68, flush=True)


# ============================================================
# GRADIO UI
# ============================================================

print("Building Gradio interface...", flush=True)

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
# START GRADIO SERVER
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
