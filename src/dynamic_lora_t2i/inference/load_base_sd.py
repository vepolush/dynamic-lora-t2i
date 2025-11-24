from __future__ import annotations
import torch
from diffusers import StableDiffusionPipeline
from src.dynamic_lora_t2i.config import (
    DEFAULT_BASE_MODEL_ID,
    DEFAULT_IMAGE_WIDTH,
    DEFAULT_IMAGE_HEIGHT,
    DEFAULT_NUM_INFERENCE_STEPS,
    DEFAULT_GUIDANCE_SCALE,
    EXPERIMENT_RESULTS_DIR,
    ensure_project_directories,
)


def load_base_sd_pipeline_cpu_fp32(model_id: str = DEFAULT_BASE_MODEL_ID) -> StableDiffusionPipeline:
    """
    Loads Stable Diffusion 1.5 on the CPU in float32
    """
    device = "cpu"
    torch_dtype = torch.float32

    print(f"[INFO] torch version: {torch.__version__}")
    try:
        import diffusers
        print(f"[INFO] diffusers version: {diffusers.__version__}")
    except Exception:
        print("[WARN] Cannot import diffusers to print version.")

    print(f"[INFO] Loading SD 1.5 base model: {model_id}")
    print(f"[INFO] Device: {device}, dtype: {torch_dtype}")

    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        use_safetensors=True,
        safety_checker=None,
    )

    pipe = pipe.to(device)
    pipe.enable_attention_slicing()

    return pipe


def generate_test_image_cpu(pipe: StableDiffusionPipeline) -> None:
    """
    Generates one test image per CPU and saves it in experiments/results/
    """
    ensure_project_directories()

    prompt = "a cute cat reading a book, cinematic lighting, 4k, highly detailed"

    print("[INFO] Generating test image on CPU...")

    with torch.no_grad():
        result = pipe(
            prompt=prompt,
            width=DEFAULT_IMAGE_WIDTH,
            height=DEFAULT_IMAGE_HEIGHT,
            num_inference_steps=DEFAULT_NUM_INFERENCE_STEPS,
            guidance_scale=DEFAULT_GUIDANCE_SCALE,
        )

    image = result.images[0]

    import numpy as np

    arr = np.array(image)
    print(f"[DEBUG] image pixel range: min={arr.min()}, max={arr.max()}")

    EXPERIMENT_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = EXPERIMENT_RESULTS_DIR / "sd15_base_sanity_check.png"
    image.save(out_path)

    print(f"[INFO] Test image saved to: {out_path}")


def main() -> None:
    pipe = load_base_sd_pipeline_cpu_fp32()
    generate_test_image_cpu(pipe)


if __name__ == "__main__":
    main()
