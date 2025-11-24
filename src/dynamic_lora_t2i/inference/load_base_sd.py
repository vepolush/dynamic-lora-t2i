from __future__ import annotations
import json
import logging
import random
from datetime import datetime
from pathlib import Path
from typing import Any
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
    setup_logging,
)


logger = logging.getLogger(__name__)


def load_base_sd_pipeline_cpu_fp32(model_id: str = DEFAULT_BASE_MODEL_ID) -> StableDiffusionPipeline:
    """Loads Stable Diffusion 1.5 on the CPU in float32."""
    device = "cpu"
    torch_dtype = torch.float32

    logger.info("Loading Stable Diffusion 1.5 base model")
    logger.info("torch version: %s", torch.__version__)
    try:
        import diffusers

        logger.info("diffusers version: %s", diffusers.__version__)
    except Exception as exc:
        logger.warning("Cannot import diffusers to print version: %s", exc)

    logger.info("Model id: %s", model_id)
    logger.info("Device: %s, dtype: %s", device, torch_dtype)

    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        use_safetensors=True,
        safety_checker=None,
    )
    pipe.model_id = model_id

    pipe = pipe.to(device)
    pipe.enable_attention_slicing()

    logger.info("Pipeline loaded and moved to %s", device)
    return pipe


def _save_generation_metadata(
    metadata_path: Path,
    pipe: StableDiffusionPipeline,
    prompt: str,
    negative_prompt: str | None,
    num_inference_steps: int,
    guidance_scale: float,
    seed: int,
    width: int,
    height: int,
    extra: dict[str, Any] | None = None,
) -> None:
    """
    Save parameters of a single generation run to JSON.
    """
    metadata: dict[str, Any] = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "num_inference_steps": num_inference_steps,
        "guidance_scale": guidance_scale,
        "seed": seed,
        "width": width,
        "height": height,
        "model_id": getattr(pipe, "model_id", None),
    }

    try:
        metadata["device"] = str(pipe.device)
    except Exception:
        metadata["device"] = None

    try:
        metadata["dtype"] = str(pipe.unet.dtype)
    except Exception:
        metadata["dtype"] = None

    if extra:
        metadata.update(extra)

    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    logger.info("Saved generation metadata to %s", metadata_path)


def generate_image_cpu(
    pipe: StableDiffusionPipeline,
    prompt: str,
    negative_prompt: str | None = None,
    num_inference_steps: int = DEFAULT_NUM_INFERENCE_STEPS,
    guidance_scale: float = DEFAULT_GUIDANCE_SCALE,
    seed: int | None = None,
    metadata_path: Path | None = None,
):
    """
    Generates one image on CPU (float32).
    """
    ensure_project_directories()

    if seed is None:
        seed = random.randint(0, 2**32 - 1)

    generator = torch.Generator(device="cpu").manual_seed(seed)
    logger.info("Generating image on CPU...")
    logger.info("seed=%d", seed)
    logger.info("prompt=%r", prompt)
    if negative_prompt:
        logger.info("negative_prompt=%r", negative_prompt)
    logger.info("steps=%d, guidance_scale=%s", num_inference_steps, guidance_scale)

    with torch.no_grad():
        result = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=DEFAULT_IMAGE_WIDTH,
            height=DEFAULT_IMAGE_HEIGHT,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
        )

    image = result.images[0]

    if metadata_path is not None:
        _save_generation_metadata(
            metadata_path=metadata_path,
            pipe=pipe,
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            seed=seed,
            width=DEFAULT_IMAGE_WIDTH,
            height=DEFAULT_IMAGE_HEIGHT,
            extra=None,
        )

    return image


def generate_test_image_cpu(pipe: StableDiffusionPipeline) -> None:
    """
    Generates one test image on CPU and saves it in experiments/results/.
    """
    ensure_project_directories()

    prompt = "a cute cat reading a book, cinematic lighting, 4k, highly detailed"
    negative_prompt = "bad quality, blurriness"
    out_path = EXPERIMENT_RESULTS_DIR / "sd15_base_sanity_check.png"
    metadata_path = out_path.with_suffix(".json")

    image = generate_image_cpu(
        pipe=pipe,
        prompt=prompt,
        negative_prompt=negative_prompt,
        seed=None,
        metadata_path=metadata_path,
    )

    import numpy as np

    arr = np.array(image)
    logger.debug("image pixel range: min=%s, max=%s", arr.min(), arr.max())

    EXPERIMENT_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    image.save(out_path)
    logger.info("Test image saved to: %s", out_path)


def main() -> None:
    setup_logging()
    pipe = load_base_sd_pipeline_cpu_fp32()
    generate_test_image_cpu(pipe)


if __name__ == "__main__":
    main()
