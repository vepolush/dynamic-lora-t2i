# src/dynamic_lora_t2i/inference/load_base_sd.py

from __future__ import annotations

import json
import logging
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Protocol, runtime_checkable

import torch
from diffusers import AutoPipelineForText2Image, DiffusionPipeline
from PIL import Image

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


@runtime_checkable
class Text2ImagePipeline(Protocol):
    device: torch.device

    def __call__(
        self,
        *,
        prompt: str,
        negative_prompt: Optional[str] = None,
        width: int,
        height: int,
        num_inference_steps: int,
        guidance_scale: float,
        generator: torch.Generator,
    ) -> Any: ...


def _pick_device_and_dtype() -> tuple[str, torch.dtype]:
    if torch.backends.mps.is_available():
        return "mps", torch.float16
    return "cpu", torch.float32


def load_base_pipeline_cpu_fp32(model_id: str = DEFAULT_BASE_MODEL_ID) -> DiffusionPipeline:
    device, dtype = _pick_device_and_dtype()

    logger.info("Loading base model via AutoPipelineForText2Image")
    logger.info("Model id: %s", model_id)
    logger.info("Device: %s, dtype: %s", device, dtype)

    try:
        pipe: DiffusionPipeline = AutoPipelineForText2Image.from_pretrained(
            model_id,
            dtype=dtype,
            use_safetensors=True,
        )
    except TypeError:
        pipe = AutoPipelineForText2Image.from_pretrained(
            model_id,
            torch_dtype=dtype,
            use_safetensors=True,
        )

    setattr(pipe, "model_id", model_id)

    if hasattr(pipe, "safety_checker"):
        setattr(pipe, "safety_checker", None)
    if hasattr(pipe, "requires_safety_checker"):
        setattr(pipe, "requires_safety_checker", False)

    pipe = pipe.to(device)
    pipe.enable_attention_slicing()

    if device == "mps" and hasattr(pipe, "vae") and getattr(pipe, "vae") is not None:
        pipe.vae.to(dtype=torch.float32)

    logger.info("Pipeline loaded and moved to %s", device)
    return pipe


def _save_generation_metadata(
    metadata_path: Path,
    pipe: Text2ImagePipeline,
    prompt: str,
    negative_prompt: str | None,
    num_inference_steps: int,
    guidance_scale: float,
    seed: int,
    width: int,
    height: int,
    extra: dict[str, Any] | None = None,
) -> None:

    timestamp = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

    model_id = getattr(pipe, "model_id", None)

    device_obj = getattr(pipe, "device", None)
    device_str = str(device_obj) if device_obj is not None else None

    unet = getattr(pipe, "unet", None)
    unet_dtype = getattr(unet, "dtype", None) if unet is not None else None
    dtype_str = str(unet_dtype) if unet_dtype is not None else None

    metadata: dict[str, Any] = {
        "timestamp": timestamp,
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "num_inference_steps": num_inference_steps,
        "guidance_scale": guidance_scale,
        "seed": seed,
        "width": width,
        "height": height,
        "model_id": model_id,
        "device": device_str,
        "dtype": dtype_str,
    }

    if extra:
        metadata.update(extra)

    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    logger.info("Saved generation metadata to %s", metadata_path)


def generate_image_cpu(
    pipe: Text2ImagePipeline,
    prompt: str,
    negative_prompt: str | None = None,
    num_inference_steps: int = DEFAULT_NUM_INFERENCE_STEPS,
    guidance_scale: float = DEFAULT_GUIDANCE_SCALE,
    seed: int | None = None,
    metadata_path: Path | None = None,
) -> Image.Image:
    ensure_project_directories()

    if seed is None:
        seed = random.randint(0, 2**32 - 1)

    generator = torch.Generator(device=str(pipe.device)).manual_seed(seed)

    logger.info("Generating image...")
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

    image: Image.Image = result.images[0]

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
        )

    return image


def generate_test_image_cpu(pipe: Text2ImagePipeline) -> None:
    ensure_project_directories()

    prompt = "a red race car"
    negative_prompt = "worst quality, normal quality, low quality, low res, blurry, distortion, text, watermark, logo, banner, extra digits, cropped, jpeg artifacts, signature, username, error, sketch, duplicate, ugly, monochrome, horror, geometry, mutation, disgusting, bad anatomy, bad proportions, bad quality, deformed, disconnected limbs, out of frame, out of focus, dehydrated, disfigured, extra arms, extra limbs, extra hands, fused fingers, gross proportions, long neck, jpeg, malformed limbs, mutated, mutated hands, mutated limbs, missing arms, missing fingers, picture frame, poorly drawn hands, poorly drawn face, collage, pixel, pixelated, grainy, color aberration, amputee, autograph, bad illustration, beyond the borders, blank background, body out of frame, boring background, branding, cut off, dismembered, disproportioned, distorted, draft, duplicated features, extra fingers, extra legs, fault, flaw, grains, hazy, identifying mark, improper scale, incorrect physiology, incorrect ratio, indistinct, kitsch, low resolution, macabre, malformed, mark, misshapen, missing hands, missing legs, mistake, morbid, mutilated, off-screen, outside the picture, poorly drawn feet, printed words, render, repellent, replicate, reproduce, revolting dimensions, script, shortened, sign, split image, squint, storyboard, tiling, trimmed, unfocused, unattractive, unnatural pose, unreal engine, unsightly, written language"
    out_path = EXPERIMENT_RESULTS_DIR / "test_picture.png"
    metadata_path = out_path.with_suffix(".json")

    image = generate_image_cpu(
        pipe=pipe,
        prompt=prompt,
        negative_prompt=negative_prompt,
        seed=1,
        metadata_path=metadata_path,
    )

    try:
        import numpy as np

        arr = np.array(image)
        logger.debug("image pixel range: min=%s, max=%s", arr.min(), arr.max())
    except ImportError:
        logger.debug("numpy is not installed; skipping pixel range debug")

    EXPERIMENT_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    image.save(out_path)
    logger.info("Test image saved to: %s", out_path)


def main() -> None:
    setup_logging()
    pipe = load_base_pipeline_cpu_fp32()
    generate_test_image_cpu(pipe)


if __name__ == "__main__":
    main()
