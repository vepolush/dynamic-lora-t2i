# src/dynamic_lora_t2i/inference/load_base_sd.py

from __future__ import annotations

import json
import logging
import os
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Protocol, runtime_checkable

import torch
from diffusers import AutoPipelineForText2Image, DiffusionPipeline, DPMSolverMultistepScheduler
from PIL import Image

try:
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

    try:
        from src.dynamic_lora_t2i.config import (
            get_device as _get_device,
            get_torch_dtype as _get_torch_dtype,
            DIFFUSERS_CACHE as _DIFFUSERS_CACHE,
            HF_HOME as _HF_HOME,
        )
    except Exception:
        _get_device = None
        _get_torch_dtype = None
        _DIFFUSERS_CACHE = None
        _HF_HOME = None

except Exception:
    from src.dynamic_lora_t2i.utils.config import (  # type: ignore
        DEFAULT_BASE_MODEL_ID,
        DEFAULT_IMAGE_WIDTH,
        DEFAULT_IMAGE_HEIGHT,
        DEFAULT_NUM_INFERENCE_STEPS,
        DEFAULT_GUIDANCE_SCALE,
        EXPERIMENT_RESULTS_DIR,
        ensure_project_directories,
        setup_logging,
    )

    try:
        from src.dynamic_lora_t2i.utils.config import (  # type: ignore
            get_device as _get_device,
            get_torch_dtype as _get_torch_dtype,
            DIFFUSERS_CACHE as _DIFFUSERS_CACHE,
            HF_HOME as _HF_HOME,
        )
    except Exception:  # pragma: no cover
        _get_device = None
        _get_torch_dtype = None
        _DIFFUSERS_CACHE = None
        _HF_HOME = None


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


def _get_cache_dir() -> Optional[str]:
    if _DIFFUSERS_CACHE is not None:
        try:
            return str(_DIFFUSERS_CACHE)
        except Exception:
            pass

    env_diff = os.getenv("DIFFUSERS_CACHE")
    if env_diff:
        return env_diff

    if _HF_HOME is not None:
        try:
            return str(_HF_HOME)
        except Exception:
            pass

    env_hf = os.getenv("HF_HOME")
    if env_hf:
        return env_hf

    return None


def _pick_device_and_dtype() -> tuple[str, torch.dtype]:
    if _get_device is not None:
        device = _get_device()
    else:
        if torch.cuda.is_available():
            device = "cuda"
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    if _get_torch_dtype is not None:
        dtype = _get_torch_dtype()
        return device, dtype

    if device == "cuda":
        use_bf16 = os.getenv("DYNAMIC_LORA_T2I_USE_BF16", "0").strip().lower() in ("1", "true", "yes", "y")
        if use_bf16 and torch.cuda.is_bf16_supported():
            return device, torch.bfloat16
        return device, torch.float16

    if device == "mps":
        return device, torch.float16

    return device, torch.float32


def _maybe_enable_xformers(pipe: DiffusionPipeline) -> None:
    try:
        pipe.enable_xformers_memory_efficient_attention()
        logger.info("Enabled xFormers memory efficient attention")
    except Exception as e:
        logger.warning("xFormers not enabled (%s). Falling back to attention slicing.", e)
        try:
            pipe.enable_attention_slicing("auto")
            logger.info("Enabled attention slicing (auto)")
        except Exception:
            logger.warning("Attention slicing not available for this pipeline")


def _apply_cuda_perf_tweaks() -> None:
    if not torch.cuda.is_available():
        return
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    except Exception:
        pass


def load_base_pipeline(
    model_id: str = DEFAULT_BASE_MODEL_ID,
    device: Optional[str] = None,
    dtype: Optional[torch.dtype] = None,
    enable_xformers: bool = True,
) -> DiffusionPipeline:
    ensure_project_directories()
    _apply_cuda_perf_tweaks()

    if device is None or dtype is None:
        picked_device, picked_dtype = _pick_device_and_dtype()
        device = device or picked_device
        dtype = dtype or picked_dtype

    cache_dir = _get_cache_dir()

    logger.info("Loading base model via AutoPipelineForText2Image")
    logger.info("Model id: %s", model_id)
    logger.info("Device: %s, dtype: %s", device, dtype)
    if device == "cuda":
        try:
            idx = torch.cuda.current_device()
            logger.info("CUDA device: %s", torch.cuda.get_device_name(idx))
            logger.info("CUDA capability: %s", torch.cuda.get_device_capability(idx))
        except Exception:
            pass
        logger.info("CUDA available memory (rough): %.2f GB", torch.cuda.get_device_properties(0).total_memory / 1e9)

    load_kwargs: dict[str, Any] = {
        "use_safetensors": True,
    }
    if cache_dir:
        load_kwargs["cache_dir"] = cache_dir

    try:
        pipe: DiffusionPipeline = AutoPipelineForText2Image.from_pretrained(
            model_id,
            dtype=dtype,
            **load_kwargs,
        )
    except TypeError:
        pipe = AutoPipelineForText2Image.from_pretrained(
            model_id,
            torch_dtype=dtype,
            **load_kwargs,
        )

    setattr(pipe, "model_id", model_id)

    if hasattr(pipe, "safety_checker"):
        try:
            setattr(pipe, "safety_checker", None)
        except Exception:
            pass
    if hasattr(pipe, "requires_safety_checker"):
        try:
            setattr(pipe, "requires_safety_checker", False)
        except Exception:
            pass

    pipe = pipe.to(device)

    try:
        pipe: DiffusionPipeline = AutoPipelineForText2Image.from_pretrained(
            model_id,
            dtype=dtype,
            **load_kwargs,
        )
    except TypeError:
        pipe = AutoPipelineForText2Image.from_pretrained(
            model_id,
            torch_dtype=dtype,
            **load_kwargs,
        )

    setattr(pipe, "model_id", model_id)

    try:
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            pipe.scheduler.config,
            use_karras_sigmas=True,
        )
        logger.info("Scheduler set: DPMSolverMultistepScheduler (karras sigmas)")
    except Exception as e:
        logger.warning("Could not set DPMSolverMultistepScheduler: %s", e)

    if hasattr(pipe, "safety_checker"):
        try:
            setattr(pipe, "safety_checker", None)
        except Exception:
            pass
    if hasattr(pipe, "requires_safety_checker"):
        try:
            setattr(pipe, "requires_safety_checker", False)
        except Exception:
            pass

    pipe = pipe.to(device)

    try:
        pipe.enable_vae_slicing()
    except Exception:
        pass

    if device == "cuda" and enable_xformers:
        _maybe_enable_xformers(pipe)
    else:
        try:
            pipe.enable_attention_slicing("auto")
        except Exception:
            pass

    if device == "mps" and hasattr(pipe, "vae") and getattr(pipe, "vae") is not None:
        try:
            pipe.vae.to(dtype=torch.float32)
            logger.info("MPS: cast VAE to float32 for stability")
        except Exception:
            pass

    try:
        setattr(pipe, "_dynamic_lora_device_str", str(device))
    except Exception:
        pass

    logger.info("Pipeline loaded and moved to %s", device)
    return pipe


def load_base_pipeline_cpu_fp32(model_id: str = DEFAULT_BASE_MODEL_ID) -> DiffusionPipeline:
    return load_base_pipeline(model_id=model_id)


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
    device_str = str(device_obj) if device_obj is not None else getattr(pipe, "_dynamic_lora_device_str", None)

    unet = getattr(pipe, "unet", None)
    unet_dtype = getattr(unet, "dtype", None) if unet is not None else None
    dtype_str = str(unet_dtype) if unet_dtype is not None else None

    meta: dict[str, Any] = {
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

    if torch.cuda.is_available():
        try:
            idx = torch.cuda.current_device()
            meta["cuda_device_name"] = torch.cuda.get_device_name(idx)
            meta["cuda_capability"] = list(torch.cuda.get_device_capability(idx))
        except Exception:
            pass

    if extra:
        meta.update(extra)

    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    logger.info("Saved generation metadata to %s", metadata_path)


def generate_image(
    pipe: Text2ImagePipeline,
    prompt: str,
    negative_prompt: str | None = None,
    num_inference_steps: int = DEFAULT_NUM_INFERENCE_STEPS,
    guidance_scale: float = DEFAULT_GUIDANCE_SCALE,
    seed: int | None = None,
    width: int = DEFAULT_IMAGE_WIDTH,
    height: int = DEFAULT_IMAGE_HEIGHT,
    metadata_path: Path | None = None,
) -> Image.Image:
    ensure_project_directories()

    if seed is None:
        seed = random.randint(0, 2**32 - 1)

    device_str = "cpu"
    try:
        device_str = str(pipe.device)
    except Exception:
        device_str = getattr(pipe, "_dynamic_lora_device_str", "cpu") or "cpu"

    generator = torch.Generator(device=device_str).manual_seed(seed)

    logger.info("Generating image...")
    logger.info("device=%s seed=%d", device_str, seed)
    logger.info("prompt=%r", prompt)
    if negative_prompt:
        logger.info("negative_prompt=%r", negative_prompt)
    logger.info("steps=%d, guidance_scale=%s, width=%d, height=%d", num_inference_steps, guidance_scale, width, height)

    autocast_dtype = None
    try:
        unet = getattr(pipe, "unet", None)
        autocast_dtype = getattr(unet, "dtype", None) if unet is not None else None
    except Exception:
        autocast_dtype = None

    if device_str.startswith("cuda"):
        with torch.inference_mode():
            with torch.autocast("cuda", dtype=autocast_dtype or torch.float16):
                result = pipe(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    width=width,
                    height=height,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    generator=generator,
                )
    elif device_str.startswith("mps"):
        with torch.inference_mode():
            result = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                width=width,
                height=height,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=generator,
            )
    else:
        with torch.no_grad():
            result = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                width=width,
                height=height,
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
            width=width,
            height=height,
        )

    return image


def generate_image_cpu(
    pipe: Text2ImagePipeline,
    prompt: str,
    negative_prompt: str | None = None,
    num_inference_steps: int = DEFAULT_NUM_INFERENCE_STEPS,
    guidance_scale: float = DEFAULT_GUIDANCE_SCALE,
    seed: int | None = None,
    metadata_path: Path | None = None,
) -> Image.Image:
    return generate_image(
        pipe=pipe,
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        seed=seed,
        width=DEFAULT_IMAGE_WIDTH,
        height=DEFAULT_IMAGE_HEIGHT,
        metadata_path=metadata_path,
    )


def generate_test_image(pipe: Text2ImagePipeline) -> None:
    ensure_project_directories()

    prompt = "portrait photo of a 20-year-old man with visible tattoos, sharp focus, high detail, realistic skin texture, studio lighting, 35mm photo, natural colors, ultra detailed"
    negative_prompt = "worst quality, low quality, blurry, out of focus, jpeg artifacts, watermark, text, logo, deformed, disfigured, bad anatomy, bad hands, extra fingers, missing fingers, mutated hands, cross-eye, poorly drawn face"
    out_path = EXPERIMENT_RESULTS_DIR / "test_picture.png"
    metadata_path = out_path.with_suffix(".json")

    image = generate_image(
        pipe=pipe,
        prompt=prompt,
        negative_prompt=negative_prompt,
        seed=None,
        width=DEFAULT_IMAGE_WIDTH,
        height=DEFAULT_IMAGE_HEIGHT,
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


def generate_test_image_cpu(pipe: Text2ImagePipeline) -> None:
    return generate_test_image(pipe)


def main() -> None:
    setup_logging()
    pipe = load_base_pipeline()
    generate_test_image(pipe)


if __name__ == "__main__":
    main()
