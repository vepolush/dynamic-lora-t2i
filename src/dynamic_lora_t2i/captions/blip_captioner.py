# src/dynamic_lora_t2i/captions/blip_captioner.py

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Iterable, Optional

import torch
from PIL import Image
from transformers import BlipForConditionalGeneration, BlipProcessor

logger = logging.getLogger(__name__)

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp"}


try:
    from src.dynamic_lora_t2i.config import (
        get_device as _get_device,
        get_torch_dtype as _get_torch_dtype,
        TRANSFORMERS_CACHE as _TRANSFORMERS_CACHE,
        HF_HOME as _HF_HOME,
    )
except Exception:
    try:
        from src.dynamic_lora_t2i.utils.config import (  # type: ignore
            get_device as _get_device,
            get_torch_dtype as _get_torch_dtype,
            TRANSFORMERS_CACHE as _TRANSFORMERS_CACHE,
            HF_HOME as _HF_HOME,
        )
    except Exception:
        _get_device = None
        _get_torch_dtype = None
        _TRANSFORMERS_CACHE = None
        _HF_HOME = None


def _iter_images(entity_dir: Path) -> Iterable[Path]:
    paths = [p for p in entity_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
    for p in sorted(paths):
        yield p


def _device() -> str:
    if _get_device is not None:
        try:
            return _get_device()
        except Exception:
            pass

    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _dtype_for_device(device: str) -> torch.dtype:
    if _get_torch_dtype is not None:
        try:
            return _get_torch_dtype()
        except Exception:
            pass

    if device == "cuda":
        use_bf16 = os.getenv("DYNAMIC_LORA_T2I_USE_BF16", "0").strip().lower() in ("1", "true", "yes", "y")
        if use_bf16 and torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16
    if device == "mps":
        return torch.float16
    return torch.float32


def _cache_dir() -> Optional[str]:
    if _TRANSFORMERS_CACHE is not None:
        try:
            return str(_TRANSFORMERS_CACHE)
        except Exception:
            pass

    env_tf = os.getenv("TRANSFORMERS_CACHE")
    if env_tf:
        return env_tf

    if _HF_HOME is not None:
        try:
            return str(_HF_HOME)
        except Exception:
            pass

    env_hf = os.getenv("HF_HOME")
    if env_hf:
        return env_hf

    return None


def _ensure_token(caption: str, token: str) -> str:
    caption = (caption or "").strip()
    token = token.strip()

    if not caption:
        return f"photo of {token}"

    if token in caption:
        return caption

    return f"photo of {token}, {caption}"


class BlipCaptioner:
    def __init__(self, model_id: str = "Salesforce/blip-image-captioning-base"):
        self.model_id = model_id
        self.device_str = _device()
        self.dtype = _dtype_for_device(self.device_str)
        self.cache_dir = _cache_dir()

        if self.device_str == "cuda":
            try:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
            except Exception:
                pass

        self.processor = BlipProcessor.from_pretrained(model_id, cache_dir=self.cache_dir)

        model_kwargs = {"cache_dir": self.cache_dir, "use_safetensors": True}
        if self.device_str == "cuda":
            model_kwargs["torch_dtype"] = self.dtype
            model_kwargs["low_cpu_mem_usage"] = True

        self.model = BlipForConditionalGeneration.from_pretrained(model_id, **model_kwargs)
        self.model.eval()
        self.model.to(self.device_str)

        logger.info(
            "BLIP captioner loaded: %s on %s (dtype=%s, cache_dir=%s)",
            model_id,
            self.device_str,
            self.dtype,
            self.cache_dir,
        )

    @torch.no_grad()
    def caption(self, image: Image.Image, max_new_tokens: int = 40) -> str:
        image = image.convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt")

        inputs = {k: v.to(self.device_str) for k, v in inputs.items()}

        if self.device_str == "cuda":
            with torch.inference_mode():
                with torch.autocast("cuda", dtype=self.dtype):
                    out = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        else:
            out = self.model.generate(**inputs, max_new_tokens=max_new_tokens)

        text = self.processor.decode(out[0], skip_special_tokens=True)
        return text.strip()

    @torch.no_grad()
    def caption_batch(self, images: list[Image.Image], max_new_tokens: int = 40) -> list[str]:
        if not images:
            return []

        images = [im.convert("RGB") for im in images]
        inputs = self.processor(images=images, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device_str) for k, v in inputs.items()}

        if self.device_str == "cuda":
            with torch.inference_mode():
                with torch.autocast("cuda", dtype=self.dtype):
                    out = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        else:
            out = self.model.generate(**inputs, max_new_tokens=max_new_tokens)

        captions: list[str] = []
        for seq in out:
            captions.append(self.processor.decode(seq, skip_special_tokens=True).strip())
        return captions


def generate_entity_captions(
    entity_dir: Path,
    placeholder_token: str,
    *,
    overwrite: bool = False,
    caption_ext: str = ".txt",
    model_id: str = "Salesforce/blip-image-captioning-base",
    batch_size: int = 4,
    max_new_tokens: int = 40,
) -> int:
    entity_dir = Path(entity_dir).resolve()
    if not entity_dir.exists():
        raise FileNotFoundError(f"Entity dir not found: {entity_dir}")

    if not caption_ext.startswith("."):
        caption_ext = "." + caption_ext

    captioner = BlipCaptioner(model_id=model_id)

    written = 0
    batch_imgs: list[Image.Image] = []
    batch_targets: list[Path] = []

    def flush_batch() -> int:
        nonlocal batch_imgs, batch_targets
        if not batch_imgs:
            return 0

        try:
            caps = captioner.caption_batch(batch_imgs, max_new_tokens=max_new_tokens)
        finally:
            for im in batch_imgs:
                try:
                    im.close()
                except Exception:
                    pass

        local_written = 0
        for cap_path, auto_cap in zip(batch_targets, caps):
            final_cap = _ensure_token(auto_cap, placeholder_token)
            cap_path.write_text(final_cap + "\n", encoding="utf-8")
            local_written += 1

        batch_imgs = []
        batch_targets = []
        return local_written

    for img_path in _iter_images(entity_dir):
        cap_path = img_path.with_suffix(caption_ext)
        if cap_path.exists() and not overwrite:
            continue

        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            logger.warning("Failed to open image %s: %s", img_path, e)
            continue

        batch_imgs.append(image)
        batch_targets.append(cap_path)

        if len(batch_imgs) >= max(1, int(batch_size)):
            written += flush_batch()

    written += flush_batch()

    logger.info("Captions generated: %d (dir=%s, model=%s)", written, entity_dir, model_id)
    return written
