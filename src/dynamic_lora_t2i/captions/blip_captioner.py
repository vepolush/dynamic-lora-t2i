# src/dynamic_lora_t2i/captions/blip_captioner.py

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable

import torch
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

logger = logging.getLogger(__name__)

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp"}


def _iter_images(entity_dir: Path) -> Iterable[Path]:
    for p in entity_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            yield p


def _device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


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
        self.device = _device()

        self.processor = BlipProcessor.from_pretrained(model_id)
        self.model = BlipForConditionalGeneration.from_pretrained(model_id)
        self.model.to(self.device)

        logger.info("BLIP captioner loaded: %s on %s", model_id, self.device)

    @torch.no_grad()
    def caption(self, image: Image.Image, max_new_tokens: int = 40) -> str:
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        out = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        text = self.processor.decode(out[0], skip_special_tokens=True)
        return text.strip()


def generate_entity_captions(
    entity_dir: Path,
    placeholder_token: str,
    *,
    overwrite: bool = False,
    caption_ext: str = ".txt",
    model_id: str = "Salesforce/blip-image-captioning-base",
) -> int:
    entity_dir = entity_dir.resolve()
    if not entity_dir.exists():
        raise FileNotFoundError(f"Entity dir not found: {entity_dir}")

    captioner = BlipCaptioner(model_id=model_id)

    written = 0
    for img_path in _iter_images(entity_dir):
        cap_path = img_path.with_suffix(caption_ext)

        if cap_path.exists() and not overwrite:
            continue

        image = Image.open(img_path).convert("RGB")
        auto_cap = captioner.caption(image)
        final_cap = _ensure_token(auto_cap, placeholder_token)

        cap_path.write_text(final_cap + "\n", encoding="utf-8")
        written += 1

    logger.info("Captions generated: %d (dir=%s)", written, entity_dir)
    return written
