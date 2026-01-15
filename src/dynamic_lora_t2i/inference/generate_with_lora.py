# src/dynamic_lora_t2i/inference/generate_with_lora.py

from __future__ import annotations

import json
import logging
import random
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Tuple, Union

import torch
from PIL import Image

from src.dynamic_lora_t2i.entities.entity_config import EntityConfig
from src.dynamic_lora_t2i.inference.loader import attach_lora, load_base_pipeline, unload_loras
from src.dynamic_lora_t2i.utils.entity_zip import sanitize_entity_name

try:
    from src.dynamic_lora_t2i.config import (
        DEFAULT_GUIDANCE_SCALE,
        DEFAULT_IMAGE_HEIGHT,
        DEFAULT_IMAGE_WIDTH,
        DEFAULT_NUM_INFERENCE_STEPS,
        ENTITIES_DIR,
        EXPERIMENT_RESULTS_DIR,
        ensure_project_directories,
    )
except Exception:
    from src.dynamic_lora_t2i.utils.config import (  # type: ignore
        DEFAULT_GUIDANCE_SCALE,
        DEFAULT_IMAGE_HEIGHT,
        DEFAULT_IMAGE_WIDTH,
        DEFAULT_NUM_INFERENCE_STEPS,
        ENTITIES_DIR,
        EXPERIMENT_RESULTS_DIR,
        ensure_project_directories,
    )

logger = logging.getLogger(__name__)


def _utc_now_z() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _ensure_placeholder_in_prompt(prompt: str, token: str) -> str:
    prompt = (prompt or "").strip()
    token = (token or "").strip()
    if not token:
        return prompt

    if not prompt:
        return f"photo of {token}"

    if token in prompt:
        return prompt

    return f"photo of {token}, {prompt}"


def _maybe_append_class_prompt(prompt: str, class_prompt: str) -> str:
    p = (prompt or "").strip()
    cp = (class_prompt or "").strip()
    if not cp:
        return p

    if cp.lower() in p.lower():
        return p

    if not p:
        return cp

    return f"{p}, {cp}"


def _find_latest_adapter_dir(adapters_dir: Path) -> Path:
    adapters_dir = adapters_dir.resolve()
    if not adapters_dir.exists():
        raise FileNotFoundError(f"Adapters dir not found: {adapters_dir}")

    subdirs = [p for p in adapters_dir.iterdir() if p.is_dir()]
    if not subdirs:
        return adapters_dir

    subdirs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return subdirs[0]


def _default_out_paths(entity_name: str) -> tuple[Path, Path]:
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SZ")
    base = EXPERIMENT_RESULTS_DIR / f"{entity_name}__{ts}"
    return base.with_suffix(".png"), base.with_suffix(".json")


@dataclass
class InferenceParams:
    negative_prompt: Optional[str] = None
    num_inference_steps: int = DEFAULT_NUM_INFERENCE_STEPS
    guidance_scale: float = DEFAULT_GUIDANCE_SCALE
    seed: Optional[int] = None

    width: int = DEFAULT_IMAGE_WIDTH
    height: int = DEFAULT_IMAGE_HEIGHT

    lora_scale: float = 1.0
    run_name: Optional[str] = None
    adapter_source: Optional[Union[str, Path]] = None

    auto_add_placeholder_token: bool = True
    auto_add_class_prompt: bool = True

    save_outputs: bool = True
    out_image_path: Optional[Union[str, Path]] = None
    out_metadata_path: Optional[Union[str, Path]] = None

    unload_before_attach: bool = True


def generate_with_lora(
    prompt: str,
    entity_id: str,
    params: Union[InferenceParams, dict[str, Any], None] = None,
    *,
    pipe: Any = None,
) -> Tuple[Image.Image, dict[str, Any], Any]:
    ensure_project_directories()

    safe_entity = sanitize_entity_name(entity_id)

    if params is None:
        p = InferenceParams()
    elif isinstance(params, dict):
        p = InferenceParams(**params)
    else:
        p = params

    if p.seed is None:
        p.seed = random.randint(0, 2**32 - 1)

    if p.width <= 0 or p.height <= 0:
        raise ValueError("width/height must be positive")
    if p.num_inference_steps <= 0:
        raise ValueError("num_inference_steps must be positive")

    entity_cfg_path = (ENTITIES_DIR / safe_entity / "entity_config.json").resolve()
    if not entity_cfg_path.exists():
        raise FileNotFoundError(
            f"entity_config.json not found for entity '{safe_entity}'. Expected: {entity_cfg_path}"
        )

    ent = EntityConfig.load(entity_cfg_path, resolve_paths=True, validate=True)
    assert ent.adapters_dir is not None

    placeholder = str(ent.placeholder_token)
    base_model_id = str(ent.model.base_model_id)

    class_prompt = str((ent.meta or {}).get("class_prompt", "")).strip()

    if p.adapter_source is not None:
        adapter_source = Path(p.adapter_source).expanduser().resolve()
    elif p.run_name is not None:
        adapter_source = (Path(ent.adapters_dir) / str(p.run_name)).resolve()
    else:
        adapter_source = _find_latest_adapter_dir(Path(ent.adapters_dir))

    if not adapter_source.exists():
        raise FileNotFoundError(f"LoRA adapter source not found: {adapter_source}")

    if pipe is None:
        pipe = load_base_pipeline(model_id=base_model_id)

    if p.unload_before_attach:
        try:
            unload_loras(pipe)
        except Exception as e:
            logger.warning("unload_loras() failed (ignored): %s", e)

    adapter_name = safe_entity
    attach_lora(pipe, adapter_source, adapter_name=adapter_name, scale=float(p.lora_scale))

    final_prompt = prompt

    if p.auto_add_placeholder_token:
        final_prompt = _ensure_placeholder_in_prompt(final_prompt, placeholder)

    if p.auto_add_class_prompt:
        final_prompt = _maybe_append_class_prompt(final_prompt, class_prompt)

    gen = torch.Generator(device=str(getattr(pipe, "device", "cpu"))).manual_seed(int(p.seed))

    with torch.no_grad():
        res = pipe(
            prompt=str(final_prompt),
            negative_prompt=p.negative_prompt,
            width=int(p.width),
            height=int(p.height),
            num_inference_steps=int(p.num_inference_steps),
            guidance_scale=float(p.guidance_scale),
            generator=gen,
        )

    img: Image.Image = res.images[0]

    meta: dict[str, Any] = {
        "timestamp": _utc_now_z(),
        "entity_id": safe_entity,
        "placeholder_token": placeholder,
        "class_prompt": class_prompt,
        "base_model_id": base_model_id,
        "adapter_name": adapter_name,
        "adapter_source": str(adapter_source),
        "lora_scale": float(p.lora_scale),
        "prompt_input": prompt,
        "prompt_final": final_prompt,
        "negative_prompt": p.negative_prompt,
        "num_inference_steps": int(p.num_inference_steps),
        "guidance_scale": float(p.guidance_scale),
        "seed": int(p.seed),
        "width": int(p.width),
        "height": int(p.height),
        "device": str(getattr(pipe, "device", None)),
    }

    if p.save_outputs:
        if p.out_image_path is None or p.out_metadata_path is None:
            out_img, out_json = _default_out_paths(safe_entity)
        else:
            out_img = Path(p.out_image_path).expanduser().resolve()
            out_json = Path(p.out_metadata_path).expanduser().resolve()

        out_img.parent.mkdir(parents=True, exist_ok=True)
        img.save(out_img)

        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

        logger.info("Saved image: %s", out_img)
        logger.info("Saved metadata: %s", out_json)

        meta["saved_image_path"] = str(out_img)
        meta["saved_metadata_path"] = str(out_json)

    return img, meta, pipe


def main() -> None:
    import argparse

    from src.dynamic_lora_t2i.config import setup_logging

    setup_logging()

    ap = argparse.ArgumentParser()
    ap.add_argument("--entity", required=True)
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--negative", default=None)
    ap.add_argument("--steps", type=int, default=DEFAULT_NUM_INFERENCE_STEPS)
    ap.add_argument("--guidance", type=float, default=DEFAULT_GUIDANCE_SCALE)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--run", default=None, help="Optional run_name inside adapters_dir")
    ap.add_argument("--adapter", default=None, help="Optional explicit adapter dir or weights file")
    args = ap.parse_args()

    params = InferenceParams(
        negative_prompt=args.negative,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance,
        seed=args.seed,
        lora_scale=args.scale,
        run_name=args.run,
        adapter_source=args.adapter,
        save_outputs=True,
        auto_add_placeholder_token=True,
        auto_add_class_prompt=True,
    )

    _, meta, _ = generate_with_lora(args.prompt, args.entity, params=params)
    print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
