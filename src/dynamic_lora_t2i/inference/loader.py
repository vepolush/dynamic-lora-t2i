# src/dynamic_lora_t2i/inference/loader.py

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

import torch

logger = logging.getLogger(__name__)


try:
    from src.dynamic_lora_t2i.config import (
        DEFAULT_BASE_MODEL_ID,
        ensure_project_directories,
        get_device,
        get_torch_dtype,
    )
except Exception:
    from src.dynamic_lora_t2i.utils.config import (  # type: ignore
        DEFAULT_BASE_MODEL_ID,
        ensure_project_directories,
        get_device,
        get_torch_dtype,
    )


@dataclass(frozen=True)
class LoRASpec:
    source: Path
    name: str
    scale: float = 1.0


_KNOWN_LORA_WEIGHT_FILES = (
    "pytorch_lora_weights.safetensors",
    "unet_lora.safetensors",
    "adapter_model.safetensors",
    "pytorch_lora_weights.bin",
    "adapter_model.bin",
    "unet_lora.pt",
)


def _pick_device_dtype(
    device: Optional[str] = None,
    dtype: Optional[torch.dtype] = None,
) -> tuple[str, torch.dtype]:
    dev = device or get_device()
    dt = dtype or get_torch_dtype()
    return dev, dt


def load_base_pipeline(
    model_id: str = DEFAULT_BASE_MODEL_ID,
    *,
    device: Optional[str] = None,
    dtype: Optional[torch.dtype] = None,
    enable_attention_slicing: bool = True,
) -> Any:
    ensure_project_directories()
    dev, dt = _pick_device_dtype(device=device, dtype=dtype)

    logger.info("Loading base pipeline")
    logger.info("model_id=%s", model_id)
    logger.info("device=%s dtype=%s", dev, dt)

    try:
        from diffusers import AutoPipelineForText2Image
    except ImportError as e:
        raise ImportError("diffusers is required. Install: pip install diffusers") from e

    try:
        pipe = AutoPipelineForText2Image.from_pretrained(
            model_id,
            dtype=dt,
            use_safetensors=True,
        )
    except TypeError:
        pipe = AutoPipelineForText2Image.from_pretrained(
            model_id,
            torch_dtype=dt,
            use_safetensors=True,
        )

    setattr(pipe, "model_id", model_id)

    if hasattr(pipe, "safety_checker"):
        setattr(pipe, "safety_checker", None)
    if hasattr(pipe, "requires_safety_checker"):
        setattr(pipe, "requires_safety_checker", False)

    pipe = pipe.to(dev)

    if enable_attention_slicing:
        try:
            pipe.enable_attention_slicing()
        except Exception:
            pass

    if dev == "mps" and hasattr(pipe, "vae") and getattr(pipe, "vae") is not None:
        try:
            pipe.vae.to(dtype=torch.float32)
        except Exception:
            pass

    logger.info("Base pipeline loaded OK")
    return pipe


def _discover_lora_dir_and_weight_name(source: Path) -> tuple[Path, Optional[str]]:
    source = source.expanduser().resolve()

    if source.is_file():
        return source.parent, source.name

    if not source.exists():
        raise FileNotFoundError(f"LoRA source not found: {source}")

    if not source.is_dir():
        raise ValueError(f"LoRA source must be a file or directory: {source}")

    for fname in _KNOWN_LORA_WEIGHT_FILES:
        cand = source / fname
        if cand.exists() and cand.is_file():
            return source, fname

    safes = sorted([p.name for p in source.glob("*.safetensors") if p.is_file()])
    if len(safes) == 1:
        return source, safes[0]

    if len(safes) > 1:
        for preferred in ("pytorch_lora_weights.safetensors", "adapter_model.safetensors", "unet_lora.safetensors"):
            if preferred in safes:
                return source, preferred
        raise ValueError(f"Multiple *.safetensors found in {source}. Specify a file directly.")

    bins = sorted([p.name for p in source.glob("*.bin") if p.is_file()])
    if len(bins) == 1:
        return source, bins[0]

    pts = sorted([p.name for p in source.glob("*.pt") if p.is_file()])
    if len(pts) == 1:
        return source, pts[0]

    raise FileNotFoundError(
        f"Could not find LoRA weights in {source}. Expected one of: {list(_KNOWN_LORA_WEIGHT_FILES)} or *.safetensors"
    )


def attach_lora(
    pipe: Any,
    lora_source: Path,
    *,
    adapter_name: str,
    scale: float = 1.0,
) -> str:
    from pathlib import Path

    lora_dir, weight_name = _discover_lora_dir_and_weight_name(lora_source)

    if not hasattr(pipe, "load_lora_weights"):
        raise RuntimeError("This pipeline does not support LoRA loading (no load_lora_weights).")

    if float(scale) <= 0.0:
        logger.info(
            "LoRA scale <= 0.0, skipping LoRA load: name=%s source=%s weight_name=%s scale=%s",
            adapter_name,
            lora_dir,
            weight_name,
            scale,
        )
        return adapter_name

    try:
        from src.dynamic_lora_t2i.utils.lora_io import ensure_compatible_lora_weights

        desired = weight_name or "pytorch_lora_weights.safetensors"
        weight_name = ensure_compatible_lora_weights(Path(lora_dir), desired)
    except Exception as e:
        logger.warning("LoRA auto-fix skipped/failed (%s). Proceeding with original weights.", repr(e))

    logger.info(
        "Loading LoRA: name=%s source=%s weight_name=%s scale=%s",
        adapter_name,
        lora_dir,
        weight_name,
        scale,
    )

    try:
        if weight_name is not None:
            pipe.load_lora_weights(str(lora_dir), weight_name=weight_name, adapter_name=adapter_name)
        else:
            pipe.load_lora_weights(str(lora_dir), adapter_name=adapter_name)
    except TypeError:
        if weight_name is not None:
            pipe.load_lora_weights(str(lora_dir), weight_name=weight_name)
        else:
            pipe.load_lora_weights(str(lora_dir))

    scale_f = float(scale)
    try:
        pipe.set_adapters([adapter_name], adapter_weights=[scale_f])
    except TypeError:
        try:
            pipe.set_adapters(adapter_name)
        except Exception:
            pass
        pipe.set_adapters([adapter_name], adapter_weights=[scale_f])

    _try_set_adapter_scale(pipe, adapter_name=adapter_name, scale=scale_f)

    return adapter_name


def _try_set_adapter_scale(pipe: Any, *, adapter_name: str, scale: float) -> None:
    if scale is None:
        return

    if hasattr(pipe, "set_adapters"):
        try:
            pipe.set_adapters(adapter_name, adapter_weights=scale)
            return
        except Exception:
            pass
        try:
            pipe.set_adapters([adapter_name], adapter_weights=[float(scale)])
            return
        except Exception:
            pass

    if hasattr(pipe, "fuse_lora"):
        try:
            pipe.fuse_lora(lora_scale=float(scale))
            logger.warning("set_adapters() not available; used fuse_lora(). This is less dynamic to unload.")
            return
        except Exception:
            pass

    logger.debug("Could not set adapter scale (pipeline has no compatible adapters API).")


def attach_loras(pipe: Any, loras: Sequence[LoRASpec]) -> list[str]:
    names: list[str] = []
    for spec in loras:
        names.append(attach_lora(pipe, spec.source, adapter_name=spec.name, scale=spec.scale))

    if hasattr(pipe, "set_adapters") and len(names) > 1:
        try:
            pipe.set_adapters(names, adapter_weights=[float(s.scale) for s in loras])
        except Exception:
            pass

    return names


def list_lora_adapters(pipe: Any) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for attr in ("get_list_adapters", "get_active_adapters", "active_adapters", "adapters"):
        if hasattr(pipe, attr):
            try:
                v = getattr(pipe, attr)
                out[attr] = v() if callable(v) else v
            except Exception:
                out[attr] = "<error>"
    return out


def unload_loras(pipe: Any, *, adapter_names: Optional[Iterable[str]] = None) -> None:
    if adapter_names is None:
        if hasattr(pipe, "unload_lora_weights"):
            try:
                pipe.unload_lora_weights()
                return
            except Exception:
                pass
        adapter_names = []
        info = list_lora_adapters(pipe)
        for v in info.values():
            if isinstance(v, (list, tuple)):
                adapter_names.extend([str(x) for x in v])

    names = list(adapter_names or [])
    if not names:
        return

    if hasattr(pipe, "delete_adapters"):
        try:
            pipe.delete_adapters(names)
            return
        except Exception:
            pass
        try:
            for n in names:
                pipe.delete_adapters(n)
            return
        except Exception:
            pass

    if hasattr(pipe, "unload_lora_weights"):
        try:
            pipe.unload_lora_weights()
        except Exception:
            pass


def load_pipeline_with_loras(
    *,
    model_id: str = DEFAULT_BASE_MODEL_ID,
    loras: Optional[Sequence[LoRASpec]] = None,
    device: Optional[str] = None,
    dtype: Optional[torch.dtype] = None,
) -> Any:
    pipe = load_base_pipeline(model_id=model_id, device=device, dtype=dtype)
    if loras:
        attach_loras(pipe, loras)
    return pipe


def main() -> None:
    import argparse

    try:
        from src.dynamic_lora_t2i.config import setup_logging  # type: ignore
    except Exception:
        from src.dynamic_lora_t2i.utils.config import setup_logging  # type: ignore

    setup_logging()

    p = argparse.ArgumentParser(description="Load base pipeline and attach LoRA adapter(s).")
    p.add_argument("--model", default=DEFAULT_BASE_MODEL_ID, help="Base model id")
    p.add_argument("--lora", required=True, help="LoRA directory or weights file")
    p.add_argument("--adapter", default="adapter", help="Adapter name in pipeline")
    p.add_argument("--scale", type=float, default=1.0, help="LoRA scale (weight)")
    p.add_argument("--prompt", required=True, help="Prompt")
    p.add_argument("--negative", default=None, help="Negative prompt")
    p.add_argument("--steps", type=int, default=30, help="Inference steps")
    p.add_argument("--guidance", type=float, default=7.5, help="Guidance scale")
    p.add_argument("--seed", type=int, default=42, help="Seed")
    p.add_argument("--out", default="experiments/results/lora_smoke.png", help="Output image path")
    args = p.parse_args()

    pipe = load_base_pipeline(args.model)
    attach_lora(pipe, Path(args.lora), adapter_name=args.adapter, scale=float(args.scale))

    gen = torch.Generator(device=str(getattr(pipe, "device", "cpu"))).manual_seed(int(args.seed))
    with torch.no_grad():
        res = pipe(
            prompt=str(args.prompt),
            negative_prompt=args.negative,
            num_inference_steps=int(args.steps),
            guidance_scale=float(args.guidance),
            width=1024,
            height=1024,
            generator=gen,
        )
    img = res.images[0]
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)
    logger.info("Saved: %s", out_path)


if __name__ == "__main__":
    main()
