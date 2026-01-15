# src/dynamic_lora_t2i/utils/lora_io.py

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Dict

import torch
from safetensors.torch import load_file, save_file


def _strip_once(s: str, prefixes: tuple[str, ...]) -> str:
    for p in prefixes:
        if s.startswith(p):
            return s[len(p):]
    return s


def normalize_lora_key(k: str) -> str:
    for comp in ("unet", "text_encoder", "text_encoder_2"):
        pref = comp + "."
        if k.startswith(pref):
            rest = k[len(pref):]
            rest = _strip_once(rest, ("base_model.model.", "base_model."))
            return pref + rest

    k = _strip_once(k, ("base_model.model.", "base_model."))
    return k


def normalize_lora_state_dict(sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    fixed: Dict[str, torch.Tensor] = {}
    for k, v in sd.items():
        nk = normalize_lora_key(k)
        if nk in fixed:
            raise RuntimeError(f"LoRA key collision after normalize: {k} -> {nk}")
        fixed[nk] = v
    return fixed


def ensure_compatible_lora_weights(
    lora_dir: Path,
    weight_name: str = "pytorch_lora_weights.safetensors",
) -> str:
    src = lora_dir / weight_name
    if not src.exists():
        raise FileNotFoundError(src)

    sd = load_file(str(src))
    sample = list(sd.keys())[:50]
    looks_ok = all(
        (k.startswith("unet.") or k.startswith("text_encoder.") or k.startswith("text_encoder_2."))
        and "base_model." not in k
        for k in sample
    )
    if looks_ok:
        return weight_name

    fixed_name = "pytorch_lora_weights.fixed.safetensors"
    fixed_path = lora_dir / fixed_name
    if fixed_path.exists():
        return fixed_name

    bak = lora_dir / "pytorch_lora_weights.orig.safetensors"
    if not bak.exists():
        shutil.copy2(src, bak)

    fixed = normalize_lora_state_dict(sd)
    save_file(fixed, str(fixed_path))

    cfg_path = lora_dir / "adapter_config.json"
    if cfg_path.exists():
        cfg = json.loads(cfg_path.read_text())
        tm = cfg.get("target_modules")
        if isinstance(tm, list):
            def _norm_tm(t: str) -> str:
                for p in ("unet.", "text_encoder.", "text_encoder_2."):
                    if t.startswith(p):
                        t = t[len(p):]
                t = t.replace("base_model.model.", "").replace("base_model.", "")
                return t

            cfg["target_modules"] = [_norm_tm(t) for t in tm]
            cfg_path.write_text(json.dumps(cfg, indent=2, ensure_ascii=False))

    return fixed_name
