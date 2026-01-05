# src/dynamic_lora_t2i/training/training_log.py

from __future__ import annotations

import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional


def _utc_now_z() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _safe_version(import_name: str) -> Optional[str]:
    try:
        mod = __import__(import_name)
        return getattr(mod, "__version__", None)
    except Exception:
        return None


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def init_training_log(
    *,
    cfg_dict: dict[str, Any],
    run_name: str,
    entity_name: str,
    device: str,
    torch_dtype: str,
    max_train_steps: int,
    num_update_steps_per_epoch: int,
    num_images: int,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "status": "running",
        "created_at": _utc_now_z(),
        "finished_at": None,
        "duration_sec": None,
        "run": {
            "run_name": run_name,
            "entity_name": entity_name,
            "device": device,
            "torch_dtype": torch_dtype,
            "num_images": int(num_images),
            "num_update_steps_per_epoch": int(num_update_steps_per_epoch),
            "max_train_steps": int(max_train_steps),
        },
        "hyperparams": {
            "train_config": cfg_dict,
        },
        "metrics": {
            "global_step": 0,
            "avg_loss": None,
            "epochs": [],
            "step_history": [],
            "step_history_truncated": False,
        },
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "torch": _safe_version("torch"),
            "diffusers": _safe_version("diffusers"),
            "accelerate": _safe_version("accelerate"),
            "peft": _safe_version("peft"),
            "transformers": _safe_version("transformers"),
        },
        "_internal": {
            "t0": time.time(),
        },
    }


def append_step(
    log: dict[str, Any],
    *,
    epoch: int,
    global_step: int,
    loss: float,
    lr: Optional[float],
    max_steps_kept: int = 20000,
) -> None:
    m = log["metrics"]
    if m["step_history_truncated"]:
        return

    m["step_history"].append(
        {
            "ts": _utc_now_z(),
            "epoch": int(epoch),
            "global_step": int(global_step),
            "loss": float(loss),
            "lr": float(lr) if lr is not None else None,
        }
    )

    if len(m["step_history"]) > int(max_steps_kept):
        m["step_history"] = m["step_history"][: int(max_steps_kept)]
        m["step_history_truncated"] = True


def append_epoch(
    log: dict[str, Any],
    *,
    epoch: int,
    update_steps: int,
    loss_mean: float,
    loss_last: float,
    lr_mean: Optional[float],
    lr_last: Optional[float],
    global_step_end: int,
) -> None:
    log["metrics"]["epochs"].append(
        {
            "epoch": int(epoch),
            "update_steps": int(update_steps),
            "loss_mean": float(loss_mean),
            "loss_last": float(loss_last),
            "lr_mean": float(lr_mean) if lr_mean is not None else None,
            "lr_last": float(lr_last) if lr_last is not None else None,
            "global_step_end": int(global_step_end),
        }
    )


def finalize_training_log(
    log: dict[str, Any],
    *,
    status: str,
    global_step: int,
    avg_loss: Optional[float],
    error: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    t0 = float(log.get("_internal", {}).get("t0", time.time()))
    dt = max(0.0, time.time() - t0)

    log["status"] = status
    log["finished_at"] = _utc_now_z()
    log["duration_sec"] = dt

    log["metrics"]["global_step"] = int(global_step)
    log["metrics"]["avg_loss"] = float(avg_loss) if avg_loss is not None else None

    if error is not None:
        log["error"] = error

    log.pop("_internal", None)
    return log
