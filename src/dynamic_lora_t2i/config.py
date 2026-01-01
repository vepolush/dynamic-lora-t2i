# src/dynamic_lora_t2i/utils/config.py

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional


def _find_project_root(start: Path) -> Path:
    markers = ("pyproject.toml", "setup.cfg", "requirements.txt", ".git")
    start = start.resolve()

    for p in (start, *start.parents):
        if any((p / m).exists() for m in markers):
            return p
        if (p / "src").exists():
            if (p / "src" / "dynamic_lora_t2i").exists():
                return p

    try:
        return start.parents[3]
    except IndexError:
        return start.parent


PROJECT_ROOT = Path(os.getenv("DYNAMIC_LORA_T2I_PROJECT_ROOT", "")).expanduser()
if not PROJECT_ROOT:
    PROJECT_ROOT = _find_project_root(Path(__file__))

SRC_DIR = PROJECT_ROOT / "src"
PACKAGE_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
ENTITIES_DIR = DATA_DIR / "entities"

EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
EXPERIMENT_CONFIGS_DIR = EXPERIMENTS_DIR / "configs"
EXPERIMENT_RESULTS_DIR = EXPERIMENTS_DIR / "results"

LORA_ADAPTERS_DIR = PROJECT_ROOT / "lora_adapters"
LORA_BASE_DIR = LORA_ADAPTERS_DIR / "base"
LORA_USER_ENTITIES_DIR = LORA_ADAPTERS_DIR / "user_entities"

NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"

REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

STREAMLIT_APP_DIR = SRC_DIR / "streamlit_app"
TESTS_DIR = PROJECT_ROOT / "tests"
LOGS_DIR = PROJECT_ROOT / "logs"

_default_cache_base = Path("/workspace/.cache") if Path("/workspace").exists() else (PROJECT_ROOT / ".cache")
CACHE_BASE_DIR = Path(os.getenv("DYNAMIC_LORA_T2I_CACHE_DIR", str(_default_cache_base))).expanduser()

HF_HOME = Path(os.getenv("HF_HOME", str(CACHE_BASE_DIR / "huggingface"))).expanduser()
TRANSFORMERS_CACHE = Path(os.getenv("TRANSFORMERS_CACHE", str(HF_HOME / "transformers"))).expanduser()
DIFFUSERS_CACHE = Path(os.getenv("DIFFUSERS_CACHE", str(HF_HOME / "diffusers"))).expanduser()
TORCH_HOME = Path(os.getenv("TORCH_HOME", str(CACHE_BASE_DIR / "torch"))).expanduser()


def ensure_project_directories() -> None:
    for path in [
        DATA_DIR,
        RAW_DATA_DIR,
        PROCESSED_DATA_DIR,
        ENTITIES_DIR,
        EXPERIMENTS_DIR,
        EXPERIMENT_CONFIGS_DIR,
        EXPERIMENT_RESULTS_DIR,
        LORA_ADAPTERS_DIR,
        LORA_BASE_DIR,
        LORA_USER_ENTITIES_DIR,
        NOTEBOOKS_DIR,
        REPORTS_DIR,
        FIGURES_DIR,
        STREAMLIT_APP_DIR,
        TESTS_DIR,
        LOGS_DIR,
        CACHE_BASE_DIR,
        HF_HOME,
        TRANSFORMERS_CACHE,
        DIFFUSERS_CACHE,
        TORCH_HOME,
    ]:
        path.mkdir(parents=True, exist_ok=True)


DEFAULT_BASE_MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"
DEFAULT_REFINER_MODEL_ID = None

USE_FP16 = True
USE_BF16 = os.getenv("DYNAMIC_LORA_T2I_USE_BF16", "0").strip().lower() in ("1", "true", "yes", "y")

DEFAULT_LORA_RANK = 16
DEFAULT_LORA_ALPHA = 16
DEFAULT_LORA_DROPOUT = 0.0

DEFAULT_NUM_EPOCHS = 1
DEFAULT_TRAIN_BATCH_SIZE = 1
DEFAULT_GRADIENT_ACCUM_STEPS = 1
DEFAULT_LEARNING_RATE = 1e-4
DEFAULT_WARMUP_STEPS = 0

DEFAULT_IMAGE_WIDTH = 1024
DEFAULT_IMAGE_HEIGHT = 1024

DEFAULT_NUM_INFERENCE_STEPS = 20
DEFAULT_GUIDANCE_SCALE = 7.5


def get_device() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda"

    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"

    return "cpu"


def get_torch_dtype():
    import torch

    device = get_device()

    if device == "cuda" and USE_BF16:
        return torch.bfloat16

    if device in ("cuda", "mps") and USE_FP16:
        return torch.float16

    return torch.float32


def get_cuda_capabilities() -> Optional[str]:
    import torch

    if not torch.cuda.is_available():
        return None
    idx = torch.cuda.current_device()
    name = torch.cuda.get_device_name(idx)
    cc = torch.cuda.get_device_capability(idx)
    return f"{name} (capability {cc[0]}.{cc[1]})"


def setup_logging(level: int = logging.INFO) -> None:
    ensure_project_directories()
    log_file = LOGS_DIR / "app.log"

    root = logging.getLogger()
    if root.handlers:
        root.setLevel(level)
        return

    root.setLevel(level)

    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s - %(message)s")

    sh = logging.StreamHandler()
    sh.setLevel(level)
    sh.setFormatter(fmt)

    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(level)
    fh.setFormatter(fmt)

    root.addHandler(sh)
    root.addHandler(fh)
