# src/dynamic_lora_t2i/utils/config.py

from pathlib import Path
import logging

PROJECT_ROOT = Path(__file__).resolve().parents[2]

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
    ]:
        path.mkdir(parents=True, exist_ok=True)


DEFAULT_BASE_MODEL_ID = "runwayml/stable-diffusion-v1-5"
# DEFAULT_BASE_MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"

DEFAULT_REFINER_MODEL_ID = None

USE_FP16 = True

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

DEFAULT_NUM_INFERENCE_STEPS = 15
DEFAULT_GUIDANCE_SCALE = 7.5


def get_device() -> str:
    """
    Returns device: 'cuda', 'mps' or 'cpu'.
    """
    import torch

    if torch.cuda.is_available():
        return "cuda"

    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"

    return "cpu"


def get_torch_dtype():
    """
    Returns torch.float16 or torch.float32 depending on USE_FP16 and device.
    """
    import torch

    device = get_device()

    if device in ("cuda", "mps") and USE_FP16:
        return torch.float16

    return torch.float32


def setup_logging(level: int = logging.INFO) -> None:
    """
    Configure basic logging for the project (console + file).
    """
    ensure_project_directories()
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    log_file = LOGS_DIR / "app.log"

    if logging.getLogger().handlers:
        return

    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file, encoding="utf-8"),
        ],
    )
