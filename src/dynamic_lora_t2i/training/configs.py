# src/dynamic_lora_t2i/training/configs.py

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any, Optional

from src.dynamic_lora_t2i.config import (
    DEFAULT_BASE_MODEL_ID,
    DEFAULT_REFINER_MODEL_ID,
    USE_FP16,
    DEFAULT_LORA_RANK,
    DEFAULT_LORA_ALPHA,
    DEFAULT_LORA_DROPOUT,
    DEFAULT_NUM_EPOCHS,
    DEFAULT_TRAIN_BATCH_SIZE,
    DEFAULT_GRADIENT_ACCUM_STEPS,
    DEFAULT_LEARNING_RATE,
    DEFAULT_WARMUP_STEPS,
    DEFAULT_IMAGE_WIDTH,
    DEFAULT_IMAGE_HEIGHT,
    ENTITIES_DIR,
    LORA_USER_ENTITIES_DIR,
    EXPERIMENT_RESULTS_DIR,
    LOGS_DIR,
    ensure_project_directories,
)

from src.dynamic_lora_t2i.utils.entity_zip import sanitize_entity_name

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    base_model_id: str = DEFAULT_BASE_MODEL_ID
    refiner_model_id: Optional[str] = DEFAULT_REFINER_MODEL_ID


@dataclass
class EntityDataConfig:
    entity_name: str = "my_entity"
    placeholder_token: str = "my_entity_token"

    entity_dir: Optional[Path] = None

    captions_ext: str = ".txt"

    image_exts: list[str] = field(default_factory=lambda: [".png", ".jpg", ".jpeg", ".webp"])

    width: int = DEFAULT_IMAGE_WIDTH
    height: int = DEFAULT_IMAGE_HEIGHT

    max_images: Optional[int] = None


@dataclass
class LoRAConfig:
    rank: int = DEFAULT_LORA_RANK
    alpha: int = DEFAULT_LORA_ALPHA
    dropout: float = DEFAULT_LORA_DROPOUT

    target_modules: Optional[list[str]] = None


@dataclass
class TrainHyperparams:
    num_epochs: int = DEFAULT_NUM_EPOCHS
    train_batch_size: int = DEFAULT_TRAIN_BATCH_SIZE
    gradient_accum_steps: int = DEFAULT_GRADIENT_ACCUM_STEPS
    learning_rate: float = DEFAULT_LEARNING_RATE
    warmup_steps: int = DEFAULT_WARMUP_STEPS

    seed: int = 42

    mixed_precision: str = "fp16" if USE_FP16 else "no"

    max_train_steps: Optional[int] = None
    lr_scheduler: str = "constant"
    optimizer: str = "adamw"

    validation_prompt: Optional[str] = None
    validation_steps: int = 0


@dataclass
class OutputConfig:
    run_name: str = "run"

    output_dir: Optional[Path] = None

    results_dir: Optional[Path] = None

    logs_dir: Optional[Path] = None

    save_every_n_steps: int = 0


@dataclass
class TrainConfig:
    schema_version: int = 1
    model: ModelConfig = field(default_factory=ModelConfig)
    data: EntityDataConfig = field(default_factory=EntityDataConfig)
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    train: TrainHyperparams = field(default_factory=TrainHyperparams)
    output: OutputConfig = field(default_factory=OutputConfig)

    meta: dict[str, Any] = field(default_factory=dict)

    def resolve_paths(self) -> "TrainConfig":
        ensure_project_directories()

        safe_entity = sanitize_entity_name(self.data.entity_name)
        self.data.entity_name = safe_entity

        if self.data.entity_dir is None:
            self.data.entity_dir = (ENTITIES_DIR / safe_entity).resolve()

        if self.output.output_dir is None:
            self.output.output_dir = (LORA_USER_ENTITIES_DIR / safe_entity / self.output.run_name).resolve()

        if self.output.results_dir is None:
            self.output.results_dir = (EXPERIMENT_RESULTS_DIR / self.output.run_name).resolve()

        if self.output.logs_dir is None:
            self.output.logs_dir = (LOGS_DIR / self.output.run_name).resolve()

        return self

    def validate(self) -> None:
        if self.schema_version != 1:
            raise ValueError(f"Unsupported schema_version: {self.schema_version}")

        if not self.data.entity_name:
            raise ValueError("data.entity_name is empty")

        if not self.data.placeholder_token or " " in self.data.placeholder_token:
            raise ValueError("data.placeholder_token must be non-empty and contain no spaces")

        if self.data.width <= 0 or self.data.height <= 0:
            raise ValueError("data.width/height must be positive")

        if self.train.train_batch_size <= 0:
            raise ValueError("train.train_batch_size must be positive")

        if self.train.gradient_accum_steps <= 0:
            raise ValueError("train.gradient_accum_steps must be positive")

        if self.train.learning_rate <= 0:
            raise ValueError("train.learning_rate must be positive")

        if self.train.mixed_precision not in {"fp16", "bf16", "no"}:
            raise ValueError("train.mixed_precision must be one of: fp16, bf16, no")

        if self.data.entity_dir is None:
            raise ValueError("data.entity_dir is None (call resolve_paths())")

        if not Path(self.data.entity_dir).exists():
            raise FileNotFoundError(f"Entity dir not found: {self.data.entity_dir}")

        if self.output.output_dir is None or self.output.results_dir is None or self.output.logs_dir is None:
            raise ValueError("output dirs are None (call resolve_paths())")

    def prepare_dirs(self) -> None:
        self.resolve_paths()
        assert self.output.output_dir and self.output.results_dir and self.output.logs_dir
        Path(self.output.output_dir).mkdir(parents=True, exist_ok=True)
        Path(self.output.results_dir).mkdir(parents=True, exist_ok=True)
        Path(self.output.logs_dir).mkdir(parents=True, exist_ok=True)


def _to_serializable(obj: Any) -> Any:
    if is_dataclass(obj):
        return {k: _to_serializable(v) for k, v in asdict(obj).items()}
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {k: _to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_serializable(x) for x in obj]
    return obj


def train_config_to_dict(cfg: TrainConfig) -> dict[str, Any]:
    return _to_serializable(cfg)


def _path_or_none(v: Any) -> Optional[Path]:
    if v is None:
        return None
    return Path(v)


def train_config_from_dict(d: dict[str, Any]) -> TrainConfig:
    model_d = d.get("model", {}) or {}
    data_d = d.get("data", {}) or {}
    lora_d = d.get("lora", {}) or {}
    train_d = d.get("train", {}) or {}
    out_d = d.get("output", {}) or {}

    cfg = TrainConfig(
        schema_version=int(d.get("schema_version", 1)),
        model=ModelConfig(
            base_model_id=model_d.get("base_model_id", DEFAULT_BASE_MODEL_ID),
            refiner_model_id=model_d.get("refiner_model_id", DEFAULT_REFINER_MODEL_ID),
        ),
        data=EntityDataConfig(
            entity_name=data_d.get("entity_name", "my_entity"),
            placeholder_token=data_d.get("placeholder_token", "my_entity_token"),
            entity_dir=_path_or_none(data_d.get("entity_dir")),
            captions_ext=data_d.get("captions_ext", ".txt"),
            image_exts=list(data_d.get("image_exts", [".png", ".jpg", ".jpeg", ".webp"])),
            width=int(data_d.get("width", DEFAULT_IMAGE_WIDTH)),
            height=int(data_d.get("height", DEFAULT_IMAGE_HEIGHT)),
            max_images=data_d.get("max_images", None),
        ),
        lora=LoRAConfig(
            rank=int(lora_d.get("rank", DEFAULT_LORA_RANK)),
            alpha=int(lora_d.get("alpha", DEFAULT_LORA_ALPHA)),
            dropout=float(lora_d.get("dropout", DEFAULT_LORA_DROPOUT)),
            target_modules=lora_d.get("target_modules", None),
        ),
        train=TrainHyperparams(
            num_epochs=int(train_d.get("num_epochs", DEFAULT_NUM_EPOCHS)),
            train_batch_size=int(train_d.get("train_batch_size", DEFAULT_TRAIN_BATCH_SIZE)),
            gradient_accum_steps=int(train_d.get("gradient_accum_steps", DEFAULT_GRADIENT_ACCUM_STEPS)),
            learning_rate=float(train_d.get("learning_rate", DEFAULT_LEARNING_RATE)),
            warmup_steps=int(train_d.get("warmup_steps", DEFAULT_WARMUP_STEPS)),
            seed=int(train_d.get("seed", 42)),
            mixed_precision=str(train_d.get("mixed_precision", "fp16" if USE_FP16 else "no")),
            max_train_steps=train_d.get("max_train_steps", None),
            lr_scheduler=str(train_d.get("lr_scheduler", "constant")),
            optimizer=str(train_d.get("optimizer", "adamw")),
            validation_prompt=train_d.get("validation_prompt", None),
            validation_steps=int(train_d.get("validation_steps", 0)),
        ),
        output=OutputConfig(
            run_name=out_d.get("run_name", "run"),
            output_dir=_path_or_none(out_d.get("output_dir")),
            results_dir=_path_or_none(out_d.get("results_dir")),
            logs_dir=_path_or_none(out_d.get("logs_dir")),
            save_every_n_steps=int(out_d.get("save_every_n_steps", 0)),
        ),
        meta=dict(d.get("meta", {}) or {}),
    )
    return cfg


def _is_yaml(path: Path) -> bool:
    return path.suffix.lower() in {".yaml", ".yml"}


def load_train_config(path: Path, *, resolve_paths: bool = True, validate: bool = True) -> TrainConfig:
    path = path.resolve()
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")

    if _is_yaml(path):
        try:
            import yaml
        except ImportError as e:
            raise ImportError("PyYAML is required to load .yaml/.yml configs. Install: pip install pyyaml") from e
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    else:
        data = json.loads(path.read_text(encoding="utf-8"))

    cfg = train_config_from_dict(data)

    if resolve_paths:
        cfg.resolve_paths()
    if validate:
        cfg.validate()

    logger.info("Loaded train config: %s", path)
    return cfg


def save_train_config(cfg: TrainConfig, path: Path) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)

    d = train_config_to_dict(cfg)

    if _is_yaml(path):
        try:
            import yaml  # type: ignore
        except ImportError as e:
            raise ImportError("PyYAML is required to save .yaml/.yml configs. Install: pip install pyyaml") from e
        path.write_text(yaml.safe_dump(d, sort_keys=False, allow_unicode=True), encoding="utf-8")
    else:
        path.write_text(json.dumps(d, ensure_ascii=False, indent=2), encoding="utf-8")

    logger.info("Saved train config: %s", path)


def _default_run_name(entity_name: str) -> str:
    return f"{sanitize_entity_name(entity_name)}_lora_v1"


def init_config_file(out_path: Path, *, entity_name: str, token: str) -> None:
    cfg = TrainConfig()
    cfg.data.entity_name = entity_name
    cfg.data.placeholder_token = token
    cfg.output.run_name = _default_run_name(entity_name)
    cfg.resolve_paths()
    save_train_config(cfg, out_path)


def main() -> None:
    import argparse
    from src.dynamic_lora_t2i.config import setup_logging

    setup_logging()

    p = argparse.ArgumentParser(description="Training config utilities (init/load/save).")
    sub = p.add_subparsers(dest="cmd", required=True)

    p_init = sub.add_parser("init", help="Create a starter config file (.yaml/.json).")
    p_init.add_argument("--out", required=True, help="Output path, e.g. experiments/configs/my_entity.yaml")
    p_init.add_argument("--entity", required=True, help="Entity name (folder name in data/entities/)")
    p_init.add_argument("--token", required=True, help="Placeholder token, e.g. vlad_object")

    p_check = sub.add_parser("check", help="Load + resolve + validate a config.")
    p_check.add_argument("--path", required=True, help="Path to config (.yaml/.json)")

    args = p.parse_args()

    if args.cmd == "init":
        init_config_file(Path(args.out), entity_name=args.entity, token=args.token)
        print(f"OK: created {args.out}")
    elif args.cmd == "check":
        cfg = load_train_config(Path(args.path), resolve_paths=True, validate=True)
        print("OK: config is valid")
        print(json.dumps(train_config_to_dict(cfg), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
