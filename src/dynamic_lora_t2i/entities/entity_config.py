# src/dynamic_lora_t2i/entities/entity_config.py

from __future__ import annotations

import json
import logging
import re
from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from src.dynamic_lora_t2i.config import (
    DEFAULT_BASE_MODEL_ID,
    DEFAULT_REFINER_MODEL_ID,
    DEFAULT_LORA_ALPHA,
    DEFAULT_LORA_DROPOUT,
    DEFAULT_LORA_RANK,
    ENTITIES_DIR,
    LORA_USER_ENTITIES_DIR,
    ensure_project_directories,
)
from src.dynamic_lora_t2i.utils.entity_zip import sanitize_entity_name

logger = logging.getLogger(__name__)


def _utc_now_z() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


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


def _path_or_none(v: Any) -> Optional[Path]:
    if v is None:
        return None
    return Path(v)


def _validate_placeholder_token(token: str) -> None:
    token = (token or "").strip()
    if not token:
        raise ValueError("placeholder_token is empty")

    if " " in token:
        raise ValueError("placeholder_token must not contain spaces")

    # Для простоти: дозволимо [a-zA-Z0-9_-]
    if not re.fullmatch(r"[A-Za-z0-9_\-]+", token):
        raise ValueError("placeholder_token must match: [A-Za-z0-9_-]+")


@dataclass
class EntityModelConfig:
    base_model_id: str = DEFAULT_BASE_MODEL_ID
    refiner_model_id: Optional[str] = DEFAULT_REFINER_MODEL_ID


@dataclass
class EntityLoRAConfig:
    rank: int = DEFAULT_LORA_RANK
    alpha: int = DEFAULT_LORA_ALPHA
    dropout: float = DEFAULT_LORA_DROPOUT

    # Якщо захочеш SDXL-специфіку: наприклад ["to_k", "to_q", "to_v", "to_out.0"]
    target_modules: Optional[list[str]] = None


@dataclass
class EntityConfig:
    """
    Entity-level config (стійкий, довгоживучий):
      - хто така сутність (name/description/token)
      - де лежать дані
      - під яку базову модель тренуємо
      - базові LoRA параметри (rank/alpha/dropout/target_modules)
      - де складати LoRA-адаптери цієї сутності

    Це НЕ “run config”. Run-конфіг (epochs/lr/etc) лишаємо в TrainConfig.
    """

    schema_version: int = 1

    name: str = "my_entity"
    description: str = ""

    placeholder_token: str = "my_entity_token"

    created_at: str = field(default_factory=_utc_now_z)
    updated_at: str = field(default_factory=_utc_now_z)

    # Де лежить сутність (за замовчуванням data/entities/<name>)
    entity_dir: Optional[Path] = None

    # Де зберігати треновані адаптери (за замовчуванням lora_adapters/user_entities/<name>/)
    adapters_dir: Optional[Path] = None

    captions_ext: str = ".txt"
    image_exts: list[str] = field(default_factory=lambda: [".png", ".jpg", ".jpeg", ".webp"])

    model: EntityModelConfig = field(default_factory=EntityModelConfig)
    lora: EntityLoRAConfig = field(default_factory=EntityLoRAConfig)

    meta: dict[str, Any] = field(default_factory=dict)

    def resolve_paths(self) -> "EntityConfig":
        ensure_project_directories()

        safe = sanitize_entity_name(self.name)
        self.name = safe

        if self.entity_dir is None:
            self.entity_dir = (ENTITIES_DIR / safe).resolve()

        if self.adapters_dir is None:
            self.adapters_dir = (LORA_USER_ENTITIES_DIR / safe).resolve()

        return self

    def config_path(self) -> Path:
        """
        Де зберігаємо entity_config.json (всередині entity_dir).
        """
        self.resolve_paths()
        assert self.entity_dir is not None
        return Path(self.entity_dir) / "entity_config.json"

    def validate(self, *, strict: bool = False) -> None:
        if self.schema_version != 1:
            raise ValueError(f"Unsupported schema_version: {self.schema_version}")

        if not self.name:
            raise ValueError("name is empty")

        _validate_placeholder_token(self.placeholder_token)

        if self.captions_ext and not self.captions_ext.startswith("."):
            raise ValueError("captions_ext must start with '.' (e.g. .txt)")

        if not self.image_exts:
            raise ValueError("image_exts is empty")

        if self.lora.rank <= 0:
            raise ValueError("lora.rank must be > 0")
        if self.lora.alpha <= 0:
            raise ValueError("lora.alpha must be > 0")
        if self.lora.dropout < 0.0 or self.lora.dropout >= 1.0:
            raise ValueError("lora.dropout must be in [0.0, 1.0)")

        self.resolve_paths()
        assert self.entity_dir is not None
        assert self.adapters_dir is not None

        if strict:
            if not Path(self.entity_dir).exists():
                raise FileNotFoundError(f"entity_dir not found: {self.entity_dir}")

            # Перевірка, що є хоч 1 зображення
            img_count = 0
            for p in Path(self.entity_dir).rglob("*"):
                if p.is_file() and p.suffix.lower() in {e.lower() for e in self.image_exts}:
                    img_count += 1
                    break
            if img_count == 0:
                raise ValueError(f"No images found in entity_dir: {self.entity_dir}")

    def to_dict(self) -> dict[str, Any]:
        return _to_serializable(self)

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "EntityConfig":
        model_d = d.get("model", {}) or {}
        lora_d = d.get("lora", {}) or {}

        cfg = EntityConfig(
            schema_version=int(d.get("schema_version", 1)),
            name=str(d.get("name", "my_entity")),
            description=str(d.get("description", "")),
            placeholder_token=str(d.get("placeholder_token", "my_entity_token")),
            created_at=str(d.get("created_at", _utc_now_z())),
            updated_at=str(d.get("updated_at", _utc_now_z())),
            entity_dir=_path_or_none(d.get("entity_dir")),
            adapters_dir=_path_or_none(d.get("adapters_dir")),
            captions_ext=str(d.get("captions_ext", ".txt")),
            image_exts=list(d.get("image_exts", [".png", ".jpg", ".jpeg", ".webp"])),
            model=EntityModelConfig(
                base_model_id=str(model_d.get("base_model_id", DEFAULT_BASE_MODEL_ID)),
                refiner_model_id=model_d.get("refiner_model_id", DEFAULT_REFINER_MODEL_ID),
            ),
            lora=EntityLoRAConfig(
                rank=int(lora_d.get("rank", DEFAULT_LORA_RANK)),
                alpha=int(lora_d.get("alpha", DEFAULT_LORA_ALPHA)),
                dropout=float(lora_d.get("dropout", DEFAULT_LORA_DROPOUT)),
                target_modules=lora_d.get("target_modules", None),
            ),
            meta=dict(d.get("meta", {}) or {}),
        )
        return cfg

    @staticmethod
    def load(path: Path, *, resolve_paths: bool = True, validate: bool = True) -> "EntityConfig":
        path = path.resolve()
        if not path.exists():
            raise FileNotFoundError(f"EntityConfig not found: {path}")

        data = json.loads(path.read_text(encoding="utf-8"))
        cfg = EntityConfig.from_dict(data)

        if resolve_paths:
            cfg.resolve_paths()
        if validate:
            cfg.validate(strict=False)

        logger.info("Loaded EntityConfig: %s", path)
        return cfg

    def save(self, path: Optional[Path] = None) -> Path:
        self.updated_at = _utc_now_z()
        self.resolve_paths()
        self.validate(strict=False)

        if path is None:
            path = self.config_path()

        path = path.resolve()
        path.parent.mkdir(parents=True, exist_ok=True)

        path.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info("Saved EntityConfig: %s", path)
        return path


def init_entity_config(
    entity_name: str,
    placeholder_token: str,
    *,
    description: str = "",
    base_model_id: str = DEFAULT_BASE_MODEL_ID,
) -> EntityConfig:
    cfg = EntityConfig(
        name=entity_name,
        description=description,
        placeholder_token=placeholder_token,
        model=EntityModelConfig(base_model_id=base_model_id, refiner_model_id=DEFAULT_REFINER_MODEL_ID),
        lora=EntityLoRAConfig(),
    )
    cfg.resolve_paths()
    cfg.save()
    return cfg


def main() -> None:
    """
    Usage:
      python -m src.dynamic_lora_t2i.entities.entity_config init --entity my_entity --token vlad_object --desc "..."
      python -m src.dynamic_lora_t2i.entities.entity_config check --entity my_entity
    """
    import argparse
    from src.dynamic_lora_t2i.config import setup_logging

    setup_logging()

    p = argparse.ArgumentParser(description="EntityConfig utilities (init/check).")
    sub = p.add_subparsers(dest="cmd", required=True)

    p_init = sub.add_parser("init", help="Create data/entities/<entity>/entity_config.json")
    p_init.add_argument("--entity", required=True, help="Entity name (folder in data/entities/)")
    p_init.add_argument("--token", required=True, help="Placeholder token, e.g. vlad_object")
    p_init.add_argument("--desc", default="", help="Human description")
    p_init.add_argument("--base-model", default=DEFAULT_BASE_MODEL_ID, help="Base model id")

    p_check = sub.add_parser("check", help="Load + validate entity_config.json")
    p_check.add_argument("--entity", required=True, help="Entity name (folder in data/entities/)")

    args = p.parse_args()

    if args.cmd == "init":
        cfg = init_entity_config(
            entity_name=args.entity,
            placeholder_token=args.token,
            description=args.desc,
            base_model_id=args.base_model,
        )
        print(f"OK: created {cfg.config_path()}")
        print(json.dumps(cfg.to_dict(), ensure_ascii=False, indent=2))

    elif args.cmd == "check":
        safe = sanitize_entity_name(args.entity)
        path = (ENTITIES_DIR / safe / "entity_config.json").resolve()
        cfg = EntityConfig.load(path, resolve_paths=True, validate=True)
        # strict=True якщо хочеш вимагати наявність зображень
        cfg.validate(strict=False)
        print("OK: entity config is valid")
        print(json.dumps(cfg.to_dict(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
