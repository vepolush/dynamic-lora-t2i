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
    target_modules: Optional[list[str]] = None


@dataclass
class EntityInferenceConfig:
    class_prompt: str = ""
    default_adapter: Optional[str] = None
    recommended_lora_scale: float = 1.0


@dataclass
class EntityConfig:
    schema_version: int = 2

    name: str = "my_entity"
    description: str = ""

    placeholder_token: str = "my_entity_token"

    created_at: str = field(default_factory=_utc_now_z)
    updated_at: str = field(default_factory=_utc_now_z)

    entity_dir: Optional[Path] = None
    adapters_dir: Optional[Path] = None

    captions_ext: str = ".txt"
    image_exts: list[str] = field(default_factory=lambda: [".png", ".jpg", ".jpeg", ".webp"])

    model: EntityModelConfig = field(default_factory=EntityModelConfig)
    lora: EntityLoRAConfig = field(default_factory=EntityLoRAConfig)

    inference: EntityInferenceConfig = field(default_factory=EntityInferenceConfig)

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
        self.resolve_paths()
        assert self.entity_dir is not None
        return Path(self.entity_dir) / "entity_config.json"

    def validate(self, *, strict: bool = False) -> None:
        if int(self.schema_version) not in (1, 2):
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

        if self.inference is None:
            raise ValueError("inference config is None")

        scale = float(getattr(self.inference, "recommended_lora_scale", 1.0))
        if scale < 0.0 or scale > 4.0:
            raise ValueError("inference.recommended_lora_scale must be in [0.0, 4.0]")

        da = (self.inference.default_adapter or "").strip()
        if self.inference.default_adapter is not None and not da:
            raise ValueError("inference.default_adapter is empty string (use null/omit or a non-empty value)")

        self.resolve_paths()
        assert self.entity_dir is not None
        assert self.adapters_dir is not None

        if strict:
            if not Path(self.entity_dir).exists():
                raise FileNotFoundError(f"entity_dir not found: {self.entity_dir}")

            img_exts = {e.lower() for e in self.image_exts}
            found = False
            for p in Path(self.entity_dir).rglob("*"):
                if p.is_file() and p.suffix.lower() in img_exts:
                    found = True
                    break
            if not found:
                raise ValueError(f"No images found in entity_dir: {self.entity_dir}")

    def to_dict(self) -> dict[str, Any]:
        return _to_serializable(self)

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "EntityConfig":
        model_d = d.get("model", {}) or {}
        lora_d = d.get("lora", {}) or {}
        inf_d = d.get("inference", {}) or {}

        # backward compatibility: старі проєкти могли зберігати class_prompt у meta
        meta = dict(d.get("meta", {}) or {})
        legacy_class_prompt = str(meta.get("class_prompt", "") or "").strip()

        inf = EntityInferenceConfig(
            class_prompt=str(inf_d.get("class_prompt", "")) if inf_d is not None else "",
            default_adapter=(inf_d.get("default_adapter", None) if inf_d is not None else None),
            recommended_lora_scale=float(inf_d.get("recommended_lora_scale", 1.0)) if inf_d is not None else 1.0,
        )
        if not inf.class_prompt and legacy_class_prompt:
            inf.class_prompt = legacy_class_prompt

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
            inference=inf,
            meta=meta,
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

        self.schema_version = 2

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
    class_prompt: str = "",
) -> EntityConfig:
    cfg = EntityConfig(
        schema_version=2,
        name=entity_name,
        description=description,
        placeholder_token=placeholder_token,
        model=EntityModelConfig(base_model_id=base_model_id, refiner_model_id=DEFAULT_REFINER_MODEL_ID),
        lora=EntityLoRAConfig(),
        inference=EntityInferenceConfig(class_prompt=class_prompt),
    )
    cfg.resolve_paths()
    cfg.save()
    return cfg


def set_default_adapter(
    entity_name: str,
    *,
    adapter: str,
    recommended_scale: float = 1.0,
    class_prompt: Optional[str] = None,
) -> Path:
    safe = sanitize_entity_name(entity_name)
    path = (ENTITIES_DIR / safe / "entity_config.json").resolve()
    cfg = EntityConfig.load(path, resolve_paths=True, validate=True)

    cfg.inference.default_adapter = str(adapter).strip()
    cfg.inference.recommended_lora_scale = float(recommended_scale)

    if class_prompt is not None:
        cfg.inference.class_prompt = str(class_prompt)

    return cfg.save(path)


def main() -> None:
    import argparse
    from src.dynamic_lora_t2i.config import setup_logging

    setup_logging()

    p = argparse.ArgumentParser(description="EntityConfig utilities (init/check/set-adapter).")
    sub = p.add_subparsers(dest="cmd", required=True)

    p_init = sub.add_parser("init", help="Create data/entities/<entity>/entity_config.json")
    p_init.add_argument("--entity", required=True, help="Entity name (folder in data/entities/)")
    p_init.add_argument("--token", required=True, help="Placeholder token, e.g. vlad_object")
    p_init.add_argument("--desc", default="", help="Human description")
    p_init.add_argument("--base-model", default=DEFAULT_BASE_MODEL_ID, help="Base model id")
    p_init.add_argument("--class-prompt", default="", help="Optional class prompt (e.g. 'toy car')")

    p_check = sub.add_parser("check", help="Load + validate entity_config.json")
    p_check.add_argument("--entity", required=True, help="Entity name (folder in data/entities/)")

    p_set = sub.add_parser("set-adapter", help="Bind default LoRA adapter + recommended scale to entity")
    p_set.add_argument("--entity", required=True, help="Entity name (folder in data/entities/)")
    p_set.add_argument("--adapter", required=True, help="Run name / relative path inside adapters_dir / absolute path")
    p_set.add_argument("--scale", type=float, default=1.0, help="Recommended LoRA scale (e.g. 0.6..1.2)")
    p_set.add_argument("--class-prompt", default=None, help="Optional override class_prompt")

    args = p.parse_args()

    if args.cmd == "init":
        cfg = init_entity_config(
            entity_name=args.entity,
            placeholder_token=args.token,
            description=args.desc,
            base_model_id=args.base_model,
            class_prompt=args.class_prompt,
        )
        print(f"OK: created {cfg.config_path()}")
        print(json.dumps(cfg.to_dict(), ensure_ascii=False, indent=2))

    elif args.cmd == "check":
        safe = sanitize_entity_name(args.entity)
        path = (ENTITIES_DIR / safe / "entity_config.json").resolve()
        cfg = EntityConfig.load(path, resolve_paths=True, validate=True)
        cfg.validate(strict=False)
        print("OK: entity config is valid")
        print(json.dumps(cfg.to_dict(), ensure_ascii=False, indent=2))

    elif args.cmd == "set-adapter":
        saved = set_default_adapter(
            entity_name=args.entity,
            adapter=args.adapter,
            recommended_scale=float(args.scale),
            class_prompt=args.class_prompt,
        )
        print(f"OK: updated {saved}")
        cfg = EntityConfig.load(saved, resolve_paths=True, validate=True)
        print(json.dumps(cfg.to_dict(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
