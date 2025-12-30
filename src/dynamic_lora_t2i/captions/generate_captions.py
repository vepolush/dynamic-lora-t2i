# src/dynamic_lora_t2i/captions/generate_captions.py

from __future__ import annotations

import argparse
from pathlib import Path

from src.dynamic_lora_t2i.config import ENTITIES_DIR, setup_logging
from src.dynamic_lora_t2i.captions.blip_captioner import generate_entity_captions


def main() -> None:
    setup_logging()

    parser = argparse.ArgumentParser()
    parser.add_argument("--entity", required=True, help="Entity folder name inside data/entities/")
    parser.add_argument("--token", required=True, help="Placeholder token, e.g. vlad_object")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing .txt captions")
    args = parser.parse_args()

    entity_dir = (ENTITIES_DIR / args.entity).resolve()
    n = generate_entity_captions(entity_dir=entity_dir, placeholder_token=args.token, overwrite=args.overwrite)

    print(f"OK: generated {n} captions in {entity_dir}")


if __name__ == "__main__":
    main()
