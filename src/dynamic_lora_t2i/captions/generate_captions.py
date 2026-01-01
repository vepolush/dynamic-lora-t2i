# src/dynamic_lora_t2i/captions/generate_captions.py

from __future__ import annotations

import argparse
from pathlib import Path

try:
    from src.dynamic_lora_t2i.config import ENTITIES_DIR, setup_logging
except Exception:
    from src.dynamic_lora_t2i.utils.config import ENTITIES_DIR, setup_logging  # type: ignore

from src.dynamic_lora_t2i.captions.blip_captioner import generate_entity_captions


def main() -> None:
    setup_logging()

    parser = argparse.ArgumentParser(description="Generate BLIP captions for images in an entity folder.")
    parser.add_argument(
        "--entity",
        required=True,
        help="Entity folder name inside data/entities/ (will be resolved under ENTITIES_DIR)",
    )
    parser.add_argument("--token", required=True, help="Placeholder token, e.g. vlad_object")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing caption files")
    parser.add_argument("--caption-ext", default=".txt", help="Caption extension (default: .txt)")
    parser.add_argument("--model-id", default="Salesforce/blip-image-captioning-base", help="BLIP model id")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size (useful on CUDA)")
    parser.add_argument("--max-new-tokens", type=int, default=40, help="Max new tokens for generation")

    args = parser.parse_args()

    entity_dir = (Path(ENTITIES_DIR) / args.entity).resolve()
    n = generate_entity_captions(
        entity_dir=entity_dir,
        placeholder_token=args.token,
        overwrite=args.overwrite,
        caption_ext=args.caption_ext,
        model_id=args.model_id,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
    )

    print(f"OK: generated {n} captions in {entity_dir}")


if __name__ == "__main__":
    main()
