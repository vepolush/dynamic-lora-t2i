# src/dynamic_lora_t2i/training/lora_trainer.py

from __future__ import annotations

import json
import logging
import math
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from PIL import Image

from src.dynamic_lora_t2i.config import (
    DEFAULT_IMAGE_HEIGHT,
    DEFAULT_IMAGE_WIDTH,
    ensure_project_directories,
    setup_logging,
)
from src.dynamic_lora_t2i.training.configs import (
    TrainConfig,
    load_train_config,
    save_train_config,
    train_config_to_dict,
)

logger = logging.getLogger(__name__)

IMAGE_EXTS_DEFAULT = {".png", ".jpg", ".jpeg", ".webp"}


# -----------------------------
# Dataset
# -----------------------------
class EntityCaptionDataset(Dataset):
    def __init__(
        self,
        entity_dir: Path,
        placeholder_token: str,
        *,
        captions_ext: str = ".txt",
        image_exts: Optional[list[str]] = None,
        width: int = DEFAULT_IMAGE_WIDTH,
        height: int = DEFAULT_IMAGE_HEIGHT,
        max_images: Optional[int] = None,
    ):
        self.entity_dir = entity_dir.resolve()
        self.placeholder_token = placeholder_token
        self.captions_ext = captions_ext
        self.width = int(width)
        self.height = int(height)

        exts = set((image_exts or list(IMAGE_EXTS_DEFAULT)))
        exts = {e.lower() for e in exts}
        self.image_paths = [p for p in self.entity_dir.rglob("*") if p.is_file() and p.suffix.lower() in exts]
        self.image_paths.sort()

        if max_images is not None:
            self.image_paths = self.image_paths[: int(max_images)]

        if not self.image_paths:
            raise ValueError(f"No images found in entity_dir: {self.entity_dir}")

        self._use_torchvision = False
        self._tv = None
        try:
            from torchvision import transforms  # type: ignore

            self._tv = transforms
            self._use_torchvision = True
        except Exception:
            self._use_torchvision = False

    def __len__(self) -> int:
        return len(self.image_paths)

    def _read_caption(self, img_path: Path) -> str:
        cap_path = img_path.with_suffix(self.captions_ext)
        if cap_path.exists():
            txt = cap_path.read_text(encoding="utf-8", errors="ignore").strip()
            if txt:
                return txt

        # fallback — мінімально придатний caption
        return f"photo of {self.placeholder_token}"

    def _pil_to_tensor_manual(self, img: Image.Image) -> torch.Tensor:
        # resize -> (H,W) and normalize to [-1, 1]
        img = img.resize((self.width, self.height), resample=Image.BICUBIC)
        arr = torch.from_numpy(__import__("numpy").array(img)).float() / 255.0  # (H,W,C)
        if arr.ndim == 2:
            arr = arr.unsqueeze(-1)
        arr = arr.permute(2, 0, 1)  # (C,H,W)
        if arr.shape[0] == 1:
            arr = arr.repeat(3, 1, 1)
        arr = (arr - 0.5) / 0.5
        return arr

    def _transform(self, img: Image.Image) -> torch.Tensor:
        img = img.convert("RGB")
        if self._use_torchvision and self._tv is not None:
            transforms = self._tv
            tfm = transforms.Compose(
                [
                    transforms.Resize((self.height, self.width), interpolation=transforms.InterpolationMode.BICUBIC),
                    transforms.CenterCrop((self.height, self.width)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
                ]
            )
            return tfm(img)
        # manual path (requires numpy)
        return self._pil_to_tensor_manual(img)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        img_path = self.image_paths[idx]
        caption = self._read_caption(img_path)
        img = Image.open(img_path)
        pixel_values = self._transform(img)
        return {
            "pixel_values": pixel_values,
            "prompt": caption,
            "path": str(img_path),
        }


def collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
    pixel_values = torch.stack([x["pixel_values"] for x in batch], dim=0)
    prompts = [x["prompt"] for x in batch]
    paths = [x["path"] for x in batch]
    return {"pixel_values": pixel_values, "prompts": prompts, "paths": paths}


# -----------------------------
# SDXL helpers
# -----------------------------
def _is_mps(device: torch.device) -> bool:
    return device.type == "mps"


def _pick_torch_dtype(mixed_precision: str, device: torch.device) -> torch.dtype:
    if mixed_precision in ("fp16", "bf16") and device.type in ("cuda", "mps"):
        return torch.float16 if mixed_precision == "fp16" else torch.bfloat16
    return torch.float32


def _make_time_ids(batch_size: int, height: int, width: int, device: torch.device) -> torch.Tensor:
    # SDXL expects: [orig_h, orig_w, crop_top, crop_left, target_h, target_w]
    t = torch.tensor([height, width, 0, 0, height, width], device=device, dtype=torch.float32)
    return t.unsqueeze(0).repeat(batch_size, 1)


def _encode_prompts_sdxl(pipe: Any, prompts: list[str], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Returns: (prompt_embeds, pooled_prompt_embeds)
    Handles different diffusers versions.
    """
    if not hasattr(pipe, "encode_prompt"):
        raise RuntimeError("Pipeline has no encode_prompt(); please upgrade diffusers or use SDXL pipeline.")

    out = pipe.encode_prompt(
        prompts,
        device=device,
        num_images_per_prompt=1,
        do_classifier_free_guidance=False,
    )

    if isinstance(out, tuple):
        if len(out) == 2:
            prompt_embeds, pooled = out
            return prompt_embeds, pooled
        if len(out) == 4:
            prompt_embeds, _, pooled, _ = out
            return prompt_embeds, pooled

    raise RuntimeError(f"Unexpected encode_prompt() output format: {type(out)}")


def _try_enable_gradient_checkpointing(unet: Any) -> None:
    try:
        if hasattr(unet, "enable_gradient_checkpointing"):
            unet.enable_gradient_checkpointing()
            logger.info("Enabled gradient checkpointing for UNet.")
    except Exception as e:
        logger.warning("Could not enable gradient checkpointing: %s", e)


# -----------------------------
# LoRA
# -----------------------------
def add_lora_to_unet(unet: torch.nn.Module, cfg: TrainConfig) -> torch.nn.Module:
    try:
        from peft import LoraConfig, get_peft_model  # type: ignore
    except ImportError as e:
        raise ImportError(
            "peft is required for LoRA training. Install: pip install peft"
        ) from e

    target_modules = cfg.lora.target_modules or ["to_k", "to_q", "to_v", "to_out.0"]

    lora_config = LoraConfig(
        r=int(cfg.lora.rank),
        lora_alpha=int(cfg.lora.alpha),
        lora_dropout=float(cfg.lora.dropout),
        target_modules=list(target_modules),
        bias="none",
    )

    unet = get_peft_model(unet, lora_config)
    unet.print_trainable_parameters()
    return unet


def extract_lora_state_dict(unet: torch.nn.Module) -> dict[str, torch.Tensor]:
    """
    Return only LoRA weights (PEFT adapter state dict if possible).
    """
    try:
        from peft.utils import get_peft_model_state_dict  # type: ignore

        sd = get_peft_model_state_dict(unet)
        return {k: v.detach().cpu() for k, v in sd.items()}
    except Exception:
        sd_full = unet.state_dict()
        sd = {k: v.detach().cpu() for k, v in sd_full.items() if ("lora_" in k or ".lora" in k)}
        if not sd:
            # last resort: save whole state dict (heavy)
            sd = {k: v.detach().cpu() for k, v in sd_full.items()}
        return sd


def save_lora_weights(
    pipe: Any,
    unet: torch.nn.Module,
    out_dir: Path,
    *,
    meta: dict[str, Any],
) -> Path:
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    state_dict = extract_lora_state_dict(unet)

    # Try to convert to diffusers LoRA format (if available)
    try:
        from diffusers.utils import convert_state_dict_to_diffusers  # type: ignore

        state_dict = convert_state_dict_to_diffusers(state_dict)
    except Exception:
        pass

    # Prefer diffusers native saving (so you can load via pipe.load_lora_weights later)
    saved_path: Optional[Path] = None
    try:
        if hasattr(pipe, "save_lora_weights"):
            pipe.save_lora_weights(str(out_dir), unet_lora_layers=state_dict)
            # diffusers typically writes: pytorch_lora_weights.safetensors
            cand = out_dir / "pytorch_lora_weights.safetensors"
            if cand.exists():
                saved_path = cand
    except Exception as e:
        logger.warning("pipe.save_lora_weights failed, will fallback: %s", e)

    # Fallback: save ourselves
    if saved_path is None:
        try:
            from safetensors.torch import save_file  # type: ignore

            saved_path = out_dir / "unet_lora.safetensors"
            save_file(state_dict, str(saved_path))
        except Exception:
            saved_path = out_dir / "unet_lora.pt"
            torch.save(state_dict, saved_path)

    (out_dir / "lora_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Saved LoRA weights: %s", saved_path)
    return saved_path


# -----------------------------
# Training
# -----------------------------
def train_one_entity(cfg: TrainConfig) -> dict[str, Any]:
    ensure_project_directories()
    cfg.resolve_paths()
    cfg.validate()
    cfg.prepare_dirs()

    # Lazy imports so the script errors nicely if deps are missing
    try:
        from accelerate import Accelerator  # type: ignore
        from accelerate.utils import set_seed  # type: ignore
    except ImportError as e:
        raise ImportError("accelerate is required. Install: pip install accelerate") from e

    try:
        from diffusers import StableDiffusionXLPipeline  # type: ignore
        from diffusers.optimization import get_scheduler  # type: ignore
    except ImportError as e:
        raise ImportError("diffusers is required. Install: pip install diffusers") from e

    mp = str(cfg.train.mixed_precision)

    if mp != "no" and torch.backends.mps.is_available() and not torch.cuda.is_available():
        logger.warning("MPS detected (no CUDA): forcing mixed_precision='no' for stability.")
        mp = "no"

    cfg.train.mixed_precision = mp

    accelerator = Accelerator(
        gradient_accumulation_steps=int(cfg.train.gradient_accum_steps),
        mixed_precision=mp,
    )
    device = accelerator.device

    set_seed(int(cfg.train.seed))

    # Save resolved config near the run outputs
    resolved_cfg_path = Path(cfg.output.output_dir) / "train_config_resolved.json"
    save_train_config(cfg, resolved_cfg_path)

    # Dataset + loader
    dataset = EntityCaptionDataset(
        entity_dir=Path(cfg.data.entity_dir),
        placeholder_token=cfg.data.placeholder_token,
        captions_ext=cfg.data.captions_ext,
        image_exts=cfg.data.image_exts,
        width=cfg.data.width,
        height=cfg.data.height,
        max_images=cfg.data.max_images,
    )

    train_loader = DataLoader(
        dataset,
        batch_size=int(cfg.train.train_batch_size),
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn,
        drop_last=True,
    )

    # Steps
    num_update_steps_per_epoch = math.ceil(len(train_loader) / int(cfg.train.gradient_accum_steps))
    max_train_steps = cfg.train.max_train_steps
    if max_train_steps is None:
        max_train_steps = int(cfg.train.num_epochs) * num_update_steps_per_epoch

    # Model/pipeline
    torch_dtype = _pick_torch_dtype(mp, device)
    if device.type == "mps":
        torch_dtype = torch.float32

    logger.info("Loading SDXL pipeline: %s", cfg.model.base_model_id)

    try:
        pipe = StableDiffusionXLPipeline.from_pretrained(
            cfg.model.base_model_id,
            dtype=torch_dtype,
            use_safetensors=True,
        )
    except TypeError:
        pipe = StableDiffusionXLPipeline.from_pretrained(
            cfg.model.base_model_id,
            torch_dtype=torch_dtype,
            use_safetensors=True,
        )

    # speed/memory knobs
    try:
        pipe.enable_attention_slicing()
    except Exception:
        pass
    try:
        pipe.enable_vae_slicing()
    except Exception:
        pass

    # Move to device
    pipe.to(device)

    if device.type == "mps":
        # ensure consistent dtype everywhere
        pipe.unet.to(dtype=torch.float32)
        if getattr(pipe, "text_encoder", None) is not None:
            pipe.text_encoder.to(dtype=torch.float32)
        if getattr(pipe, "text_encoder_2", None) is not None:
            pipe.text_encoder_2.to(dtype=torch.float32)
        if getattr(pipe, "vae", None) is not None:
            pipe.vae.to(dtype=torch.float32)

    # Freeze everything except LoRA
    pipe.unet.requires_grad_(False)
    if getattr(pipe, "text_encoder", None) is not None:
        pipe.text_encoder.requires_grad_(False)
    if getattr(pipe, "text_encoder_2", None) is not None:
        pipe.text_encoder_2.requires_grad_(False)
    if getattr(pipe, "vae", None) is not None:
        pipe.vae.requires_grad_(False)

    # For MPS stability: keep VAE in fp32
    if _is_mps(device) and getattr(pipe, "vae", None) is not None:
        try:
            pipe.vae.to(dtype=torch.float32)
        except Exception:
            pass

    # Add LoRA on UNet
    pipe.unet = add_lora_to_unet(pipe.unet, cfg)
    _try_enable_gradient_checkpointing(pipe.unet)

    # Optimizer
    params_to_optimize = [p for p in pipe.unet.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params_to_optimize, lr=float(cfg.train.learning_rate))

    # LR scheduler
    lr_scheduler = get_scheduler(
        name=str(cfg.train.lr_scheduler),
        optimizer=optimizer,
        num_warmup_steps=int(cfg.train.warmup_steps),
        num_training_steps=int(max_train_steps),
    )

    # Prepare with accelerator
    pipe.unet, optimizer, train_loader, lr_scheduler = accelerator.prepare(
        pipe.unet, optimizer, train_loader, lr_scheduler
    )

    # Noise scheduler (from pipeline)
    noise_scheduler = pipe.scheduler
    prediction_type = getattr(noise_scheduler.config, "prediction_type", "epsilon")

    # Scaling factor for latents
    scaling_factor = 0.13025
    if getattr(pipe, "vae", None) is not None:
        scaling_factor = float(getattr(getattr(pipe.vae, "config", None), "scaling_factor", scaling_factor))

    logger.info(
        "Training start | images=%d | batch=%d | accum=%d | steps=%d | dtype=%s | device=%s",
        len(dataset),
        int(cfg.train.train_batch_size),
        int(cfg.train.gradient_accum_steps),
        int(max_train_steps),
        str(torch_dtype),
        str(device),
    )

    global_step = 0
    running_loss = 0.0

    # optional tqdm
    pbar = None
    try:
        from tqdm.auto import tqdm  # type: ignore

        pbar = tqdm(total=int(max_train_steps), disable=not accelerator.is_local_main_process)
    except Exception:
        pbar = None

    pipe.unet.train()

    for epoch in range(int(cfg.train.num_epochs)):
        for step, batch in enumerate(train_loader):
            with accelerator.accumulate(pipe.unet):
                pixel_values = batch["pixel_values"].to(device=device, dtype=torch.float32)
                prompts: list[str] = batch["prompts"]

                # VAE encode -> latents
                with torch.no_grad():
                    latents = pipe.vae.encode(pixel_values).latent_dist.sample()
                    latents = latents * scaling_factor

                # Sample noise + timestep
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                timesteps = torch.randint(
                    0,
                    noise_scheduler.config.num_train_timesteps,
                    (bsz,),
                    device=latents.device,
                    dtype=torch.int64,
                )

                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # Text embeddings (SDXL)
                with torch.no_grad():
                    prompt_embeds, pooled_prompt_embeds = _encode_prompts_sdxl(pipe, prompts, device=device)

                if device.type == "mps":
                    prompt_embeds = prompt_embeds.to(dtype=torch.float32)
                    pooled_prompt_embeds = pooled_prompt_embeds.to(dtype=torch.float32)

                time_ids = _make_time_ids(bsz, int(cfg.data.height), int(cfg.data.width), device=device)
                added_cond_kwargs = {"text_embeds": pooled_prompt_embeds, "time_ids": time_ids}

                # Predict noise
                model_out = pipe.unet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=prompt_embeds,
                    added_cond_kwargs=added_cond_kwargs,
                )
                model_pred = model_out.sample if hasattr(model_out, "sample") else model_out[0]

                if prediction_type == "epsilon":
                    target = noise
                elif prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(f"Unsupported prediction_type: {prediction_type}")

                loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            # logging
            if accelerator.sync_gradients:
                global_step += 1
                running_loss += float(loss.detach().item())

                if pbar is not None:
                    pbar.update(1)
                    pbar.set_postfix({"loss": f"{loss.detach().item():.4f}"})

                if global_step >= int(max_train_steps):
                    break

        if global_step >= int(max_train_steps):
            break

    if pbar is not None:
        pbar.close()

    avg_loss = running_loss / max(1, global_step)
    logger.info("Training finished. steps=%d, avg_loss=%.6f", global_step, avg_loss)

    # Save LoRA
    meta = {
        "entity_name": cfg.data.entity_name,
        "placeholder_token": cfg.data.placeholder_token,
        "base_model_id": cfg.model.base_model_id,
        "global_step": global_step,
        "avg_loss": avg_loss,
        "train_config": train_config_to_dict(cfg),
    }

    out_dir = Path(cfg.output.output_dir)
    weights_path = save_lora_weights(pipe, pipe.unet, out_dir, meta=meta)

    # Optional validation image
    val_img_path = None
    if cfg.train.validation_prompt:
        try:
            pipe.set_progress_bar_config(disable=True)
        except Exception:
            pass

        try:
            pipe.unet.eval()
            with torch.no_grad():
                img = pipe(
                    prompt=cfg.train.validation_prompt,
                    num_inference_steps=20,
                    guidance_scale=7.0,
                    width=int(cfg.data.width),
                    height=int(cfg.data.height),
                ).images[0]
            val_img_path = str(Path(cfg.output.results_dir) / "validation.png")
            Path(val_img_path).parent.mkdir(parents=True, exist_ok=True)
            img.save(val_img_path)
            logger.info("Saved validation image: %s", val_img_path)
        except Exception as e:
            logger.warning("Validation generation failed: %s", e)

    summary = {
        "ok": True,
        "global_step": global_step,
        "avg_loss": avg_loss,
        "weights_path": str(weights_path),
        "validation_image": val_img_path,
        "output_dir": str(out_dir),
        "results_dir": str(cfg.output.results_dir),
        "logs_dir": str(cfg.output.logs_dir),
    }

    (out_dir / "training_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    setup_logging()

    import argparse

    p = argparse.ArgumentParser(description="Train LoRA for one entity (SDXL).")
    p.add_argument("--config", required=True, help="Path to training config (.yaml/.yml/.json)")
    args = p.parse_args()

    cfg = load_train_config(Path(args.config), resolve_paths=True, validate=True)
    summary = train_one_entity(cfg)

    print("OK")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
