# src/dynamic_lora_t2i/training/lora_trainer.py

from __future__ import annotations

import json
import logging
import math
import os
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from PIL import Image

try:
    from src.dynamic_lora_t2i.config import (
        DEFAULT_IMAGE_HEIGHT,
        DEFAULT_IMAGE_WIDTH,
        ensure_project_directories,
        setup_logging,
        HF_HOME,
        TRANSFORMERS_CACHE,
        DIFFUSERS_CACHE,
        get_device,
    )
except Exception:
    from src.dynamic_lora_t2i.utils.config import (  # type: ignore
        DEFAULT_IMAGE_HEIGHT,
        DEFAULT_IMAGE_WIDTH,
        ensure_project_directories,
        setup_logging,
        HF_HOME,
        TRANSFORMERS_CACHE,
        DIFFUSERS_CACHE,
        get_device,
    )

from src.dynamic_lora_t2i.training.configs import (
    TrainConfig,
    load_train_config,
    save_train_config,
    train_config_to_dict,
)

logger = logging.getLogger(__name__)

IMAGE_EXTS_DEFAULT = {".png", ".jpg", ".jpeg", ".webp"}


def _ensure_hf_cache_env() -> None:
    try:
        if "HF_HOME" not in os.environ and HF_HOME is not None:
            os.environ["HF_HOME"] = str(HF_HOME)
        if "TRANSFORMERS_CACHE" not in os.environ and TRANSFORMERS_CACHE is not None:
            os.environ["TRANSFORMERS_CACHE"] = str(TRANSFORMERS_CACHE)
        if "DIFFUSERS_CACHE" not in os.environ and DIFFUSERS_CACHE is not None:
            os.environ["DIFFUSERS_CACHE"] = str(DIFFUSERS_CACHE)
    except Exception:
        pass


def _apply_cuda_perf_tweaks() -> None:
    if not torch.cuda.is_available():
        return
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    except Exception:
        pass
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass


def _num_workers_for_dataloader() -> int:
    try:
        cpu = os.cpu_count() or 4
    except Exception:
        cpu = 4
    return max(2, min(8, cpu - 1))


def _maybe_enable_xformers(pipe: Any) -> None:
    if not torch.cuda.is_available():
        return
    if os.getenv("DYNAMIC_LORA_T2I_DISABLE_XFORMERS", "0").strip().lower() in ("1", "true", "yes", "y"):
        return
    try:
        pipe.enable_xformers_memory_efficient_attention()
        logger.info("Enabled xFormers memory efficient attention")
    except Exception as e:
        logger.warning("xFormers not enabled (%s). Falling back to attention slicing.", e)
        try:
            pipe.enable_attention_slicing("auto")
            logger.info("Enabled attention slicing (auto)")
        except Exception:
            logger.warning("Attention slicing not available for this pipeline")


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
        self.entity_dir = Path(entity_dir).resolve()
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

        self._np = None
        if not self._use_torchvision:
            try:
                import numpy as np

                self._np = np
            except Exception as e:
                raise ImportError("torchvision is not available and numpy is missing. Install one of them.") from e

    def __len__(self) -> int:
        return len(self.image_paths)

    def _read_caption(self, img_path: Path) -> str:
        cap_path = img_path.with_suffix(self.captions_ext)
        if cap_path.exists():
            txt = cap_path.read_text(encoding="utf-8", errors="ignore").strip()
            if txt:
                return txt
        return f"photo of {self.placeholder_token}"

    def _pil_to_tensor_manual(self, img: Image.Image) -> torch.Tensor:
        img = img.resize((self.width, self.height), resample=Image.BICUBIC)
        np = self._np
        assert np is not None

        arr = np.asarray(img, dtype=np.float32) / 255.0
        if arr.ndim == 2:
            arr = arr[..., None]
        if arr.shape[2] == 1:
            arr = np.repeat(arr, 3, axis=2)

        t = torch.from_numpy(arr).permute(2, 0, 1)
        t = (t - 0.5) / 0.5
        return t

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

        return self._pil_to_tensor_manual(img)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        img_path = self.image_paths[idx]
        caption = self._read_caption(img_path)

        with Image.open(img_path) as img:
            pixel_values = self._transform(img)

        return {"pixel_values": pixel_values, "prompt": caption, "path": str(img_path)}


def collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
    pixel_values = torch.stack([x["pixel_values"] for x in batch], dim=0)
    prompts = [x["prompt"] for x in batch]
    paths = [x["path"] for x in batch]
    return {"pixel_values": pixel_values, "prompts": prompts, "paths": paths}


def _pick_pipeline_dtype(mixed_precision: str, device: torch.device) -> torch.dtype:
    if device.type == "mps":
        return torch.float32
    if mixed_precision == "bf16" and device.type == "cuda":
        return torch.bfloat16
    if mixed_precision == "fp16" and device.type == "cuda":
        return torch.float16
    return torch.float32


def _make_time_ids(batch_size: int, height: int, width: int, device: torch.device) -> torch.Tensor:
    t = torch.tensor([height, width, 0, 0, height, width], device=device, dtype=torch.float32)
    return t.unsqueeze(0).repeat(batch_size, 1)


def _encode_prompts_sdxl(pipe: Any, prompts: list[str], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
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


def add_lora_to_unet(unet: torch.nn.Module, cfg: TrainConfig) -> torch.nn.Module:
    try:
        from peft import LoraConfig, get_peft_model
    except ImportError as e:
        raise ImportError("peft is required for LoRA training. Install: pip install peft") from e

    target_modules = cfg.lora.target_modules or ["to_k", "to_q", "to_v", "to_out.0"]

    lora_config = LoraConfig(
        r=int(cfg.lora.rank),
        lora_alpha=int(cfg.lora.alpha),
        lora_dropout=float(cfg.lora.dropout),
        target_modules=list(target_modules),
        bias="none",
    )

    unet = get_peft_model(unet, lora_config)

    try:
        unet.print_trainable_parameters()
    except Exception:
        pass

    return unet


def extract_lora_state_dict(unet: torch.nn.Module) -> dict[str, torch.Tensor]:
    try:
        from peft.utils import get_peft_model_state_dict  # type: ignore

        sd = get_peft_model_state_dict(unet)
        return {k: v.detach().cpu() for k, v in sd.items()}
    except Exception:
        sd_full = unet.state_dict()
        sd = {k: v.detach().cpu() for k, v in sd_full.items() if ("lora_" in k or ".lora" in k)}
        if not sd:
            sd = {k: v.detach().cpu() for k, v in sd_full.items()}
        return sd


def save_lora_weights(
    pipe: Any,
    unet: torch.nn.Module,
    out_dir: Path,
    *,
    meta: dict[str, Any],
) -> Path:
    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    adapter_dir = out_dir / "peft_adapter"
    saved_adapter = False
    try:
        if hasattr(unet, "save_pretrained"):
            adapter_dir.mkdir(parents=True, exist_ok=True)
            # peft supports safe_serialization=True
            unet.save_pretrained(str(adapter_dir), safe_serialization=True)
            saved_adapter = True
            logger.info("Saved PEFT adapter to %s", adapter_dir)
    except Exception as e:
        logger.warning("Saving PEFT adapter failed: %s", e)

    state_dict = extract_lora_state_dict(unet)

    try:
        from diffusers.utils import convert_state_dict_to_diffusers

        state_dict = convert_state_dict_to_diffusers(state_dict)
    except Exception:
        pass

    saved_path: Optional[Path] = None
    try:
        from safetensors.torch import save_file

        saved_path = out_dir / "unet_lora.safetensors"
        save_file(state_dict, str(saved_path))
    except Exception:
        saved_path = out_dir / "unet_lora.pt"
        torch.save(state_dict, saved_path)

    (out_dir / "lora_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Saved LoRA weights file: %s (adapter_saved=%s)", saved_path, saved_adapter)
    return saved_path


def train_one_entity(cfg: TrainConfig) -> dict[str, Any]:
    ensure_project_directories()
    _ensure_hf_cache_env()
    _apply_cuda_perf_tweaks()

    cfg.resolve_paths()
    cfg.validate()
    cfg.prepare_dirs()

    try:
        from accelerate import Accelerator
        from accelerate.utils import set_seed
    except ImportError as e:
        raise ImportError("accelerate is required. Install: pip install accelerate") from e

    try:
        from diffusers import StableDiffusionXLPipeline
        from diffusers.optimization import get_scheduler
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

    resolved_cfg_path = Path(cfg.output.output_dir) / "train_config_resolved.json"
    if accelerator.is_main_process:
        save_train_config(cfg, resolved_cfg_path)
    accelerator.wait_for_everyone()

    dataset = EntityCaptionDataset(
        entity_dir=Path(cfg.data.entity_dir),
        placeholder_token=cfg.data.placeholder_token,
        captions_ext=cfg.data.captions_ext,
        image_exts=cfg.data.image_exts,
        width=cfg.data.width,
        height=cfg.data.height,
        max_images=cfg.data.max_images,
    )

    use_pin = device.type == "cuda"
    num_workers = int(os.getenv("DYNAMIC_LORA_T2I_NUM_WORKERS", str(_num_workers_for_dataloader())))

    train_loader = DataLoader(
        dataset,
        batch_size=int(cfg.train.train_batch_size),
        shuffle=True,
        num_workers=num_workers,
        pin_memory=use_pin,
        persistent_workers=(num_workers > 0),
        collate_fn=collate_fn,
        drop_last=True,
    )

    num_update_steps_per_epoch = math.ceil(len(train_loader) / int(cfg.train.gradient_accum_steps))
    max_train_steps = cfg.train.max_train_steps
    if max_train_steps is None:
        max_train_steps = int(cfg.train.num_epochs) * num_update_steps_per_epoch

    pipe_dtype = _pick_pipeline_dtype(mp, device)

    logger.info("Loading SDXL pipeline: %s", cfg.model.base_model_id)
    logger.info("device=%s mixed_precision=%s load_dtype=%s", device, mp, pipe_dtype)

    cache_dir = None
    try:
        cache_dir = str(DIFFUSERS_CACHE)
    except Exception:
        cache_dir = os.getenv("DIFFUSERS_CACHE") or None

    load_kwargs: dict[str, Any] = {
        "use_safetensors": True,
    }
    if cache_dir:
        load_kwargs["cache_dir"] = cache_dir

    variant = os.getenv("DYNAMIC_LORA_T2I_MODEL_VARIANT", "").strip() or None
    if variant:
        load_kwargs["variant"] = variant

    try:
        pipe = StableDiffusionXLPipeline.from_pretrained(
            cfg.model.base_model_id,
            dtype=pipe_dtype,
            **load_kwargs,
        )
    except TypeError:
        pipe = StableDiffusionXLPipeline.from_pretrained(
            cfg.model.base_model_id,
            torch_dtype=pipe_dtype,
            **load_kwargs,
        )

    try:
        pipe.enable_vae_slicing()
    except Exception:
        pass

    _maybe_enable_xformers(pipe)

    pipe.to(device)

    pipe.unet.requires_grad_(False)
    if getattr(pipe, "text_encoder", None) is not None:
        pipe.text_encoder.requires_grad_(False)
    if getattr(pipe, "text_encoder_2", None) is not None:
        pipe.text_encoder_2.requires_grad_(False)
    if getattr(pipe, "vae", None) is not None:
        pipe.vae.requires_grad_(False)

    pipe.unet = add_lora_to_unet(pipe.unet, cfg)
    _try_enable_gradient_checkpointing(pipe.unet)

    params_to_optimize = [p for p in pipe.unet.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params_to_optimize, lr=float(cfg.train.learning_rate))

    lr_scheduler = get_scheduler(
        name=str(cfg.train.lr_scheduler),
        optimizer=optimizer,
        num_warmup_steps=int(cfg.train.warmup_steps),
        num_training_steps=int(max_train_steps),
    )

    pipe.unet, optimizer, train_loader, lr_scheduler = accelerator.prepare(
        pipe.unet, optimizer, train_loader, lr_scheduler
    )

    noise_scheduler = pipe.scheduler
    prediction_type = getattr(noise_scheduler.config, "prediction_type", "epsilon")

    scaling_factor = 0.13025
    if getattr(pipe, "vae", None) is not None:
        scaling_factor = float(getattr(getattr(pipe.vae, "config", None), "scaling_factor", scaling_factor))

    vae_dtype = getattr(pipe.vae, "dtype", torch.float32) if getattr(pipe, "vae", None) is not None else torch.float32
    try:
        unet_param = next(iter(pipe.unet.parameters()))
        unet_dtype = unet_param.dtype
    except Exception:
        unet_dtype = pipe_dtype

    logger.info(
        "Training start | images=%d | batch=%d | accum=%d | max_steps=%d | unet_dtype=%s | vae_dtype=%s | device=%s | workers=%d",
        len(dataset),
        int(cfg.train.train_batch_size),
        int(cfg.train.gradient_accum_steps),
        int(max_train_steps),
        str(unet_dtype),
        str(vae_dtype),
        str(device),
        num_workers,
    )

    pbar = None
    try:
        from tqdm.auto import tqdm

        pbar = tqdm(total=int(max_train_steps), disable=not accelerator.is_local_main_process)
    except Exception:
        pbar = None

    pipe.unet.train()

    global_step = 0
    running_loss = 0.0

    save_every = int(getattr(cfg.output, "save_every_n_steps", 0) or 0)
    val_every = int(getattr(cfg.train, "validation_steps", 0) or 0)
    do_val = bool(cfg.train.validation_prompt and val_every > 0)

    for epoch in range(int(cfg.train.num_epochs)):
        for _, batch in enumerate(train_loader):
            with accelerator.accumulate(pipe.unet):
                pixel_values = batch["pixel_values"].to(device=device, non_blocking=True)
                prompts: list[str] = batch["prompts"]

                with torch.no_grad():
                    with accelerator.autocast():
                        pv = pixel_values.to(dtype=vae_dtype)
                        latents = pipe.vae.encode(pv).latent_dist.sample()
                        latents = latents * scaling_factor

                latents = latents.to(dtype=unet_dtype)

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

                with torch.no_grad():
                    with accelerator.autocast():
                        prompt_embeds, pooled_prompt_embeds = _encode_prompts_sdxl(pipe, prompts, device=device)

                time_ids = _make_time_ids(bsz, int(cfg.data.height), int(cfg.data.width), device=device)
                added_cond_kwargs = {"text_embeds": pooled_prompt_embeds, "time_ids": time_ids}

                with accelerator.autocast():
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

            if accelerator.sync_gradients:
                global_step += 1
                running_loss += float(loss.detach().item())

                if pbar is not None:
                    pbar.update(1)
                    pbar.set_postfix({"loss": f"{loss.detach().item():.4f}"})

                if save_every > 0 and (global_step % save_every == 0):
                    if accelerator.is_main_process:
                        out_dir = Path(cfg.output.output_dir) / f"checkpoint_step_{global_step:06d}"
                        meta = {
                            "entity_name": cfg.data.entity_name,
                            "placeholder_token": cfg.data.placeholder_token,
                            "base_model_id": cfg.model.base_model_id,
                            "global_step": global_step,
                            "avg_loss_so_far": running_loss / max(1, global_step),
                            "train_config": train_config_to_dict(cfg),
                        }
                        unwrapped_unet = accelerator.unwrap_model(pipe.unet)
                        save_lora_weights(pipe, unwrapped_unet, out_dir, meta=meta)
                    accelerator.wait_for_everyone()

                if do_val and (global_step % val_every == 0) and accelerator.is_main_process:
                    try:
                        pipe.set_progress_bar_config(disable=True)
                    except Exception:
                        pass
                    try:
                        pipe.unet = accelerator.unwrap_model(pipe.unet)
                        pipe.unet.eval()
                        with torch.no_grad():
                            with torch.autocast("cuda", dtype=unet_dtype) if device.type == "cuda" else torch.no_grad():
                                img = pipe(
                                    prompt=str(cfg.train.validation_prompt),
                                    num_inference_steps=20,
                                    guidance_scale=7.0,
                                    width=int(cfg.data.width),
                                    height=int(cfg.data.height),
                                ).images[0]
                        val_path = Path(cfg.output.results_dir) / f"validation_step_{global_step:06d}.png"
                        val_path.parent.mkdir(parents=True, exist_ok=True)
                        img.save(val_path)
                        logger.info("Saved validation image: %s", val_path)
                    except Exception as e:
                        logger.warning("Validation generation failed at step %d: %s", global_step, e)

                if global_step >= int(max_train_steps):
                    break

        if global_step >= int(max_train_steps):
            break

    if pbar is not None:
        pbar.close()

    avg_loss = running_loss / max(1, global_step)
    logger.info("Training finished. steps=%d, avg_loss=%.6f", global_step, avg_loss)

    out_dir = Path(cfg.output.output_dir)
    weights_path = None
    val_img_path = None

    if accelerator.is_main_process:
        meta = {
            "entity_name": cfg.data.entity_name,
            "placeholder_token": cfg.data.placeholder_token,
            "base_model_id": cfg.model.base_model_id,
            "global_step": global_step,
            "avg_loss": avg_loss,
            "train_config": train_config_to_dict(cfg),
        }

        unwrapped_unet = accelerator.unwrap_model(pipe.unet)
        weights_path = str(save_lora_weights(pipe, unwrapped_unet, out_dir, meta=meta))

        if cfg.train.validation_prompt:
            try:
                pipe.set_progress_bar_config(disable=True)
            except Exception:
                pass
            try:
                pipe.unet = unwrapped_unet
                pipe.unet.eval()
                with torch.no_grad():
                    if device.type == "cuda":
                        with torch.autocast("cuda", dtype=unet_dtype):
                            img = pipe(
                                prompt=str(cfg.train.validation_prompt),
                                num_inference_steps=20,
                                guidance_scale=7.0,
                                width=int(cfg.data.width),
                                height=int(cfg.data.height),
                            ).images[0]
                    else:
                        img = pipe(
                            prompt=str(cfg.train.validation_prompt),
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
                logger.warning("Final validation generation failed: %s", e)

    accelerator.wait_for_everyone()

    summary = {
        "ok": True,
        "global_step": global_step,
        "avg_loss": avg_loss,
        "weights_path": weights_path,
        "validation_image": val_img_path,
        "output_dir": str(out_dir),
        "results_dir": str(cfg.output.results_dir),
        "logs_dir": str(cfg.output.logs_dir),
        "device": str(device),
        "mixed_precision": mp,
    }

    if accelerator.is_main_process:
        (out_dir / "training_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    return summary


def main() -> None:
    setup_logging()

    import argparse

    p = argparse.ArgumentParser(description="Train LoRA for one entity (SDXL) on CUDA (RunPod).")
    p.add_argument("--config", required=True, help="Path to training config (.yaml/.yml/.json)")
    args = p.parse_args()

    cfg = load_train_config(Path(args.config), resolve_paths=True, validate=True)
    summary = train_one_entity(cfg)

    print("OK")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
