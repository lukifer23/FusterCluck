"""Multimodal training loop for Stage 3/4 (vision-text alignment)."""

from __future__ import annotations

import json
import logging
import math
import time
from pathlib import Path
from typing import Iterator

import sentencepiece as spm
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, IterableDataset

from fustercluck.models.components import FusterCluckDecoder
from fustercluck.models.vision import MultimodalAdapter
from fustercluck.train.config import StageVisionConfig, TrainerConfig
from fustercluck.train.utils import ThroughputTracker, format_timespan
from fustercluck.utils.checkpoint import CheckpointManager
from fustercluck.utils.misc import amp_autocast, ensure_dir, get_device

import webdataset as wds

LOGGER = logging.getLogger(__name__)


class MultimodalDataset(IterableDataset):
    """Streams multimodal samples from WebDataset shards."""

    def __init__(
        self,
        cfg: StageVisionConfig,
        tokenizer: spm.SentencePieceProcessor,
        image_preprocess,
    ) -> None:
        super().__init__()
        if not cfg.vision_shards:
            raise ValueError("Stage 3/4 requires 'vision_shards' entries in config")
        self.cfg = cfg
        self.image_token_id = cfg.image_token_id or tokenizer.PieceToId(cfg.image_token)
        self.seq_len = cfg.seq_len
        self.transform = image_preprocess
        urls = cfg.vision_shards
        pipeline = (
            wds.WebDataset(urls, handler="warn")
            .shuffle(cfg.shuffle_buffer)
            .decode("pil")
            .to_tuple("json", "jpg")
            .repeat()
        )
        self.pipeline = pipeline

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:  # type: ignore[override]
        for meta, image in self.pipeline:
            if isinstance(meta, (bytes, bytearray)):
                meta = json.loads(meta)
            elif isinstance(meta, str):
                meta = json.loads(meta)
            if "input_ids" not in meta:
                continue
            tokens = meta["input_ids"]
            if not isinstance(tokens, list):
                continue
            if len(tokens) < 2:
                continue
            # Pad / truncate to seq_len
            if len(tokens) > self.seq_len:
                tokens = tokens[: self.seq_len]
            else:
                tokens = tokens + [0] * (self.seq_len - len(tokens))
            token_tensor = torch.tensor(tokens, dtype=torch.long)
            if (token_tensor == self.image_token_id).sum().item() == 0:
                continue  # require at least one <image> token
            image_tensor = self.transform(image)
            yield {"tokens": token_tensor, "image": image_tensor}


class StageMultimodalTrainer:
    """Multimodal trainer handling Stage 3/4 alignment."""

    def __init__(self, cfg: StageVisionConfig, trainer_cfg: TrainerConfig) -> None:
        self.cfg = cfg
        self.trainer_cfg = trainer_cfg
        self.device = get_device(trainer_cfg.device)
        ensure_dir(cfg.checkpoint_dir)
        self.checkpoint = CheckpointManager(cfg.checkpoint_dir)

        tokenizer = spm.SentencePieceProcessor()
        tokenizer.Load(str(cfg.tokenizer_path))
        if cfg.image_token_id is None:
            cfg.image_token_id = tokenizer.PieceToId(cfg.image_token)
        self.image_token_id = cfg.image_token_id

        vocab_size = tokenizer.GetPieceSize()
        self.model = FusterCluckDecoder(
            vocab_size=vocab_size,
            dim=cfg.model_dim,
            num_layers=cfg.model_layers,
            num_heads=cfg.model_heads,
            num_kv_heads=cfg.model_kv_heads,
            mlp_ratio=cfg.mlp_ratio,
            rope_theta=cfg.rope_theta,
            dropout=cfg.dropout,
        ).to(self.device)

        self.adapter = MultimodalAdapter(
            hidden_dim=cfg.model_dim,
            fusion=cfg.adapter.fusion,
            num_queries=cfg.adapter.num_queries,
            trainable_backbone=cfg.adapter.train_backbone,
            image_size=cfg.adapter.image_size,
        ).to(self.device)

        from torchvision.models.vision_transformer import ViT_B_16_Weights

        preprocess = ViT_B_16_Weights.IMAGENET1K_V1.transforms()

        self.dataset = MultimodalDataset(cfg, tokenizer, preprocess)
        self.dataloader = self._make_iterator()

        self.optimizer = torch.optim.AdamW(
            list(self.model.parameters()) + list(self.adapter.parameters()),
            lr=cfg.optimizer.lr,
            betas=cfg.optimizer.betas,
            eps=cfg.optimizer.eps,
            weight_decay=cfg.optimizer.weight_decay,
        )
        self.step = 0
        self.total_elapsed = 0.0
        self.throughput = ThroughputTracker()

        torch.backends.cuda.matmul.allow_tf32 = True  # type: ignore
        torch.backends.cudnn.allow_tf32 = True

        LOGGER.info(
            "Stage3/4 setup: device=%s precision=%s seq_len=%d micro_batch=%d grad_accum=%d",
            self.device,
            self.trainer_cfg.precision,
            cfg.seq_len,
            cfg.micro_batch_size,
            cfg.gradient_accumulation,
        )

    def _make_iterator(self):
        loader_kwargs = {
            "batch_size": self.cfg.micro_batch_size,
            "drop_last": True,
        }
        if self.trainer_cfg.dataloader_workers:
            loader_kwargs["num_workers"] = self.trainer_cfg.dataloader_workers
            loader_kwargs["persistent_workers"] = self.trainer_cfg.persistent_workers
        loader_kwargs["pin_memory"] = self.trainer_cfg.pin_memory
        return iter(DataLoader(self.dataset, **loader_kwargs))

    def _next_batch(self):
        try:
            return next(self.dataloader)
        except StopIteration:
            self.dataloader = self._make_iterator()
            return next(self.dataloader)

    def resume_from_checkpoint(self, path: Path) -> None:
        state = torch.load(path, map_location=self.device)
        self.model.load_state_dict(state["model"], strict=False)
        adapter_state = state.get("adapter")
        if adapter_state:
            self.adapter.load_state_dict(adapter_state, strict=False)
        self.optimizer.load_state_dict(state["optimizer"])
        metadata = state.get("metadata", {})
        self.step = int(metadata.get("step", 0))
        self.total_elapsed = float(metadata.get("elapsed", 0.0))
        LOGGER.info("Resumed multimodal trainer from %s at step=%d", path, self.step)
        self.throughput.reset()

    def train(self) -> None:
        grad_accum = self.cfg.gradient_accumulation
        scaler = None
        if self.device.type == "cuda" and self.trainer_cfg.precision == "fp16":
            scaler = torch.cuda.amp.GradScaler()

        tokens_per_update = self.cfg.seq_len * self.cfg.micro_batch_size * grad_accum
        while self.step < self.cfg.max_steps:
            loss_accum = 0.0
            step_start = time.time()
            for _ in range(grad_accum):
                batch = self._next_batch()
                tokens = batch["tokens"].to(self.device)
                images = batch["image"].to(self.device)
                image_mask = tokens == self.image_token_id
                if image_mask.sum(dim=1).min().item() == 0:
                    continue  # skip batches without image tokens
                input_ids = tokens[:, :-1]
                targets = tokens[:, 1:]
                vision_mask = image_mask[:, :-1]
                with amp_autocast(self.trainer_cfg.precision, self.device):
                    vision_embeddings = self.adapter(images)
                    logits = self.model(
                        input_ids,
                        vision_embeddings=vision_embeddings,
                        vision_mask=vision_mask,
                    )
                    loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
                    loss = loss / grad_accum
                if scaler is not None:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()
                loss_accum += loss.item()
            torch.nn.utils.clip_grad_norm_(
                list(self.model.parameters()) + list(self.adapter.parameters()),
                self.trainer_cfg.grad_clip,
            )
            if scaler is not None:
                scaler.step(self.optimizer)
                scaler.update()
            else:
                self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)
            self.step += 1
            elapsed = time.time() - step_start
            self.throughput.update(tokens_per_update, elapsed)
            self.total_elapsed += elapsed

            if self.step % self.cfg.log_interval == 0:
                avg_step = self.total_elapsed / self.step if self.step else 0.0
                remaining = max(self.cfg.max_steps - self.step, 0)
                eta_seconds = avg_step * remaining
                LOGGER.info(
                    "step=%d loss=%.4f tokens/s=%.1f eta=%s",
                    self.step,
                    loss_accum * grad_accum,
                    self.throughput.tokens_per_second,
                    format_timespan(eta_seconds) if self.step else "--:--",
                )
                self.throughput.reset()

            if self.step % self.cfg.eval_interval == 0:
                batch = self._next_batch()
                tokens = batch["tokens"].to(self.device)
                images = batch["image"].to(self.device)
                mask = tokens == self.image_token_id
                if mask.sum(dim=1).min().item() == 0:
                    continue
                input_ids = tokens[:, :-1]
                targets = tokens[:, 1:]
                vision_embeddings = self.adapter(images)
                logits = self.model(input_ids, vision_embeddings=vision_embeddings, vision_mask=mask[:, :-1])
                eval_loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
                LOGGER.info("eval_loss=%.4f", float(eval_loss))
                self.checkpoint.save(
                    self.step,
                    self.model,
                    self.optimizer,
                    extra={"adapter": self.adapter.state_dict()},
                    step=self.step,
                    eval_loss=float(eval_loss),
                    elapsed=self.total_elapsed,
                )

            if math.isnan(loss_accum):
                raise RuntimeError("NaN detected in loss")


__all__ = ["StageMultimodalTrainer"]
