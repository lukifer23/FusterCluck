"""Stage 0: tokenizer sanity check and 50M-token overfit."""

from __future__ import annotations

import logging
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, IterableDataset

from fustercluck.data.tokenized_dataset import TokenizedDataset
from fustercluck.models.components import FusterCluckDecoder
from fustercluck.train.config import Stage0Config, TrainerConfig
from fustercluck.utils.checkpoint import CheckpointManager
from fustercluck.utils.misc import amp_autocast, ensure_dir, get_device
from fustercluck.train.utils import ThroughputTracker, format_timespan

LOGGER = logging.getLogger(__name__)


class PackedSequenceDataset(IterableDataset):
    """Streaming dataset that packs tokenized sequences to a fixed length."""

    def __init__(self, dataset: TokenizedDataset, seq_len: int, shuffle: bool = True) -> None:
        super().__init__()
        self.dataset = dataset
        self.seq_len = seq_len
        self.shuffle = shuffle

    def __iter__(self):  # type: ignore[override]
        indices = torch.randperm(len(self.dataset)) if self.shuffle else torch.arange(len(self.dataset))
        buffer: list[int] = []
        for idx in indices.tolist():
            sample = self.dataset[idx].tolist()
            while sample:
                remaining = self.seq_len - len(buffer)
                take = min(len(sample), remaining)
                buffer.extend(sample[:take])
                sample = sample[take:]
                if len(buffer) == self.seq_len:
                    yield torch.tensor(buffer, dtype=torch.long)
                    buffer = []
        if buffer:
            padded = buffer + [0] * (self.seq_len - len(buffer))
            yield torch.tensor(padded, dtype=torch.long)


def create_dataloader(
    dataset: TokenizedDataset,
    cfg: Stage0Config,
    trainer_cfg: TrainerConfig | None = None,
) -> DataLoader:
    iterable = PackedSequenceDataset(dataset, cfg.seq_len)
    loader_kwargs = {
        "batch_size": cfg.micro_batch_size,
        "drop_last": True,
    }
    if trainer_cfg is not None:
        num_workers = getattr(trainer_cfg, "dataloader_workers", 0)
        if num_workers:
            loader_kwargs["num_workers"] = num_workers
            loader_kwargs["persistent_workers"] = getattr(trainer_cfg, "persistent_workers", False)
        loader_kwargs["pin_memory"] = getattr(trainer_cfg, "pin_memory", False)
    return DataLoader(iterable, **loader_kwargs)


def apply_compile(model: torch.nn.Module, trainer_cfg: TrainerConfig) -> torch.nn.Module:
    if trainer_cfg.use_compile and hasattr(torch, "compile"):
        LOGGER.info("Compiling model with mode=%s", trainer_cfg.compile_mode)
        return torch.compile(model, mode=trainer_cfg.compile_mode, fullgraph=False)  # type: ignore[attr-defined]
    return model


def load_vocab(tokenizer_path: Path) -> int:
    import sentencepiece as spm

    if not tokenizer_path.exists():
        raise FileNotFoundError(f"Tokenizer model not found: {tokenizer_path}")
    processor = spm.SentencePieceProcessor()
    processor.Load(str(tokenizer_path))
    return processor.GetPieceSize()


@torch.no_grad()
def evaluate(model: FusterCluckDecoder, batch: torch.Tensor, device: torch.device) -> float:
    model.eval()
    input_ids = batch[:, :-1].to(device)
    targets = batch[:, 1:].to(device)
    logits = model(input_ids)
    loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
    model.train()
    return float(loss.item())


class Stage0Trainer:
    def __init__(self, cfg: Stage0Config, trainer_cfg: TrainerConfig) -> None:
        self.cfg = cfg
        self.trainer_cfg = trainer_cfg
        self.device = get_device(trainer_cfg.device)
        ensure_dir(cfg.checkpoint_dir)
        self.checkpoint = CheckpointManager(cfg.checkpoint_dir)

        vocab_size = load_vocab(cfg.tokenizer_path)
        self.model = FusterCluckDecoder(
            vocab_size=vocab_size,
            dim=cfg.model_dim,
            num_layers=cfg.model_layers,
            num_heads=cfg.model_heads,
            num_kv_heads=cfg.model_kv_heads,
            mlp_ratio=cfg.mlp_ratio,
            rope_theta=cfg.rope_theta,
            dropout=cfg.dropout,
        )
        self.model.to(self.device)

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=cfg.optimizer.lr,
            betas=cfg.optimizer.betas,
            eps=cfg.optimizer.eps,
            weight_decay=cfg.optimizer.weight_decay,
        )

        self.model = apply_compile(self.model, trainer_cfg)

        self.dataset = TokenizedDataset(cfg.dataset_path, cfg.idx_path)
        self.dataloader = self._make_iterator()
        self.step = 0
        self.throughput = ThroughputTracker()
        self.total_elapsed = 0.0
        effective_batch = cfg.seq_len * cfg.micro_batch_size * cfg.gradient_accumulation
        if self.device.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True  # type: ignore
            torch.backends.cudnn.allow_tf32 = True
        LOGGER.info(
            "Stage0 setup: device=%s precision=%s seq_len=%d micro_batch=%d grad_accum=%d effective_tokens=%d",
            self.device,
            self.trainer_cfg.precision,
            cfg.seq_len,
            cfg.micro_batch_size,
            cfg.gradient_accumulation,
            effective_batch,
        )

    def _make_iterator(self):
        return iter(create_dataloader(self.dataset, self.cfg, self.trainer_cfg))

    def _next_batch(self) -> torch.Tensor:
        try:
            return next(self.dataloader)
        except StopIteration:
            self.dataloader = self._make_iterator()
            return next(self.dataloader)

    def train(self) -> None:
        logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")
        grad_accum = self.cfg.gradient_accumulation
        scaler = None
        if self.device.type == "cuda" and self.trainer_cfg.precision == "fp16":
            scaler = torch.cuda.amp.GradScaler()

        tokens_per_update = self.cfg.seq_len * self.cfg.micro_batch_size * grad_accum
        while self.step < self.cfg.max_steps:
            loss_accum = 0.0
            step_start = time.time()
            for micro_step in range(grad_accum):
                try:
                    batch = self._next_batch()
                except StopIteration:  # pragma: no cover - defensive, should not hit
                    raise RuntimeError("Dataset iterator exhausted; provide more packed tokens")
                batch = batch.to(self.device)
                input_ids = batch[:, :-1]
                targets = batch[:, 1:]
                with amp_autocast(self.trainer_cfg.precision, self.device):
                    logits = self.model(input_ids)
                    loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
                    loss = loss / grad_accum
                if scaler is not None:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()
                loss_accum += loss.item()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.trainer_cfg.grad_clip)
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
                remaining_steps = max(self.cfg.max_steps - self.step, 0)
                eta_seconds = avg_step * remaining_steps
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
                eval_loss = evaluate(self.model, batch.to(self.device), self.device)
                LOGGER.info("eval_loss=%.4f", eval_loss)
                self.checkpoint.save(
                    self.step,
                    self.model,
                    self.optimizer,
                    step=self.step,
                    eval_loss=eval_loss,
                    elapsed=self.total_elapsed,
                )

            if math.isnan(loss_accum):
                raise RuntimeError("NaN detected in loss")

    def resume_from_checkpoint(self, checkpoint_path: Path) -> None:
        state = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(state["model"])  # type: ignore[index]
        self.optimizer.load_state_dict(state["optimizer"])  # type: ignore[index]
        metadata = state.get("metadata", {})
        resume_step = int(metadata.get("step", self._step_from_filename(checkpoint_path)))
        self.step = resume_step
        self.total_elapsed = float(metadata.get("elapsed", self.total_elapsed))
        LOGGER.info("Resumed from %s at step=%d", checkpoint_path, self.step)
        self.throughput.reset()

    @staticmethod
    def _step_from_filename(path: Path) -> int:
        try:
            return int(path.stem.split("-")[-1])
        except ValueError:
            return 0


def run_stage0(cfg: Stage0Config, trainer_cfg: TrainerConfig | None = None) -> None:
    trainer_cfg = trainer_cfg or TrainerConfig(device="mps")
    trainer = Stage0Trainer(cfg, trainer_cfg)
    trainer.train()
