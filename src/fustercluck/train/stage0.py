"""Generic text-stage trainer: tokenizer sanity and full pre-training runs."""

from __future__ import annotations

import logging
import math
import signal
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, IterableDataset

from fustercluck.data.tokenized_dataset import TokenizedDataset
from fustercluck.models.components import FusterCluckDecoder
from fustercluck.train.config import StageConfig, TrainerConfig
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
        # Use full dataset for training
        max_sequences = len(self.dataset)

        if self.shuffle:
            indices = torch.randperm(max_sequences, dtype=torch.long)
        else:
            indices = torch.arange(max_sequences, dtype=torch.long)

        # Use a simpler, more robust packing approach
        all_tokens = []
        for idx in indices:
            sample = self.dataset[idx]
            all_tokens.extend(sample.tolist())

        # Pack into fixed-size chunks
        for i in range(0, len(all_tokens), self.seq_len):
            chunk = all_tokens[i:i + self.seq_len]
            if len(chunk) < self.seq_len:
                # Pad the last chunk
                chunk.extend([0] * (self.seq_len - len(chunk)))
            yield torch.tensor(chunk, dtype=torch.long)


def create_dataloader(
    dataset: TokenizedDataset,
    cfg: StageConfig,
    trainer_cfg: TrainerConfig | None = None,
) -> DataLoader:
    iterable = PackedSequenceDataset(dataset, cfg.seq_len)
    loader_kwargs = {
        "batch_size": cfg.micro_batch_size,
        "drop_last": True,
    }
    if trainer_cfg is not None:
        num_workers = getattr(trainer_cfg, "dataloader_workers", 0)
        if num_workers > 0:
            loader_kwargs["num_workers"] = num_workers
            loader_kwargs["persistent_workers"] = getattr(trainer_cfg, "persistent_workers", True)
            loader_kwargs["prefetch_factor"] = getattr(trainer_cfg, "prefetch_factor", 2)
        loader_kwargs["pin_memory"] = getattr(trainer_cfg, "pin_memory", False)
    return DataLoader(iterable, **loader_kwargs)


def apply_compile(model: torch.nn.Module, trainer_cfg: TrainerConfig) -> torch.nn.Module:
    if trainer_cfg.use_compile and hasattr(torch, "compile"):
        LOGGER.info("Compiling model with mode=%s", trainer_cfg.compile_mode)
        try:
            return torch.compile(model, mode=trainer_cfg.compile_mode, fullgraph=False)  # type: ignore[attr-defined]
        except RuntimeError as exc:
            LOGGER.warning("torch.compile failed (%s); continuing without compilation", exc)
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


class TextStageTrainer:
    """Trainer capable of running Stage 0 sanity or full text pre-training."""

    def __init__(self, cfg: StageConfig, trainer_cfg: TrainerConfig) -> None:
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
        self.model = self.model.to(self.device)

        if trainer_cfg.gradient_checkpointing:
            LOGGER.info("Enabling gradient checkpointing")
            self.model.gradient_checkpointing_enable()

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
        self.tokens_per_update = cfg.seq_len * cfg.micro_batch_size * cfg.gradient_accumulation

        if self.device.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True  # type: ignore[attr-defined]
            torch.backends.cudnn.allow_tf32 = True  # type: ignore[attr-defined]
            torch.backends.cudnn.benchmark = True  # type: ignore[attr-defined]
            torch.backends.cudnn.deterministic = False  # type: ignore[attr-defined]
        elif self.device.type == "mps":
            torch.set_float32_matmul_precision("medium")

        LOGGER.info(
            "Trainer setup: device=%s precision=%s seq_len=%d micro_batch=%d grad_accum=%d tokens/update=%d",
            self.device,
            self.trainer_cfg.precision,
            cfg.seq_len,
            cfg.micro_batch_size,
            cfg.gradient_accumulation,
            self.tokens_per_update,
        )

    def _make_iterator(self):
        return iter(create_dataloader(self.dataset, self.cfg, self.trainer_cfg))

    def _next_batch(self) -> torch.Tensor:
        try:
            return next(self.dataloader)
        except StopIteration:
            self.dataloader = self._make_iterator()
            return next(self.dataloader)

    def _install_signal_handlers(self) -> None:
        def handler(signum, _frame):
            LOGGER.info("Received signal %d; writing emergency checkpoint", signum)
            try:
                self.checkpoint.save(
                    self.step,
                    self.model,
                    self.optimizer,
                    eval_loss=0.0,
                    elapsed=self.total_elapsed,
                    emergency=True,
                )
            except Exception as exc:  # pragma: no cover - best effort
                LOGGER.error("Emergency checkpoint failed: %s", exc)
            finally:
                sys.exit(1)

        signal.signal(signal.SIGINT, handler)
        signal.signal(signal.SIGTERM, handler)

    def train(self) -> None:
        logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")
        grad_accum = self.cfg.gradient_accumulation
        scaler = None
        if self.device.type == "cuda" and self.trainer_cfg.precision == "fp16":
            scaler = torch.cuda.amp.GradScaler()

        # Try to resume from checkpoint
        self._load_checkpoint()

        self._install_signal_handlers()

        print("[DEBUG] About to enter training loop...")
        while self.step < self.cfg.max_steps:
            try:
                print(f"[DEBUG] Starting step {self.step}")
                loss_accum = torch.tensor(0.0, device=self.device)
                step_start = time.time()

                for accum_step in range(grad_accum):
                    batch = self._next_batch()
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
                    loss_accum += loss.detach()  # Accumulate tensor directly for better precision

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.trainer_cfg.grad_clip)
                if scaler is not None:
                    scaler.step(self.optimizer)
                    scaler.update()
                else:
                    self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)

                # Memory cleanup for MPS
                if hasattr(torch, 'mps') and torch.mps.is_available():
                    torch.mps.empty_cache()

                self.step += 1

                elapsed = time.time() - step_start
                self.throughput.update(self.tokens_per_update, elapsed)
                self.total_elapsed += elapsed

                if self.step % self.cfg.log_interval == 0:
                    print(f"[DEBUG] Step {self.step} reached log interval, computing stats...")
                    avg_step = self.total_elapsed / self.step if self.step else 0.0
                    remaining_steps = max(self.cfg.max_steps - self.step, 0)
                    eta_seconds = avg_step * remaining_steps
                    loss_value = loss_accum.item() * grad_accum

                    # Memory usage info
                    memory_info = ""
                    if hasattr(torch, 'mps') and torch.mps.is_available():
                        try:
                            mem_mb = torch.mps.current_allocated_memory() / (1024**2)
                            memory_info = f" mem={mem_mb:.0f}MB"
                        except:
                            pass

                    print(f"[TRAINING] step={self.step} loss={loss_value:.4f} "
                          f"tokens/s={self.throughput.tokens_per_second:.1f} eta={format_timespan(eta_seconds) if self.step else '--:--'}{memory_info}")
                    LOGGER.info(
                        "step=%d loss=%.4f tokens/s=%.1f eta=%s",
                        self.step,
                        loss_value,
                        self.throughput.tokens_per_second,
                        format_timespan(eta_seconds) if self.step else "--:--",
                    )
                    self.throughput.reset()

                if self.step % self.cfg.eval_interval == 0:
                    eval_batch = self._next_batch().to(self.device)
                    eval_loss = evaluate(self.model, eval_batch, self.device)
                    LOGGER.info("eval_loss=%.4f", eval_loss)
                    checkpoint_path = self.checkpoint.save(
                        self.step,
                        self.model,
                        self.optimizer,
                        eval_loss=eval_loss,
                        elapsed=self.total_elapsed,
                    )
                    LOGGER.info("Checkpoint saved to %s", checkpoint_path)

                if math.isnan(loss_accum):
                    raise RuntimeError("NaN detected in loss")

            except Exception as e:
                LOGGER.error("Training step %d failed: %s", self.step, e)
                # Save emergency checkpoint on error
                try:
                    emergency_path = self.checkpoint.save(
                        self.step,
                        self.model,
                        self.optimizer,
                        eval_loss=0.0,
                        elapsed=self.total_elapsed,
                        emergency=True,
                        error=str(e)
                    )
                    LOGGER.info("Emergency checkpoint saved to %s", emergency_path)
                except Exception as save_error:
                    LOGGER.error("Failed to save emergency checkpoint: %s", save_error)

                # Re-raise the exception to exit
                raise

    def _load_checkpoint(self) -> None:
        """Load latest checkpoint if available."""
        checkpoint_data = self.checkpoint.load()
        if checkpoint_data:
            try:
                self.model.load_state_dict(checkpoint_data["model"])
                self.optimizer.load_state_dict(checkpoint_data["optimizer"])
                metadata = checkpoint_data.get("metadata", {})
                self.step = metadata.get("step", self.step)
                self.total_elapsed = metadata.get("elapsed", 0.0)

                # Move model to device after loading
                self.model.to(self.device)

                LOGGER.info("Resumed from checkpoint at step %d", self.step)
            except Exception as e:
                LOGGER.warning("Failed to load checkpoint: %s", e)
        else:
            LOGGER.info("No checkpoint found, starting from scratch")

    def resume_from_checkpoint(self, checkpoint_path: Path) -> None:
        state = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(state["model"])  # type: ignore[index]
        self.optimizer.load_state_dict(state["optimizer"])  # type: ignore[index]
        metadata = state.get("metadata", {})
        self.step = int(metadata.get("step", self._step_from_filename(checkpoint_path)))
        self.total_elapsed = float(metadata.get("elapsed", self.total_elapsed))
        LOGGER.info("Resumed from %s at step=%d", checkpoint_path, self.step)
        self.throughput.reset()

    @staticmethod
    def _step_from_filename(path: Path) -> int:
        try:
            return int(path.stem.split("-")[-1])
        except ValueError:
            return 0


def run_stage(cfg: StageConfig, trainer_cfg: TrainerConfig | None = None) -> None:
    trainer_cfg = trainer_cfg or TrainerConfig(device="mps")
    trainer = TextStageTrainer(cfg, trainer_cfg)
    trainer.train()


__all__ = ["TextStageTrainer", "run_stage", "create_dataloader", "PackedSequenceDataset"]
