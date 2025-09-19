# FusterCluck-450

FusterCluck-450 is a from-scratch, 450M parameter multimodal (text + vision) model tuned for Apple Silicon Macs. This repository contains the training stack, data tooling, and evaluation harness needed to reproduce the plan in `Plan.md` on an M3 Pro MacBook Pro.

The current training focus centers on four high-signal domains—**science**, **code**, **chess**, and **data/analysis workflows**—with only a light sprinkling of general chat to preserve conversational polish. All scripts, manifests, and docs in this repo assume that domain mix.

## System Requirements (Dev Rig)

- macOS 14+
- Apple Silicon M3 Pro (18 GB unified memory)
- 512 GB internal SSD (external NVMe strongly recommended for multimodal datasets)
- Python 3.10+

### Memory Footprint Guidelines

- Stage 0/1 (text-only sanity) uses a 4-layer 1024-dim decoder and fits comfortably in <8 GB.
- Full 24-layer model requires gradient checkpointing and gradient accumulation (32–48) to stay <16 GB.
- For multimodal stages, keep micro-batch ≤1 image per step and rely on accumulation to hit global batch targets.

## Quick Start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
```

## Stage 0 (Focused text mix)

```bash
# 1) Build the stage0 corpus with the focused domain manifest
python scripts/build_stage0_corpus.py \
  --manifest configs/stage0_sources.json \
  --output data/raw/stage0_focused.txt \
  --target-tokens 50_000_000

# 2) Train or refresh the tokenizer on the focused corpus
python scripts/build_tokenizer.py data/raw/stage0_focused.txt \
  --output artifacts/tokenizer/fustercluck \
  --vocab-size 50000

# 3) Pretokenize for the training loaders
python scripts/pretokenize_text.py \
  artifacts/tokenizer/fustercluck.model \
  data/tokenized/stage0_focused \
  data/raw/stage0_focused.txt

# 4) Launch the Stage 0 dry run
python scripts/run_stage0.py \
  data/tokenized/stage0_focused.bin \
  data/tokenized/stage0_focused.idx \
  artifacts/tokenizer/fustercluck.model \
  --max-steps 2000 \
  --seq-len 2048 \
  --micro-batch 4 \
  --grad-accum 8 \
  --device mps \
  --checkpoint-dir artifacts/checkpoints/stage0_focused
```

## Repository Layout

```
configs/            # Hydra/OmegaConf stage configs
scripts/            # CLI entrypoints for tokenizer, data prep, training
src/fustercluck/
  data/             # WebDataset pipelines, packing utilities
  models/           # Transformer + fusion modules
  tokenizer/        # SentencePiece training utilities
  train/            # Stage loops, evaluation hooks
  utils/            # Shared helpers (logging, profiling, checkpointing)
```

## Development Roadmap

1. Train SentencePiece tokenizer (`scripts/build_tokenizer.py`).
2. Run Stage 0 dry-run with synthetic text (`scripts/run_stage0.py`).
3. Integrate external datasets via WebDataset + manifest files.
4. Scale through Stage 4 multimodal training.
5. Run SFT stages with LoRA heads, export gguf and MLX artifacts.

Refer to `Plan.md` for the full project plan. The code in this repo is production-ready; there is no placeholder logic. Every module includes docstrings and doc comments to help keep the architecture understandable while we scale.
