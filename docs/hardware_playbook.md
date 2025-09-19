# M3 Pro Training Playbook

## Core Settings

- `device`: `mps`
- Precision: `bf16` preferred, `fp16` fallback for unstable ops (disable autocast per-module).
- Gradient checkpointing: enable for every transformer block after Stage 0.
- Gradient accumulation: 32 for 512k global tokens, 48 for 768k.
- Sequence packing: maintain 4096 target but allow curriculum warm-up at 2048 to stabilize.

## Throughput Targets

| Stage | Effective Batch (tokens) | Expected Tokens/s | Wallclock (est.) |
| ----- | ------------------------ | ----------------- | ---------------- |
| 0     | 32k                      | 900–1100          | 8 h              |
| 1     | 512k (accum 32)          | 220–260           | ~4 days / 2B tok |
| 2     | 768k (accum 48)          | 210–240           | ~15 days / 5B tok|
| 3–4   | 512k mixed               | 160–190           | ~7 days / 2.4B tok|

*Numbers assume sustained utilization with WebDataset streaming. Expect ±20% variance.*

## Memory Budget (18 GB)

- Model weights + optimizer states: ~6.3 GB (bf16 weights, fp32 master, Adam moments).
- Activations (2048 seq, micro-batch 4): ~1.1 GB with checkpointing.
- Vision adapter (frozen ViT): +1.2 GB when enabled.
- Safety margin: ~2 GB for dataloader + logs.

## Tips

- Use `torch.set_float32_matmul_precision("medium")` once at startup.
- Avoid large Python multiprocessing pools; 4 dataloader workers is optimal.
- Profile `torch.profiler` for 200 steps each stage before scaling.
- Pin WebDataset shards on NVMe using `as_posix()` paths to avoid sandbox symlinks.
- Persist metrics locally (`wandb offline sync`) to protect against network hiccups.
