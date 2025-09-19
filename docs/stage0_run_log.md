# Stage 0 Dry Run (2025-09-19)

- **Corpus**: `data/raw/stage0_corpus.txt` (446,792 lines)
- **Tokenizer**: `artifacts/tokenizer/fustercluck.model` (50K vocab, coverage 99.95%)
- **Tokenized Shard**: `data/tokenized/stage0.bin` (`uint16`, 62,472,280 tokens)
- **Training Run**: 24×1024 decoder truncated to 4 layers (Stage 0 config)
  - `seq_len=512`, micro-batch 2, grad accumulation 4
  - LR 5e-4, precision `bf16`, device `mps`
  - 60 steps (~6 minutes wall-clock)
- **Metrics**:
  - Step 10 loss 29.99 → Step 60 loss 26.56 (steady descent)
  - Eval (step 30): 6.84 CE; Eval (step 60): 6.35 CE
  - No NaNs or divergence observed
- **Artifacts**: Checkpoint `artifacts/checkpoints/stage0/step-0000060.pt` with optimizer state and manifest JSON.

This confirms the Stage 0 shard, tokenizer, and training harness operate end-to-end on the M3 Pro prior to scaling sequence length and batch size.
