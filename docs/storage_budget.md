# Storage Budget (Local Development)

| Component                                   | Tokens (effective) | Raw Size | Tokenized Size | Notes |
| ------------------------------------------- | ------------------ | -------- | -------------- | ----- |
| Text pretrain (RefinedWeb/SlimPajama/Dolma) | 7.5B (11B eff.)    | ~42 GB   | ~13 GB         | Pre-tokenize to uint16 `.bin/.idx` |
| High-signal books/encyclopedia/Q&A          | 0.5B (0.8B eff.)   | ~3 GB    | ~1 GB          | Curated, deduped |
| Caption datasets (LAION/CC/COCO)            | 0.4B (1.0B eff.)   | ~700 GB  | n/a            | Store on external NVMe as WebDataset |
| OCR/DocVQA corpora                          | 0.35B (0.9B eff.)  | ~450 GB  | n/a            | Includes PDF raster caches |
| UI/Screenshot corpora                       | 0.2B (0.6B eff.)   | ~250 GB  | n/a            | Synthetic + ScreenQA |
| Synthetic OCR cache (per epoch)             | 0.1B (0.3B eff.)   | ~40 GB   | n/a            | Regenerate per curriculum stage |
| Instruction/CoT SFT                         | 0.3B               | ~6 GB    | ~2 GB          | Stored tokenized |
| Tool-call SFT                               | 0.12B              | ~3 GB    | ~1 GB          | JSON traces |
| Checkpoints (7 kept)                        | n/a                | ~42 GB   | n/a            | Model + optimizer states |
| Eval artifacts/logs                         | n/a                | ~10 GB   | n/a            | WandB offline exports |

**Total active storage requirement:** ~1.5 TB. Keep tokenized text and checkpoints on the internal SSD (<120 GB). Store image/WebDataset shards on the external NVMe. Rotate synthetic caches to stay within 200 GB headroom.

### Incremental Strategy (before external drive)

1. Focus on text pipeline + Stage 0 dry run (requires <30 GB).
2. Generate small synthetic OCR batches (≤10k samples, ~2 GB) to validate multimodal dataloader.
3. After NVMe arrival, sync curated WebDataset shards and expand synthetic caches as needed.
