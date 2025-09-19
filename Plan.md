# FusterCluck‑450: Refined Training Plan (Apple Silicon)

> A compact, natively multimodal (text + images/screenshots) LLM trained from scratch, optimized for Apple Silicon (M‑series) with PyTorch MPS. Target size ≈ **450M params** with an attached vision encoder and lightweight fusion. Delivers chat, CoT, JSON/function calling, and strong OCR on UI screenshots and documents.

---

## 0) High‑level Changes vs. Original Plan

* **Kept** the 450M target, CoT, function‑calling, and OCR goals.
* **Refined** architecture to hit \~450M more tightly (24 layers × 1,024 dim + 50K vocab + vision adapter ≈ 420–470M depending on heads/ffn scale).
* **Clearer token budget**: use a *curriculum* with staged caps, aggressive dedup, and heavy quality filtering; emphasize **RefinedWeb/SlimPajama/Dolma** over raw C4/OpenWebText.
* **Multimodal path**: prefer **SigLIP/CLIP ViT‑B/16** encoders; add **tiny Q‑Former or Perceiver‑Resampler** instead of a thick MLP for better sample efficiency in OCR/GUI.
* **Apple Silicon realism**: no Flash‑Attention on MPS; rely on PyTorch SDPA, gradient checkpointing, activation recompute, sequence packing, fused ops where available.
* **Evaluation-first training**: small, frequent checkpoints with a fixed eval battery; stop early if curves flatten by < 3–5% over two eval windows.

---

## 1) Model Spec (≈450M)

### 1.1 Text Backbone (Decoder‑only)

* **Layers**: 24
* **Hidden size (d\_model)**: 1,024
* **Attention heads**: 16 (head\_dim = 64)
* **FFN expansion**: 4× (intermediate\_size = 4,096)
* **Positional**: RoPE (base 10,000), **ALiBi** optional ablation
* **Context length**: 4,096 (8,192 ablation later)
* **Norm**: RMSNorm pre‑LN
* **Attention**: SDPA (scaled dot‑product attention) with **GQA** (e.g., 16 q, 4 k/v heads) for memory savings
* **Dropout**: 0.0 for pretrain, 0.05 for SFT
* **Tokenizer**: SentencePiece (Unigram), **50K** vocab, UTF‑8 clean, digit/URL preservation; add special tokens: `<image>`, `<im_end>`, `<tool>`, `<json>`, `<sys>`

**Param sanity**: \~12·L·d² ≈ 12·24·(1,024²) ≈ 302M, + embeddings (\~51M), + norms/ln/heads ≈ 15–25M → ≈ 370–380M for text. With multimodal projection & Q‑Former (\~40–70M), total ≈ **420–450M**.

### 1.2 Vision Branch

* **Encoder**: SigLIP or CLIP **ViT‑B/16**

  * Frozen for stage 1, partial unfreeze late SFT only if needed.
* **Image resolution**: 224–336 short‑side, keep aspect ratio, center‑crop/pad; for screenshots, allow 448 for small fonts.
* **Input types**: natural images, UI screenshots, documents (RGB); optional PDF page rasterization.

### 1.3 Fusion Options (choose one, keep module tiny)

* **Q‑Former‑mini**: 2–4 transformer layers, 32–64 learnable queries, cross‑attend to ViT tokens → project to d\_model.
* **Perceiver‑Resampler**: downsamples ViT tokens to 64 latents via cross‑attention → linear to d\_model.
* **(Ablation)** Simple MLP projector from \[CLS] or pooled tokens (cheapest but least flexible).

### 1.4 Training Losses

* **Language**: next‑token cross‑entropy (causal)
* **Vision‑text alignment**: contrastive (InfoNCE) on image/text pairs for a small %, plus supervised captioning loss
* **Multimodal causal**: next‑token loss on sequences with `<image>` token preambles

---

## 2) Data Plan & Token Budget (Quality‑first)

### 2.1 Overall Budget (curriculum)

* **Total target tokens**: **12–16B** effective tokens

  * **Text‑only pretrain**: 7–9B
  * **Vision‑text pairs**: 2.5–3.5B (caption tokens + OCR prompts)
* **Instruction/CoT SFT**: 200–400M
* **Function-calling SFT**: 50–150M
* **Eval/holdout**: \~1–2% of each split

> *Rationale*: 20–25 tokens/parameter baseline for small models, + multimodal overhead. We stop when evals plateau; no virtue in hitting a number if validation stalls.

**Domain emphasis (text curriculum)**

- Science + research prose ≈ 35–40%
- Data & analysis workflows (statistics, experiment logs, notebooks) ≈ 20%
- Code for scientific/ML stacks ≈ 25%
- Chess corpora (annotated PGNs, commentary) ≈ 15%
- General chat ballast ≤5% to keep conversational tone without diluting specialization

### 2.2 Text Pretrain Sources (favor curated, deduped)

* **RefinedWeb (v1/v2)**, **SlimPajama**, **Dolma** (web mix) → primary
* Books/encyclopedic subsets with permissive licenses
* High‑signal forums/Q\&A (license‑friendly), math/science writeups
* **Optional code**: a tiny sprinkle (≤3%) from permissive repos to improve JSON/tool formatting

**Preprocessing**: URL/domain filtering, language id, length & perplexity filter, MinHash dedup (near‑dup cluster removal), HTML/boilerplate strip, emoji/UTF‑8 keep.

### 2.3 Vision‑Text (caption + OCR + UI)

* **Generic**: LAION‑Aesthetics (scored), CC3M/CC12M, COCO‑caps
* **OCR‑centric**: TextVQA, ST‑VQA, TextCaps, SynthText (rendered), ICDAR (scene text), IIIT‑5K, **DocVQA/InfographicVQA/ChartQA** (for charts, tables, PDFs), **Screen2Words**, **UIE/ScreenQA** for app/web UIs
* **Screenshot & document synthesis**: generate diverse programmatic screenshots/docs (varied fonts, DPI, dark/light modes, overlays), then label exact text boxes to create ultra‑clean OCR pairs

**Balancing**: Cap natural‑image captions at \~60–65% of vision tokens; ensure **≥35–40% explicitly text‑heavy images** (docs, UIs, signs) to specialize OCR.

### 2.4 Instruction/CoT

* **Seed**: Open‑licensed chat/instruction sets (Alpaca‑style, Dolly‑15K‑like), self‑instruct from the base model as it matures
* **CoT formatting**: Use explicit tags (e.g., `<reasoning>…</reasoning>`) during SFT; suppress at inference via stop tokens or system prompt
* **Safety/quality filters**: remove leakage of copyrighted/PII; reject pattern‑copying from proprietary assistants

### 2.5 Function‑Calling

* Create a **tool schema pack** (JSON Schema) with 10–30 realistic tools: calculator, calendar, web‑search stub, weather stub, file I/O stub, image OCR stub, etc.
* **Synthesize** task → tool JSON calls → tool responses → final answers. Include both **single‑tool** and **multi‑tool** chains; randomize argument order/whitespace; include failure modes and retries.
* Hold out a **blind tool set** (unseen schemas) for generalization.

---

## 3) Training Stages & Schedules

| Stage                      | Purpose                                                                    | Data/Size                      | Batch & LR                                                          | Epochs        | Expected Wallclock (M3 Pro/Max, MPS)\* |
| -------------------------- | -------------------------------------------------------------------------- | ------------------------------ | ------------------------------------------------------------------- | ------------- | -------------------------------------- |
| **0. Tokenizer & Dry‑run** | Build 50K SentencePiece; overfit a 50M‑token shard to verify loss plumbing | 50M tokens                     | bs=64 seq=2k; LR 1e‑3 warmup 2k                                     | until overfit | 6–10 h                                 |
| **1. Text Pretrain A**     | Core language                                                              | 2B tokens                      | eff. bs≈2–4M tokens; LR 2.5e‑4; cosine; warmup 2k                   | 1 pass        | 1.5–3 d                                |
| **2. Text Pretrain B**     | Scale + curriculum                                                         | +5B tokens                     | same LR schedule, pack sequences                                    | 1 pass        | 3–6 d                                  |
| **3. Multimodal Intro**    | Align image encoder + projector                                            | 300M pairs (\~0.6–0.8B tokens) | bs tuned to VRAM, freeze ViT; LR 1.5e‑4                             | 1 pass        | 0.8–1.5 d                              |
| **4. Multimodal Full**     | Caption + OCR + UI                                                         | +1.8–2.6B tokens               | unfreeze projector, Q‑Former; optional last‑block ViT unfreeze late | 1 pass        | 2–4 d                                  |
| **5. SFT (Chat/CoT)**      | Helpful, CoT formatting                                                    | 200–400M tokens                | bs smaller; LR 8e‑5; dropout 0.05                                   | 1–2           | 8–24 h                                 |
| **6. SFT (Tools)**         | JSON/function calling                                                      | 50–150M tokens                 | same as above                                                       | 1             | 6–12 h                                 |

\* *Wallclock is **illustrative**; depends heavily on dataloader throughput, mixed precision stability, and seq length. Use these as planning ranges, not promises. Early‑stop on eval saturation.*

### Core Tricks on MPS

* **Precision**: bf16 where stable; fall back to fp16 for specific ops if needed; keep master weights fp32.
* **Memory**: gradient checkpointing, sequence packing, sliced attention/kv‑cache, smaller activation dtype.
* **Throughput**: multiple workers, pinned memory, WebDataset shards, cache pre‑tokenized text.
* **Stability**: grad clip 1.0, EMA weights (0.999) checkpointed each eval window.

---

## 4) Implementation Stack

* **Framework**: PyTorch 2.3+ (MPS), Transformers, Accelerate.
* **Trainer**: HF Trainer/TRL or a **custom loop** for fine control over packing, curriculum, multi‑loss weighting.
* **PEFT**: LoRA for SFT stages (r=8–16; α=16–32; target q,k,v,o + MLP in small subsets). Keep **full‑fine‑tune** for base pretrain.
* **Data**: WebDataset/tar shards for images; HF streaming for text; on‑the‑fly tokenization → packer.
* **Logging**: wandb or tensorboard; record **tokens/sec, throughput/GB, loss, ppl, grad‑norms, NaN counts**.
* **Checkpoints**: every 2–8B tokens for base; every 20–50M for SFT.

**Loss weighting during multimodal**: start with 0.8 (LM) / 0.2 (contrastive), anneal to 0.9/0.1 by end; OCR‑heavy batches can go full causal.

---

## 5) Evaluation Plan (fixed battery)

**Text**

* Perplexity on held‑out RefinedWeb/SlimPajama shards
* **TruthfulQA (MC1/2)**, **HellaSwag**, **GSM8K (few‑shot)**, **MMLU‑Lite**

**Function‑Calling**

* ToolBench‑style exact‑match on arguments; schema generalization with held‑out tools

**Multimodal / OCR**

* **TextVQA**, **TextCaps**, **ST‑VQA** exact‑match/F1
* **DocVQA** subsets (layout‑aware Q/A)
* **ScreenQA/UIE** for UI understanding

**Chat/Helpful**

* MT‑Bench‑Lite / AlpacaEval‑style pairwise preference win‑rate

**Acceptance Gates (suggested)**

* PPL ≤ 10 on held‑out web
* OCR exact‑match ≥ 80% on TextVQA‑like split
* Tool‑call success ≥ 90% on in‑distribution; ≥ 75% on unseen schemas
* MT‑Bench‑Lite win‑rate ≥ 65%

---

## 6) Inference, Packaging, and UX

* **Quantization**: 4‑bit (nf4/gguf‑q4\_k) for CPU/MPS inference; aim < 2–2.5 GB RAM.
* **Runtimes**: PyTorch eager for dev; export to **llama.cpp‑style** (custom for projector) or **MLX** for Apple‑native inference.
* **Serving**: Gradio local app; CLI with `--vision` flag; JSON function‑call schema validation.
* **Prompt conventions**: system prompt with tool registry; `<image>` preamble for each image; stop sequences for `<reasoning>`.

---

## 7) Risks & Mitigations

| Risk                          | Likelihood | Impact | Mitigation                                                                                   |
| ----------------------------- | ---------- | ------ | -------------------------------------------------------------------------------------------- |
| Throughput bottlenecks on MPS | Med        | High   | WebDataset, pre‑tokenize, larger packs, profile dataloader, reduce seq length during warm‑up |
| Mixed‑precision instability   | Med        | Med    | Grad clip, bf16→fp16 fallbacks, disable autocast on known‑bad ops                            |
| Overfitting small OCR sets    | Med        | Med    | Heavy synthetic UI/doc data, augmentations, font diversity                                   |
| Vision encoder mismatch       | Low        | Med    | Start frozen; only unfreeze last block late with small LR                                    |
| Function‑calling overfitting  | Med        | Med    | Hold‑out tool schemas, randomize argument order/whitespace                                   |
| Hallucination in CoT          | Med        | Med    | Train with `<reasoning>` tags; inference suppress; add verifiable tasks/tools                |

---

## 8) Concrete To‑Dos (Actionable Checklist)

**Architecture & Repo**

* [ ] Implement 24×1024 decoder with RoPE, RMSNorm, GQA
* [ ] Add Q‑Former‑mini (2–4 layers, 64 queries) and Perceiver‑Resampler options
* [ ] Special tokens & tokenizer build scripts (SentencePiece 50K)
* [ ] Data packer with variable seq lengths and packing

**Data**

* [ ] Stand up text pipeline (RefinedWeb/SlimPajama/Dolma) + MinHash dedup
* [ ] Build screenshot/document **synthetic OCR generator** (fonts, DPI, noise)
* [ ] Create WebDataset shards for vision sets; store crops + transcripts
* [ ] Assemble tool schema pack; synthesize tool‑call datasets with noise/failures

**Training**

* [ ] Stage 0 dry‑run to overfit 50M tokens
* [ ] Stage 1–2 text pretrain with frequent evals & ckpts
* [ ] Stage 3–4 multimodal alignment then full mixed batches
* [ ] Stage 5–6 SFT for chat/CoT and tools (LoRA)

**Eval**

* [ ] Implement fixed eval battery & dashboards
* [ ] Define acceptance gates & early‑stop rules

**Packaging**

* [ ] Export paths: PyTorch → gguf/MLX (custom projector)
* [ ] Gradio demo (chat + image upload + tool calls)

---

## 9) Milestones & Go/No‑Go Gates

1. **M0 (Week 0)**: Repo scaffolding, tokenizer, synthetic OCR generator ✅
2. **M1 (Week 1)**: 50M‑token overfit demo; end‑to‑end data → loss ✅ / ❌
3. **M2 (Week 2–3)**: 2B‑token text pretrain; PPL trend improves ≥ 15% vs. M1 ✅ / ❌
4. **M3 (Week 3–4)**: Multimodal intro; contrastive alignment stable; OCR EM ≥ 60% on small split ✅ / ❌
5. **M4 (Week 4–5)**: Full multimodal; OCR EM ≥ 75%; PPL ≤ 11 ✅ / ❌
6. **M5 (Week 5)**: SFT (chat/CoT/tools); tool EM ≥ 90% in‑dist; ≥ 75% unseen ✅ / ❌
7. **M6 (Week 6)**: Freeze v1.0; quantized demo < 2.5 GB; pass acceptance gates ✅ / ❌

---

## 10) Appendix: Hyperparam Templates

**Pretrain (text)**

```yaml
seq_len: 4096
optimizer: adamw
betas: [0.9, 0.95]
weight_decay: 0.1
lr: 2.5e-4
lr_schedule: cosine
warmup_steps: 2000
grad_clip: 1.0
precision: bf16
batch_tokens_eff: 2_000_000  # use grad accumulation to hit this
ckpt_interval_tokens: 100_000_000
```

**Multimodal (alignment)**

```yaml
image_size: 224
fuse: qformer_mini  # or perceiver_resampler
num_queries: 64
loss_weights: {lm: 0.8, contrastive: 0.2}
unfreeze_vit_last_block_after_steps: 50_000
lr: 1.5e-4
```

**SFT (chat/CoT/tools)**

```yaml
precision: bf16
lora: {r: 16, alpha: 32, target_modules: [q_proj,k_proj,v_proj,o_proj,up_proj,down_proj]}
lr: 8e-5
dropout: 0.05
max_grad_norm: 1.0
seq_len: 4096
```

**Prompt/Format Snippets**

```text
<sys>You are PixelSage, a compact multimodal assistant. Use tools when appropriate. Keep JSON strict.</sys>

<image>
User: Extract the invoice total and due date.
Assistant: <reasoning>…</reasoning>{"total":"$318.77","due_date":"2025-10-15"}
```

---

## 11) Naming (for fun)

* **FusterClucke** (primary)
* SightLine • OCRacle • TinEyelet • Scribelet • OptiMuse

---

## 12) What to Defer / Nice‑to‑Haves

* 15K context & multi‑image conversations
* LayoutLM‑style 2D positional encodings for documents
* Lightweight RLAIF for helpfulness & harmlessness
* On‑device CoreML export path (post‑v1)

---

**Bottom line**: This version trades a sliver of model complexity for *data quality, eval discipline, and Apple‑Silicon realism*. If we enforce acceptance gates and early‑stop policy, we ship a tight 450M multimodal model that’s genuinely useful for chat, OCR on screenshots, and tool use—on a single Mac.
