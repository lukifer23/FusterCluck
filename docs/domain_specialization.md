# Domain Specialization Plan (Stage 0 → Stage 4)

We are refocusing FusterCluck-450 on a narrower but deeper slice of knowledge:

1. **Science** – physics, math, ML research prose, experiment reports
2. **Data & Analysis** – statistics, data engineering, analytics writeups, experiment tracking
3. **Code** – scientific/ML workloads in Python, C/C++, Rust (numerics, systems, tooling)
4. **Chess** – notation, strategy commentary, annotated games
5. **General chat ballast** – minimal coverage to keep conversational quality

Vision/OCR will move to a *post-base* alignment stage. The base 12–16B token curriculum is purely text, reducing complexity until we have a strong domain foundation.

## Token Budget Outline (Text-only)

| Domain        | Target Tokens | Example Sources                                                               |
| ------------- | ------------- | ----------------------------------------------------------------------------- |
| Science       | 4.5B          | arXiv (math/physics/cs), PubMed Central OA subset, NASA/ESA docs, textbooks   |
| Data & Analysis | 2.5B        | Stats/ML textbooks, experiment logs, data engineering blogs, notebooks        |
| Code          | 3.0B          | The Stack v1 Python/C/C++/Rust, scientific repos (NumPy, PyTorch, OpenMM)     |
| Chess         | 1.5B          | Lichess PGN dumps, Chess.com commentary, Chess StackExchange, annotated PDFs  |
| General Chat  | 0.5B          | RefinedWeb science portals, Wikipedia science portals, curated dialogs        |
| Instruction/SFT | 0.5B       | Domain-specific instructions, reasoning, tool calls                           |
| Eval/Holdout  | 0.2B          | Carved per domain                                                             |

Total pretrain ≈ 12.7B tokens before SFT. Remaining 0.5–1.0B tokens can flex toward science or code depending on eval trends.

## Vision/OCR Strategy

- **Deferred to Stage 4.5+**: once the base LM is trained, introduce multimodal adapters using synthetic science PDFs, code screenshots, and chessboard renders.
- Advantages: keeps early training simple, reduces MPS memory pressure, lets us tailor vision datasets specifically to our domains.

## Domain-specific Goals

- **Science**: concise equation handling, accurate summarization, ability to explain derivations.
- **Data & Analysis**: reason about statistics pipelines, interpret experiment logs, produce clean data narratives.
- **Code**: generate and critique ML training scripts, CUDA kernels, scientific simulations.
- **Chess**: reason about positions, provide annotations, perform SAN/PGN manipulation, basic tactics.
- **General chat**: maintain helpful tone and follow instructions without domain drift.

## Upcoming Tasks

1. Stage 0 shard rebuilt using curated domain samples (50M tokens) to validate tokenizer + pipeline.
2. Stage 1 dataset manifests per domain with token accounting.
3. Evaluation harness additions: 
   - Science comprehension (arXiv/PubMed QA)
   - Data & analysis benchmarking (forecast backtests, stats problem sets)
   - Code tests (unit snippets, docstring completion)
   - Chess move prediction/tactic classification.

This specialization keeps the project ambitious but grounded in real strengths, paving the way for meaningful tooling and demos.
