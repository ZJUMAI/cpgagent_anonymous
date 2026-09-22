# guideline.retrieve

Retrieve focused, auditable snippets from the local Markdown guideline corpus.
The default path is two-stage retrieval:

1. DashScope `text-embedding-v4` document embeddings are cached locally, and
   FAISS is used when installed.
2. The retrieved candidate snippets are reranked by the configured core model.
   `MEDCLAW_GUIDELINE_RERANK_PROVIDER` defaults to `local_openai`; set it to
   `qwen` to use `MEDCLAW_QWEN_MODEL` and DashScope instead.

Use this skill when the agent needs guideline evidence for cancer workup, staging,
molecular testing, pathology confirmation, systemic therapy, radiotherapy, follow-up, or
when the benchmark asks for guideline-aligned recommendations.

Inputs:

- `case_id` is required and should be the current benchmark case id.
- `query` is optional but strongly recommended. Ask a focused clinical question, for
  example: `子宫内膜癌 分期 分子分型 MMR POLE 辅助治疗 随访`.
- `cancer_type` can be `auto`, `all`, or any corpus tag such as `nsclc`, `sclc`,
  or `ucec`.
- `guideline_family` can restrict the corpus to `nccn` or `csco`. With `auto`,
  an explicitly named family in the query is used.
- `guideline_version` can restrict the corpus to a four-digit version such as
  `2010` or `2025`. With `auto`, a year adjacent to NCCN or CSCO in the query
  is used.
- `max_snippets` controls how many guideline chunks are returned.
- `chunk_mode` defaults to `chapter`, reusing `guideline_planner`'s first-level
  heading chunks so traditional RAG can be compared against latent
  guideline-memory planner slots. Explicitly use `page` only for legacy
  page/window retrieval experiments.
- `retrieval_mode` defaults to `auto`. Use `vector` to require cached/generated
  embeddings, or `lexical` for deterministic keyword fallback.
- `rerank_mode` defaults to `auto`. Use `llm` to require core-model reranking,
  or `none` to preserve retrieval order.
- `rerank_candidate_count` controls how many retrieved candidates are sent to
  the core model before returning `max_snippets`.
- `force_rebuild_embeddings=true` rebuilds document embeddings and the local FAISS index.

Runtime/cache behavior:

- This skill runs with the same active Python interpreter as MedClaw and does
  not require a separate per-skill environment.
- Missing document embeddings are generated with `DASHSCOPE_API_KEY`.
- Chapters that exceed the embedding API input limit are internally segmented;
  segment vectors are mean-pooled back into one chapter vector, so chapter-mode
  retrieval still returns one candidate per first-level guideline section.
- Cached document embeddings and repeated query embeddings are reused to avoid
  repeated token spend.
- LLM reranking uses the same `DASHSCOPE_API_KEY`, `MEDCLAW_QWEN_MODEL`, and
  `MEDCLAW_QWEN_BASE_URL` as the core model. In `auto` mode, rerank failure
  falls back to the original retrieval order with a warning; in `llm` mode, it
  fails clearly.
- Guideline Markdown files are discovered recursively under
  `medclaw/knowledge/guidelines/`. The bundled corpus is grouped by guideline
  family, including `csco/` and `nccn/`.
- Cache path defaults to `medclaw/knowledge/guidelines/.embedding_cache/`.
  Override it with `MEDCLAW_GUIDELINE_EMBEDDING_CACHE_DIR`.
- Embedding model defaults to `text-embedding-v4`; override with
  `MEDCLAW_GUIDELINE_EMBEDDING_MODEL`.

Output:

- `findings.snippets` contains source guideline name, page number, vector score,
  matched terms, and a bounded text excerpt.
- A JSON artifact is saved for audit, but the full guideline files are not sent to the
  model.

Safety:

- This tool retrieves guideline text for benchmark research only.
- It does not decide treatment by itself. The final answer must integrate patient stage,
  pathology, biomarkers, performance status, comorbidities, and available evidence.
