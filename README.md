# default-rag

Question answering over your own documents. You upload a file, ask a question, and
get back an answer built only from the fragments that were retrieved for it.

Documents are converted and chunked by docling-serve, embedded, and stored in
Qdrant as both dense vectors and BM25 sparse vectors. A question runs a hybrid
search over both, reranks what comes back, and generates the answer from the top
fragments.

The embedder, reranker and LLM are reached over OpenAI-compatible HTTP, so vLLM,
TGI, Ollama or a hosted API all work.

## How it works

Ingestion is asynchronous. `POST /ingest` validates the upload, spools it to a
shared volume and enqueues a job, then answers `202` with a job id. A worker picks
the job up, sends the file to docling-serve for conversion and chunking, embeds the
chunks in concurrent batches, and upserts the points into Qdrant. Each point holds
the chunk text, its heading trail, and whatever metadata you attached. If you passed
a `webhook_url`, the terminal status is POSTed there; either way it is readable from
`GET /ingest/{job_id}`.

Document ids are derived from a digest of the chunk texts, and point ids from the
document id. Re-ingesting the same file overwrites instead of duplicating.

Answering is synchronous. The query is embedded, then searched two ways at once,
dense vector similarity and server-side BM25, with the two result sets merged by
reciprocal rank fusion. The fused candidates go to the reranker, which keeps the
best `top_n`. With verification enabled the LLM is first asked whether those
fragments contain the answer at all; if they don't, the service says so instead of
generating. Otherwise the fragments become the grounding context for the answer.

Architecture diagrams are in [`.docs/c4/`](.docs/c4) as a LikeC4 model. Render them
with `likec4 start .docs/c4`.

## Requirements

- Docker with Compose, or Python 3.13 and Poetry for a local run
- An OpenAI-compatible embedder, reranker and LLM. They are not part of this stack
  and can run wherever you like.

Redis, Qdrant and docling-serve ship with `deploy/docker-compose.yml`.

## Getting started

Two files are needed and neither one is in the repository.

### 1. `.env`

Copy the template and fill in every URL. API keys can stay empty if your upstreams
don't require them.

```
cp .env.example .env
```

| Variable | Points at |
| --- | --- |
| `RAG_QUEUE__REDIS_URL` | `redis://redis:6379/0` |
| `RAG_DOCLING__URL` | `http://docling:5001` |
| `RAG_QDRANT__URL` | `http://qdrant:6333` |
| `RAG_EMBEDDER__URL` | your embedder, OpenAI-compatible base path |
| `RAG_RERANKER__URL` | your reranker, Jina-style `/rerank` |
| `RAG_LLM__URL` | your LLM, OpenAI-compatible base path |

The first three are service names inside the compose network. For an upstream
running on the host, use `host.docker.internal`, since `localhost` inside a
container is the container itself. The compose file already maps that name to the
host gateway.

### 2. `config.local.yaml`

`config.yaml` holds the defaults that are the same everywhere and leaves
deployment-specific values blank. Fill those in here:

```yaml
rag:
  docling:
    chunking:
      max_tokens: 480

  embedder:
    model: intfloat/multilingual-e5-small
    query_prefix: query
    passage_prefix: passage

  qdrant:
    dense_vector: dense
    sparse_vector: bm25
    bm25:
      language: english
      tokenizer: multilingual

  reranker:
    model: BAAI/bge-reranker-base

  llm:
    model: Qwen/Qwen3-8B
    timeout_sec: 120
    temperature: 0.2
    max_tokens: 1024
```

Two of these are easy to get wrong:

- `chunking.max_tokens` needs headroom. It is counted by the embedder's tokenizer
  without special tokens, and the `passage_prefix` is prepended afterwards. If you
  set it to your embedder's context window, the largest chunks overflow and the
  embedding request fails. Leave 5-10% spare.
- `bm25.language` selects the stemmer and stopword list, and has to match the
  language of your documents rather than the language of the code. An English
  stemmer over a Russian corpus degrades the sparse half of every search without
  reporting anything.

This file has to exist before `docker build`, because the image copies it in.

### 3. Run

```
docker compose -f deploy/docker-compose.yml up --build
```

The API listens on `8000`. Add `-f deploy/docker-compose.gpu.yml` to run
docling-serve on CUDA instead of CPU.

Without Docker, from the repository root:

```
poetry install --with api
cd rag && python main.py                          # API
cd rag && arq src.worker.tasks.WorkerSettings      # worker, separate shell
```

Both processes read `config.yaml`, `config.local.yaml`, `prompts.yaml` and `.env`
relative to the working directory.

## API

Interactive docs are served at `/docs`.

### `POST /ask`

```bash
curl -X POST localhost:8000/ask -H 'Content-Type: application/json' -d '{
  "query": "What is the retention period for audit logs?",
  "collection": "handbook",
  "include_context": true
}'
```

| Field | Type | Notes |
| --- | --- | --- |
| `query` | string | 1-8192 characters |
| `collection` | string | Qdrant collection to search |
| `metadata_filter` | object | Optional, restricts retrieval, see below |
| `include_context` | bool | Return the grounding fragments, default `false` |

The response is `{"answer": "...", "context": [...]}`, where each context entry
carries the fragment text, its rerank score, its document id and its metadata.
`context` is `null` unless you asked for it.

Whatever metadata you attached at ingest time is queryable:

```json
{
  "query": "...",
  "collection": "handbook",
  "metadata_filter": {"department": "legal", "year": [2025, 2026]}
}
```

Every key has to match. A list value matches a fragment carrying any one of its
items. Values may be strings, integers or booleans.

### `POST /ingest`

```bash
curl -X POST localhost:8000/ingest \
  -F file=@handbook.pdf \
  -F collection=handbook \
  -F 'metadata={"department":"legal","year":2026}' \
  -F webhook_url=https://example.com/hooks/rag
```

| Field | Notes |
| --- | --- |
| `file` | The document. Extension must be in `ingest.allowed_extensions` |
| `collection` | Created on first ingest if missing |
| `metadata` | Optional JSON object, copied onto every chunk. Default `{}` |
| `webhook_url` | Optional, receives the terminal status |

Answers `202 {"job_id": "..."}`. Malformed metadata or an unsupported extension get
`400`, an upload over `max_upload_bytes` gets `413`.

### `GET /ingest/{job_id}`

Returns `{"job_id", "status", "result", "error"}`. `status` is one of `queued`,
`in_progress`, `success`, `failed`, `canceled`, `not_found`. On success, `result`
holds the document id, collection and chunk count; on failure, `error` holds the
message.

### `POST /ingest/{job_id}/cancel`

Aborts a queued or running job. Returns `202` with `{"canceled": true}` when the
abort was confirmed, and `409` with `{"canceled": false}` when it wasn't confirmed
within `queue.cancel_timeout_sec`. A `409` doesn't mean the job survived: the abort
is already registered in Redis and takes effect when a worker sees it.

### `GET /health`

`200` when Qdrant and Redis both answer, `503` otherwise. The body reports each
dependency separately, so a failing check names itself.

## Configuration

Settings are merged from three sources, later wins:

1. `config.yaml`, defaults, tracked in git
2. `config.local.yaml`, deployment values, not tracked
3. environment, prefix `RAG_`, `__` between levels (`RAG_QDRANT__URL`)

Empty environment variables are ignored rather than treated as empty strings, so a
blank `RAG_LLM__API_KEY` in `.env` leaves the setting unset. API keys are held as
secrets and never appear in logs or error responses.

| Section | Key settings |
| --- | --- |
| `server` | `host`, `port` |
| `queue` | `job_timeout`, `concurrency` (parallel jobs per worker), `cancel_timeout_sec` |
| `logging` | `level` |
| `ingest` | `max_upload_bytes`, `allowed_extensions`, `spool_dir` |
| `docling` | `chunker` (`hybrid` or `hierarchical`), poll and result timeouts, `convert.*` conversion flags, `chunking.max_tokens`, `chunking.merge_peers` |
| `embedder` | `model`, `batch_size`, `max_concurrency`, `query_prefix`, `passage_prefix` |
| `qdrant` | `upsert_batch_size`, `dense_vector` and `sparse_vector` names, `bm25.language`, `bm25.tokenizer` |
| `reranker` | `model`, `timeout_sec` |
| `llm` | `model`, `temperature`, `max_tokens`, `timeout_sec` |
| `retrieve` | `top_k`, `top_n`, `prefetch_multiplier`, `verify` |
| `webhook` | `timeout_sec` |

Retrieval widths compose: each branch prefetches `top_k * prefetch_multiplier`
candidates, fusion narrows them to `top_k`, and the reranker keeps `top_n`.

`retrieve.verify` costs an extra LLM round trip on every question, and in return
the service refuses to answer from insufficient context. Turn it off if latency
matters more.

`docling.timeout_sec` bounds a whole conversion. Left as `null` it never gives up,
and a stuck document holds its worker slot until `queue.job_timeout`.