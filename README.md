# Enterprise Insurance AI Chat Agent

A production-grade, RBAC-enforced, RAG-powered insurance assistant built on FastAPI, LangGraph, PostgreSQL (with PGVector + RLS), Redis, and React.

## Features

- **Persistent chat sessions.** Every conversation is a `ChatSession` row with its `ChatMessage` history in PostgreSQL, scoped to the owning user. Sessions survive page reloads, logouts, and container restarts (as long as the `postgres_data` volume sticks around), the sidebar lists them by most recently updated, and each one gets auto-titled from its first message.
- **RBAC-enforced access.** Customers, employees, and administrators see different slices of policies, documents, and chat history, enforced both at the API layer and via Postgres row-level security.
- **RAG-powered answers.** Hybrid retrieval (BM25 + dense embeddings + reranking) over ingested policy documents, with retrieved sources shown in the UI so answers stay traceable.
- **Agentic tool use.** A LangGraph state machine routes each turn to the right tool (policy lookup, document search, memory recall) and shows its routing decision in the UI.
- **Short- and long-term memory.** Per-session rolling summarization in Redis, plus long-term preference/fact extraction that persists across sessions in Postgres with pgvector.
- **Guardrails.** Inline PII detection/redaction (Presidio) and prompt-injection defenses on both input and output.
- **Immutable audit log.** Hash-chained, append-only log of security-relevant events, readable only by administrators.

## Quick start

```bash
cp .env.example .env
# Edit .env: set OPENAI_API_KEY and JWT_SECRET_KEY (openssl rand -hex 32)
docker compose up --build
```

Then open:
- Frontend: http://localhost:5173
- API docs: http://localhost:8000/docs
- Default admin: `admin@example.com` / `admin-change-me` (change immediately)

## Module map

| Path | Purpose |
|---|---|
| `backend/app/core/` | Config, security primitives, logging, exceptions |
| `backend/app/db/` | SQLAlchemy models, session management, RLS helpers, migrations |
| `backend/app/auth/` | Login, JWT, refresh tokens, MFA hooks |
| `backend/app/rbac/` | Roles, permissions, tool-level authorization |
| `backend/app/guardrails/` | PII (Presidio), prompt injection, output validation |
| `backend/app/rag/` | Ingestion pipeline, hybrid retrieval (BM25 + dense + rerank) |
| `backend/app/agent/` | LangGraph state machine, router, tool nodes, memory extraction |
| `backend/app/memory/` | Short-term (Redis checkpointer) and long-term memory |
| `backend/app/audit/` | Immutable, hash-chained audit log |
| `backend/app/api/` | FastAPI routes: auth, chat (SSE), documents, admin |
| `backend/tests/` | Unit + integration tests |
| `frontend/` | React (Vite) minimal UI: login, chat, session list |
| `infra/` | docker-compose, Postgres init scripts, Alembic env |
| `docs/` | Architecture doc, threat model, ops runbook |

## Testing

```bash
cd backend && pytest -v                    # unit tests
docker compose -f docker-compose.test.yml up --abort-on-container-exit  # integration
```

See `docs/ARCHITECTURE.docx` for the full component walkthrough.
