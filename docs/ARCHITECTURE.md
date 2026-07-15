---
title: "Enterprise Insurance AI Chat Agent"
subtitle: "Architecture, Components, and Operational Guide"
author: "System Documentation"
date: "January 2026"
---

# Overview

The Enterprise Insurance AI Chat Agent is a production-grade, RBAC-enforced, RAG-powered assistant that serves customers, employees, and administrators of an insurance carrier. It combines Retrieval-Augmented Generation, agentic tool use, short- and long-term memory, and defense-in-depth Role-Based Access Control, while meeting the compliance, observability, and reliability expectations of a regulated financial services environment.

This document describes the system's architecture, the responsibility of each module, the security posture, the test strategy, and how to operate the deployment. Read it end to end if you're new to the codebase; skim the "Module map" section if you know what you're looking for.

# Guiding principles

Three cross-cutting principles inform every component and are worth stating up front, because they are what distinguish this build from a demo-quality one.

**Defense in depth for data access.** RBAC is enforced at multiple layers: at the database (Postgres row-level security policies keyed off session variables), at the retrieval layer (metadata `WHERE` clauses applied before vector search), and at the tool layer (parameterized query templates run under a read-only Postgres role). No single failure — including a fully jailbroken agent — should be able to leak another user's data. Every layer is a genuine gate, not a rubber stamp.

**Observability and evaluation as first-class components.** Structured logging, hash-chained audit records, retrieval and routing evaluation suites, and per-turn tracing hooks are part of the initial build, not bolted on. Every LLM call, tool invocation, retrieval event, and RBAC decision is traceable. This is what lets you debug production incidents and improve the system over time; without it, you're flying blind.

**Compliance-aware by design.** The audit log is immutable and tamper-evident. PII detection and redaction run inline. Long-term memory extraction excludes sensitive categories by policy. Cascade delete supports right-to-erasure. Retention windows are configurable to align with jurisdiction-specific insurance regulations (commonly five to seven years).

# High-level architecture

The system is composed of the following runtime components:

- **FastAPI backend** — the HTTP interface: authentication, chat, document upload, admin, policies.
- **PostgreSQL 16 with pgvector** — the source of truth: users, policies, chat history, long-term memory, audit log, and the vector store for RAG.
- **Redis** — short-term memory (per-session state), the refresh-token revocation list, rate limiting, and the ingestion job queue.
- **Ingestion worker (Arq)** — a separate process that consumes document ingestion jobs from Redis, parses, chunks, embeds, and writes to Postgres.
- **React frontend (Vite)** — the customer- and employee-facing chat UI, plus a minimal admin surface.
- **LiteLLM** — a provider abstraction sitting in front of OpenAI (primary), Anthropic, and Azure OpenAI for failover.

The backend is a single container that hosts the FastAPI app and a set of internal services. The ingestion worker is a second container using the same code base but a different entrypoint. This separation matters: uploading a large PDF must not consume request handler capacity, and worker crashes must not take the API down.

The React frontend is a static Single-Page Application that talks to the API over HTTPS. All access tokens are short-lived (15 minutes); refresh tokens are rotated on use and revocable at the server.

# Module map

The backend source tree is organized by responsibility. Each module has a narrow contract and its own tests.

| Path | Purpose |
| --- | --- |
| `backend/app/core/` | Configuration, structured logging, security primitives, domain exceptions |
| `backend/app/db/` | SQLAlchemy models, Alembic migrations, session factories, RLS policies, seed data |
| `backend/app/auth/` | Login, JWT signing/verification, refresh token revocation, security principal loader |
| `backend/app/rbac/` | Permission engine, tool-level authorization decorators, scope checks |
| `backend/app/guardrails/` | PII detection (Presidio + insurance-specific rules), prompt injection classification, output grounding and citation validation |
| `backend/app/rag/` | Document parsers, chunker, embedder (LiteLLM/OpenAI + fake for tests), hybrid retriever (BM25 + dense), reranker, ingestion service, Arq worker |
| `backend/app/agent/` | LLM client wrapper, router, tool implementations, LangGraph-style orchestrator |
| `backend/app/memory/` | Short-term store (Redis + in-memory fallback) with rolling summarization; long-term extraction, storage, and semantic recall |
| `backend/app/audit/` | Hash-chained append-only audit log with a chain verifier |
| `backend/app/api/` | FastAPI routers for auth, chat (SSE), documents, admin, policies; shared dependencies |
| `backend/app/schemas/` | Pydantic request/response schemas |
| `backend/tests/` | Unit and integration tests (97 tests, all deterministic, SQLite-backed) |
| `frontend/` | React + Vite SPA: login, session list, streaming chat |
| `infra/` | Postgres init scripts, RLS policy SQL |
| `docs/` | This document and operational notes |

# Component-by-component walkthrough

## Configuration and logging

All settings load from environment variables via a single `Settings` object (`app/core/config.py`). Nothing else in the codebase reads `os.environ` directly — this makes tests trivially overridable and keeps the deployment contract in one place.

Logging is structured JSON via `structlog`. Every log line carries the fields you'll want to correlate: `request_id`, `user_id`, `session_id`, `tool_name`, `event`. Human-friendly colored output is used in development; production emits machine-parsable JSON.

Domain exceptions live in `app/core/exceptions.py` as a small taxonomy: `AuthenticationError`, `PermissionDenied`, `GuardrailViolation` (with a category label), `RateLimitExceeded`, `NotFoundError`, `ValidationError`. Each carries an `http_status`, so the FastAPI exception handler maps them declaratively without a chain of `if isinstance(...)` checks.

Security primitives are Argon2id for password hashing (with bcrypt fallback for existing hashes during migration) and HS256 JWTs. Access tokens are 15 minutes by default; refresh tokens are 7 days and rotate on use. Both are opaque to the client apart from decode-time verification.

## Database and models

The data model is defined in `app/db/models.py` using SQLAlchemy 2.0's `Mapped[...]` style. There are fifteen tables:

- Identity and access: `users`, `roles`, `permissions`, `role_permissions`, `user_roles`, `departments`, `employee_profiles`
- Insurance domain: `insurance_policies`, `claims`
- Conversations: `chat_sessions`, `chat_messages`
- Memory: `long_term_memories`
- Knowledge base: `documents`, `document_chunks`
- Audit: `audit_logs`

Portable column types (`GUID` and `JSONType` in `app/db/types.py`) let the same models work against Postgres (with native UUID and JSONB) and against SQLite (used for hermetic unit tests). This is deliberate: unit tests should not require a running Postgres, and they don't.

The Postgres-only extras — the `vector` column on `document_chunks` and `long_term_memories`, the `tsvector` column and its update trigger, the HNSW and GIN indexes — are applied in the second Alembic migration (`0002_pgvector_and_bm25.py`). On other dialects, this migration is a no-op.

Sessions are opened through async context managers in `app/db/session.py`. The factories are lazy so importing the module never fails just because a driver is missing in a hermetic test environment. Every session sets three Postgres session variables — `app.current_user_id`, `app.current_user_type`, `app.current_department_id` — which RLS policies key off. `set_config(..., true)` scopes them to the transaction so a leaked pooled connection cannot carry stale identity into the next request.

A separate read-only engine, connected as the `insurance_ro` Postgres role, is used by all agent tool queries. That role has `SELECT` grants only; even if a jailbroken agent generated a mutating statement, the database would reject it. This is defense in depth in the plainest form.

## Row-level security

Row-level security policies (`app/db/rls.py`) enforce the same rules the application code enforces, so the database is the last line of defense:

- `insurance_policies`: customers can only see rows where `customer_id` matches the session variable; employees see rows whose `policy_type` matches the department linked to their session variable; administrators see everything.
- `claims`: inherit from policies via subquery, so the policy's RLS automatically applies.
- `chat_sessions`, `chat_messages`, `long_term_memories`: owner-only, with an administrator override for review purposes.
- `audit_logs`: administrator read only; nobody can `UPDATE` or `DELETE` (enforced at the grant level as well as the policy level).

This works because the app sets the session variables at the top of every session. If a bug in the application forgets to set them, the RLS policies return zero rows rather than everything — a safe failure mode.

## Authentication

The login flow is straightforward: verify credentials against `users.hashed_password`, issue an access token and a refresh token, log the event in the audit log (success or failure). The uniform error message on wrong password vs. unknown email prevents user enumeration.

The `SecurityPrincipal` (`app/auth/service.py`) is the resolved identity for a request: user id, email, user type, list of role names, set of `(resource, action)` permission tuples, department id and name. Loading a principal is one eagerly-loaded query — no lazy loads leak into request handlers. Everything RBAC needs travels with the principal, so downstream code never queries the identity subsystem again mid-request.

Refresh tokens use the standard rotation-on-use pattern. When a client calls `/api/auth/refresh`, the presented token is immediately revoked (its `jti` added to the denylist) and a fresh pair is issued. The denylist is in-memory for tests and Redis-backed in production; either way, decisions are quick and revocations propagate globally.

## RBAC engine

The RBAC engine (`app/rbac/engine.py`) offers three interfaces:

- Assertions (`assert_permission`, `assert_any_permission`) for use inside handlers.
- Scope checks (`can_view_policy`, `visible_document_visibilities`, `visible_departments`) for domain-specific policies that combine role permissions with row-level context.
- The `@authorize_tool(...)` decorator, which wraps agent tool coroutines so they check permissions before touching the DB or vector store.

The last is the important one for agentic architectures. Every tool checks its permissions *before it executes*, so even an agent that has convinced itself to call `search_knowledge_base` on behalf of a customer who lacks the internal-document permission will fail cleanly. Combined with the retrieval-layer metadata filter and Postgres RLS, this is a genuine three-layer gate.

## Guardrails

Guardrails are implemented as three composable components: PII detection, prompt injection classification, and output validation. An orchestrator ties them together into an `InputGuardrails` and an `OutputGuardrails` pair used by the agent.

The **PII detector** (`app/guardrails/pii.py`) uses Presidio in production with an insurance-specific extension for policy numbers (`LIFE-100001`, `HOME-200001`, `VEHICLE-300001`) and claim numbers (`CLM-*`). When Presidio is unavailable — in unit tests, or in stripped-down deployments — it falls back to a rule-based detector that covers SSN, credit card, email, US phone, US ZIP, and the insurance-specific categories. The output of `.scan(text)` is a list of matches with category, span, matched text, and confidence. `.redact(text)` returns the text with matches replaced by `[REDACTED:CATEGORY]` placeholders.

The **prompt injection classifier** (`app/guardrails/injection.py`) is rule-based by default and scores six categories: instruction override ("ignore previous instructions"), role hijack ("you are now DAN mode"), system prompt exfiltration ("reveal your initial instructions"), encoding smuggling (obvious base64 payloads with wrappers), SQL injection patterns, and tool abuse phrasing. Each category has its own confidence weight; the decision to block is a threshold on the maximum. In production this is augmented with a small classifier (Prompt Guard 86M or equivalent) whose score combines with the rule signals.

The **output validator** (`app/guardrails/output.py`) checks that the assistant's answer is grounded in the retrieved sources. It splits the answer into sentences, classifies each as factual or hedge/greeting, and for factual sentences requires that (a) it contains an inline citation of the form `[source: ID]` and (b) the sentence has meaningful lexical overlap with the cited chunk. Both requirements are tunable. Sentences that fail these checks are collected as problems and the answer is either downgraded to a safe fallback or regeneration is triggered. The production implementation swaps the lexical heuristic for an NLI-based entailment check.

The orchestrator (`app/guardrails/orchestrator.py`) runs input checks in order (injection first, PII second — injection is blocking, PII is redacting but non-blocking) and output checks after synthesis. Guardrail events are logged with category and count so operators can see which categories fire and how often. That's the beginning of a proper evaluation loop.

## Retrieval-augmented generation

RAG is split into ingestion and retrieval, each with clean boundaries.

**Ingestion** (`app/rag/ingestion.py`) is a pipeline: parse the uploaded document (PDF, DOCX, PPTX, or plain text) via `app/rag/parser.py`; chunk it structure-aware with parent-child hierarchy via `app/rag/chunker.py`; embed the child chunks; persist chunks with their RBAC metadata (visibility tier, department id) so the retriever can pre-filter.

Deduplication is content-hash based: uploading the same file twice returns the existing document row. Document status transitions through `queued → parsing → embedding → indexed`, or `failed` with an error message on any exception, so admins can monitor progress and diagnose failures without opening logs.

The **chunker** produces both parent chunks (large coherent sections, ~1500 tokens) and children (~300 tokens, with overlap). At retrieval time, the small chunks provide precision; the parent expansion at synthesis time provides coherent context. Insurance documents in particular have nested sections and coverage tables that naive character splitting fragments across chunks — a homeowner's declarations page loses meaning when "Coverage A: Dwelling — $250,000" is split across a chunk boundary. Splitting by structure fixes this.

The **embedder** is `text-embedding-3-large` at 1536 dimensions in production, with a deterministic fake embedder for tests. The fake produces reasonable cosine-similarity behavior via token hashing plus L2 normalization; it is not competitive with a real model, but it is competitive enough to make retrieval unit tests meaningful.

The **hybrid retriever** (`app/rag/retriever.py`) combines BM25 and dense scores. Both sides retrieve the top 20 candidates independently, their scores are min-max normalized, unioned, and combined with configurable weights (default 0.6 dense, 0.4 BM25). Insurance queries mix natural language ("what does my policy cover for water damage") with exact-match tokens (policy numbers, endorsement codes like "HO-3"), and hybrid retrieval consistently outperforms either alone.

The **reranker** (`app/rag/reranker.py`) reduces the fused candidates to the top 5. In production it is a cross-encoder (BGE-reranker-v2-m3 by default); in tests it is a lexical Jaccard-overlap fallback that produces deterministic ordering. Rerankers are cheap and consistently improve precision; this is one of the cheapest single-digit-percentage precision wins available in RAG.

Critically, **RBAC is enforced at retrieval time, not after**. The `search_knowledge_base` tool loads only the chunks whose visibility tier and department metadata match the caller's principal. Post-filtering — retrieving 100 candidates and dropping the ones the user can't see — leaks data through top-k truncation and wrecks recall metrics. Pre-filtering costs nothing in Postgres because the metadata columns are indexed.

## The agent

The agent (`app/agent/graph.py`) is a state machine with these nodes: input guard, router, execute tools (concurrently, bounded), synthesize, output guard. In production it's a LangGraph `StateGraph`; here it's implemented as a linear sequence for testability. Behavior is identical.

**Input guard** runs PII redaction and injection classification. The redacted text (not the original) is what downstream nodes see. Blocking is limited to injection; PII is redacted but not blocked, because customers legitimately ask about their own policy numbers.

**The router** (`app/agent/router.py`) classifies the query into one or more tool calls. In production it's an LLM call with structured JSON output; in tests it's a deterministic keyword-based router that produces the same shape. Both output a `RoutingDecision`: `tools` (list of tool names, whitelisted so the LLM cannot hallucinate a tool that doesn't exist), `query_rewrite` (a standalone version of the query, useful when the user asks "what about that policy?"), `reasoning`, and an optional `clarify` question when the query is ambiguous.

**Tool execution** runs the selected tools concurrently under a semaphore. Every tool uses parameterized queries and applies its own authorization check via `@authorize_tool` on top of the pre-request principal check.

The available tools are:

- `get_my_policies` — the authenticated customer's own policies.
- `get_policy_by_number` — a specific policy, with the RBAC scope check (customers see only their own; employees see only their department; admins see everything).
- `get_claims_for_policy` — claims for a policy that passes the scope check.
- `search_knowledge_base` — the hybrid retrieval pipeline, pre-filtered by the principal's visibility tier and department scope.
- `recall_long_term_memory` — the user's stored preferences and facts. Only ever accesses memories belonging to the authenticated user, keyed off `user_id`.

Any of these can fail closed and return an empty result rather than an exception; the synthesis node then works with what it has and cites what it retrieved.

**Synthesis** takes the tool outputs and produces the final answer with inline citations of the form `[source: ID]`. The prompt is explicit: use only the provided sources; if you cannot answer with them, say so.

**Output guard** checks the answer. On failure, we do not fail hard — we downgrade to a safe fallback message ("I don't have enough grounded information to answer that reliably...") and log the block reason. This is intentional: a brittle guardrail that raises 500s at the user is worse than one that returns a graceful non-answer.

Every turn logs `turn_started` and `turn_completed` (or `turn_blocked`) events to the audit log with the tools used, source count, and any block reason.

## Memory

Short-term memory (`app/memory/short_term.py`) is a per-session dict with a rolling summary. State is serialized to Redis (or an in-memory store for tests) keyed by `session_id`. When the message list exceeds 20 messages, the oldest are summarized and dropped so context stays bounded. Summarization is a stub in tests (just appends "prior turns condensed") and an LLM call in production.

Long-term memory (`app/memory/long_term.py`) is different from short-term in two ways: it persists across sessions in Postgres, and it is *extracted*, not stored verbatim. The extractor scans user messages (never assistant messages — the assistant's suggestions must not be laundered into "the user said") for preferences, facts, and context. Messages with SSN patterns are skipped wholesale as a defense-in-depth measure on top of PII redaction.

Recall is semantic in production (pgvector cosine similarity) and keyword-based in tests. When a new preference conflicts with an old one (both mention "prefer email" vs "prefer phone"), the older row is marked `is_active=false` rather than deleted, preserving audit history. `last_referenced_at` is updated on recall hits, which is the beginning of recency-based decay.

Right-to-erasure is straightforward: cascade delete on the user row propagates to memories, chat sessions, and chat messages. The audit log retains a record of the erasure itself.

## Audit log

The audit log (`app/audit/service.py`) is append-only and hash-chained. Every row carries a `prev_hash` and a `chain_hash`; each `chain_hash` = sha256(prev_hash || row_hash). Genesis is 64 zeros. Tamper with any historical row — change a payload, delete a row, insert a fabricated one — and the chain breaks at that point.

`verify_chain()` walks the log and returns `(ok, first_broken_id)`. On success, `first_broken_id` is `None`. On failure, it points at the first row whose recomputed chain hash disagrees with the stored one. This is the row where tamper occurred, or the row just after a deletion.

In Postgres, the `audit_logs` table also has `UPDATE` and `DELETE` grants revoked at the SQL level, and RLS restricts `SELECT` to administrators. Combined with the chain, three layers protect the log: table grants prevent mutations, RLS prevents unauthorized reads, and the chain detects any tampering that somehow slips through.

The audit log records: authentication events (success and failure), RBAC denials, agent turn started/completed/blocked, tool calls, document uploads and deletions, and any explicit administrative action. Retention is configurable to match jurisdiction (commonly 5–7 years for insurance).

## HTTP API

The API (`app/api/`) is FastAPI, organized by resource. Every route depends on `current_principal`, which resolves the JWT and loads a `SecurityPrincipal` from the database. Permission-gated routes use `require_permission("resource", "action")` as a declarative dependency; role-gated routes use `require_user_type("administrator")`.

Chat messages are streamed via Server-Sent Events. The event types are:

- `routing` — which tools the router picked.
- `sources` — the retrieved chunks, so the UI can render provenance.
- `answer_chunk` — incremental deltas of the answer text.
- `done` — the final message id and any block reason.

SSE is chosen over WebSockets because the server-to-client stream is one-way — no reason to pay the WebSocket complexity tax.

Document upload uses multipart form-data. On upload, the file is saved to disk (under `uploads/`, mounted as a Docker volume), registered as a `Document` row in `queued` status, and enqueued to the Arq worker. If Redis is unavailable the request handler falls back to inline ingestion — fine for dev, not for prod.

Administrative endpoints expose the audit log, the chain verifier, and a user listing.

The FastAPI app configures CORS from the environment, adds a request-id middleware that binds the id into structlog's context vars so every log line during a request is correlated, and maps `InsuranceAgentError` subclasses to their `http_status`.

## Frontend

The frontend (`frontend/`) is a Vite-built React SPA with two views: login and chat. It authenticates against `/api/auth/login`, keeps the access token in memory (not `localStorage` — that's XSS bait), and streams chat responses via `fetch` + manual SSE parsing.

The UI is deliberately minimal but functional. It shows the user's identity and role, a session list, message history, a composer, and — importantly — the routing decision (which tools the agent used) and the retrieved sources for the current turn. Making the agent's reasoning visible in the UI is a foundational transparency feature; users trust systems whose choices they can see.

# Test strategy

The suite contains 97 tests across the following files:

| File | Tests | Coverage focus |
| --- | --- | --- |
| `test_db.py` | 5 | Model registration, seed idempotence |
| `test_auth.py` | 8 | Login, JWT roundtrip, token type confusion, refresh revocation, principal loading |
| `test_rbac.py` | 10 | Permission assertions, policy scope, document visibility tiers, department scoping, tool decorator |
| `test_guardrails.py` | 21 | Injection categories, PII patterns, redaction, output grounding, citation enforcement |
| `test_rag.py` | 11 | Chunking, embedder determinism, BM25 exact match, hybrid retrieval, RBAC pre-filter, reranker |
| `test_audit.py` | 6 | Chain linking, verification, payload tamper detection, deletion detection |
| `test_memory.py` | 9 | Short-term put/get, rolling summarization, extraction, persistence, conflict handling |
| `test_agent.py` | 13 | Router, tool authorization, cross-customer isolation, full agent flow, PII redaction end-to-end, audit recording |
| `test_api.py` | 14 | HTTP auth, policy scoping over HTTP, chat session isolation, audit-log admin gates |

The default test path is fully hermetic: SQLite in-memory database, no Redis, no LLM calls. A deterministic `FakeLLM` and `FakeEmbedder` allow the full agent flow to be exercised without API keys. This is a design goal: unit tests must run in under a minute on a laptop with no external dependencies, so contributors actually run them. Integration tests requiring Postgres and Redis are marked `@pytest.mark.integration` and can be run against a docker-compose test stack.

The most important tests in the suite are the RBAC boundary tests — the ones that verify a customer cannot see another customer's policy, an employee in Life cannot see a vehicle policy, and an unauthorized session id returns 404. If any of these break, we have a data-leakage regression, and any change that touches auth, RBAC, or the retriever must pass them.

# Operational notes

## Local development

```bash
cp .env.example .env
# Set OPENAI_API_KEY and generate JWT_SECRET_KEY (openssl rand -hex 32)
docker compose up --build
```

The stack comes up with:
- Postgres 16 with pgvector, seeded with demo users and policies.
- Redis 7 for short-term memory and the ingestion queue.
- Backend on port 8000 (Swagger UI at `/docs`).
- Ingestion worker.
- Frontend on port 5173.

Default credentials seeded by `app/db/seed.py`:

- Admin: `admin@example.com` / `admin-change-me` — change immediately.
- Customers: `alice@example.com`, `bob@example.com`, `carol@example.com` / `customer-pass`.
- Employees: `life.employee@example.com`, `home.employee@example.com`, `vehicle.employee@example.com` / `employee-pass`.

## Running tests

```bash
cd backend
pip install -r requirements.txt
python -m pytest -v
```

All 97 tests should pass in about a minute. There are no external dependencies.

## Production deployment

Docker Compose is fine for local development. For production, deploy the backend and ingestion worker containers to Kubernetes or ECS, use a managed Postgres (with pgvector), a managed Redis, and put an ingress in front of the backend with TLS termination. Secrets belong in a real secret manager (Vault, AWS Secrets Manager, GCP Secret Manager) — not in a shipped `.env`.

Environment variables to set in production: `OPENAI_API_KEY`, `JWT_SECRET_KEY`, `POSTGRES_*`, `REDIS_URL`, `CORS_ORIGINS`, and optionally `LANGSMITH_API_KEY` for tracing. Everything else has a sensible default.

## Cost and performance controls

The primary cost lever is model routing. The router uses `gpt-4o-mini`, synthesis uses `gpt-4o`. Embedding uses `text-embedding-3-large` (batched, up to 100 inputs per call). Semantic caching for FAQ-style queries is straightforward to bolt on via a Redis key on the query hash — this is the single biggest cost lever after model routing.

Per-user token budgets are configured via `MAX_TOKENS_PER_USER_PER_DAY` and enforced at request admission. Rate limiting is per user and per role, applied at the ingress level via Redis counters.

## Observability

Instrumentation for LangSmith (or Langfuse) is wired into the LLM client wrapper via LiteLLM; set `LANGSMITH_API_KEY` and `LANGSMITH_PROJECT` and traces appear per turn. Structured logs are the second observability leg; a log aggregator (ELK, Datadog, Loki) consumes them and enables correlation with the request id present in every trace and log line.

Metrics worth tracking: retrieval precision@k against a golden dataset, router classification accuracy against a labeled routing dataset, guardrail firing rates by category, per-user token spend, p50/p95/p99 latency by endpoint, and (in production) the audit chain verification result on a nightly cron. The last is your tamper-evidence signal — if the chain verifier ever returns `ok=false`, someone or something has touched the audit log directly.

## Threat model summary

The primary threats and how the system addresses them:

**Data exfiltration via a jailbroken agent.** Addressed by defense in depth: the tool decorator checks permissions before execution; the retriever pre-filters by visibility and department; the read-only Postgres role prevents any mutation; RLS enforces per-row visibility at the database. Even complete LLM compromise cannot fetch data the caller is not authorized to see.

**Prompt injection via document content.** The document ingestion pipeline does not execute retrieved content; retrieved chunks are passed to the LLM as reference material with explicit "use only the provided sources" instructions. Adversarial content in documents is treated as data, not as instructions. Output validation catches attempts to answer from unstated sources.

**Session hijacking.** Access tokens are short-lived and refresh tokens are rotate-on-use with a server-side denylist. Compromised tokens can be revoked immediately. Admin accounts require MFA (interface wired, enforcement recommended).

**Audit log tampering.** The hash chain makes tampering detectable; grants prevent it at the database level; RLS restricts reads to administrators.

**PII leakage into memory or logs.** PII detection runs on input and output; long-term memory extraction skips messages with detected PII; the audit log stores hashes, not raw payloads, where sensitive fields are involved.

**Denial of service via expensive queries.** Per-user token budgets, rate limits, and bounded tool-execution parallelism (semaphore of 3) prevent runaway loops from exhausting quotas.

# Extension points

The architecture is designed for the following extensions:

**Cross-encoder reranker upgrade.** The reranker interface is stable; swap `LexicalReranker` for `CrossEncoderReranker` and set `RERANKER_ENABLED=true`. In production this is the default.

**A different LLM provider.** LiteLLM is already in the client wrapper; add the provider name to the `PRIMARY_LLM_MODEL` env var and (if needed) additional API keys.

**Multi-tenant deployment.** The schema is single-tenant. Multi-tenancy would add a `tenant_id` column to identity and domain tables, propagate a tenant filter to RLS policies, and add tenant isolation to the retriever's pre-filter.

**Persistent MFA and passwordless login.** The `User` model has `mfa_secret` and `mfa_enabled` columns wired but not enforced. Add a TOTP challenge in the login endpoint and a WebAuthn flow for passwordless.

**Feedback loop for continuous evaluation.** The chat API can be extended with thumbs-up/thumbs-down endpoints that write to a `feedback` table joined with the audit log; the training/evaluation team consumes this to build a golden dataset over time.

# Closing

This system is a serious attempt at production-grade agentic RAG in a regulated vertical. Every design decision — the read-only DB role, the hash-chained audit log, RBAC at three layers, the deterministic test path, the pre-filtered retrieval — is the response to a specific failure mode encountered when building systems that handle real money and real personal information.

It is not, however, a place to stop. Cross-encoder rerankers, proper NLI-based output validation, semantic caching, evaluation datasets in CI, per-tenant fine-tuning, WebAuthn — these are the next steps. The foundations here should let you take them without ripping anything up.
