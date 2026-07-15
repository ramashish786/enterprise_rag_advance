# Changelog

## v2 — Full dependency refresh (2026-07-11)

All Python packages upgraded to their latest stable release. The 97-test suite continues to pass in full (~73s) and the live HTTP smoke test passes on all endpoints.

### Notable version bumps

| Package | v1 | v2 | Notes |
| --- | --- | --- | --- |
| fastapi | 0.115.6 | 0.139.0 | No code changes needed |
| sqlalchemy | 2.0.36 | 2.0.51 | Patch bump |
| pydantic | 2.10.4 | 2.13.4 | No code changes |
| pydantic-settings | 2.7.0 | 2.14.2 | No code changes |
| redis | 5.2.1 | **5.3.1** | Pinned at 5.x because `arq 0.28.0` requires `redis<6`. To use redis 8.x, swap the queue library. |
| arq | 0.26.3 | 0.28.0 | Patch bump |
| langgraph | 0.2.60 | **1.2.9** | 1.0 release. The reference `InsuranceAgent` never imported langgraph directly (it's a linear state-machine impl), so no code changes needed. |
| langchain | 0.3.13 | **1.3.13** | 1.0 release. Not used in test path (FakeLLM). Deferred imports in agent/llm.py and rag/embedder.py continue to work. |
| langchain-openai | 0.2.14 | **1.3.5** | Same as langchain. |
| openai | 1.58.1 | **2.45.0** | Only used via litellm, deferred import — no code changes needed. |
| tiktoken | 0.8.0 | 0.13.0 | Not imported in test path |
| sentence-transformers | 3.3.1 | 5.6.0 | Deferred import in rag/reranker.py |
| pypdf | 5.1.0 | **6.14.2** | The parser API (`PdfReader`, `page.extract_text()`) is unchanged. |
| structlog | 24.4.0 | **26.1.0** | Config API unchanged for our usage. |
| pytest | 8.3.4 | **9.1.1** | asyncio-mode config unchanged. |
| pytest-asyncio | 0.25.0 | **1.4.0** | The `asyncio_mode = auto` setting in pytest.ini continues to work. |
| pytest-cov | 6.0.0 | 7.1.0 | No config changes |
| faker | 33.1.0 | 40.28.1 | Not currently invoked; installed for future test data generation. |

### Removed packages

- **`passlib`** — unmaintained since 2020 and still pulled in `crypt` (deprecated in Python 3.13). Replaced with direct `argon2-cffi` usage in `app/core/security.py`. Same public API for hashing/verifying, with an added `needs_rehash()` helper for silent credential-strength upgrades on login.
- **`python-jose`** — unmaintained. Replaced with `PyJWT`. Same public API for `create_access_token`, `create_refresh_token`, `decode_token`.
- **`detoxify`** — was in `requirements.txt` but never imported anywhere. Dropped.

### Code changes required

Only one file changed: `backend/app/core/security.py`. It now imports `argon2` and `jwt` (PyJWT) directly instead of going through passlib and jose. The public API (`hash_password`, `verify_password`, `create_access_token`, `create_refresh_token`, `decode_token`) is byte-compatible — nothing else in the codebase or the test suite needed to change.

The Argon2id parameters follow OWASP's 2024 recommendation: `time_cost=3`, `memory_cost=64 MiB`, `parallelism=4`, `hash_len=32`, `salt_len=16`. This is stronger than passlib's defaults.

### Verification

- **Import check**: all 49 backend modules import cleanly on the new stack.
- **Test suite**: 97/97 tests pass in ~73 seconds.
- **Live HTTP smoke test**: 10/10 endpoints return the expected status and payload.
- **Alembic**: migration graph valid (`0001 → 0002`, head at `0002`).
- **Frontend**: React SPA builds cleanly with esbuild (1.1 MB bundle).
