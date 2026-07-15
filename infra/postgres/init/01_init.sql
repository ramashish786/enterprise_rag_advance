-- Extensions
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Read-only role used by agent tools (defense in depth: even a jailbroken
-- agent that somehow crafts SQL cannot mutate data).
-- The application role (insurance_app) is created by Postgres from env vars.
DO $$
BEGIN
    IF NOT EXISTS (SELECT FROM pg_catalog.pg_roles WHERE rolname = 'insurance_ro') THEN
        CREATE ROLE insurance_ro LOGIN PASSWORD 'change-me-too';
    END IF;
END $$;

-- Grants applied after Alembic creates the schema (see backend/app/db/rls.py).
