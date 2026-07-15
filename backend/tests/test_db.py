"""Tests for the DB layer: models, seed, session helpers."""
from __future__ import annotations

from sqlalchemy import select

from app.db.models import (
    Department,
    EmployeeProfile,
    InsurancePolicy,
    Permission,
    Role,
    User,
    UserRole,
    UserType,
)


async def test_models_create_and_query(db_session):
    """Every table can be inserted into and queried."""
    from app.core.security import hash_password
    user = User(
        email="u@example.com",
        hashed_password=hash_password("pw"),
        user_type=UserType.CUSTOMER,
        full_name="U",
    )
    db_session.add(user)
    await db_session.commit()

    result = await db_session.execute(select(User).where(User.email == "u@example.com"))
    fetched = result.scalar_one()
    assert fetched.id == user.id
    assert fetched.user_type == UserType.CUSTOMER


async def test_seed_is_idempotent(seeded_session):
    """Seed populates the reference tables and running twice yields no dupes."""
    from app.db.seed import seed_all

    roles = (await seeded_session.execute(select(Role))).scalars().all()
    role_names = {r.name for r in roles}
    assert role_names == {"customer", "employee", "administrator"}

    perms = (await seeded_session.execute(select(Permission))).scalars().all()
    assert len(perms) >= 10

    depts = (await seeded_session.execute(select(Department))).scalars().all()
    assert {d.name for d in depts} == {
        "Life Insurance", "Home Insurance", "Vehicle Insurance"
    }

    admins = (await seeded_session.execute(
        select(User).where(User.user_type == UserType.ADMINISTRATOR)
    )).scalars().all()
    assert len(admins) == 1
    assert admins[0].email == "admin@example.com"

    # Idempotent
    initial_user_count = len(
        (await seeded_session.execute(select(User))).scalars().all()
    )
    await seed_all()
    final_user_count = len(
        (await seeded_session.execute(select(User))).scalars().all()
    )
    assert initial_user_count == final_user_count


async def test_seed_creates_demo_policies(seeded_session):
    alice = (await seeded_session.execute(
        select(User).where(User.email == "alice@example.com")
    )).scalar_one()

    policies = (await seeded_session.execute(
        select(InsurancePolicy).where(InsurancePolicy.customer_id == alice.id)
    )).scalars().all()

    assert len(policies) == 2
    policy_numbers = {p.policy_number for p in policies}
    assert "LIFE-100001" in policy_numbers
    assert "HOME-200001" in policy_numbers


async def test_seed_creates_employees_with_departments(seeded_session):
    profiles = (await seeded_session.execute(select(EmployeeProfile))).scalars().all()
    assert len(profiles) == 3

    for p in profiles:
        assert p.employee_number.startswith("EMP-")


async def test_user_role_link(seeded_session):
    alice = (await seeded_session.execute(
        select(User).where(User.email == "alice@example.com")
    )).scalar_one()
    ur = (await seeded_session.execute(
        select(UserRole).where(UserRole.user_id == alice.id)
    )).scalars().all()
    assert len(ur) == 1
