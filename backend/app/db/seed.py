"""Idempotent seed data: roles, permissions, departments, an admin user, and
a handful of demo customers/employees/policies for local dev.

Run with: python -m app.db.seed
"""
from __future__ import annotations

import asyncio
from datetime import date, timedelta
from decimal import Decimal

from sqlalchemy import select

from app.core.logging import configure_logging, get_logger
from app.core.security import hash_password
from app.db.models import (
    Department,
    EmployeeProfile,
    InsurancePolicy,
    Permission,
    PolicyStatus,
    PolicyType,
    Role,
    RolePermission,
    User,
    UserRole,
    UserType,
)
from app.db.session import get_session

logger = get_logger(__name__)


# --- Permission catalog ---

PERMISSIONS = [
    # policy
    ("policy", "read_own",       "Read own policies"),
    ("policy", "read_department", "Read policies in own department"),
    ("policy", "read_any",       "Read any policy"),
    # document
    ("document", "read_public",   "Read public documents"),
    ("document", "read_internal", "Read internal documents"),
    ("document", "read_restricted","Read restricted documents"),
    ("document", "upload",        "Upload knowledge base documents"),
    ("document", "delete",        "Delete knowledge base documents"),
    ("document", "approve_deletion", "Confirm a document deletion requested by another admin"),
    # chat
    ("chat", "create_session",    "Create chat sessions"),
    ("chat", "read_own_history",  "Read own chat history"),
    ("chat", "read_any_history",  "Read any chat history"),
    # claims / policy mutations (human-in-the-loop approval flows)
    ("claim", "file",              "File an insurance claim"),
    ("claim", "approve",           "Approve or reject filed claims"),
    ("policy", "request_cancellation", "Request cancellation of an own policy"),
    ("policy", "approve_cancellation", "Approve or reject policy cancellation requests"),
    ("policy", "request_change",   "Request a coverage change on an own policy"),
    ("policy", "approve_change",   "Approve or reject policy change requests"),
    # user management
    ("user", "manage",            "Create/update/delete users"),
    ("role", "assign",            "Assign roles"),
    # audit
    ("audit_log", "read",         "Read the audit log"),
]

ROLE_PERMISSIONS = {
    "customer": [
        ("policy", "read_own"),
        ("document", "read_public"),
        ("chat", "create_session"),
        ("chat", "read_own_history"),
        ("claim", "file"),
        ("policy", "request_cancellation"),
        ("policy", "request_change"),
    ],
    "employee": [
        ("policy", "read_department"),
        ("document", "read_public"),
        ("document", "read_internal"),
        ("chat", "create_session"),
        ("chat", "read_own_history"),
        ("claim", "approve"),
        ("policy", "approve_cancellation"),
        ("policy", "approve_change"),
    ],
    "administrator": [(r, a) for (r, a, _) in PERMISSIONS],
}


async def seed_all() -> None:
    async with get_session() as s:
        # Permissions
        existing = {(p.resource, p.action) for p in (await s.execute(select(Permission))).scalars()}
        for resource, action, desc in PERMISSIONS:
            if (resource, action) not in existing:
                s.add(Permission(resource=resource, action=action, description=desc))
        await s.commit()

        perm_map = {
            (p.resource, p.action): p
            for p in (await s.execute(select(Permission))).scalars()
        }

        # Roles
        existing_roles = {r.name: r for r in (await s.execute(select(Role))).scalars()}
        for role_name in ROLE_PERMISSIONS:
            if role_name not in existing_roles:
                s.add(Role(name=role_name, description=f"{role_name.title()} role"))
        await s.commit()
        existing_roles = {r.name: r for r in (await s.execute(select(Role))).scalars()}

        # Role-Permission bindings
        for role_name, tuples in ROLE_PERMISSIONS.items():
            role = existing_roles[role_name]
            existing_bindings = {
                rp.permission_id
                for rp in (
                    await s.execute(
                        select(RolePermission).where(RolePermission.role_id == role.id)
                    )
                ).scalars()
            }
            for t in tuples:
                perm = perm_map[t]
                if perm.id not in existing_bindings:
                    s.add(RolePermission(role_id=role.id, permission_id=perm.id))
        await s.commit()

        # Departments
        existing_depts = {d.name: d for d in (await s.execute(select(Department))).scalars()}
        for name, ptype in [
            ("Life Insurance", PolicyType.LIFE),
            ("Home Insurance", PolicyType.HOME),
            ("Vehicle Insurance", PolicyType.VEHICLE),
        ]:
            if name not in existing_depts:
                s.add(Department(name=name, policy_type=ptype))
        await s.commit()
        existing_depts = {d.name: d for d in (await s.execute(select(Department))).scalars()}

        # Default administrator
        admin_email = "admin@example.com"
        admin = (
            await s.execute(select(User).where(User.email == admin_email))
        ).scalar_one_or_none()
        if not admin:
            admin = User(
                email=admin_email,
                hashed_password=hash_password("admin-change-me"),
                user_type=UserType.ADMINISTRATOR,
                full_name="System Administrator",
            )
            s.add(admin)
            await s.flush()
            s.add(UserRole(user_id=admin.id, role_id=existing_roles["administrator"].id))
            logger.warning("seeded default administrator; change the password immediately",
                           email=admin_email)

        # Demo customers
        demo_customers = [
            ("alice@example.com",   "Alice Anderson",   "customer-pass"),
            ("bob@example.com",     "Bob Brown",        "customer-pass"),
            ("carol@example.com",   "Carol Chen",       "customer-pass"),
        ]
        for email, name, pw in demo_customers:
            u = (await s.execute(select(User).where(User.email == email))).scalar_one_or_none()
            if not u:
                u = User(
                    email=email,
                    hashed_password=hash_password(pw),
                    user_type=UserType.CUSTOMER,
                    full_name=name,
                )
                s.add(u)
                await s.flush()
                s.add(UserRole(user_id=u.id, role_id=existing_roles["customer"].id))

        # Demo employees, one per department
        demo_employees = [
            ("life.employee@example.com",    "Ella Life",    "Life Insurance",    "EMP-L001"),
            ("home.employee@example.com",    "Henry Home",   "Home Insurance",    "EMP-H001"),
            ("vehicle.employee@example.com", "Vera Vehicle", "Vehicle Insurance", "EMP-V001"),
        ]
        for email, name, dept_name, empno in demo_employees:
            u = (await s.execute(select(User).where(User.email == email))).scalar_one_or_none()
            if not u:
                u = User(
                    email=email,
                    hashed_password=hash_password("employee-pass"),
                    user_type=UserType.EMPLOYEE,
                    full_name=name,
                )
                s.add(u)
                await s.flush()
                s.add(UserRole(user_id=u.id, role_id=existing_roles["employee"].id))
                s.add(EmployeeProfile(
                    user_id=u.id,
                    department_id=existing_depts[dept_name].id,
                    employee_number=empno,
                ))

        await s.commit()

        # Demo policies for alice
        alice = (await s.execute(select(User).where(User.email == "alice@example.com"))).scalar_one()
        bob = (await s.execute(select(User).where(User.email == "bob@example.com"))).scalar_one()
        today = date.today()
        demo_policies = [
            ("LIFE-100001",    alice.id, PolicyType.LIFE,    Decimal("500000"), Decimal("60.00")),
            ("HOME-200001",    alice.id, PolicyType.HOME,    Decimal("300000"), Decimal("110.00")),
            ("VEHICLE-300001", bob.id,   PolicyType.VEHICLE, Decimal("40000"),  Decimal("80.00")),
        ]
        for pnum, cust_id, ptype, coverage, premium in demo_policies:
            existing = (
                await s.execute(select(InsurancePolicy).where(InsurancePolicy.policy_number == pnum))
            ).scalar_one_or_none()
            if not existing:
                s.add(InsurancePolicy(
                    policy_number=pnum,
                    customer_id=cust_id,
                    policy_type=ptype,
                    coverage_amount=coverage,
                    premium=premium,
                    start_date=today - timedelta(days=180),
                    end_date=today + timedelta(days=185),
                    status=PolicyStatus.ACTIVE,
                    details={},
                ))
        await s.commit()

    logger.info("seed complete")


def main() -> None:
    configure_logging()
    asyncio.run(seed_all())


if __name__ == "__main__":
    main()
