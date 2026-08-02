"""
intelligence/security.py — Security Layer
Stage 7.2: RBAC Engine (Role-Based Access Control)
Stage 7.4: Audit Logging System

Responsibilities:
- Define user roles and permissions
- Filter restricted tables/columns per user
- Log all queries for enterprise compliance
- Detect suspicious usage patterns

Role/permission model rewritten to match ibes AG's real department
structure, from the official table classification spreadsheet (Tier 1/2,
owning department, staff/manager access, sensitivity). Previously this
used an invented role system (admin/analyst/finance/sales/viewer) with
no connection to ibes's actual org chart or access rules — see
ROLE_ALIASES below for how old role strings map onto the real ones so
existing call sites (batch_test_runner --role, query_service.py) keep
working unchanged.
"""

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional

from erp42_analytics.core.models import AuditEntry, QueryStatus
from erp42_analytics.repositories.history_repo import AuditRepository

logger = logging.getLogger(__name__)


# ─── Role Definitions (ibes AG real departments) ───────────────────────────
# Source: ibes table classification spreadsheet + Theresa's clarification.
# 6 real roles, forming a hierarchy — NOT a flat set of independent roles:
#   Geschäftsleitung  — top level, full read access to everything
#   Vertrieb, Vertrieb+Einkauf, Buchhaltung/Sekretariat, Lager/Fertigung
#                     — second-level department roles, each with more
#                       rights than Mitarbeiter
#   Mitarbeiter       — base level; every other role has AT LEAST these
#                       rights on top of their own. Concretely: wherever
#                       "Mitarbeiter" appears in a table's staff access
#                       column, EVERY role can read that table — not
#                       just whichever specific department is also
#                       listed alongside it on that row. See
#                       _build_permissions() for the inheritance logic.

class UserRole(Enum):
    VERTRIEB = "vertrieb"                              # Sales
    VERTRIEB_EINKAUF = "vertrieb_einkauf"               # Sales+Purchasing
    BUCHHALTUNG_SEKRETARIAT = "buchhaltung_sekretariat" # Accounting/Secretariat
    LAGER_FERTIGUNG = "lager_fertigung"                 # Warehouse/Manufacturing
    GESCHAEFTSLEITUNG = "geschaeftsleitung"             # Management — full access
    MITARBEITER = "mitarbeiter"                         # Base level — inherited by all other roles



# Backward-compat: old placeholder role strings still work, mapped onto
# the nearest real department. "viewer" and "analyst" both land on
# MITARBEITER (the broadest general-access tier) since there's no
# narrower "read-only" department in ibes's actual structure — flag this
# to Theresa/Paul if a stricter read-only tier turns out to be needed;
# for now this errs toward "at least as permissive as before", not more
# restrictive, since UAT/testing (--role analyst default) shouldn't
# suddenly lose access to tables it could see before.
ROLE_ALIASES = {
    "admin": UserRole.GESCHAEFTSLEITUNG,
    "analyst": UserRole.MITARBEITER,
    "finance": UserRole.BUCHHALTUNG_SEKRETARIAT,
    "sales": UserRole.VERTRIEB,
    "viewer": UserRole.MITARBEITER,
}


def _resolve_role(role: str) -> UserRole:
    """Normalize a role string (real department name or legacy alias) to UserRole."""
    role_lower = role.lower().strip()
    try:
        return UserRole(role_lower)
    except ValueError:
        pass
    if role_lower in ROLE_ALIASES:
        return ROLE_ALIASES[role_lower]
    logger.warning(f"Unknown role '{role}', defaulting to mitarbeiter")
    return UserRole.MITARBEITER


# ─── Table Access Classification (from ibes table classification sheet) ────
# One entry per table ibes has officially classified. Fields:
#   tier: 1 (primary business table) or 2 (supporting/reference table)
#   owning_dept: department responsible for the table's data
#   staff_departments: UserRole values whose staff-level access includes
#       this table (this is the actual access grant used for filtering)
#   sensitive: True if flagged Sensitive=YES in the sheet
#   masked_columns: columns to mask in results, even for roles that can
#       otherwise see the table (currently only benutzer/Passwort — the
#       sheet's Tier 1 rows had no Sensitive=YES entries)
#
# Manager access is "Geschäftsleitung" on every single row in the sheet
# with no exceptions, and Executive access was blank throughout — so
# GESCHAEFTSLEITUNG is modeled as full access below rather than encoded
# per-row here. Tables NOT in this dict (anything outside ibes's official
# Tier 1/2 classification sheet) are blocked by default
# for every role except GESCHAEFTSLEITUNG — see get_permissions().
TableAccess = dict  # {tier, owning_dept, staff_departments, sensitive, masked_columns}

_V = UserRole.VERTRIEB
_VE = UserRole.VERTRIEB_EINKAUF
_B = UserRole.BUCHHALTUNG_SEKRETARIAT
_L = UserRole.LAGER_FERTIGUNG
_M = UserRole.MITARBEITER

TABLE_ACCESS: dict = {
    # ── Tier 1 ──────────────────────────────────────────────────────────
    "abposten": {"tier": 1, "owning_dept": "Buchhaltung/Sekretariat", "staff_departments": [_B]},
    "abschlag": {"tier": 1, "owning_dept": "Buchhaltung/Sekretariat", "staff_departments": [_B]},
    "afposten": {"tier": 1, "owning_dept": "Vertrieb", "staff_departments": [_V, _VE]},
    "anfrage":  {"tier": 1, "owning_dept": "Vertrieb", "staff_departments": [_V, _VE]},
    "angebot":  {"tier": 1, "owning_dept": "Vertrieb", "staff_departments": [_V, _VE, _M]},
    "anposten": {"tier": 1, "owning_dept": "Vertrieb", "staff_departments": [_V, _VE, _M]},
    "auftrag":  {"tier": 1, "owning_dept": "Vertrieb", "staff_departments": [_V, _VE, _M]},
    "auposten": {"tier": 1, "owning_dept": "Vertrieb", "staff_departments": [_V, _VE, _M]},
    "beposten": {"tier": 1, "owning_dept": "Vertrieb+Einkauf", "staff_departments": [_VE, _M]},
    "bestell":  {"tier": 1, "owning_dept": "Vertrieb+Einkauf", "staff_departments": [_VE, _M]},
    "bestells": {"tier": 1, "owning_dept": "Vertrieb+Einkauf", "staff_departments": [_VE, _M]},
    "bsposten": {"tier": 1, "owning_dept": "Vertrieb+Einkauf", "staff_departments": [_VE, _M]},
    "feposten": {"tier": 1, "owning_dept": "Lager/Fertigung", "staff_departments": [_L, _M]},
    "fertigun": {"tier": 1, "owning_dept": "Lager/Fertigung", "staff_departments": [_L, _M]},
    "fmposten": {"tier": 1, "owning_dept": "Lager/Fertigung", "staff_departments": [_L, _M]},
    "guposten": {"tier": 1, "owning_dept": "Buchhaltung/Sekretariat", "staff_departments": [_B, _M]},
    "gutschri": {"tier": 1, "owning_dept": "Buchhaltung/Sekretariat", "staff_departments": [_B, _M]},
    "kunde":    {"tier": 1, "owning_dept": "Vertrieb", "staff_departments": [_V, _VE, _M]},
    "lager":    {"tier": 1, "owning_dept": "Vertrieb", "staff_departments": [_V, _VE, _M]},
    "lagernum": {"tier": 1, "owning_dept": "Lager/Fertigung", "staff_departments": [_L, _M]},
    "lansprec": {"tier": 1, "owning_dept": "Vertrieb+Einkauf", "staff_departments": [_VE, _M]},
    "liefer":   {"tier": 1, "owning_dept": "Vertrieb+Einkauf", "staff_departments": [_V, _VE, _M]},
    "lieferan": {"tier": 1, "owning_dept": "Vertrieb+Einkauf", "staff_departments": [_VE, _M]},
    "lieferun": {"tier": 1, "owning_dept": "Vertrieb+Einkauf", "staff_departments": [_VE, _M]},
    "liposten": {"tier": 1, "owning_dept": "Vertrieb+Einkauf", "staff_departments": [_VE, _M]},
    "lsposten": {"tier": 1, "owning_dept": "Vertrieb+Einkauf", "staff_departments": [_V, _VE, _M]},
    "meldung":  {"tier": 1, "owning_dept": "Lager/Fertigung", "staff_departments": [_L, _M]},
    "produkt":  {"tier": 1, "owning_dept": "Vertrieb", "staff_departments": [_V, _VE]},
    "proforma": {"tier": 1, "owning_dept": "Vertrieb", "staff_departments": [_V, _VE]},
    "projekt":  {"tier": 1, "owning_dept": "Vertrieb", "staff_departments": [_V, _VE, _M]},
    "prposten": {"tier": 1, "owning_dept": "Vertrieb", "staff_departments": [_V, _VE]},
    "rechnung": {"tier": 1, "owning_dept": "Buchhaltung/Sekretariat", "staff_departments": [_B, _M]},
    "reparatu": {"tier": 1, "owning_dept": "Lager/Fertigung", "staff_departments": [_L, _M]},
    "reposten": {"tier": 1, "owning_dept": "Buchhaltung/Sekretariat", "staff_departments": [_B, _M]},
    "rlposten": {"tier": 1, "owning_dept": "Lager/Fertigung", "staff_departments": [_L, _M]},
    "rma":      {"tier": 1, "owning_dept": "Lager/Fertigung", "staff_departments": [_L]},
    "rmposten": {"tier": 1, "owning_dept": "Lager/Fertigung", "staff_departments": [_L]},
    "rpposten": {"tier": 1, "owning_dept": "Lager/Fertigung", "staff_departments": [_L, _M]},
    "ruecklie": {"tier": 1, "owning_dept": "Lager/Fertigung", "staff_departments": [_L, _M]},
    "ruecknah": {"tier": 1, "owning_dept": "Buchhaltung/Sekretariat", "staff_departments": [_B, _M]},
    "ruposten": {"tier": 1, "owning_dept": "Buchhaltung/Sekretariat", "staff_departments": [_B, _M]},
    "storno":   {"tier": 1, "owning_dept": "Buchhaltung/Sekretariat", "staff_departments": [_B, _M]},
    "stposten": {"tier": 1, "owning_dept": "Buchhaltung/Sekretariat", "staff_departments": [_B, _M]},
    "stueckli": {"tier": 1, "owning_dept": "Lager/Fertigung", "staff_departments": [_L]},
    "untergru": {"tier": 1, "owning_dept": "Buchhaltung/Sekretariat", "staff_departments": [_B]},
    "waposten": {"tier": 1, "owning_dept": "Buchhaltung/Sekretariat", "staff_departments": [_B, _M]},
    "waren":    {"tier": 1, "owning_dept": "Vertrieb+Einkauf", "staff_departments": [_VE, _M]},
    "wartung":  {"tier": 1, "owning_dept": "Buchhaltung/Sekretariat", "staff_departments": [_B, _M]},
    "weposten": {"tier": 1, "owning_dept": "Vertrieb+Einkauf", "staff_departments": [_VE, _M]},
    "xprojektzeiten": {"tier": 1, "owning_dept": "Buchhaltung/Sekretariat", "staff_departments": [_B]},

    # ── Tier 2 ──────────────────────────────────────────────────────────
    "anschrif": {"tier": 2, "owning_dept": "Vertrieb", "staff_departments": [_V, _VE]},
    "ansprech": {"tier": 2, "owning_dept": "Vertrieb", "staff_departments": [_V, _VE, _M]},
    "belegart": {"tier": 2, "owning_dept": "Buchhaltung/Sekretariat", "staff_departments": [_B]},
    "benutzer": {"tier": 2, "owning_dept": "Geschäftsleitung", "staff_departments": [],
                 "sensitive": True, "masked_columns": ["Passwort"]},
    "bestatus": {"tier": 2, "owning_dept": "Buchhaltung/Sekretariat", "staff_departments": [_B, _M]},
    "chefinfo": {"tier": 2, "owning_dept": "Buchhaltung/Sekretariat", "staff_departments": [_B]},
    "debitor":  {"tier": 2, "owning_dept": "Buchhaltung/Sekretariat", "staff_departments": [_B, _M]},
    "fakturat": {"tier": 2, "owning_dept": "Buchhaltung/Sekretariat", "staff_departments": [_B, _M]},
    "gruppe":   {"tier": 2, "owning_dept": "Buchhaltung/Sekretariat", "staff_departments": [_B, _M]},
    "kdsgvo":   {"tier": 2, "owning_dept": "Vertrieb", "staff_departments": [_V, _VE]},
    "kostenst": {"tier": 2, "owning_dept": "Buchhaltung/Sekretariat", "staff_departments": [_B, _M]},
    "kreditor": {"tier": 2, "owning_dept": "Buchhaltung/Sekretariat", "staff_departments": [_B]},
    "kundeart": {"tier": 2, "owning_dept": "Vertrieb", "staff_departments": [_V, _VE]},
    "kundenbr": {"tier": 2, "owning_dept": "Vertrieb", "staff_departments": [_V, _VE, _M]},
    "kundenei": {"tier": 2, "owning_dept": "Vertrieb", "staff_departments": [_V, _VE]},
    "kundesta": {"tier": 2, "owning_dept": "Buchhaltung/Sekretariat", "staff_departments": [_B]},
    "kundeums": {"tier": 2, "owning_dept": "Buchhaltung/Sekretariat", "staff_departments": [_B]},
    "lagerbe":  {"tier": 2, "owning_dept": "Vertrieb", "staff_departments": [_V, _VE, _M]},
    "lagerum":  {"tier": 2, "owning_dept": "Vertrieb", "staff_departments": [_V, _VE, _M]},
    "lagerump": {"tier": 2, "owning_dept": "Vertrieb", "staff_departments": [_V, _VE, _M]},
    "lagerzu":  {"tier": 2, "owning_dept": "Vertrieb", "staff_departments": [_V, _VE]},
    "lanschri": {"tier": 2, "owning_dept": "Vertrieb+Einkauf", "staff_departments": [_VE, _M]},
    "lbedarf":  {"tier": 2, "owning_dept": "Vertrieb+Einkauf", "staff_departments": [_VE]},
    "liefsta":  {"tier": 2, "owning_dept": "Vertrieb+Einkauf", "staff_departments": [_VE]},
    "liefums":  {"tier": 2, "owning_dept": "Vertrieb+Einkauf", "staff_departments": [_VE]},
    "mehrlage": {"tier": 2, "owning_dept": "Vertrieb+Einkauf", "staff_departments": [_VE]},
    "mitarbei": {"tier": 2, "owning_dept": "Geschäftsleitung", "staff_departments": []},
    "postbuch": {"tier": 2, "owning_dept": "Buchhaltung/Sekretariat", "staff_departments": [_B]},
    "preislis": {"tier": 2, "owning_dept": "Vertrieb", "staff_departments": [_V, _VE]},
    "prodbere": {"tier": 2, "owning_dept": "Vertrieb", "staff_departments": [_V, _VE]},
    "textbaus": {"tier": 2, "owning_dept": "Buchhaltung/Sekretariat", "staff_departments": [_V, _VE, _M]},
    "typecode": {"tier": 2, "owning_dept": "Buchhaltung/Sekretariat", "staff_departments": [_B]},
    "xkunde_produkteigenschaft": {"tier": 2, "owning_dept": "Vertrieb", "staff_departments": [_V, _VE]},
    "xprodukteigenschaft": {"tier": 2, "owning_dept": "Vertrieb", "staff_departments": [_V, _VE]},
    "xproduktinfo": {"tier": 2, "owning_dept": "Vertrieb", "staff_departments": [_V, _VE]},
    "xproduktkunde": {"tier": 2, "owning_dept": "Vertrieb", "staff_departments": [_V, _VE]},
    "xstatus":  {"tier": 2, "owning_dept": "Vertrieb", "staff_departments": [_V, _VE, _M]},
    "zahlziel": {"tier": 2, "owning_dept": "Buchhaltung/Sekretariat", "staff_departments": [_B, _M]},
}

# Tables ibes has explicitly flagged Sensitive=YES — checked independently
# from TABLE_ACCESS so this can't silently drop out of sync if the dict
# above is edited. Kept in sync with semantic_service.SENSITIVE_TABLES
# and sql_service.SQLGuardrailService.SENSITIVE_TABLES (three independent
# checks, on purpose — see comments in those files).
SENSITIVE_TABLES = {
    name: info["masked_columns"]
    for name, info in TABLE_ACCESS.items()
    if info.get("sensitive")
}


@dataclass
class RolePermissions:
    """Computed permissions for a resolved UserRole."""
    role: UserRole
    allowed_tables: list = field(default_factory=list)   # empty = all (Geschäftsleitung only)
    blocked_tables: list = field(default_factory=list)
    masked_columns_by_table: dict = field(default_factory=dict)
    max_rows: int = 500
    can_export: bool = True
    can_add_golden: bool = False
    can_rescan: bool = False


def _build_permissions(role: UserRole) -> RolePermissions:
    """
    Derive permissions for a real department role from TABLE_ACCESS,
    rather than hand-maintaining a separate table list per role — this
    keeps the two from silently drifting out of sync as the ibes
    classification sheet grows (currently 87 tables: 49 Tier 1 + 38
    Tier 2; a handful of other known tables — fibubuch, liefbew,
    wiedervo, aufgabe — are Tier 3 and, per Theresa's confirmation,
    restricted to Geschäftsleitung, which is what happens by default
    below since they're absent from TABLE_ACCESS entirely).

    Hierarchy (confirmed by Theresa — ibes's real ERP role model):
      Geschäftsleitung  — top level, full read access to everything
      Vertrieb, Vertrieb+Einkauf, Buchhaltung/Sekretariat, Lager/Fertigung
                        — second-level department roles
      Mitarbeiter       — base level; "any other role has more rights
                          than this". Critically: wherever "Mitarbeiter"
                          appears in a table's staff access, EVERY role
                          has access to it — not just whichever specific
                          department also happens to be listed alongside
                          it. This is a hierarchy/inheritance rule, not a
                          flat set-membership check.
    """
    if role == UserRole.GESCHAEFTSLEITUNG:
        # Full access — matches "Manager access = Geschäftsleitung" on
        # every row of the sheet with no exceptions. Still respects
        # per-table masked_columns (sensitivity isn't a rank thing).
        return RolePermissions(
            role=role,
            allowed_tables=[],  # empty = unrestricted
            blocked_tables=[],
            masked_columns_by_table={
                name: info["masked_columns"]
                for name, info in TABLE_ACCESS.items()
                if info.get("sensitive")
            },
            max_rows=1000,
            can_export=True,
            can_add_golden=True,
            can_rescan=True,
        )

    if role == UserRole.MITARBEITER:
        # Baseline role itself: only the tables explicitly tagged
        # Mitarbeiter — no department-specific tables beyond that.
        allowed = [
            name for name, info in TABLE_ACCESS.items()
            if UserRole.MITARBEITER in info["staff_departments"]
        ]
    else:
        # Department role: own department's tables, UNION every table
        # tagged Mitarbeiter (baseline inheritance) — regardless of
        # which other specific departments are also listed on that row.
        allowed = [
            name for name, info in TABLE_ACCESS.items()
            if role in info["staff_departments"]
            or UserRole.MITARBEITER in info["staff_departments"]
        ]

    masked = {
        name: info["masked_columns"]
        for name, info in TABLE_ACCESS.items()
        if info.get("sensitive") and name in allowed
    }
    return RolePermissions(
        role=role,
        allowed_tables=allowed,
        blocked_tables=[],  # enforced via whitelist (allowed_tables) instead
        masked_columns_by_table=masked,
        max_rows=500,
        can_export=True,
        can_add_golden=(role == UserRole.MITARBEITER),
        can_rescan=False,
    )


ROLE_PERMISSIONS = {role: _build_permissions(role) for role in UserRole}


class RBACEngine:
    """
    Role-Based Access Control Engine.
    Stage 7.2 — enforces data access policies.
    """

    def __init__(self, enabled: bool = True):
        self.enabled = enabled

    def get_permissions(self, role: str) -> RolePermissions:
        """Get permissions for a role string (real department or legacy alias)."""
        return ROLE_PERMISSIONS[_resolve_role(role)]

    def filter_tables(
        self, tables: dict, role: str = "mitarbeiter"
    ) -> dict:
        """
        Filter all_tables dict based on role permissions.
        Unclassified tables (not in TABLE_ACCESS at all — ibes hasn't
        reviewed them yet) are excluded for every role except
        Geschäftsleitung, since we have no basis to say they're safe.
        """
        if not self.enabled:
            return tables

        perms = self.get_permissions(role)

        if not perms.allowed_tables and not perms.blocked_tables:
            return tables  # Geschäftsleitung — unrestricted

        filtered = {}
        for table_name, info in tables.items():
            if table_name in perms.blocked_tables:
                continue
            if table_name not in perms.allowed_tables:
                continue
            filtered[table_name] = info

        return filtered

    def check_sql_permissions(
        self, sql: str, role: str = "mitarbeiter"
    ) -> tuple:
        """
        Check if SQL respects role permissions.
        Returns (allowed: bool, reason: str)
        """
        if not self.enabled:
            return True, ""

        perms = self.get_permissions(role)

        # Geschäftsleitung: unrestricted table access, but sensitive
        # tables are still blocked outright at the SQL level (not just
        # masked) — this mirrors the hard-block in SQLGuardrailService
        # and semantic_service.py. Masking sensitive data only in the
        # RESULT is not sufficient; the query itself shouldn't reference
        # it, matching the classification sheet's intent for benutzer.
        for sensitive_table in SENSITIVE_TABLES:
            if re.search(rf'\b{sensitive_table}\b', sql, re.IGNORECASE):
                return False, (
                    f"Access denied: table '{sensitive_table}' is "
                    f"sensitive (masked columns: "
                    f"{SENSITIVE_TABLES[sensitive_table]}) — not "
                    f"queryable directly regardless of role"
                )

        if not perms.allowed_tables and not perms.blocked_tables:
            return True, ""  # Geschäftsleitung, non-sensitive table

        # Whitelist enforcement: parse the actual tables referenced in
        # the SQL (rather than regex-scanning only the tables we happen
        # to have classified) so unclassified tables are caught too —
        # ibes has only classified 87 tables in Tier 1/2 so far — the
        # and an unreviewed table is not automatically safe just because
        # it isn't in TABLE_ACCESS.
        try:
            from sql_metadata import Parser
            referenced_tables = {
                t.split(".")[-1].lower().strip("[]")
                for t in Parser(sql).tables
            }
        except Exception as e:
            logger.debug(f"SQL table parse failed in RBAC check: {e}")
            referenced_tables = set()

        for table_name in referenced_tables:
            if table_name not in perms.allowed_tables:
                return False, (
                    f"Access denied: table '{table_name}' not allowed "
                    f"for role '{role}'"
                    + (" (unclassified — not in ibes table access sheet)"
                       if table_name not in TABLE_ACCESS else "")
                )

        return True, ""

    def mask_result(
        self, columns: list, rows: list, role: str = "mitarbeiter",
        table_name: Optional[str] = None,
    ) -> tuple:
        """
        Mask sensitive columns in result based on role and (if known)
        which table they came from. table_name is optional for backward
        compatibility with existing call sites that don't pass it —
        without it, this falls back to masking any column name matching
        ANY sensitive column across ALL tables the role can see, which
        is more conservative (may over-mask) but never under-masks.
        """
        if not self.enabled:
            return columns, rows

        perms = self.get_permissions(role)
        if not perms.masked_columns_by_table:
            return columns, rows

        if table_name and table_name in perms.masked_columns_by_table:
            masked_cols = perms.masked_columns_by_table[table_name]
        else:
            masked_cols = [
                c for cols in perms.masked_columns_by_table.values()
                for c in cols
            ]

        if not masked_cols:
            return columns, rows

        mask_indices = [
            i for i, col in enumerate(columns)
            if any(m.lower() in col.lower() for m in masked_cols)
        ]

        if not mask_indices:
            return columns, rows

        masked_rows = []
        for row in rows:
            row_list = list(row)
            for idx in mask_indices:
                if idx < len(row_list) and row_list[idx] is not None:
                    row_list[idx] = "***"
            masked_rows.append(row_list)

        return columns, masked_rows

    def get_max_rows(self, role: str = "mitarbeiter") -> int:
        return self.get_permissions(role).max_rows

    def can_export(self, role: str = "mitarbeiter") -> bool:
        return self.get_permissions(role).can_export

    def can_add_golden(self, role: str = "mitarbeiter") -> bool:
        return self.get_permissions(role).can_add_golden

    def can_rescan(self, role: str = "mitarbeiter") -> bool:
        return self.get_permissions(role).can_rescan


class AuditLogger:
    """
    Enterprise Audit Logging System.
    Stage 7.4 — logs all queries for compliance.
    """

    def __init__(self):
        self.repo = AuditRepository()

    def log_query(
        self,
        user_id: str,
        session_id: str,
        question: str,
        sql: str,
        tables: list,
        status: QueryStatus,
        trust_score: float,
        exec_time: float,
        row_count: int,
        error: Optional[str] = None,
    ):
        """Log a query to audit trail."""
        entry = AuditEntry(
            timestamp=datetime.now(),
            user_id=user_id,
            session_id=session_id,
            question=question,
            sql_generated=sql,
            tables_accessed=tables,
            status=status,
            trust_score=trust_score,
            exec_time=exec_time,
            row_count=row_count,
            error=error,
        )
        self.repo.log(entry)

    def get_recent_for_user(
        self, user_id: str, limit: int = 20
    ) -> list:
        """Get recent audit entries for a user."""
        return self.repo.get_recent(limit=limit, user_id=user_id)

    def get_all_recent(self, limit: int = 50) -> list:
        """Get all recent audit entries (admin only)."""
        return self.repo.get_recent(limit=limit)

    def detect_suspicious_activity(
        self, user_id: str
    ) -> list:
        """
        Stage 7.7 — Error Tracking & Incident Management: detect suspicious usage patterns.
        Returns list of warnings.
        """
        warnings = []
        recent = self.get_recent_for_user(user_id, limit=20)

        if not recent:
            return warnings

        # Check for high failure rate
        failures = sum(1 for r in recent if r.get("status") == "failed")
        if failures > 5:
            warnings.append(
                f"High failure rate detected for user '{user_id}': "
                f"{failures} failed queries"
            )

        # Check for rapid queries (potential abuse)
        if len(recent) >= 10:
            warnings.append(
                f"High query volume for user '{user_id}': "
                f"{len(recent)} recent queries"
            )

        return warnings


# ─── Singletons ───────────────────────────────────────────────────────────────
_rbac: Optional[RBACEngine] = None
_audit: Optional[AuditLogger] = None


def get_rbac() -> RBACEngine:
    global _rbac
    if _rbac is None:
        from erp42_analytics.core.config import config
        _rbac = RBACEngine(enabled=config.security.rbac_enabled)
    return _rbac


def get_audit_logger() -> AuditLogger:
    global _audit
    if _audit is None:
        _audit = AuditLogger()
    return _audit
