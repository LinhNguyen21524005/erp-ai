"""
intelligence/intent_engine.py — Intent Understanding Engine
Stage 3.8: Parse user question into structured business intent.

Responsibilities:
- Detect query intent type (aggregation, ranking, listing, comparison)
- Extract metrics, dimensions, filters, time ranges
- Determine required tables via synonym KB
- Detect if JOIN is needed
- Build structured QueryIntent object
"""

import logging
import re
from dataclasses import dataclass, field
from typing import Optional

from erp42_analytics.core.models import QueryIntent
from erp42_analytics.intelligence.synonym_kb import get_synonym_kb
from erp42_analytics.intelligence.join_graph import get_join_graph

logger = logging.getLogger(__name__)


# ─── Intent Patterns ──────────────────────────────────────────────────────────

# Aggregation patterns (COUNT, SUM, AVG)
AGGREGATION_PATTERNS = [
    r'(?i)\bwie viele\b',           # wie viele Aufträge
    r'(?i)\banzahl\b',              # Anzahl der Kunden
    r'(?i)\bhow many\b',            # how many orders
    r'(?i)\bcount\b',               # count orders
    r'(?i)\bgesamtumsatz\b',        # Gesamtumsatz
    r'(?i)\btotal\b',               # total revenue
    r'(?i)\bsumme\b',               # Summe der Bestellungen
    r'(?i)\bsumm(?:iere|ary)?\b',   # summarize
    r'(?i)\bdurchschnitt\b',        # Durchschnitt
    r'(?i)\baverage\b',             # average
    r'(?i)\bmittelwert\b',          # Mittelwert
]

# Ranking patterns (TOP N)
RANKING_PATTERNS = [
    r'(?i)\btop\s+\d+\b',           # top 10
    r'(?i)\bdie\s+\d+\s+\w+sten\b', # die 10 umsatzstärksten
    r'(?i)\bbest(?:en?)?\b',        # beste
    r'(?i)\blargest\b',             # largest
    r'(?i)\bhighest\b',             # highest
    r'(?i)\bmost\b',                # most
    r'(?i)\bmeisten\b',             # meisten
    r'(?i)\bfirst\s+\d+\b',        # first 5
]

# Listing patterns (SELECT all/many)
LISTING_PATTERNS = [
    r'(?i)\bzeige\b',               # zeige alle
    r'(?i)\balle\b',                # alle Lieferanten
    r'(?i)\bshow\b',                # show all
    r'(?i)\blist\b',                # list customers
    r'(?i)\bauflist\w*\b',          # auflisten
    r'(?i)\banzeigen\b',            # anzeigen
]

# Time patterns
TIME_PATTERNS = {
    "year": [
        r'(?i)\bim\s+jahr\s+(\d{4})\b',   # im Jahr 2023
        r'(?i)\bin\s+(\d{4})\b',           # in 2023
        r'(?i)\b(\d{4})\b',               # 2023
    ],
    "month": [
        r'(?i)\bim\s+monat\b',             # im Monat
        r'(?i)\bmonthly\b',               # monthly
        r'(?i)\bmonatlich\b',             # monatlich
    ],
    "quarter": [
        r'(?i)\bquartal\b',               # Quartal
        r'(?i)\bquarterly\b',             # quarterly
        r'(?i)\bq[1-4]\b',               # Q1, Q2...
    ],
}

# Join-required patterns (signals multi-table query)
JOIN_SIGNALS = [
    r'(?i)\bmit\s+kunden\w*\b',           # mit Kundeninformationen
    r'(?i)\bwith\s+customer\b',           # with customer
    r'(?i)\bpro\s+kunde\b',              # pro Kunde
    r'(?i)\bje\s+kunde\b',               # je Kunde
    r'(?i)\bpro\s+mitarbeiter\b',        # pro Mitarbeiter
    r'(?i)\bje\s+mitarbeiter\b',         # je Mitarbeiter
    r'(?i)\bpro\s+projekt\b',            # pro Projekt
    r'(?i)\bpro\s+lieferant\b',          # pro Lieferant
    r'(?i)\bjeder\s+kunde\b',            # jeder Kunde
    r'(?i)\beach\s+customer\b',          # each customer
    r'(?i)\bkundeninformation\w*\b',     # Kundeninformationen
    r'(?i)\bwith\s+details\b',           # with details
]

# "All" patterns (no TOP N)
ALL_PATTERNS = [
    r'(?i)\balle\b',        # alle
    r'(?i)\bjeder\b',       # jeder
    r'(?i)\bjede\b',        # jede
    r'(?i)\bjedes\b',       # jedes
    r'(?i)\ball\b',         # all
    r'(?i)\bevery\b',       # every
    r'(?i)\beach\b',        # each
]


class IntentEngine:
    """
    Parses natural language questions into structured QueryIntent.
    Used to help SQL generator make better decisions.
    """

    def __init__(self):
        self.synonym_kb = get_synonym_kb()
        self.join_graph = get_join_graph()

    def parse(self, question: str) -> QueryIntent:
        """
        Parse question into structured QueryIntent.
        Main entry point.
        """
        intent = QueryIntent(raw_question=question)

        # Detect language
        intent.language = self._detect_language(question)

        # Detect intent type
        intent.intent_type = self._detect_intent_type(question)

        # Extract time info
        time_info = self._extract_time(question)
        if time_info:
            intent.time_grain = time_info.get("grain")
            if time_info.get("year"):
                intent.filters["year"] = time_info["year"]

        # Extract limit (for TOP N queries)
        intent.limit = self._extract_limit(question)

        # Extract tables needed via synonym KB
        tables_needed = self.synonym_kb.get_all_tables_mentioned(question)
        intent.dimensions = tables_needed

        # Detect if JOIN is needed
        intent.needs_join = self._needs_join(question, tables_needed)

        # Compute confidence
        intent.confidence = self._compute_confidence(intent, question)

        logger.debug(
            f"Intent parsed: type={intent.intent_type}, "
            f"tables={tables_needed}, "
            f"needs_join={intent.needs_join}, "
            f"limit={intent.limit}"
        )

        return intent

    # ─── Detection Methods ────────────────────────────────────────────────────

    def _detect_language(self, question: str) -> str:
        """Detect primary language of question."""
        german_words = [
            "wie", "viele", "alle", "zeige", "welche", "wer",
            "der", "die", "das", "und", "mit", "von", "nach",
            "aufträge", "kunden", "lieferanten", "jahr", "monat"
        ]
        words = re.findall(r"\b\w+\b", question.lower())
        german_count = sum(1 for w in words if w in german_words)

        if german_count >= 2:
            return "de"
        elif german_count == 1:
            return "mixed"
        return "en"

    def _detect_intent_type(self, question: str) -> str:
        """Detect type of query intent."""
        # Check ranking first (most specific)
        if any(re.search(p, question) for p in RANKING_PATTERNS):
            return "ranking"
        # Then aggregation
        if any(re.search(p, question) for p in AGGREGATION_PATTERNS):
            return "aggregation"
        # Then listing
        if any(re.search(p, question) for p in LISTING_PATTERNS):
            return "listing"
        return "retrieval"

    def _extract_time(self, question: str) -> Optional[dict]:
        """Extract time dimension from question."""
        result = {}

        # Check for specific year
        for pattern in TIME_PATTERNS["year"]:
            match = re.search(pattern, question)
            if match and match.lastindex and match.group(1).isdigit():
                year = int(match.group(1))
                if 2000 <= year <= 2099:
                    result["year"] = year
                    result["grain"] = "year"
                    return result

        # Check for month grain
        if any(re.search(p, question) for p in TIME_PATTERNS["month"]):
            result["grain"] = "month"
            return result

        # Check for quarter grain
        if any(re.search(p, question) for p in TIME_PATTERNS["quarter"]):
            result["grain"] = "quarter"
            return result

        return None

    def _extract_limit(self, question: str) -> Optional[int]:
        """Extract TOP N limit from question."""
        # "top 10", "die 10", "first 5"
        patterns = [
            r'(?i)\btop\s+(\d+)\b',
            r'(?i)\bfirst\s+(\d+)\b',
            r'(?i)\bdie\s+(\d+)\b',
            r'(?i)\bden\s+(\d+)\b',
        ]
        for pattern in patterns:
            match = re.search(pattern, question)
            if match:
                return int(match.group(1))

        # "umsatzstärksten" pattern often implies TOP 10
        if re.search(r'(?i)\b\w+sten\b', question):
            # Check if a number is mentioned nearby
            num_match = re.search(r'\b(\d+)\b', question)
            if num_match:
                return int(num_match.group(1))

        return None

    def _needs_join(self, question: str, tables: list) -> bool:
        """Determine if query requires a JOIN."""
        # Explicit join signals in question (most reliable signal)
        if any(re.search(p, question) for p in JOIN_SIGNALS):
            return True

        # Multiple DIFFERENT core tables needed
        # Exclude time-dimension duplicates (auftrag+rechnung for revenue)
        core_tables = set(tables) - {"rechnung"}  # rechnung often redundant with auftrag
        if len(core_tables) >= 2:
            return True

        return False

    def _compute_confidence(self, intent: QueryIntent, question: str) -> float:
        """Compute confidence in intent parsing."""
        score = 1.0

        # Lower confidence if no tables detected
        if not intent.dimensions:
            score -= 0.3

        # Lower if intent type is just "retrieval" (catch-all)
        if intent.intent_type == "retrieval":
            score -= 0.1

        return round(max(0.0, score), 2)

    # ─── Intent-aware prompt hints ────────────────────────────────────────────

    def get_prompt_hints(self, intent: QueryIntent) -> str:
        """
        Generate additional prompt hints based on parsed intent.
        These are injected into the LLM prompt to improve SQL generation.
        """
        hints = []

        # Time filter hint
        if "year" in intent.filters:
            year = intent.filters["year"]
            hints.append(
                f"HINT: User wants data for year {year}. "
                f"Use WHERE YEAR(Datum) = {year}"
            )

        # Aggregation hint
        if intent.intent_type == "aggregation":
            hints.append(
                "HINT: This is an aggregation query. "
                "Use COUNT/SUM/AVG without TOP N."
            )

        # Ranking hint
        if intent.intent_type == "ranking" and intent.limit:
            hints.append(
                f"HINT: User wants TOP {intent.limit} results. "
                f"Use SELECT TOP {intent.limit} with ORDER BY."
            )

        # Listing hint
        if intent.intent_type == "listing":
            hints.append(
                "HINT: User wants to see all records. "
                "Do NOT add TOP N. Return all rows."
            )

        # JOIN hint — only if tables are truly different entities
        join_tables = [t for t in intent.dimensions if t != "rechnung"]
        if intent.needs_join and len(join_tables) >= 2:
            tables_str = " and ".join(join_tables[:3])
            hints.append(
                f"HINT: This query likely needs a JOIN between "
                f"{tables_str} tables via Kundennumm or Mitarbeite."
            )

        return "\n".join(hints)


# ─── Singleton ────────────────────────────────────────────────────────────────
_intent_engine: Optional[IntentEngine] = None


def get_intent_engine() -> IntentEngine:
    global _intent_engine
    if _intent_engine is None:
        _intent_engine = IntentEngine()
    return _intent_engine
