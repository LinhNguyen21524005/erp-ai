"""
intelligence/prompt_guard.py — Prompt Guard & Input Normalization Engine
Stage 3.2: Validate and normalize user input before it enters the AI pipeline.

Responsibilities:
- Block malicious/unsafe prompts (prompt injection, SQL injection attempts)
- Detect invalid or empty requests
- Normalize language quirks (extra whitespace, encoding issues)
- Expand common abbreviations before semantic processing

This runs BEFORE intent parsing and semantic search — it's the first
gate a question passes through.
"""

import logging
import re
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class GuardResult:
    """Result of prompt guard validation."""
    allowed: bool = True
    reason: Optional[str] = None
    normalized_question: str = ""
    warnings: list = field(default_factory=list)


class PromptGuard:
    """
    Stage 3.2 — Prompt Guard & Input Normalization Engine.
    First line of defense before semantic/SQL pipeline.
    """

    # Patterns that indicate prompt injection attempts
    INJECTION_PATTERNS = [
        r'(?i)ignore\s+(all\s+)?(previous|above|prior)\s+instructions',
        r'(?i)disregard\s+(all\s+)?(previous|above|prior)',
        r'(?i)you\s+are\s+now\s+(a|an)\s+\w+',
        r'(?i)system\s*:\s*',
        r'(?i)forget\s+(everything|all)\s+(you|i)',
        r'(?i)reveal\s+(your|the)\s+(system\s+)?prompt',
        r'(?i)what\s+is\s+your\s+(system\s+)?prompt',
        r'(?i)act\s+as\s+(if|though)\s+you',
    ]

    # Patterns indicating direct SQL injection attempts in the question itself
    SQL_INJECTION_PATTERNS = [
        r'(?i)\bDROP\s+TABLE\b',
        r'(?i)\bDELETE\s+FROM\b',
        r'(?i)\bUPDATE\s+\w+\s+SET\b',
        r'(?i)\bINSERT\s+INTO\b',
        r'(?i)\bEXEC(UTE)?\s*\(',
        r'(?i)\bxp_cmdshell\b',
        r"(?i)'\s*OR\s*'?1'?\s*=\s*'?1",
        r'(?i)--\s*$',
        r'(?i);\s*DROP\b',
    ]

    # Abbreviation expansion (business shorthand)
    ABBREVIATIONS = {
        "rev": "revenue",
        "qty": "quantity",
        "cust": "customer",
        "mth": "month",
        "yr": "year",
        "amt": "amount",
        "avg": "average",
        "num": "number",
        "info": "information",
    }

    MAX_QUESTION_LENGTH = 500
    MIN_QUESTION_LENGTH = 2

    def validate(self, question: str) -> GuardResult:
        """
        Main entry point — validate and normalize a user question.
        Returns GuardResult with allowed=False if blocked.
        """
        result = GuardResult(normalized_question=question)

        # 1. Empty/whitespace check
        stripped = question.strip()
        if not stripped:
            result.allowed = False
            result.reason = "Question cannot be empty"
            return result

        # 2. Length checks
        if len(stripped) < self.MIN_QUESTION_LENGTH:
            result.allowed = False
            result.reason = "Question too short to process"
            return result

        if len(stripped) > self.MAX_QUESTION_LENGTH:
            result.allowed = False
            result.reason = (
                f"Question exceeds maximum length "
                f"({self.MAX_QUESTION_LENGTH} characters)"
            )
            return result

        # 3. Prompt injection detection
        for pattern in self.INJECTION_PATTERNS:
            if re.search(pattern, stripped):
                result.allowed = False
                result.reason = (
                    "Question contains instructions that look like an "
                    "attempt to override system behavior"
                )
                logger.warning(
                    f"Prompt injection blocked: '{stripped[:60]}'"
                )
                return result

        # 4. SQL injection detection (in the natural language question itself)
        for pattern in self.SQL_INJECTION_PATTERNS:
            if re.search(pattern, stripped):
                result.allowed = False
                result.reason = (
                    "Question contains SQL keywords that are not allowed "
                    "in natural language queries"
                )
                logger.warning(
                    f"SQL injection attempt blocked: '{stripped[:60]}'"
                )
                return result

        # 5. Normalize whitespace
        normalized = re.sub(r'\s+', ' ', stripped)

        # 6. Expand abbreviations (word-boundary safe)
        normalized = self._expand_abbreviations(normalized)

        result.normalized_question = normalized
        result.allowed = True

        if normalized != stripped:
            result.warnings.append(
                f"Question normalized for processing"
            )
            logger.debug(
                f"Normalized: '{stripped}' → '{normalized}'"
            )

        return result

    def _expand_abbreviations(self, text: str) -> str:
        """Expand common business abbreviations word-by-word."""
        words = text.split(" ")
        expanded = []
        for word in words:
            # Strip trailing punctuation for matching, keep it after
            clean = re.sub(r'[^\w]', '', word.lower())
            if clean in self.ABBREVIATIONS:
                # Preserve original casing pattern roughly (lowercase expand)
                expanded.append(self.ABBREVIATIONS[clean])
            else:
                expanded.append(word)
        return " ".join(expanded)


# ─── Singleton ────────────────────────────────────────────────────────────────
_prompt_guard: Optional[PromptGuard] = None


def get_prompt_guard() -> PromptGuard:
    global _prompt_guard
    if _prompt_guard is None:
        _prompt_guard = PromptGuard()
    return _prompt_guard
