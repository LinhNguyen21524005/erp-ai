"""
intelligence/synonym_kb.py — Synonym Knowledge Base
Stage 3.4: Business synonym understanding for German/English ERP queries.

Responsibilities:
- Map business terms to canonical ERP concepts
- Handle German/English/mixed language synonyms
- Industry-specific ERP terminology
- Expandable via Admin panel (human feedback)
"""

import logging
import re
from typing import Optional

from erp42_analytics.repositories.history_repo import HistoryRepository

logger = logging.getLogger(__name__)


# ─── Built-in Synonym Map ─────────────────────────────────────────────────────
# Maps user terms → canonical ERP42 table/concept
# Format: {user_term: {"tables": [...], "concept": "...", "domain": "..."}}

BUILTIN_SYNONYMS = {
    # ── Revenue / Sales ───────────────────────────────────────────────────────
    "umsatz":           {"tables": ["auftrag"], "concept": "revenue", "domain": "finance"},
    "gesamtumsatz":     {"tables": ["auftrag"], "concept": "total_revenue", "domain": "finance"},  # auftrag only
    "jahresumsatz":     {"tables": ["auftrag"], "concept": "annual_revenue", "domain": "finance"},
    "monatsumsatz":     {"tables": ["auftrag"], "concept": "monthly_revenue", "domain": "finance"},
    "umsatzauswertung": {"tables": ["auftrag"], "concept": "revenue_analysis", "domain": "finance"},
    "umsatzstaerksten":  {"tables": ["auftrag", "kunde"], "concept": "top_revenue", "domain": "finance"},
    "umsatzstaerkste":   {"tables": ["auftrag", "kunde"], "concept": "top_revenue", "domain": "finance"},
    "umsatzstärksten":   {"tables": ["auftrag", "kunde"], "concept": "top_revenue", "domain": "finance"},
    "umsatzstärkste":    {"tables": ["auftrag", "kunde"], "concept": "top_revenue", "domain": "finance"},
    "revenue":          {"tables": ["auftrag"], "concept": "revenue", "domain": "finance"},
    "sales":            {"tables": ["auftrag"], "concept": "sales", "domain": "sales"},
    "turnover":         {"tables": ["auftrag"], "concept": "revenue", "domain": "finance"},
    "income":           {"tables": ["rechnung"], "concept": "income", "domain": "finance"},
    "bruttosumme":      {"tables": ["auftrag"], "concept": "gross_amount", "domain": "finance"},
    "nettosumme":       {"tables": ["auftrag"], "concept": "net_amount", "domain": "finance"},

    # ── Orders ────────────────────────────────────────────────────────────────
    "auftrag":          {"tables": ["auftrag"], "concept": "order", "domain": "sales"},
    "aufträge":         {"tables": ["auftrag"], "concept": "orders", "domain": "sales"},
    "auftraege":        {"tables": ["auftrag"], "concept": "orders", "domain": "sales"},
    "bestellung":       {"tables": ["bestell"], "concept": "purchase_order", "domain": "procurement"},
    "bestellungen":     {"tables": ["bestell"], "concept": "purchase_orders", "domain": "procurement"},
    "order":            {"tables": ["auftrag"], "concept": "order", "domain": "sales"},
    "orders":           {"tables": ["auftrag"], "concept": "orders", "domain": "sales"},
    "sales order":      {"tables": ["auftrag"], "concept": "sales_order", "domain": "sales"},
    "purchase order":   {"tables": ["bestell"], "concept": "purchase_order", "domain": "procurement"},

    # ── Customers ─────────────────────────────────────────────────────────────
    "kunde":            {"tables": ["kunde"], "concept": "customer", "domain": "crm"},
    "kunden":           {"tables": ["kunde"], "concept": "customers", "domain": "crm"},
    "kundenstamm":      {"tables": ["kunde"], "concept": "customer_master", "domain": "crm"},
    "customer":         {"tables": ["kunde"], "concept": "customer", "domain": "crm"},
    "customers":        {"tables": ["kunde"], "concept": "customers", "domain": "crm"},
    "client":           {"tables": ["kunde"], "concept": "customer", "domain": "crm"},
    "käufer":           {"tables": ["kunde"], "concept": "buyer", "domain": "crm"},
    "abnehmer":         {"tables": ["kunde"], "concept": "customer", "domain": "crm"},
    "kundeninformation":  {"tables": ["kunde"], "concept": "customer_info", "domain": "crm"},
    "kundeninformationen":{"tables": ["kunde"], "concept": "customer_details", "domain": "crm"},
    "kundendaten":        {"tables": ["kunde"], "concept": "customer_data", "domain": "crm"},
    "kundennamen":        {"tables": ["kunde"], "concept": "customer_name", "domain": "crm"},
    "customer info":      {"tables": ["kunde"], "concept": "customer_info", "domain": "crm"},
    "customer details":   {"tables": ["kunde"], "concept": "customer_details", "domain": "crm"},
    "customer data":      {"tables": ["kunde"], "concept": "customer_data", "domain": "crm"},

    # ── Invoices ─────────────────────────────────────────────────────────────
    "rechnung":         {"tables": ["rechnung"], "concept": "invoice", "domain": "finance"},
    "rechnungen":       {"tables": ["rechnung"], "concept": "invoices", "domain": "finance"},
    "faktura":          {"tables": ["rechnung"], "concept": "invoice", "domain": "finance"},
    "invoice":          {"tables": ["rechnung"], "concept": "invoice", "domain": "finance"},
    "invoices":         {"tables": ["rechnung"], "concept": "invoices", "domain": "finance"},
    "bill":             {"tables": ["rechnung"], "concept": "invoice", "domain": "finance"},

    # ── Suppliers ─────────────────────────────────────────────────────────────
    "lieferant":        {"tables": ["lieferan"], "concept": "supplier", "domain": "procurement"},
    "lieferanten":      {"tables": ["lieferan"], "concept": "suppliers", "domain": "procurement"},
    "lieferantenstamm": {"tables": ["lieferan"], "concept": "supplier_master", "domain": "procurement"},
    "supplier":         {"tables": ["lieferan"], "concept": "supplier", "domain": "procurement"},
    "suppliers":        {"tables": ["lieferan"], "concept": "suppliers", "domain": "procurement"},
    "vendor":           {"tables": ["lieferan"], "concept": "supplier", "domain": "procurement"},
    "vendors":          {"tables": ["lieferan"], "concept": "suppliers", "domain": "procurement"},
    "hersteller":       {"tables": ["lieferan"], "concept": "manufacturer", "domain": "procurement"},

    # ── Products / Inventory ──────────────────────────────────────────────────
    "artikel":          {"tables": ["lager"], "concept": "article", "domain": "inventory"},
    "lager":            {"tables": ["lager"], "concept": "warehouse", "domain": "inventory"},
    "lagerbestand":     {"tables": ["lager"], "concept": "stock_level", "domain": "inventory"},
    "bestand":          {"tables": ["lager"], "concept": "stock", "domain": "inventory"},
    "produkt":          {"tables": ["lager"], "concept": "product", "domain": "inventory"},
    "produkte":         {"tables": ["lager"], "concept": "products", "domain": "inventory"},
    "ware":             {"tables": ["lager"], "concept": "goods", "domain": "inventory"},
    "waren":            {"tables": ["lager"], "concept": "goods", "domain": "inventory"},
    "product":          {"tables": ["lager"], "concept": "product", "domain": "inventory"},
    "products":         {"tables": ["lager"], "concept": "products", "domain": "inventory"},
    "inventory":        {"tables": ["lager"], "concept": "inventory", "domain": "inventory"},
    "stock":            {"tables": ["lager"], "concept": "stock", "domain": "inventory"},
    "item":             {"tables": ["lager"], "concept": "item", "domain": "inventory"},

    # ── Employees ─────────────────────────────────────────────────────────────
    "mitarbeiter":      {"tables": ["mitarbei"], "concept": "employee", "domain": "hr"},
    "angestellter":     {"tables": ["mitarbei"], "concept": "employee", "domain": "hr"},
    "personal":         {"tables": ["mitarbei"], "concept": "staff", "domain": "hr"},
    "employee":         {"tables": ["mitarbei"], "concept": "employee", "domain": "hr"},
    "employees":        {"tables": ["mitarbei"], "concept": "employees", "domain": "hr"},
    "staff":            {"tables": ["mitarbei"], "concept": "staff", "domain": "hr"},
    "sales rep":        {"tables": ["mitarbei"], "concept": "sales_rep", "domain": "hr"},
    "vertreter":        {"tables": ["mitarbei"], "concept": "representative", "domain": "hr"},

    # ── Projects ──────────────────────────────────────────────────────────────
    "projekt":          {"tables": ["projekt"], "concept": "project", "domain": "pm"},
    "projekte":         {"tables": ["projekt"], "concept": "projects", "domain": "pm"},
    "project":          {"tables": ["projekt"], "concept": "project", "domain": "pm"},
    "projects":         {"tables": ["projekt"], "concept": "projects", "domain": "pm"},

    # ── Offers / Quotes ───────────────────────────────────────────────────────
    "angebot":          {"tables": ["angebot"], "concept": "quote", "domain": "sales"},
    "angebote":         {"tables": ["angebot"], "concept": "quotes", "domain": "sales"},
    "quote":            {"tables": ["angebot"], "concept": "quote", "domain": "sales"},
    "quotation":        {"tables": ["angebot"], "concept": "quotation", "domain": "sales"},
    "offer":            {"tables": ["angebot"], "concept": "offer", "domain": "sales"},
    "angebot erstellt": {"tables": ["angebot"], "concept": "quote_created", "domain": "sales"},

    # ── Finance / Accounting ──────────────────────────────────────────────────
    "buchung":          {"tables": ["fibubuch"], "concept": "booking", "domain": "finance"},
    "buchungen":        {"tables": ["fibubuch"], "concept": "bookings", "domain": "finance"},
    "buchhaltung":      {"tables": ["fibubuch"], "concept": "accounting", "domain": "finance"},
    "fibu":             {"tables": ["fibubuch"], "concept": "financial_accounting", "domain": "finance"},
    "booking":          {"tables": ["fibubuch"], "concept": "booking", "domain": "finance"},
    "accounting":       {"tables": ["fibubuch"], "concept": "accounting", "domain": "finance"},

    # ── Time Dimensions ───────────────────────────────────────────────────────
    "jahr":             {"tables": ["auftrag"], "concept": "year", "domain": "time"},
    "monat":            {"tables": ["auftrag"], "concept": "month", "domain": "time"},
    "quartal":          {"tables": ["auftrag"], "concept": "quarter", "domain": "time"},
    "woche":            {"tables": ["auftrag"], "concept": "week", "domain": "time"},
    "year":             {"tables": ["auftrag"], "concept": "year", "domain": "time"},
    "month":            {"tables": ["auftrag"], "concept": "month", "domain": "time"},
    "quarter":          {"tables": ["auftrag"], "concept": "quarter", "domain": "time"},
    "yearly":           {"tables": ["auftrag"], "concept": "annual", "domain": "time"},
    "monthly":          {"tables": ["auftrag"], "concept": "monthly", "domain": "time"},
    "quarterly":        {"tables": ["auftrag"], "concept": "quarterly", "domain": "time"},
}


class SynonymKB:
    """
    Business Synonym Knowledge Base.
    Combines built-in synonyms with user-added synonyms from DB.
    """

    def __init__(self):
        self.repo = HistoryRepository()
        self._kb: dict = {}
        self._build()

    def _build(self):
        """Build combined KB from built-in + DB synonyms."""
        # Start with built-in
        self._kb = dict(BUILTIN_SYNONYMS)

        # Add user-defined synonyms from DB
        try:
            db_synonyms = self.repo.get_synonyms()
            for term, synonyms in db_synonyms.items():
                if term not in self._kb:
                    self._kb[term] = {
                        "tables": [], "concept": term, "domain": "custom"
                    }
                for syn in synonyms:
                    if syn in self._kb:
                        # Merge tables from synonym entry
                        self._kb[term]["tables"].extend(
                            self._kb[syn].get("tables", [])
                        )
        except Exception as e:
            logger.warning(f"Failed to load DB synonyms: {e}")

    def lookup(self, term: str) -> Optional[dict]:
        """Look up a term in the KB."""
        term_lower = term.lower().strip()
        return self._kb.get(term_lower)

    def get_tables_for_term(self, term: str) -> list:
        """Get relevant tables for a business term."""
        entry = self.lookup(term)
        if entry:
            return entry.get("tables", [])
        return []

    def extract_concepts(self, question: str) -> dict:
        """
        Extract business concepts from a natural language question.
        Returns: {concept: {tables, domain, matched_term}}
        """
        question_lower = question.lower()
        words = re.findall(r"[a-zA-ZäöüÄÖÜß]+", question_lower)

        # Also try multi-word matches (2-word phrases)
        bigrams = [
            f"{words[i]} {words[i+1]}"
            for i in range(len(words) - 1)
        ]

        concepts = {}
        matched_terms = set()

        # Check bigrams first (more specific)
        for bigram in bigrams:
            if bigram in self._kb:
                entry = self._kb[bigram]
                concept = entry["concept"]
                if concept not in concepts:
                    concepts[concept] = {
                        "tables": entry["tables"],
                        "domain": entry["domain"],
                        "matched_term": bigram,
                    }
                    matched_terms.update(bigram.split())

        # Then single words
        for word in words:
            if word in matched_terms:
                continue
            if word in self._kb:
                entry = self._kb[word]
                concept = entry["concept"]
                if concept not in concepts:
                    concepts[concept] = {
                        "tables": entry["tables"],
                        "domain": entry["domain"],
                        "matched_term": word,
                    }

        return concepts

    def get_all_tables_mentioned(self, question: str) -> list:
        """Get unique list of tables implied by question vocabulary."""
        concepts = self.extract_concepts(question)
        tables = []
        for concept_info in concepts.values():
            for table in concept_info["tables"]:
                if table not in tables:
                    tables.append(table)
        return tables

    def add_synonym(self, term: str, synonym: str,
                    domain: str = "custom") -> bool:
        """Add new synonym (human feedback)."""
        try:
            self.repo.save_synonym(term, synonym, domain)
            # Update in-memory KB
            term_lower = term.lower()
            if term_lower not in self._kb:
                self._kb[term_lower] = {
                    "tables": [], "concept": term_lower, "domain": domain
                }
            self._build()  # Rebuild to include new synonym
            return True
        except Exception as e:
            logger.error(f"Failed to add synonym: {e}")
            return False

    def reload(self):
        """Reload KB from DB (after human edits)."""
        self._build()
        logger.info("Synonym KB reloaded")


# ─── Singleton ────────────────────────────────────────────────────────────────
_synonym_kb: Optional[SynonymKB] = None


def get_synonym_kb() -> SynonymKB:
    global _synonym_kb
    if _synonym_kb is None:
        _synonym_kb = SynonymKB()
    return _synonym_kb
