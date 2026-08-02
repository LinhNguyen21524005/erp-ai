"""
services/semantic_service.py — Semantic Intelligence Service
Migrates: schema_embedding_store.py + semantic_filter.py
Covers: Stage 2.4 (Column Mapping) + Stage 2.8/2.9 (Embedding + Virtual Schema) + Stage 3.7 (Hybrid Retrieval)

Responsibilities:
- Embed table descriptions into ChromaDB
- Hybrid scoring (embedding + keyword boost)
- Table filtering with important columns
- Schema text building for LLM prompt
"""

import logging
import re
from typing import Optional

import chromadb

from erp42_analytics.core.config import config
from erp42_analytics.core.models import QueryIntent

logger = logging.getLogger(__name__)


# ─── Hand-crafted descriptions for 10 core ERP42 tables ─────────────────────
# Priority 1: These override AI-generated descriptions
# Each description includes: synonyms + column meanings + use cases + joins
CORE_TABLE_DESCRIPTIONS = {
    "auftrag": (
        "Auftrag order sales order Kundenauftrag Auftraege Aufträge. "
        "Contains order ID Auftragid, order number Nummer, order date Datum, "
        "customer number Kundennumm, gross amount Bruttosumm revenue Umsatz, "
        "net amount Summe, VAT Mwst, discount Rabatt, employee Mitarbeite, "
        "order status Auftragste, currency Waehrung. "
        "Use for: Wie viele Auftraege how many orders, "
        "Gesamtumsatz im Jahr annual revenue Jahresumsatz, "
        "total sales 2023 2024 SUM(Bruttosumm) YEAR(Datum), "
        "top orders umsatzstaerkste Auftraege mit Kundeninformationen, "
        "Auftragsanzahl, revenue per customer Umsatz pro Kunde. "
        "Often joined with kunde (via Kundennumm) for customer details. "
        "Often joined with mitarbei (via Mitarbeite) for employee sales."
    ),
    "kunde": (
        "Kunde customer client Kunden Kundenstamm Kundendaten Kundeninformationen. "
        "Contains customer ID Kundenid, customer number Kundennumm, "
        "name Vorname Name, address Strasse Ort Plz Land, "
        "email Email, phone Telefon, discount Rabatt, "
        "payment terms Zahlungszi, customer group Gruppe, limit Limit. "
        "Use for: Kundeninformationen customer details, "
        "wie viele Kunden how many customers Kundenanzahl, "
        "top customers by revenue umsatzstaerkste Kunden. "
        "Often joined with auftrag (via Kundennumm) for order analysis. "
        "Often joined with rechnung (via Kundennumm) for invoice analysis."
    ),
    "rechnung": (
        "Rechnung invoice bill Rechnungen Faktura Ausgangsrechnung. "
        "Contains invoice ID Rechnungid, number Nummer, date Datum, "
        "customer Kundennumm, gross amount Bruttosumm, net Summe, "
        "VAT Mwst, payment due Faelligam, status Buchung. "
        "Use for: Wie viele Rechnungen, Gesamtsumme total invoice amount, "
        "offene Rechnungen outstanding invoices, revenue Umsatz aus Rechnungen. "
        "Often joined with kunde (via Kundennumm) for customer details."
    ),
    "lieferan": (
        "Lieferant supplier vendor Lieferanten Lieferantenstamm. "
        "Contains supplier ID Lieferanti, number Lieferante, "
        "name Vorname Name, address Strasse Ort Land, "
        "email Email, phone Telefon, payment terms Zahlungszi. "
        "Use for: Lieferantenliste supplier list, "
        "wie viele Lieferanten how many suppliers Lieferantenanzahl. "
        "Often joined with bestell (via Lieferante) for purchase orders."
    ),
    "bestell": (
        "Bestellung purchase order Bestellungen Einkaufsbestellung Einkauf. "
        "Contains order ID Bestellung, number Nummer, date Datum, "
        "supplier Lieferante, gross amount Bruttosumm, delivery Lieferdatu. "
        "Use for: Bestellungen purchase orders Einkauf, "
        "wie viele Bestellungen, Lieferantenbestellungen. "
        "Often joined with lieferan (via Lieferante) for supplier details."
    ),
    "lager": (
        "Lager warehouse stock inventory Artikel article product Produkt. "
        "Contains article Artikelnum, name Bezeichnun, group Warengrupp, "
        "sales price Verkaufspr, purchase Einkaufsko, stock Lbestand. "
        "Use for: Lagerbestand stock levels, Artikelliste product list, "
        "wie viele Artikel how many products, Inventur inventory."
    ),
    "fibubuch": (
        "Fibubuch financial booking accounting Buchung Finanzbuchhaltung. "
        "Contains booking ID Id, datetime Datumzeit, type Belegart, "
        "debit Sollkonto, credit Habenkonto, amount Betrag, tax Steuer. "
        "Use for: Buchungen financial transactions, Kontobewegungen, "
        "Fibuauswertung financial analysis. "
        "Often joined with sachkont for account details."
    ),
    "mitarbei": (
        "Mitarbeiter employee staff Personal Angestellter. "
        "Contains employee ID Mitarbeite, short name Mitkurz, "
        "full name Mitname, commission Provision, department Abteilung. "
        "Use for: Mitarbeiterliste, wie viele Mitarbeiter, "
        "Umsatz pro Mitarbeiter sales by employee. "
        "Often joined with auftrag (via Mitarbeite) for sales analysis."
    ),
    "projekt": (
        "Projekt project Projekte Projektmanagement Projektliste. "
        "Contains project ID Projektid, number Projekt, name Name, "
        "customer Kundennumm, budget Budget, status Status, date Datum. "
        "Use for: Projektliste, wie viele Projekte, Projektstatus, "
        "Projektbudget, Projekte eines Kunden. "
        "Often joined with kunde (via Kundennumm) for customer projects."
    ),
    "angebot": (
        "Angebot offer quote quotation Angebote Verkaufsangebot. "
        "Contains offer ID Angebotid, number Nummer, date Datum, "
        "customer Kundennumm, gross Bruttosumm, net Summe. "
        "Use for: Angebote quotations, wie viele Angebote, "
        "Angebotsuebersicht, offene Angebote open quotes."
    ),
    # ─── Negative hints ───────────────────────────────────────────────────────
    "kundesta": (
        "Kundesta customer statistics aggregated summary Kundenstatistik. "
        "Contains PRE-AGGREGATED statistics per customer only. "
        "NOT for individual orders — use auftrag for individual orders. "
        "NOT customer master data — use kunde for customer details."
    ),
    "aufgabe": (
        "Aufgabe task todo internal reminder Aufgaben. "
        "aufgabe means TASK not ORDER — different from auftrag. "
        "NOT related to sales orders Auftraege or revenue Umsatz. "
        "For sales orders use auftrag table instead."
    ),
    "dw_kunde": (
        "dw_kunde data warehouse customer view derived reporting table. "
        "NOT the main customer master — use kunde for Kundeninformationen. "
        "NOT for order analysis — use auftrag for orders."
    ),
    "kundenbr": (
        "Kundenbr customer industry branch classification lookup only. "
        "NOT for order queries or revenue analysis. "
        "Use kunde for customer details, auftrag for orders."
    ),
    "kundeums": (
        "Kundeums customer revenue statistics PRE-AGGREGATED summary. "
        "NOT for individual orders — use auftrag table instead."
    ),
    "dw_umsatzpo": (
        "dw_umsatzpo data warehouse revenue position derived reporting. "
        "NOT the main order table — use auftrag for individual orders."
    ),
    "wiedervo": (
        "Wiedervorlage reminder followup callback system internal. "
        "NOT related to sales orders Auftraege or revenue Umsatz."
    ),
    "liefsta": (
        "Liefsta delivery status tracking Lieferstatus. "
        "NOT for revenue Umsatz or Gesamtumsatz analysis. "
        "For revenue use auftrag table with YEAR(Datum)."
    ),
    "liefbew": (
        "Liefbew delivery movement Lieferbewegung tracking. "
        "NOT for revenue or order totals — use auftrag for sales."
    ),
}


# ─── Important columns per core table (Stage 2.4 — Column Mapping & Semantic Tagging)
# Learned from appf.py: fewer columns = more accurate SQL generation
IMPORTANT_COLUMNS = {
    "auftrag":  ["Auftragid", "Nummer", "Datum", "Kundennumm", "Bruttosumm",
                 "Summe", "Mitarbeite", "Projekt", "Auftragste", "Waehrung", "Rabatt"],
    "kunde":    ["Kundenid", "Kundennumm", "Vorname", "Name", "Ort", "Land",
                 "Email", "Telefon", "Gruppe", "Branche", "Rabatt"],
    "rechnung": ["Rechnungid", "Nummer", "Datum", "Kundennumm", "Bruttosumm",
                 "Summe", "Faelligam", "Buchung", "Projekt", "Mitarbeite"],
    "lieferan": ["Lieferanti", "Lieferante", "Name", "Vorname", "Ort", "Land",
                 "Email", "Telefon", "Zahlungszi", "Skonto"],
    "bestell":  ["Bestellung", "Nummer", "Datum", "Lieferante", "Bruttosumm",
                 "Summe", "Lieferdatu", "Projekt", "Mitarbeite"],
    "lager":    ["Lagerid", "Artikelnum", "Bezeichnun", "Warengrupp",
                 "Verkaufspr", "Einkaufsko", "Lbestand", "Einheit", "Gewicht"],
    "fibubuch": ["Id", "Datumzeit", "Belegart", "Nummer",
                 "Sollkonto", "Habenkonto", "Betrag", "Steuer"],
    "mitarbei": ["Mitarbeite", "Mitkurz", "Mitname", "Provision",
                 "Abteilung", "EMail", "Telefon"],
    "projekt":  ["Projektid", "Projekt", "Name", "Kundennumm",
                 "Bezeichnun", "Budget", "Status", "Datum"],
    "angebot":  ["Angebotid", "Nummer", "Datum", "Kundennumm",
                 "Bruttosumm", "Summe", "Projekt", "Mitarbeite"],
}


# ─── Keyword Boost Map ────────────────────────────────────────────────────────
KEYWORD_BOOST = {
    # Orders
    "auftrag": ["auftrag"], "auftraege": ["auftrag"],
    "aufträge": ["auftrag"], "orders": ["auftrag"], "order": ["auftrag"],
    "auftragsanzahl": ["auftrag"],
    # Revenue
    "umsatz": ["auftrag", "rechnung"], "umsätze": ["auftrag", "rechnung"],
    "gesamtumsatz": ["auftrag", "rechnung"], "revenue": ["auftrag", "rechnung"],
    "sales": ["auftrag", "rechnung"], "bruttosumm": ["auftrag", "rechnung"],
    "erloes": ["auftrag", "rechnung"], "umsatzstaerksten": ["auftrag", "kunde"],
    # Time
    "jahr": ["auftrag"], "year": ["auftrag"], "monat": ["auftrag"],
    "month": ["auftrag"], "quartal": ["auftrag"], "datum": ["auftrag"],
    # Customers
    "kunde": ["kunde"], "kunden": ["kunde"], "customer": ["kunde"],
    "customers": ["kunde"], "kundenanzahl": ["kunde"],
    "kundennumm": ["kunde", "auftrag"],
    # Invoices
    "rechnung": ["rechnung"], "rechnungen": ["rechnung"],
    "invoice": ["rechnung"], "invoices": ["rechnung"],
    # Suppliers
    "lieferant": ["lieferan"], "lieferanten": ["lieferan"],
    "supplier": ["lieferan"], "suppliers": ["lieferan"],
    "lieferantenanzahl": ["lieferan"],
    # Purchase orders
    "bestellung": ["bestell"], "bestellungen": ["bestell"],
    "purchase": ["bestell"], "einkauf": ["bestell", "lieferan"],
    # Warehouse
    "lager": ["lager"], "lagerbestand": ["lager"],
    "stock": ["lager"], "inventory": ["lager"],
    "artikel": ["lager"], "produkt": ["lager"], "product": ["lager"],
    "bestand": ["lager"],
    # Finance
    "buchung": ["fibubuch"], "buchungen": ["fibubuch"],
    "fibu": ["fibubuch"], "buchhaltung": ["fibubuch"],
    # Employees
    "mitarbeiter": ["mitarbei"], "mitarbeite": ["mitarbei"],
    "employee": ["mitarbei"], "employees": ["mitarbei"],
    # Projects
    "projekt": ["projekt"], "projekte": ["projekt"], "project": ["projekt"],
    # Offers
    "angebot": ["angebot"], "angebote": ["angebot"],
    "offer": ["angebot"], "quote": ["angebot"],
}


SCHEMA_COLLECTION = "schema_tables"
BOOST_AMOUNT = 0.4

# Webshop-domain table prefixes — a separate subsystem (online shop) that
# lives in the same 379-table database as the ERP core tables. UAT testing
# showed these repeatedly contaminating retrieval for ordinary ERP questions
# (e.g. "current inventory level" pulling in eueorditem alongside lager).
# Used by find_relevant_tables() to apply a stricter relevance floor.
WEBSHOP_TABLE_PREFIXES = ("eue", "sh_")

# ─── Official table classification (provided by ibes AG) ───────────────────
# Source: ibes table classification spreadsheet, cross-referenced against
# the hand-crafted CORE_TABLE_DESCRIPTIONS above. This is the authoritative
# business-validated list — Tier 1 = primary business tables, Tier 2 =
# supporting/reference tables. Used as an additional (broader) protection
# signal for retrieval, alongside CORE_TABLE_DESCRIPTIONS.
#
# Note: 6 tables in CORE_TABLE_DESCRIPTIONS (aufgabe, dw_kunde, dw_umsatzpo,
# fibubuch, liefbew, wiedervo) do NOT appear in ibes's Tier 1/2 sheet —
# wiedervo and liefbew are confirmed real tables (used successfully in UAT
# runs), so likely just Tier 3 / omitted from the sheet rather than wrong.
# Flag this list to Theresa/Paul to confirm scope before Week 2 if it matters
# for RBAC design.
OFFICIAL_TIER1_TABLES = {
    "abposten", "abschlag", "afposten", "anfrage", "angebot", "anposten",
    "auftrag", "auposten", "beposten", "bestell", "bestells", "bsposten",
    "feposten", "fertigun", "fmposten", "guposten", "gutschri", "kunde",
    "lager", "lagernum", "lansprec", "liefer", "lieferan", "lieferun",
    "liposten", "lsposten", "meldung", "produkt", "proforma", "projekt",
    "prposten", "rechnung", "reparatu", "reposten", "rlposten", "rma",
    "rmposten", "rpposten", "ruecklie", "ruecknah", "ruposten", "storno",
    "stposten", "stueckli", "untergru", "waposten", "waren", "wartung",
    "weposten", "xprojektzeiten",
}
OFFICIAL_TIER2_TABLES = {
    "anschrif", "ansprech", "belegart", "benutzer", "bestatus", "chefinfo",
    "debitor", "fakturat", "gruppe", "kdsgvo", "kostenst", "kreditor",
    "kundeart", "kundenbr", "kundenei", "kundesta", "kundeums", "lagerbe",
    "lagerum", "lagerump", "lagerzu", "lanschri", "lbedarf", "liefsta",
    "liefums", "mehrlage", "mitarbei", "postbuch", "preislis", "prodbere",
    "textbaus", "typecode", "xkunde_produkteigenschaft",
    "xprodukteigenschaft", "xproduktinfo", "xproduktkunde", "xstatus",
    "zahlziel",
}
OFFICIAL_CLASSIFIED_TABLES = OFFICIAL_TIER1_TABLES | OFFICIAL_TIER2_TABLES

# ── Sensitive tables — HARD EXCLUDE from retrieval, always ──────────────
# ibes classification marks these Sensitive=YES with specific masked
# columns. Full RBAC (Stage 4.5 — Permission-aware Query Processing /
# Sensitive Column Filtering) isn't built yet, so until per-role access
# control exists, the safest stopgap is to never let these tables surface
# as a retrieval candidate at all — not floor-filtered, not boost-exempt,
# unconditionally excluded. Extend this dict as more Sensitive=YES rows
# are confirmed from the ibes sheet.
SENSITIVE_TABLES = {
    "benutzer": ["Passwort"],  # ERP users — access restricted to Geschäftsleitung only
}


class SemanticService:
    """
    Manages semantic table search and schema text building.
    Combines embedding similarity + keyword boost (hybrid scoring).
    """

    def __init__(self):
        self.chroma_client = chromadb.PersistentClient(
            path=config.storage.chroma_path
        )
        self.collection = self.chroma_client.get_or_create_collection(
            name=SCHEMA_COLLECTION,
            metadata={"hnsw:space": "cosine"}
        )

    # ─── Index Management ─────────────────────────────────────────────────────

    def index_tables(self, all_tables: dict, force_reindex: bool = False):
        """Embed all table descriptions into ChromaDB."""
        if not force_reindex and self.collection.count() > 0:
            logger.info(f"Schema indexed: {self.collection.count()} tables")
            return

        if force_reindex:
            self.chroma_client.delete_collection(SCHEMA_COLLECTION)
            self.collection = self.chroma_client.get_or_create_collection(
                name=SCHEMA_COLLECTION,
                metadata={"hnsw:space": "cosine"}
            )

        documents, ids, metadatas = [], [], []
        core_count = 0

        for table_name, info in all_tables.items():
            description = self._build_description(table_name, info)
            cols = info.get("columns", [])
            col_names = [c["name"] if isinstance(c, dict) else c for c in cols]
            is_core = table_name in CORE_TABLE_DESCRIPTIONS
            if is_core:
                core_count += 1

            documents.append(description)
            ids.append(table_name)
            metadatas.append({
                "table_name": table_name,
                "column_count": len(col_names),
                "columns": ",".join(col_names[:20]),
                "is_core": str(is_core),
            })

        batch_size = config.app.schema_batch_size
        for i in range(0, len(documents), batch_size):
            self.collection.add(
                documents=documents[i:i+batch_size],
                ids=ids[i:i+batch_size],
                metadatas=metadatas[i:i+batch_size],
            )
            logger.info(
                f"Indexed {min(i+batch_size, len(documents))}/{len(documents)}"
            )

        logger.info(
            f"Schema indexed: {len(documents)} tables "
            f"({core_count} hand-crafted)"
        )

    def _build_description(self, table_name: str, info: dict) -> str:
        """Build description with priority: hand-crafted > AI-generated > auto."""
        # Priority 1: Hand-crafted
        if table_name in CORE_TABLE_DESCRIPTIONS:
            return CORE_TABLE_DESCRIPTIONS[table_name]
        # Priority 2: AI-generated
        ai_desc = info.get("business_description")
        if ai_desc and not ai_desc.startswith(f"Table {table_name}:"):
            return ai_desc
        # Priority 3: Auto from column names
        cols = info.get("columns", [])
        col_names = [c["name"] if isinstance(c, dict) else c for c in cols]
        return f"Table {table_name}: {', '.join(col_names)}"

    # ─── Hybrid Search ────────────────────────────────────────────────────────

    def find_relevant_tables(
        self, question: str, n_results: int = None
    ) -> list:
        """
        Hybrid scoring: embedding similarity + keyword boost.
        Ensures boosted tables are always in candidate pool.
        """
        n_results = n_results or config.query.n_relevant_tables
        if self.collection.count() == 0:
            return []

        # Get large candidate pool so keyword boost has effect
        n_candidates = min(
            max(n_results * config.query.n_candidates_multiplier,
                config.query.min_candidates),
            self.collection.count()
        )

        results = self.collection.query(
            query_texts=[question],
            n_results=n_candidates,
        )

        if not results or not results["ids"] or not results["ids"][0]:
            return []

        keyword_boosts = self._compute_keyword_boost(question)

        scored = []
        for i, table_id in enumerate(results["ids"][0]):
            # Sensitive tables (Passwort etc.) are excluded unconditionally
            # here — before scoring, before boosting, before the protection
            # logic below. No score or keyword match can rescue them back
            # into the candidate pool. See SENSITIVE_TABLES for scope.
            if table_id in SENSITIVE_TABLES:
                logger.warning(
                    f"  🔒 excluded sensitive table '{table_id}' from "
                    f"retrieval candidates (masked columns: "
                    f"{SENSITIVE_TABLES[table_id]})"
                )
                continue
            emb_score = 1 - results["distances"][0][i]
            boost = keyword_boosts.get(table_id, 0)
            final = emb_score + boost
            is_core = results["metadatas"][0][i].get("is_core") == "True"
            scored.append((table_id, final, emb_score, boost, is_core))

        scored.sort(key=lambda x: x[1], reverse=True)

        # ── Relevance floor (Step 3.5 quality gate) ─────────────────────
        # Previously this just sliced scored[:n_results] unconditionally,
        # meaning if only 1 table was truly relevant, the other 1-2 slots
        # got padded with whatever scored next — even pure embedding
        # noise (e.g. "akontakt"/"kontakt" scoring 0.45-0.48 for a plain
        # customer-count question). That noise then entered the LLM
        # prompt as a plausible-looking candidate, and the model would
        # often hallucinate a join to it.
        #
        # Protection rule: a table is EXEMPT from the floor (always kept)
        # if any of —
        #   (a) it has a hand-crafted description in CORE_TABLE_DESCRIPTIONS
        #       — checked fresh against the constant here, not against the
        #       "is_core" flag baked into ChromaDB metadata at indexing
        #       time, since that index can go stale relative to the code
        #   (b) it's in ibes's official Tier 1/2 table classification
        #       (OFFICIAL_CLASSIFIED_TABLES) — the business-validated
        #       source of truth for which tables are real ERP business
        #       tables, independent of what our own code happens to know
        #   (c) it was matched via an explicit keyword boost (boost > 0)
        #       — a deliberate word-level signal is worth more than raw
        #       embedding similarity, and the original design intent
        #       (see _compute_keyword_boost) was to guarantee inclusion
        # Everything else is pure embedding noise and must clear the
        # floor — a higher one for webshop-domain tables specifically,
        # since those are a known contamination source in the logs.
        filtered = []
        for table_id, final, emb, boost, is_core in scored:
            protected = (
                (table_id in CORE_TABLE_DESCRIPTIONS)
                or (table_id in OFFICIAL_CLASSIFIED_TABLES)
                or (boost > 0)
            )
            if protected:
                filtered.append((table_id, final, emb, boost, is_core))
                continue
            is_webshop = table_id.startswith(WEBSHOP_TABLE_PREFIXES)
            floor = (
                config.query.webshop_min_relevance_score
                if is_webshop
                else config.query.min_relevance_score
            )
            if final >= floor:
                filtered.append((table_id, final, emb, boost, is_core))
            else:
                logger.debug(
                    f"  ✗ dropped {table_id}: {final:.3f} below "
                    f"floor {floor:.2f}{' [webshop]' if is_webshop else ''}"
                )

        # Never return zero candidates — if everything fell below the
        # floor, keep at least the single best match so the pipeline has
        # something to work with (still logged/scored honestly).
        if not filtered and scored:
            filtered = [scored[0]]

        top = filtered[:n_results]

        table_names = []
        for table_id, final, emb, boost, is_core in top:
            marker = "⭐" if is_core else "  "
            boost_str = f" +{boost:.2f}" if boost > 0 else ""
            logger.info(
                f"  {marker} {table_id}: {final:.3f} "
                f"(emb:{emb:.3f}{boost_str})"
            )
            table_names.append(table_id)

        logger.info(f"Found {len(table_names)} tables for: '{question[:50]}'")
        return table_names

    def _compute_keyword_boost(self, question: str) -> dict:
        """Compute keyword boost scores from question words."""
        words = re.findall(r"[a-zA-ZäöüÄÖÜß]+", question.lower())
        boost_scores = {}
        for word in words:
            if word in KEYWORD_BOOST:
                for table in KEYWORD_BOOST[word]:
                    boost_scores[table] = (
                        boost_scores.get(table, 0) + BOOST_AMOUNT
                    )
        return boost_scores

    # ─── Schema Text Building ─────────────────────────────────────────────────

    def build_schema_text(self, filtered_tables: dict) -> str:
        """
        Build compact schema text for LLM prompt.
        Learned from Dataherald InfoRelevantColumns pattern:
        - Uses IMPORTANT_COLUMNS for core tables
        - Injects cardinality values for low-cardinality columns
        - Adds sample rows for context
        - Includes ERP42test.dbo prefix automatically
        """
        db_prefix = config.db.full_table_prefix
        parts = []

        for table_name, info in filtered_tables.items():
            cols = info.get("columns", [])
            all_col_names = [
                c["name"] if isinstance(c, dict) else c for c in cols
            ]

            # Use important columns for core tables
            if table_name in IMPORTANT_COLUMNS:
                col_names = [
                    c for c in IMPORTANT_COLUMNS[table_name]
                    if c in all_col_names
                ]
            else:
                col_names = all_col_names[:20]

            # Build column text with cardinality values (Dataherald pattern)
            col_texts = []
            col_map = {
                c["name"]: c for c in cols
                if isinstance(c, dict)
            }
            for col_name in col_names:
                col_info = col_map.get(col_name, {})
                categories = col_info.get("categories")
                if categories and col_info.get("low_cardinality"):
                    # Show sample values → helps model use correct filter values
                    sample = ", ".join(str(v) for v in categories[:10])
                    col_texts.append(f"{col_name} (values: {sample})")
                else:
                    col_texts.append(col_name)

            line = f'Table "{db_prefix}.{table_name}": {", ".join(col_texts)}'

            # Add sample rows if available (Dataherald get_table_examples pattern)
            examples = info.get("examples", [])
            if examples:
                line += "\n  Sample rows:"
                for ex in examples[:2]:  # Max 2 sample rows
                    # Only show important columns in sample
                    sample_vals = {
                        k: v for k, v in ex.items()
                        if k in col_names
                    }
                    if sample_vals:
                        line += f"\n  {sample_vals}"

            parts.append(line)

        return "\n".join(parts)

    def needs_current_time(self, question: str) -> bool:
        """Check if question needs current datetime."""
        time_keywords = [
            "heute", "today", "jetzt", "now", "aktuell", "current",
            "diese woche", "this week", "diesen monat", "this month"
        ]
        q_lower = question.lower()
        return any(kw in q_lower for kw in time_keywords)

    # ─── Status ───────────────────────────────────────────────────────────────

    def is_indexed(self) -> bool:
        return self.collection.count() > 0

    def count(self) -> int:
        return self.collection.count()
