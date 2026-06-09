"""Service helpers shared by CLI, tests, and web deployments."""

from __future__ import annotations

import csv
from io import StringIO
from typing import Iterable

from .chunking import chunk_documents
from .cli import generate_markdown_report
from .generator import LeadWinGenerator
from .models import Lead, LeadPlan
from .store import KnowledgeBase


def load_leads_from_csv_text(csv_text: str) -> list[Lead]:
    """Load prospect leads from CSV text."""

    reader = csv.DictReader(StringIO(csv_text))
    if not reader.fieldnames:
        raise ValueError("Leads CSV must include a header row")

    leads = [Lead.from_row(row) for row in reader]
    return [lead for lead in leads if lead.company or lead.website or lead.description]


def build_knowledge_base_from_texts(
    documents: Iterable[tuple[str, str]],
    chunk_size: int = 900,
    overlap: int = 120,
) -> KnowledgeBase:
    """Build a knowledge base from in-memory documents."""

    chunks = chunk_documents(documents, chunk_size=chunk_size, overlap=overlap)
    return KnowledgeBase(chunks)


def generate_plans_from_texts(
    knowledge_documents: Iterable[tuple[str, str]],
    leads_csv: str,
    top_k: int = 4,
    chunk_size: int = 900,
    overlap: int = 120,
) -> list[LeadPlan]:
    """Generate plans from uploaded knowledge text and leads CSV text."""

    knowledge_base = build_knowledge_base_from_texts(
        knowledge_documents,
        chunk_size=chunk_size,
        overlap=overlap,
    )
    generator = LeadWinGenerator(knowledge_base, top_k=top_k)
    leads = load_leads_from_csv_text(leads_csv)
    if not leads:
        raise ValueError("Leads CSV did not contain any usable lead rows")

    return [generator.generate(lead) for lead in leads]


def serialize_plan(plan: LeadPlan) -> dict:
    """Return a JSON-serializable representation of a lead plan."""

    return {
        "company": plan.lead.company,
        "industry": plan.lead.industry,
        "contact_name": plan.lead.contact_name,
        "website": plan.lead.website,
        "fit_score": plan.fit_score,
        "rationale": plan.rationale,
        "outreach_email": plan.outreach_email,
        "discovery_questions": list(plan.discovery_questions),
        "proposal_angles": list(plan.proposal_angles),
        "next_steps": list(plan.next_steps),
        "retrieved_context": [
            {
                "source": chunk.source,
                "score": round(chunk.score, 4),
                "text": chunk.text,
            }
            for chunk in plan.retrieved_context
        ],
        "markdown": plan.to_markdown(),
    }


def generate_response_payload(
    knowledge_documents: Iterable[tuple[str, str]],
    leads_csv: str,
    top_k: int = 4,
) -> dict:
    """Generate the response payload used by the Vercel API."""

    plans = generate_plans_from_texts(
        knowledge_documents=knowledge_documents,
        leads_csv=leads_csv,
        top_k=top_k,
    )
    return {
        "plans": [serialize_plan(plan) for plan in plans],
        "markdown_report": generate_markdown_report(plans),
    }
