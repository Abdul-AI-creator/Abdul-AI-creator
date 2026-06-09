"""Domain models used by the lead-generation RAG workflow."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


def _first_present(row: Mapping[str, object], *keys: str) -> str:
    normalized = {key.strip().lower().replace(" ", "_"): value for key, value in row.items()}
    for key in keys:
        value = normalized.get(key)
        if value is not None:
            text = str(value).strip()
            if text and text.lower() != "nan":
                return text
    return ""


@dataclass(frozen=True)
class KnowledgeChunk:
    """A searchable chunk of agency knowledge."""

    id: str
    source: str
    text: str


@dataclass(frozen=True)
class RetrievedChunk:
    """A chunk returned from retrieval with a cosine-similarity score."""

    id: str
    source: str
    text: str
    score: float


@dataclass(frozen=True)
class Lead:
    """A prospect record used to generate a client-winning plan."""

    company: str
    industry: str = ""
    website: str = ""
    contact_name: str = ""
    title: str = ""
    description: str = ""
    pain_points: str = ""
    location: str = ""
    employees: str = ""
    source: str = ""

    @classmethod
    def from_row(cls, row: Mapping[str, object]) -> "Lead":
        """Create a lead from a CSV/dict row with forgiving column names."""

        return cls(
            company=_first_present(row, "company", "company_name", "account", "business"),
            industry=_first_present(row, "industry", "vertical", "sector", "niche"),
            website=_first_present(row, "website", "domain", "url"),
            contact_name=_first_present(row, "contact_name", "name", "decision_maker"),
            title=_first_present(row, "title", "role", "job_title"),
            description=_first_present(row, "description", "about", "notes", "summary"),
            pain_points=_first_present(row, "pain_points", "pain_point", "problems", "needs"),
            location=_first_present(row, "location", "city", "region", "country"),
            employees=_first_present(row, "employees", "employee_count", "company_size", "size"),
            source=_first_present(row, "source", "lead_source", "channel"),
        )

    def as_query(self) -> str:
        """Return a retrieval query that combines buying context and pain."""

        parts = [
            self.company,
            self.industry,
            self.title,
            self.description,
            self.pain_points,
            self.location,
            self.employees,
        ]
        return " ".join(part for part in parts if part).strip()


@dataclass(frozen=True)
class LeadPlan:
    """Generated sales plan for a lead."""

    lead: Lead
    fit_score: int
    rationale: str
    retrieved_context: tuple[RetrievedChunk, ...]
    outreach_email: str
    discovery_questions: tuple[str, ...]
    proposal_angles: tuple[str, ...]
    next_steps: tuple[str, ...]

    def to_markdown(self) -> str:
        """Render a lead plan as Markdown for sharing with sales teams."""

        context_lines = [
            f"- `{chunk.source}` (score {chunk.score:.2f}): {chunk.text[:260].strip()}"
            for chunk in self.retrieved_context
        ]
        questions = "\n".join(f"- {question}" for question in self.discovery_questions)
        angles = "\n".join(f"- {angle}" for angle in self.proposal_angles)
        steps = "\n".join(f"- {step}" for step in self.next_steps)
        context = "\n".join(context_lines) if context_lines else "- No matching context found."

        contact = self.lead.contact_name or "there"
        title = f"{self.lead.company} - {self.lead.industry}".strip(" -")

        return f"""## {title}

**Contact:** {contact}{f" ({self.lead.title})" if self.lead.title else ""}
**Website:** {self.lead.website or "Unknown"}
**Fit score:** {self.fit_score}/100

### Why this lead is worth pursuing
{self.rationale}

### Retrieved agency context
{context}

### Outreach email
{self.outreach_email}

### Discovery questions
{questions}

### Proposal angles
{angles}

### Recommended next steps
{steps}
"""
