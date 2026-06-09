"""Sales plan generation grounded in retrieved agency knowledge."""

from __future__ import annotations

import textwrap

from .models import Lead, LeadPlan, RetrievedChunk
from .store import KnowledgeBase


HIGH_INTENT_TERMS = {
    "book calls",
    "pipeline",
    "sales qualified",
    "demo",
    "appointment",
    "revenue",
    "conversion",
    "qualified leads",
    "outbound",
    "growth",
}


class LeadWinGenerator:
    """Generate lead plans, outreach, and sales angles with RAG context."""

    def __init__(self, knowledge_base: KnowledgeBase, top_k: int = 4) -> None:
        self.knowledge_base = knowledge_base
        self.top_k = top_k

    def generate(self, lead: Lead) -> LeadPlan:
        """Generate a grounded plan for one prospect."""

        retrieved = tuple(self.knowledge_base.search(lead.as_query(), top_k=self.top_k))
        fit_score = self._score_lead(lead, retrieved)
        rationale = self._build_rationale(lead, retrieved, fit_score)
        outreach = self._build_outreach_email(lead, retrieved)
        discovery_questions = self._build_discovery_questions(lead)
        proposal_angles = self._build_proposal_angles(lead, retrieved)
        next_steps = self._build_next_steps(lead)

        return LeadPlan(
            lead=lead,
            fit_score=fit_score,
            rationale=rationale,
            retrieved_context=retrieved,
            outreach_email=outreach,
            discovery_questions=tuple(discovery_questions),
            proposal_angles=tuple(proposal_angles),
            next_steps=tuple(next_steps),
        )

    def _score_lead(self, lead: Lead, retrieved: tuple[RetrievedChunk, ...]) -> int:
        score = 35
        if lead.company:
            score += 5
        if lead.industry:
            score += 8
        if lead.contact_name or lead.title:
            score += 8
        if lead.pain_points:
            score += 12
        if lead.website:
            score += 4

        query = lead.as_query().lower()
        score += min(14, sum(3 for term in HIGH_INTENT_TERMS if term in query))

        if retrieved:
            average_relevance = sum(chunk.score for chunk in retrieved) / len(retrieved)
            score += min(14, round(average_relevance * 60))

        return max(0, min(100, score))

    def _build_rationale(
        self,
        lead: Lead,
        retrieved: tuple[RetrievedChunk, ...],
        fit_score: int,
    ) -> str:
        strongest_source = retrieved[0].source if retrieved else "the agency knowledge base"
        pain = lead.pain_points or "their growth priorities"
        industry = lead.industry or "their market"
        confidence = "high" if fit_score >= 75 else "moderate" if fit_score >= 55 else "early"
        return (
            f"This is a {confidence}-fit prospect because {lead.company or 'the company'} "
            f"operates in {industry} and appears to care about {pain}. The best matching "
            f"proof point comes from `{strongest_source}`, so outreach should connect the "
            "agency's relevant case studies, offers, and operating process to the prospect's "
            "specific acquisition problem."
        )

    def _build_outreach_email(self, lead: Lead, retrieved: tuple[RetrievedChunk, ...]) -> str:
        contact = lead.contact_name or "there"
        company = lead.company or "your team"
        industry_phrase = f" in {lead.industry}" if lead.industry else ""
        pain = lead.pain_points or "creating a predictable qualified-lead pipeline"

        proof = "we help teams turn targeted outreach into qualified sales conversations"
        if retrieved:
            proof = self._summarize_context(retrieved[0].text)

        email = f"""
        Subject: Qualified pipeline idea for {company}

        Hi {contact},

        I noticed {company}{industry_phrase} is likely focused on {pain}. We work with
        teams facing similar pipeline goals, and one relevant proof point from our playbook is:
        "{proof}"

        A practical starting point would be to map your ideal customer profile, identify
        high-intent accounts, and launch a small outbound test with personalized messaging
        tied to the buying trigger we find.

        Would it be useful if I shared 3 account segments and message angles that could
        help {company} create more qualified sales conversations?

        Best,
        Your Name
        """
        return textwrap.dedent(email).strip()

    def _build_discovery_questions(self, lead: Lead) -> list[str]:
        target = lead.industry or "your target market"
        return [
            f"Which customer segment in {target} has the highest close rate and deal value today?",
            "What channels currently create qualified calls, and where does volume or quality break down?",
            "What buying triggers usually make a prospect actively look for a solution like yours?",
            "Which objections prevent interested prospects from becoming sales-qualified opportunities?",
            "What would make a pilot successful: meetings booked, SQLs, pipeline value, or closed revenue?",
        ]

    def _build_proposal_angles(
        self,
        lead: Lead,
        retrieved: tuple[RetrievedChunk, ...],
    ) -> list[str]:
        industry = lead.industry or "the prospect's niche"
        pain = lead.pain_points or "lead quality and conversion"
        context_angle = (
            self._summarize_context(retrieved[0].text)
            if retrieved
            else "Use the agency's best case study as the core credibility proof."
        )
        return [
            f"ICP and account-list build for {industry}, prioritized by pain, buying trigger, and revenue fit.",
            f"Personalized outreach campaign positioned around {pain}, with A/B-tested offers and CTAs.",
            f"Credibility layer: {context_angle}",
            "Weekly optimization loop covering reply quality, meeting show rate, objections, and pipeline value.",
        ]

    def _build_next_steps(self, lead: Lead) -> list[str]:
        company = lead.company or "the prospect"
        return [
            f"Validate {company}'s ICP, decision makers, and current acquisition channels.",
            "Research 20-50 lookalike accounts and identify the strongest buying triggers.",
            "Send the first email with one specific pain point, one proof point, and one low-friction CTA.",
            "Prepare a short audit or account-segment teardown to increase reply-to-meeting conversion.",
        ]

    @staticmethod
    def _summarize_context(text: str, max_words: int = 34) -> str:
        words = text.split()
        summary = " ".join(words[:max_words])
        if len(words) > max_words:
            summary += "..."
        return summary
