from pathlib import Path

from leadwin_rag.cli import generate_markdown_report, load_leads
from leadwin_rag.generator import LeadWinGenerator
from leadwin_rag.models import Lead
from leadwin_rag.service import generate_response_payload, load_leads_from_csv_text
from leadwin_rag.store import KnowledgeBase


def test_knowledge_base_retrieves_relevant_context() -> None:
    knowledge_base = KnowledgeBase.from_paths(["data/sample_knowledge"], chunk_size=80, overlap=10)

    results = knowledge_base.search("cybersecurity SaaS qualified demos outbound", top_k=2)

    assert results
    assert any("cybersecurity" in result.text.lower() for result in results)
    assert results[0].score > 0


def test_generator_creates_grounded_lead_plan() -> None:
    knowledge_base = KnowledgeBase.from_paths(["data/sample_knowledge"], chunk_size=80, overlap=10)
    generator = LeadWinGenerator(knowledge_base, top_k=3)
    lead = Lead(
        company="CloudFence",
        industry="SaaS cybersecurity",
        contact_name="Maya Patel",
        title="VP Sales",
        pain_points="Needs more qualified enterprise demos and stronger outbound conversion",
        description="Security platform selling to compliance-heavy buyers",
    )

    plan = generator.generate(lead)

    assert plan.fit_score >= 70
    assert plan.retrieved_context
    assert "CloudFence" in plan.outreach_email
    assert "qualified" in plan.outreach_email.lower()
    assert plan.proposal_angles
    assert plan.discovery_questions


def test_load_leads_accepts_sample_csv() -> None:
    leads = load_leads(Path("data/sample_leads.csv"))

    assert len(leads) == 3
    assert leads[0].company == "CloudFence"
    assert leads[0].industry == "SaaS cybersecurity"


def test_markdown_report_contains_citations() -> None:
    knowledge_base = KnowledgeBase.from_paths(["data/sample_knowledge"], chunk_size=80, overlap=10)
    generator = LeadWinGenerator(knowledge_base)
    lead = Lead(company="Northstar IT Partners", industry="Managed services", pain_points="Needs SQLs")

    report = generate_markdown_report([generator.generate(lead)])

    assert "# LeadWin RAG Report" in report
    assert "Retrieved agency context" in report
    assert "data/sample_knowledge" in report


def test_service_generates_vercel_payload_from_text() -> None:
    leads_csv = (
        "company,industry,contact_name,pain_points\n"
        "CloudFence,SaaS cybersecurity,Maya Patel,Needs qualified demos\n"
    )
    knowledge = [
        (
            "playbook.md",
            "A cybersecurity SaaS client booked 31 qualified demos with trigger-based outbound.",
        )
    ]

    leads = load_leads_from_csv_text(leads_csv)
    payload = generate_response_payload(knowledge, leads_csv, top_k=2)

    assert leads[0].company == "CloudFence"
    assert payload["plans"][0]["fit_score"] > 0
    assert "CloudFence" in payload["markdown_report"]
