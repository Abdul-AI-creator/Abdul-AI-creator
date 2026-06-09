"""Streamlit app for the LeadWin RAG system."""

from __future__ import annotations

import tempfile
from pathlib import Path

import streamlit as st

from leadwin_rag.cli import load_leads
from leadwin_rag.generator import LeadWinGenerator
from leadwin_rag.store import KnowledgeBase


st.set_page_config(page_title="LeadWin RAG", layout="wide")
st.title("LeadWin RAG")
st.caption("Generate qualified lead plans, outreach, and proposal angles from agency knowledge.")

with st.sidebar:
    st.header("Inputs")
    knowledge_files = st.file_uploader(
        "Upload agency knowledge",
        type=["md", "txt", "csv"],
        accept_multiple_files=True,
        help="Use case studies, offers, ICP docs, objection handling notes, testimonials, and service pages.",
    )
    leads_file = st.file_uploader(
        "Upload leads CSV",
        type=["csv"],
        help="Recommended columns: company, industry, website, contact_name, title, pain_points, description.",
    )
    top_k = st.slider("Retrieved context chunks", min_value=1, max_value=8, value=4)
    run_button = st.button("Generate lead plans", type="primary")


def _persist_uploads(uploaded_files, directory: Path) -> list[Path]:
    paths: list[Path] = []
    for uploaded in uploaded_files:
        path = directory / uploaded.name
        path.write_bytes(uploaded.getvalue())
        paths.append(path)
    return paths


if not knowledge_files or not leads_file:
    st.info("Upload agency knowledge files and a leads CSV to start.")
    st.markdown(
        """
        **What the system produces**

        - Prospect fit score
        - Retrieved agency context with citations
        - Personalized outbound email
        - Discovery questions
        - Proposal angles and next steps
        """
    )
elif run_button:
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        knowledge_paths = _persist_uploads(knowledge_files, tmp_path)
        leads_path = tmp_path / leads_file.name
        leads_path.write_bytes(leads_file.getvalue())

        knowledge_base = KnowledgeBase.from_paths(knowledge_paths)
        generator = LeadWinGenerator(knowledge_base, top_k=top_k)
        leads = load_leads(leads_path)
        plans = [generator.generate(lead) for lead in leads]

    st.success(f"Generated {len(plans)} lead plans.")
    summary = [
        {
            "Company": plan.lead.company,
            "Industry": plan.lead.industry,
            "Contact": plan.lead.contact_name,
            "Fit Score": plan.fit_score,
            "Top Source": plan.retrieved_context[0].source if plan.retrieved_context else "",
        }
        for plan in plans
    ]
    st.dataframe(summary, use_container_width=True, hide_index=True)

    for plan in plans:
        with st.expander(f"{plan.lead.company or 'Unknown company'} - {plan.fit_score}/100", expanded=False):
            st.markdown(plan.to_markdown())
