"""Command-line interface for LeadWin RAG."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from .generator import LeadWinGenerator
from .models import Lead
from .store import KnowledgeBase


def load_leads(path: Path) -> list[Lead]:
    """Load leads from a CSV file."""

    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        leads = [Lead.from_row(row) for row in reader]

    return [lead for lead in leads if lead.company or lead.website or lead.description]


def generate_markdown_report(plans: list, title: str = "LeadWin RAG Report") -> str:
    """Render a full report from generated lead plans."""

    sections = [f"# {title}", ""]
    for plan in plans:
        sections.append(plan.to_markdown())
    return "\n\n".join(sections).strip() + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate lead qualification and outreach plans with a local RAG knowledge base."
    )
    parser.add_argument(
        "--knowledge",
        nargs="+",
        required=True,
        help="One or more Markdown, text, CSV files, or directories containing agency knowledge.",
    )
    parser.add_argument("--leads", required=True, help="CSV file containing prospect leads.")
    parser.add_argument(
        "--output",
        default="outputs/leadwin_report.md",
        help="Markdown report path. Defaults to outputs/leadwin_report.md.",
    )
    parser.add_argument("--top-k", type=int, default=4, help="Number of knowledge chunks to retrieve.")
    parser.add_argument("--chunk-size", type=int, default=900, help="Words per knowledge chunk.")
    parser.add_argument("--overlap", type=int, default=120, help="Overlapping words between chunks.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    knowledge_base = KnowledgeBase.from_paths(
        args.knowledge,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
    )
    generator = LeadWinGenerator(knowledge_base, top_k=args.top_k)
    leads = load_leads(Path(args.leads))
    plans = [generator.generate(lead) for lead in leads]

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(generate_markdown_report(plans), encoding="utf-8")

    print(f"Generated {len(plans)} lead plans at {output}")


if __name__ == "__main__":
    main()
