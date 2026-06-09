I'm a machine learning engineer and data scientist specializing in artificial intelligence, predictive analytics, and intelligent automation. I help businesses leverage AI and data science to automate workflows, predict outcomes, and gain competitive advantages through custom ML solutions.

# LeadWin RAG

LeadWin RAG is a local-first retrieval augmented generation system for lead-generation agencies. It helps an agency turn its existing sales assets into personalized lead plans that can be used to qualify prospects, write outreach, prepare discovery calls, and win clients.

The system retrieves relevant agency knowledge from playbooks, case studies, offers, objection-handling notes, testimonials, and service pages. It then generates a grounded sales plan for each prospect in a leads CSV.

## What it generates

- Prospect fit score
- Rationale for why the lead is worth pursuing
- Retrieved agency context with source citations
- Personalized outbound email
- Discovery questions for the sales call
- Proposal angles and recommended next steps

## Project structure

```text
leadwin_rag/
  chunking.py      # Loads and chunks Markdown, text, and CSV knowledge files
  store.py         # Local TF-IDF vector store and retrieval
  generator.py     # Lead scoring, outreach, and proposal generation
  models.py        # Lead, retrieved context, and generated plan models
  cli.py           # Batch command-line workflow
  service.py       # Shared service helpers for APIs and web deployments
api/generate.py    # Vercel Python serverless endpoint
public/index.html  # Vercel browser UI
app.py             # Optional local Streamlit UI
data/
  sample_knowledge # Example agency playbook and case studies
  sample_leads.csv # Example prospects
tests/             # RAG pipeline tests
vercel.json        # Vercel routing and function config
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run the batch RAG workflow

```bash
python -m leadwin_rag.cli \
  --knowledge data/sample_knowledge \
  --leads data/sample_leads.csv \
  --output outputs/leadwin_report.md
```

The generated report is written to `outputs/leadwin_report.md`.

## Run the web UI

```bash
streamlit run app.py
```

Upload agency knowledge files and a leads CSV, then click **Generate lead plans**.

## Deploy on Vercel

The Vercel deployment uses `public/index.html` for the browser UI and `api/generate.py` for the Python serverless RAG endpoint.

Preview deploy:

```bash
npx vercel
```

Production deploy:

```bash
npx vercel --prod
```

If you deploy from the Vercel dashboard, import this repository and keep the default framework preset. Vercel will serve the static UI and the `/api/generate` function from the repository root.

## Leads CSV format

Recommended columns:

```csv
company,industry,website,contact_name,title,pain_points,description,location,employees,source
```

The loader also accepts common aliases such as `company_name`, `domain`, `role`, `job_title`, `notes`, `problems`, and `lead_source`.

## Knowledge base inputs

Supported formats:

- `.md`
- `.txt`
- `.csv`

Good source material includes:

- Case studies and client results
- ICP definitions
- Service offer documents
- Sales scripts
- Objection-handling notes
- Testimonials
- Industry-specific messaging examples

## Run tests

```bash
pytest
```

## Notes for production use

This implementation uses pure-Python TF-IDF retrieval so it can run locally and on Vercel without paid APIs or compiled ML dependencies. For larger production deployments, the `KnowledgeBase` class can be swapped for neural embeddings and a hosted vector database while keeping the lead-generation interface intact.
