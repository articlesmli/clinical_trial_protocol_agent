Here is the updated `README.md` containing the new **📈 Future Roadmap** section added near the end of the file:

```markdown
# Clinical Trial Protocol Agent

An asynchronous multi-tool agent framework built with **FastAPI**, **LangGraph**, **Pydantic**, **Anthropic Claude**, **ChromaDB**, and **Sentence Transformers** designed to parse dense clinical trial protocols, extract patient inclusion/exclusion criteria, enforce strict safety guardrails, evaluate output faithfulness, orchestrate **human-in-the-loop (HITL) clinical reviews**, and execute downstream AI-powered patient-trial eligibility matching through both a CLI and an interactive web UI.

---

## Project Directory Structure

```text
clinical_trial_protocol_agent/          # Project root folder & Virtual Environment
  ├── .github/
  │   └── workflows/
  │       └── ci-cd.yml                 # GitHub Actions CI/CD pipeline automation
  ├── protocols/                        # Directory for input clinical trial PDFs
  ├── data/                             # Persistent storage directory
  │   └── indexed_trials.json           # Saved structured clinical trial records
  ├── src/                              # Source code package
  │   ├── __init__.py
  │   ├── schemas.py                    # Pydantic models & state definitions (`CriteriaExtractionOutput`, `ProtocolState`)
  │   ├── prompts.py                    # Modular prompt templates & dynamic builders
  │   ├── tools.py                      # PDF parsing (`pypdf`), Anthropic extraction, & ChromaDB vector search
  │   ├── guardrails.py                 # Clinical boundary validation & input sanitisation
  │   ├── evaluation.py                 # LLM-as-a-judge faithfulness scoring & production telemetry
  │   ├── database.py                   # Persistent JSON storage layer and keyword search engine
  │   ├── matching_engine.py            # AI-powered patient eligibility evaluation using Claude & structured output
  │   └── workflow.py                   # LangGraph stateful nodes, human review gating, and graph compilation
  ├── .env                              # Environment variables (`ANTHROPIC_API_KEY`, `HF_TOKEN`)
  ├── .gitignore                        # Files to ignore
  ├── Dockerfile                        # Multi-stage containerisation blueprint for production deployment
  ├── requirements.txt                  # Project dependencies with version pinning & sentence-transformers
  ├── run_agent.py                      # CLI entry point running Phase 1 (Extraction Workflow) & Phase 2 (Patient Matching)
  ├── web_app.py                        # FastAPI web interface application for browser-based uploads, RAG execution, and HITL endpoints
  ├── LICENSE                           # MIT License file
  └── README.md                         # Project documentation

```

---

## Key Components

* **`src/schemas.py`**: Defines the shared state schema (`ProtocolState` including `review_status` and `reviewer_notes`) and structured Pydantic models (`CriteriaExtractionOutput`, `PatientMatchOutput`).
* **`src/prompts.py`**: Houses dynamic prompt engineering utilities and specialized baseline knowledge templates.
* **`src/tools.py`**: Implements real PDF extraction via `pypdf`, criteria extraction using Anthropic's model via JSON schema structured outputs, and semantic similarity search over historical trial precedents using `ChromaDB`.
* **`src/guardrails.py`**: Enforces clinical safety constraints, boundary checks, and input sanitisation to ensure patient safety and protocol adherence.
* **`src/evaluation.py`**: Leverages Anthropic Claude as a judge to measure factual faithfulness, prevent hallucinations, and track vector distance telemetry.
* **`src/database.py`**: Handles persistent disk-based CRUD operations and fast keyword-based searching across indexed protocol criteria (`data/indexed_trials.json`).
* **`src/matching_engine.py`**: Evaluates real patient profiles against stored trial protocols using Claude structured output to determine eligibility status, rationales, and violated criteria.
* **`src/workflow.py`**: Configures the stateful LangGraph workflow, orchestrating execution from extraction through validation, automated self-correction, **human clinical review gating**, and automatic downstream database indexing.
* **`web_app.py`**: FastAPI server providing browser-based upload endpoints (`/upload-and-analyze/`), pending review tracking (`/protocols/pending-reviews`), clinical sign-off processing (`/protocols/{trial_id}/review`), and interactive Swagger UI documentation.
* **`Dockerfile`**: Provides a secure, optimised multi-stage Python runtime container image.
* **`.github/workflows/ci-cd.yml`**: Automates multi-stage build tests and validation checks via GitHub Actions.
* **`run_agent.py`**: The clean executable command-line interface entry point that runs the extraction pipeline (Phase 1) and immediately chains it into the AI patient-matching engine (Phase 2).

---

## Advanced Architecture: Two-Phase Pipeline, Reflection & Human-in-the-Loop Review

The framework utilises an adaptive layout spanning protocol ingestion, web-based interaction, and downstream clinical intelligence:

### Phase 1: Ingestion, Self-Correction & Clinical Review Workflow

1. **Therapeutic Specialisation & RAG:** The pipeline extracts text dynamically and pulls relevant historical context via ChromaDB.
2. **Guardrails & Evaluation:** Extracted criteria are tested against clinical boundaries and checked for factual faithfulness.
3. **Self-Correction (Critique Loop):** If automated validation fails, LangGraph triggers a feedback loop back to the corrector node to re-prompt the LLM with error telemetry.
4. **Human-in-the-Loop (HITL) Review:** Once automated checks pass, the workflow pauses at the `human_review` node (`PENDING_HUMAN_REVIEW`), allowing licensed clinicians to inspect and sign off on criteria via FastAPI endpoints.
5. **Persistent Indexing:** Approved protocols are automatically committed to disk via `database.py`.

### Phase 2: AI Patient Eligibility Matching

1. **Structured Retrieval:** Pulls verified and human-approved trial inclusion and exclusion rules from persistent storage.
2. **LLM-Driven Screening:** Cross-references complex patient profiles against protocol specifications using Pydantic structured outputs to produce clear eligibility determinations, decision rationales, and violation flags.

---

## Getting Started

### 1. Set Up the Virtual Environment (Local Development)

```bash
python3.10 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install --upgrade pip

```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
pip install fastapi uvicorn python-multipart python-dotenv

```

### 3. Configure Environment Variables

Create a root **`.env`** file and add your credentials:

```env
ANTHROPIC_API_KEY=sk-ant-api03-...
HF_TOKEN=hf_your_hugging_face_token_here

```

### 4. Run via Command Line (CLI)

```bash
python run_agent.py --pdf protocols/trial_protocol_104xb.pdf

```

### 5. Run via Web Interface (FastAPI & HITL Review)

Start the FastAPI server:

```bash
python web_app.py

```

Then, open your browser and navigate to **`http://127.0.0.1:8000/docs`** to access the interactive Swagger UI where researchers can:

* Upload protocol PDFs (`POST /upload-and-analyze/`)
* Check pending clinical reviews (`GET /protocols/pending-reviews`)
* Submit clinical approvals or rejections (`POST /protocols/{trial_id}/review`)

---

## Production Deployment & Containerisation

To run the agent fully sandboxed inside a secure **Docker container**:

### 1. Build the Docker Image

```bash
docker build -t clinical-trial-agent .

```

### 2. Run the Containerised Agent

```bash
docker run --rm --env-file .env clinical-trial-agent

```

---

## Future Roadmap

To optimise the PENDING_HUMAN_REVIEW bottleneck over time, the human-in-the-loop gate will be used to collect a gold-standard dataset, which will then be used to deploy specialised AI agents that pre-screen the data before it ever reaches the human reviewer. Thus, the human reviewers will become more efficient, and the automation becomes more precise.

When this data collection process runs over months, it will create a structured "Audit Trail and Feedback Dataset" of JSON objects containing the primary parts: the Context, the Discrepancy (Before vs. After), and the Root Cause.

Once hundreds of these logs are collected, it will be possible to generate a Performance Dashboard that exposes systematic flaws in the primary extraction pipeline:
### The Error Heatmap
A statistical breakdown showing exactly where the primary AI agent struggles.
### The Edge-Case Repository
A curated list of highly complex medical paragraphs that the AI agent repeatedly misinterprets.
### The "Gold Standard" Evaluation Set
A collection of 100% human-vetted protocol extractions.

---

## License

This project is licensed under the **MIT License**.

