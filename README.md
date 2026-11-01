# Clinical Trial Protocol Agent

An asynchronous multi-tool agent framework built with **FastAPI**, **LangGraph**, **Pydantic**, **Anthropic Claude 3.5 Sonnet / 5.5**, **ChromaDB**, and **Sentence Transformers** designed to parse dense clinical trial protocols, extract patient inclusion/exclusion criteria, enforce strict safety guardrails, evaluate output faithfulness, and execute downstream AI-powered patient-trial eligibility matching through both a CLI and an interactive web UI.

---

## Project Directory Structure

```text
clinical_trial_protocol_agent/          # Project root folder & Virtual Environment
  ├── .github/
  │   └── workflows/
  │       └── ci-cd.yml                 # GitHub Actions CI/CD pipeline automation
  ├── protocols/                        # Directory for input clinical trial PDFs
  │   └── trial_protocol_104xb.pdf
  ├── data/                             # Persistent storage directory
  │   ├── indexed_trials.json           # Saved structured clinical trial records
  │   └── fine_tuning/                  # Curated offline SFT datasets (ClinicalTrials.gov)
  ├── scripts/
  │   └── generate_fine_tuning_data.py  # Offline dataset curation script for fine-tuning
  ├── src/                              # Source code package
  │   ├── __init__.py
  │   ├── schemas.py                    # Pydantic models & state definitions (`CriteriaExtractionOutput`, `ProtocolState`)
  │   ├── prompts.py                    # Modular prompt templates & dynamic builders (Oncology, Cardiology, Corrections)
  │   ├── tools.py                      # PDF parsing (`pypdf`), Anthropic extraction (`claude-sonnet-5-5`), & ChromaDB vector search
  │   ├── guardrails.py                 # Clinical boundary validation & input sanitisation
  │   ├── evaluation.py                 # LLM-as-a-judge faithfulness scoring & production telemetry
  │   ├── database.py                   # Persistent JSON storage layer and keyword search engine
  │   ├── matching_engine.py            # AI-powered patient eligibility evaluation using Claude & structured output
  │   └── workflow.py                   # LangGraph stateful nodes and graph compilation (`extractor` -> `validator` -> `corrector` -> `downstream_indexer`)
  ├── .env                              # Environment variables (`ANTHROPIC_API_KEY`, `HF_TOKEN`)
  ├── .gitignore                        # Files to ignore (bin/, lib/, __pycache__, .env)
  ├── Dockerfile                        # Multi-stage containerisation blueprint for production deployment
  ├── requirements.txt                  # Project dependencies with version pinning & sentence-transformers
  ├── main.py                           # CLI entry point running Phase 1 (Extraction Workflow) & Phase 2 (Patient Matching)
  ├── web_app.py                        # FastAPI web interface application for browser-based PDF uploads & RAG execution
  ├── LICENSE                           # MIT License file
  └── README.md                         # Project documentation

```

---

## Key Components

* **`src/schemas.py`**: Defines the shared state schema (`ProtocolState`) and structured Pydantic models (`CriteriaExtractionOutput`, `PatientMatchOutput`) for reliable LLM parsing.
* **`src/prompts.py`**: Houses dynamic prompt engineering utilities. Offers specialised baseline knowledge templates for specific therapeutic fields alongside structured formatting recipes for self-correction loops.
* **`src/tools.py`**: Implements real PDF extraction via `pypdf`, criteria extraction using Anthropic's model via JSON schema structured outputs, and semantic similarity search over historical trial precedents using `ChromaDB`.
* **`src/guardrails.py`**: Enforces clinical safety constraints, boundary checks, and input sanitisation to ensure patient safety and protocol adherence.
* **`src/evaluation.py`**: Leverages Anthropic Claude as a judge to measure factual faithfulness, prevent hallucinations, and track vector distance telemetry.
* **`src/database.py`**: Handles persistent disk-based CRUD operations and fast keyword-based searching across indexed protocol criteria (`data/indexed_trials.json`).
* **`src/matching_engine.py`**: Evaluates real patient profiles against stored trial protocols using Claude 3.5 Sonnet structured output to determine eligibility status, rationales, and violated criteria.
* **`src/workflow.py`**: Configures the stateful LangGraph workflow, orchestrating execution from extraction through validation, automated self-correction, and automatic downstream database indexing.
* **`scripts/generate_fine_tuning_data.py`**: Curation script for filtering and building high-quality offline Supervised Fine-Tuning (SFT) datasets from ClinicalTrials.gov records.
* **`web_app.py`**: FastAPI server providing a robust browser-based upload interface (`/upload-and-analyze/`) and interactive Swagger UI documentation.
* **`Dockerfile`**: Provides a secure, optimised multi-stage Python runtime container image.
* **`.github/workflows/ci-cd.yml`**: Automates multi-stage build tests and validation checks via GitHub Actions.
* **`main.py`**: The clean executable command-line interface entry point that runs the extraction pipeline (Phase 1) and immediately chains it into the AI patient-matching engine (Phase 2).

---

## Advanced Architecture: Two-Phase Pipeline & Reflection

The framework utilises an adaptive layout spanning protocol ingestion, web-based interaction, and downstream clinical intelligence:

### Phase 1: Ingestion & Self-Correction Workflow

1. **Therapeutic Specialisation & RAG:** The pipeline extracts text dynamically and pulls relevant historical context via ChromaDB.
2. **Guardrails & Evaluation:** Extracted criteria are tested against clinical boundaries and checked for factual faithfulness.
3. **Self-Correction (Critique Loop):** If validation fails, LangGraph triggers a feedback loop back to the corrector node to re-prompt the LLM with error telemetry.
4. **Persistent Indexing:** Successfully validated protocols are automatically written to disk via `database.py`.

### Phase 2: AI Patient Eligibility Matching

1. **Structured Retrieval:** Pulls verified trial inclusion and exclusion rules from persistent storage.
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
pip install fastapi uvicorn python-multipart

```

### 3. Configure Environment Variables

Create a root **`.env`** file and add your credentials:

```env
ANTHROPIC_API_KEY=sk-ant-api03-...
HF_TOKEN=hf_your_hugging_face_token_here

```

### 4. Run via Command Line (CLI)

```bash
python main.py --pdf protocols/trial_protocol_104xb.pdf

```

### 5. Run via Web Interface (FastAPI)

Start the FastAPI server:

```bash
python web_app.py

```

Then, open your browser and navigate to **`http://127.0.0.1:8000/docs`** to access the interactive Swagger UI where researchers can upload protocol PDFs and trigger the RAG pipeline directly.

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

## License

This project is licensed under the **MIT License**.
