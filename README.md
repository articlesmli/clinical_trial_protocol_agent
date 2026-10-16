# Clinical Trial Protocol Agent

An asynchronous multi-tool agent framework built with **LangGraph**, **Pydantic**, **Anthropic Claude 3.5 Sonnet / 5.5**, **ChromaDB**, and **Sentence Transformers** designed to parse dense clinical trial protocols, extract patient inclusion/exclusion criteria, query vector database precedents, and run compliance audits.

---

## Project Directory Structure

```text
clinical_trial_protocol_agent/          # Project root folder & Virtual Environment
  ├── .github/
  │   └── workflows/
  │       └── ci-cd.yml                 # GitHub Actions CI/CD pipeline automation
  ├── protocols/                        # Directory for input clinical trial PDFs
  │   └── trial_protocol_104xb.pdf
  ├── src/                              # Source code package
  │   ├── __init__.py
  │   ├── schemas.py                    # Pydantic models & state definitions (`CriteriaExtractionOutput`, `ProtocolState`)
  │   ├── tools.py                      # PDF parsing (`pypdf`), Anthropic extraction (`claude-sonnet-5-5`), & ChromaDB vector search
  │   └── workflow.py                   # LangGraph stateful nodes and graph compilation (`extractor` -> `vector_search` -> `validator`)
  ├── .env                              # Environment variables (`ANTHROPIC_API_KEY`, `HF_TOKEN`)
  ├── .gitignore                        # Files to ignore (bin/, lib/, __pycache__, .env)
  ├── Dockerfile                        # Multi-stage containerization blueprint for production deployment
  ├── requirements.txt                  # Project dependencies with version pinning & sentence-transformers
  ├── main.py                           # Entry point script to run the agent pipeline
  ├── LICENSE                           # MIT License file
  └── README.md                         # Project documentation


```

---

## Key Components

* **`src/schemas.py`**: Defines the shared state schema (`ProtocolState`) and structured Pydantic models (`CriteriaExtractionOutput`) for reliable LLM parsing.


* **`src/tools.py`**: Implements real PDF extraction via `pypdf`, criteria extraction using Anthropic's `claude-sonnet-5-5` model via JSON schema structured outputs, and semantic similarity search over historical trial precedents using `ChromaDB`, Hugging Face embeddings, and `sentence-transformers`.


* **`src/workflow.py`**: Configures the stateful LangGraph workflow, orchestrating sequential execution nodes from parsing to vector retrieval and compliance reporting.


* **`Dockerfile`**: Provides a secure, optimized multi-stage Python 3.10 runtime container image with non-privileged user execution.


* **`.github/workflows/ci-cd.yml`**: Automates multi-stage build tests and validation checks via GitHub Actions.


* **`main.py`**: The clean executable command-line interface entry point that sets up the graph and prints the final clinical compliance report.



---

## Getting Started

### 1. Set Up the Virtual Environment (Local Development)

```bash
python3.10 -m venv .
source bin/activate  # On Windows: Scripts\activate
pip install --upgrade pip


```

### 2. Install Dependencies

```bash
pip install -r requirements.txt


```

### 3. Configure Environment Variables

Create a root **`.env`** file and add your credentials:

```env
ANTHROPIC_API_KEY=sk-ant-usr-...
HF_TOKEN=hf_your_hugging_face_token_here


```

### 4. Run the Agent Locally

```bash
python main.py --pdf protocols/trial_protocol_104xb.pdf


```

---

## Production Deployment & Containerization

To run the agent fully sandboxed inside a secure **Docker container**:

### 1. Build the Docker Image

```bash
docker build -t clinical-trial-agent .


```

### 2. Run the Containerized Agent

```bash
docker run --rm --env-file .env clinical-trial-agent


```

---

## License

This project is licensed under the **MIT License**.