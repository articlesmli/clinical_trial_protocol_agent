# run_agent.py
import os
import json
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from src.workflow import app
from src.matching_engine import TrialMatchingEngine

# Load environment variables (e.g., ANTHROPIC_API_KEY)
load_dotenv()

def run_clinical_pipeline(pdf_path: str):
    print("==================================================")
    print("   CLINICAL TRIAL PROTOCOL AGENT PIPELINE         ")
    print("==================================================")
    
    if os.getenv("ANTHROPIC_API_KEY"):
        print("[System] Anthropic API Key successfully loaded.")
    else:
        print("[System] Warning: ANTHROPIC_API_KEY not found. Running in offline/fallback mode.")

    print(f"[System] Target Protocol PDF: {pdf_path}\n")

    # 1. Initialize the LangGraph workflow state
    initial_state = {
        "trial_id": "NCT_104XB",
        "trial_title": "Phase Protocol 104XB Evaluation",
        "pdf_path": pdf_path,
        "therapeutic_area": "Oncology",
        "extracted_criteria": [],
        "search_results": [],
        "compliance_report": "",
        "messages": [HumanMessage(content=f"Analyze protocol {pdf_path} for trial eligibility criteria.")]
    }
    
    print("=== Phase 1: Running Extraction, Validation & Indexing Workflow ===")
    final_state = app.invoke(initial_state)
    print(f"Workflow completed with compliance status: {final_state.get('compliance_status')}")

    # 2. Run AI Patient Matching Engine against the indexed database
    print("\n=== Phase 2: Running AI Patient Matching Engine ===")
    matching_engine = TrialMatchingEngine()
    
    # Sample mock patient profile to test eligibility
    sample_patient = {
        "age": 50,
        "diagnosis": "Solid tumor malignancy",
        "metastases": "None active",
        "autoimmune_history": "None"
    }

    if os.getenv("ANTHROPIC_API_KEY"):
        match_result = matching_engine.evaluate_patient_fit(sample_patient, trial_id="NCT_104XB")
        print("\n--- Patient Eligibility Evaluation Result ---")
        print(json.dumps(match_result, indent=2))
    else:
        print("[System] Skipping patient matching evaluation: ANTHROPIC_API_KEY missing.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run Clinical Trial Protocol Agent pipeline.")
    parser.add_argument(
        "--pdf", 
        type=str, 
        default="protocols/trial_protocol_104xb.pdf",
        help="Path to the clinical trial protocol PDF file."
    )
    args = parser.parse_args()
    
    run_clinical_pipeline(args.pdf)

