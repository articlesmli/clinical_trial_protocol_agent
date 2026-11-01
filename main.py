# main.py
import os
import json
import argparse
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from src.workflow import app
from src.matching_engine import TrialMatchingEngine

load_dotenv()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clinical Trial Protocol Compliance Agent")
    parser.add_argument(
        "--pdf", 
        type=str, 
        default="protocols/trial_protocol_104xb.pdf",
        help="Path to the clinical trial protocol PDF file."
    )
    args = parser.parse_args()

    # Check for Anthropic API key
    if os.getenv("ANTHROPIC_API_KEY"):
        print("[System] Anthropic API Key successfully loaded.")
    else:
        print("[System] Warning: ANTHROPIC_API_KEY not found. Running in offline/fallback mode.")

    print(f"[System] Target Protocol PDF: {args.pdf}")

    # Initial state populated with proper trial identifiers
    initial_state = {
        "trial_id": "NCT_104XB",
        "trial_title": "Phase Protocol 104XB Evaluation",
        "pdf_path": args.pdf,
        "extracted_criteria": [],
        "search_results": [],
        "compliance_report": "",
        "messages": [HumanMessage(content=f"Analyze protocol {args.pdf} for trial eligibility criteria.")]
    }
    
    print("\n=== Phase 1: Running Extraction & Validation Workflow ===")
    final_state = app.invoke(initial_state)
    print(f"Workflow completed with status: {final_state.get('compliance_status')}")

    print("\n=== Phase 2: Running AI Patient Matching Engine ===")
    matching_engine = TrialMatchingEngine()
    
    # Example patient profile to test against the extracted protocol criteria
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