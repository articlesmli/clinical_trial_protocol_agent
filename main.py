import os
import argparse
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from src.workflow import app

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

    initial_state = {
        "pdf_path": args.pdf,
        "extracted_criteria": [],
        "search_results": [],
        "compliance_report": "",
        "messages": [HumanMessage(content=f"Analyze protocol {args.pdf} for trial eligibility criteria.")]
    }
    
    final_state = app.invoke(initial_state)
    print("\n=== FINAL OUTPUT ===")
    print(final_state["compliance_report"])