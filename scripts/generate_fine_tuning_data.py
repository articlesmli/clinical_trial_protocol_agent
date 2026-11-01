# scripts/generate_fine_tuning_data.py
import os
import json
import requests
from typing import List, Dict, Any

def fetch_and_filter_trials(search_term: str = "Oncology", max_results: int = 5) -> List[Dict[str, Any]]:
    """
    Queries ClinicalTrials.gov API v2 with filters to avoid bad/withdrawn trials,
    extracts structured criteria text, and formats them for fine-tuning.
    """
    base_url = "https://clinicaltrials.gov/api/v2/studies"
    
    # Query parameters focusing on active/completed studies to ensure high data quality
    params = {
        "query.cond": search_term,
        "filter.overallStatus": "COMPLETED,RECRUITING,ACTIVE_NOT_RECRUITING",
        "pageSize": max_results
    }
    
    print(f"[Dataset Generator] Querying ClinicalTrials.gov for condition: '{search_term}'...")
    try:
        response = requests.get(base_url, params=params, timeout=15)
        response.raise_for_status()
        data = response.json()
    except Exception as e:
        print(f"[Error] Failed to fetch data from ClinicalTrials.gov: {e}")
        return []

    studies = data.get("studies", [])
    print(f"[Dataset Generator] Found {len(studies)} studies matching criteria.")
    
    training_records = []
    
    for study in studies:
        protocol_section = study.get("protocolSection", {})
        nct_id = protocol_section.get("identificationModule", {}).get("nctId", "UNKNOWN")
        title = protocol_section.get("identificationModule", {}).get("briefTitle", "")
        eligibility_module = protocol_section.get("eligibilityModule", {})
        
        raw_text = eligibility_module.get("eligibilityCriteria", "")
        
        # --- Filtering Phase (Quality Control) ---
        if not raw_text or len(raw_text.strip()) < 150:
            print(f"[Skip] {nct_id}: Criteria text too short or empty.")
            continue
            
        if "inclusion criteria" not in raw_text.lower() and "exclusion criteria" not in raw_text.lower():
            print(f"[Skip] {nct_id}: Lacks clear inclusion/exclusion section markers.")
            continue

        print(f"[Success] {nct_id}: Passed quality filters ('{title[:40]}...')")

        # Simple string-splitting helper to populate structured targets for fine-tuning
        inclusion_lines = []
        exclusion_lines = []
        current_bucket = inclusion_lines
        
        for line in raw_text.split("\n"):
            cleaned = line.strip()
            if not cleaned:
                continue
            if "exclusion criteria" in cleaned.lower():
                current_bucket = exclusion_lines
                continue
            if "inclusion criteria" in cleaned.lower():
                current_bucket = inclusion_lines
                continue
            if cleaned.startswith("-") or cleaned.startswith("•") or (cleaned and cleaned[0].isdigit()):
                current_bucket.append(cleaned.lstrip("-•0123456789. "))

        # Format into standard fine-tuning JSONL schema aligned with CriteriaExtractionOutput
        record = {
            "trial_id": nct_id,
            "title": title,
            "messages": [
                {
                    "role": "system", 
                    "content": "You are a clinical trial protocol extraction expert. Parse the raw eligibility criteria text into structured JSON containing inclusion and exclusion criteria."
                },
                {
                    "role": "user", 
                    "content": raw_text
                },
                {
                    "role": "assistant", 
                    "content": json.dumps({
                        "inclusion_criteria": inclusion_lines if inclusion_lines else [raw_text[:300]],
                        "exclusion_criteria": exclusion_lines
                    })
                }
            ]
        }
        training_records.append(record)

    return training_records

if __name__ == "__main__":
    output_dir = "data/fine_tuning"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "clinical_criteria_sft.jsonl")

    # Generate dataset from reliable studies offline
    dataset = fetch_and_filter_trials(search_term="Oncology", max_results=5)

    if dataset:
        with open(output_path, "w", encoding="utf-8") as f:
            for entry in dataset:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        print(f"\n[Dataset Generator] Saved {len(dataset)} curated training examples to {output_path}")
    else:
        print("[Dataset Generator] No valid records generated.")