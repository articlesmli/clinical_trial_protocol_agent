# src/matching_engine.py
import json
from typing import Dict, Any
from pydantic import BaseModel, Field
from langchain_anthropic import ChatAnthropic
from database import TrialDatabase

class PatientMatchOutput(BaseModel):
    eligibility_status: str = Field(description="Must be 'ELIGIBLE', 'INELIGIBLE', or 'UNCLEAR/REQUIRES_REVIEW'")
    matching_rationale: list[str] = Field(description="Detailed bullet points explaining the match rationale.")
    violated_criteria: list[str] = Field(description="List of triggered exclusion criteria or missed inclusion criteria.")

class TrialMatchingEngine:
    """Handles AI-powered patient eligibility matching using the trial database."""
    def __init__(self, db_path: str = "data/indexed_trials.json"):
        self.db = TrialDatabase(storage_path=db_path)

    def evaluate_patient_fit(self, patient_profile: Dict[str, Any], trial_id: str) -> Dict[str, Any]:
        trial = self.db.get_trial(trial_id)
        if not trial:
            return {"status": "error", "message": f"Trial {trial_id} not found."}

        llm = ChatAnthropic(model="claude-3-5-sonnet-20241022", temperature=0)
        structured_llm = llm.with_structured_output(PatientMatchOutput, method="json_schema")

        prompt = f"""
        Evaluate whether the patient meets all inclusion criteria and violates none of the exclusion criteria.
        --- TRIAL: {trial['title']} ({trial_id}) ---
        Inclusion: {json.dumps(trial['inclusion_criteria'], indent=2)}
        Exclusion: {json.dumps(trial['exclusion_criteria'], indent=2)}
        --- PATIENT ---
        {json.dumps(patient_profile, indent=2)}
        """

        try:
            result = structured_llm.invoke(prompt)
            return {
                "trial_id": trial_id,
                "trial_title": trial["title"],
                "eligibility_status": result.eligibility_status,
                "matching_rationale": result.matching_rationale,
                "violated_criteria": result.violated_criteria
            }
        except Exception as e:
            return {"status": "error", "message": str(e)}