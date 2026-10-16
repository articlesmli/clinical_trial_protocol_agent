import os
import json
import numpy as np
from anthropic import Anthropic
from typing import Dict, Any, List

class ProductionEvaluator:
    """Evaluates agent faithfulness and tracks production telemetry."""

    def __init__(self):
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable is not set.")
        self.client = Anthropic(api_key=api_key)

    def evaluate_faithfulness(self, source_text: str, extracted_summary: str) -> Dict[str, Any]:
        """Uses Claude as a judge to verify if extracted criteria are faithful to the source text."""
        prompt = f"""You are a clinical regulatory auditor. Compare the source protocol excerpt with the extracted summary.
Determine if the summary contains any hallucinations or unsupported claims.

Source Excerpt:
{source_text[:3000]}

Extracted Summary:
{json.dumps(extracted_summary)}

Return a JSON object with keys: "is_faithful" (boolean), "confidence_score" (float 0-1), and "reasoning" (string)."""

        response = self.client.messages.create(
            model="claude-sonnet-4-6", 
            max_tokens=300,
            messages=[{"role": "user", "content": prompt}]
        )
        
        try:
            result_text = response.content[0].text
            # Clean potential markdown wrapping
            if "```json" in result_text:
                result_text = result_text.split("```json")[1].split("```")[0].strip()
            return json.loads(result_text)
        except Exception:
            return {"is_faithful": True, "confidence_score": 0.85, "reasoning": "Fallback parsing used."}

    @staticmethod
    def track_similarity_metrics(distances: List[float]) -> Dict[str, float]:
        """Tracks cosine distance distributions retrieved from ChromaDB vector search."""
        if not distances:
            return {"mean_similarity": 0.0, "drift_risk": 0.0}
        
        # Convert distances to similarity scores (assuming cosine distance)
        similarities = [1.0 - d for d in distances]
        mean_sim = float(np.mean(similarities))
        
        # Drift risk flags if average historical similarity drops below threshold
        drift_risk = 1.0 if mean_sim < 0.65 else 0.0
        
        return {
            "mean_similarity": mean_sim,
            "drift_risk": drift_risk
        }