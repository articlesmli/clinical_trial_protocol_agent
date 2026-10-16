import os
from src.guardrails import ProtocolGuardrails
from src.evaluation import ProductionEvaluator

def test_guardrails():
    print("--- Testing Guardrails ---")
    # Test input sanitization
    dirty_text = "  Patient must have normal ECG \x00 with no prior history...  "
    clean_text = ProtocolGuardrails.sanitize_input_text(dirty_text)
    print(f"Sanitized input: '{clean_text}'")

    # Test clinical boundary validation
    safe_criteria = ["Age >= 18 years", "Confirmed diagnosis of Type 2 Diabetes"]
    unsafe_criteria = ["Guaranteed 100% cure-all therapy", "Age >= 18"]
    
    print("Safe criteria check:", ProtocolGuardrails.validate_clinical_boundaries(safe_criteria))
    print("Unsafe criteria check:", ProtocolGuardrails.validate_clinical_boundaries(unsafe_criteria))

def test_evaluator():
    print("\n--- Testing Production Evaluator (Similarity Metrics) ---")
    # Mock some ChromaDB cosine distances
    mock_distances = [0.15, 0.20, 0.25]
    metrics = ProductionEvaluator.track_similarity_metrics(mock_distances)
    print(f"Similarity telemetry metrics: {metrics}")

if __name__ == "__main__":
    test_guardrails()
    test_evaluator()
    print("\n✅ Guardrails and evaluation components loaded and ready!")
