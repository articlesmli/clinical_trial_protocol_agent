from langgraph.graph import StateGraph
from src.schemas import ProtocolState
from src.guardrails import ProtocolGuardrails
from src.evaluation import ProductionEvaluator

# DO NOT instantiate ProductionEvaluator globally here anymore

def extractor_node(state: ProtocolState) -> ProtocolState:
    """Placeholder or execution node for protocol parsing."""
    return state

def validator_node(state: ProtocolState) -> ProtocolState:
    """LangGraph node enforcing guardrails and LLM-as-a-judge evaluation."""
    criteria = state.get("criteria", {})
    raw_source = state.get("raw_text", "")
    
    # 1. Run Guardrails validation
    inclusion_check = ProtocolGuardrails.validate_clinical_boundaries(criteria.get("inclusion_criteria", []))
    exclusion_check = ProtocolGuardrails.validate_clinical_boundaries(criteria.get("exclusion_criteria", []))
    
    if not inclusion_check["passed"] or not exclusion_check["passed"]:
        state["compliance_status"] = "FLAGGED_BY_GUARDRAILS"
        return state

    # 2. Run LLM-as-a-Judge Faithfulness Check (Initialized lazily here so env vars are loaded)
    evaluator = ProductionEvaluator()
    faithfulness_result = evaluator.evaluate_faithfulness(raw_source, criteria)
    state["faithfulness_score"] = faithfulness_result.get("confidence_score", 1.0)
    
    if not faithfulness_result.get("is_faithful", True):
        state["compliance_status"] = "FAILED_FAITHFULNESS_CHECK"
    else:
        state["compliance_status"] = "PASSED"
        
    return state

# Initialize the LangGraph StateGraph
workflow = StateGraph(ProtocolState)

# Add nodes to the graph
workflow.add_node("extractor", extractor_node)
workflow.add_node("validator", validator_node)

# Set graph flow edges
workflow.set_entry_point("extractor")
workflow.add_edge("extractor", "validator")
workflow.set_finish_point("validator")

# Compile the graph into 'app'
app = workflow.compile()