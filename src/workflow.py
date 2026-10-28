# src/workflow.py
import os
from langgraph.graph import StateGraph, END
from src.schemas import ProtocolState, CriteriaExtractionOutput
from src.guardrails import ProtocolGuardrails
from src.evaluation import ProductionEvaluator
from src.tools import extract_criteria_with_rag
from src.prompts import build_correction_prompt
from langchain_anthropic import ChatAnthropic

def extractor_node(state: ProtocolState) -> ProtocolState:
    """Parses text dynamically using therapeutic fields and RAG context states."""
    pdf_path = state.get("pdf_path", "")
    therapeutic_area = state.get("therapeutic_area", "General")
    historical_examples = state.get("historical_examples", "")

    # Extract criteria using dynamic prompts
    result, context_text = extract_criteria_with_rag(
        pdf_path=pdf_path,
        therapeutic_area=therapeutic_area,
        historical_examples=historical_examples
    )
    
    # Store extraction and textual context back into state
    state["criteria"] = {
        "inclusion_criteria": result.inclusion_criteria,
        "exclusion_criteria": result.exclusion_criteria
    }
    state["raw_text"] = context_text
    return state

def validator_node(state: ProtocolState) -> ProtocolState:
    """LangGraph node enforcing guardrails and LLM-as-a-judge evaluation."""
    criteria = state.get("criteria", {})
    raw_source = state.get("raw_text", "")
    
    # 1. Run Guardrails validation
    inclusion_check = ProtocolGuardrails.validate_clinical_boundaries(criteria.get("inclusion_criteria", []))
    exclusion_check = ProtocolGuardrails.validate_clinical_boundaries(criteria.get("exclusion_criteria", []))
    
    # Track discrete error metrics for prompt correction loops
    errors = []
    if not inclusion_check["passed"]:
        errors.append(f"Inclusion boundaries failed: {inclusion_check.get('reason')}")
    if not exclusion_check["passed"]:
        errors.append(f"Exclusion boundaries failed: {exclusion_check.get('reason')}")

    if errors:
        state["compliance_status"] = "FLAGGED_BY_GUARDRAILS"
        state["validation_errors"] = "; ".join(errors)
        return state

    # 2. Run LLM-as-a-Judge Faithfulness Check
    evaluator = ProductionEvaluator()
    faithfulness_result = evaluator.evaluate_faithfulness(raw_source, criteria)
    state["faithfulness_score"] = faithfulness_result.get("confidence_score", 1.0)
    
    if not faithfulness_result.get("is_faithful", True):
        state["compliance_status"] = "FAILED_FAITHFULNESS_CHECK"
        state["validation_errors"] = "Factual discrepancies/hallucinations detected by evaluation judge."
    else:
        state["compliance_status"] = "PASSED"
        state["validation_errors"] = ""
        
    return state

def corrector_node(state: ProtocolState) -> ProtocolState:
    """Self-correction graph node modifying prompts based on validation failures."""
    print("[Node: Corrector] Correcting bad outputs using validator feedback telemetry...")
    raw_text = state.get("raw_text", "")
    errors = state.get("validation_errors", "")
    criteria = state.get("criteria", {})
    
    if os.getenv("ANTHROPIC_API_KEY"):
        try:
            llm = ChatAnthropic(model="claude-sonnet-5-5")
            structured_llm = llm.with_structured_output(CriteriaExtractionOutput, method="json_schema")
            
            # Dynamic revision prompt generation
            correction_prompt = build_correction_prompt(
                current_text=raw_text,
                errors=errors,
                feedback=f"Confidence baseline: {state.get('faithfulness_score', 0.0)}",
                criteria=criteria
            )
            
            corrected_output = structured_llm.invoke(correction_prompt)
            if corrected_output:
                state["criteria"] = {
                    "inclusion_criteria": corrected_output.inclusion_criteria,
                    "exclusion_criteria": corrected_output.exclusion_criteria
                }
                print("[Node: Corrector] Generated updated structural definitions.")
        except Exception as e:
            print(f"[Error] Reflection phase failed: {e}")
            
    return state

def routing_edge(state: ProtocolState):
    """Evaluates validation state and returns target edge node names."""
    status = state.get("compliance_status")
    # If it failed guardrails or faithfulness check, route it into corrector
    if status in ["FLAGGED_BY_GUARDRAILS", "FAILED_FAITHFULNESS_CHECK"]:
        return "corrector"
    return END

# Initialize the LangGraph StateGraph
workflow = StateGraph(ProtocolState)

# Add nodes to graph layout
workflow.add_node("extractor", extractor_node)
workflow.add_node("validator", validator_node)
workflow.add_node("corrector", corrector_node)

# Set up flow control rules
workflow.set_entry_point("extractor")
workflow.add_edge("extractor", "validator")

# Conditional path checking whether corrections are needed
workflow.add_conditional_edges(
    "validator",
    routing_edge,
    {
        "corrector": "corrector",
        END: END
    }
)

# Loop corrector adjustments straight back to validator to ensure quality control
workflow.add_edge("corrector", "validator")

# Compile graph into application object
app = workflow.compile()
