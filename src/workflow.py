# src/workflow.py
import os
from langgraph.graph import StateGraph, END
from src.schemas import ProtocolState, CriteriaExtractionOutput
from src.guardrails import ProtocolGuardrails
from src.evaluation import ProductionEvaluator
from src.tools import extract_criteria_with_rag
from src.prompts import build_correction_prompt
from langchain_anthropic import ChatAnthropic
from src.database import TrialDatabase

# Initialize the persistent database instance
global_trial_database = TrialDatabase()

def extractor_node(state: ProtocolState) -> ProtocolState:
    """Parses text dynamically using therapeutic fields and RAG context states."""
    pdf_path = state.get("pdf_path", "")
    therapeutic_area = state.get("therapeutic_area", "General")
    historical_examples = state.get("historical_examples", "")

    result, context_text = extract_criteria_with_rag(
        pdf_path=pdf_path,
        therapeutic_area=therapeutic_area,
        historical_examples=historical_examples
    )
    
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
    
    inclusion_check = ProtocolGuardrails.validate_clinical_boundaries(criteria.get("inclusion_criteria", []))
    exclusion_check = ProtocolGuardrails.validate_clinical_boundaries(criteria.get("exclusion_criteria", []))
    
    errors = []
    if not inclusion_check["passed"]:
        errors.append(f"Inclusion boundaries failed: {inclusion_check.get('reason')}")
    if not exclusion_check["passed"]:
        errors.append(f"Exclusion boundaries failed: {exclusion_check.get('reason')}")

    if errors:
        state["compliance_status"] = "FLAGGED_BY_GUARDRAILS"
        state["validation_errors"] = "; ".join(errors)
        return state

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
            llm = ChatAnthropic(model="claude-3-5-sonnet-20241022")
            structured_llm = llm.with_structured_output(CriteriaExtractionOutput, method="json_schema")
            
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

def human_review_node(state: ProtocolState) -> ProtocolState:
    """Pauses the workflow for clinical expert validation before database commit."""
    print("[Node: Human Review] Pausing workflow for clinical expert sign-off...")
    if not state.get("review_status"):
        state["review_status"] = "PENDING_HUMAN_REVIEW"
    return state

def downstream_indexer_node(state: ProtocolState) -> ProtocolState:
    """Pushes verified, high-quality structured criteria into the downstream persistent database."""
    if state.get("compliance_status") == "PASSED" and state.get("review_status") == "APPROVED":
        trial_id = state.get("trial_id", "NCT_104XB")
        title = state.get("trial_title", "Clinical Trial Protocol 104XB")
        criteria = state.get("criteria", {})
        
        global_trial_database.add_trial(
            trial_id=trial_id,
            title=title,
            structured_criteria=criteria
        )
        print(f"[Node: Indexer] Successfully indexed verified trial {trial_id} into downstream database!")
        
    return state

def routing_edge(state: ProtocolState):
    """Evaluates validation state and returns target edge node names."""
    status = state.get("compliance_status")
    if status in ["FLAGGED_BY_GUARDRAILS", "FAILED_FAITHFULNESS_CHECK"]:
        return "corrector"
    return "human_review"

def review_routing_edge(state: ProtocolState):
    """Routes based on human review decision."""
    review_status = state.get("review_status")
    if review_status == "APPROVED":
        return "downstream_indexer"
    return END

# Initialize the LangGraph StateGraph
workflow = StateGraph(ProtocolState)

# Add nodes to graph layout
workflow.add_node("extractor", extractor_node)
workflow.add_node("validator", validator_node)
workflow.add_node("corrector", corrector_node)
workflow.add_node("human_review", human_review_node)
workflow.add_node("downstream_indexer", downstream_indexer_node)

# Set up flow control rules
workflow.set_entry_point("extractor")
workflow.add_edge("extractor", "validator")

workflow.add_conditional_edges(
    "validator",
    routing_edge,
    {
        "corrector": "corrector",
        "human_review": "human_review"
    }
)

workflow.add_conditional_edges(
    "human_review",
    review_routing_edge,
    {
        "downstream_indexer": "downstream_indexer",
        END: END
    }
)

workflow.add_edge("corrector", "validator")
workflow.add_edge("downstream_indexer", END)

app = workflow.compile()