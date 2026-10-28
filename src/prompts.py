# src/prompts.py

ONCOLOGY_TEMPLATE = """You are a highly specialized Oncology Clinical Trial Agent. 
When parsing criteria, pay strict attention to:
- RECIST 1.1 criteria eligibility or staging metrics.
- Prior systemic chemotherapies, radiation windows, or toxicities resolved to Grade <= 1.
- Biomarker requirements (e.g., EGFR, ALK, PD-L1 expressions)."""

CARDIOLOGY_TEMPLATE = """You are a highly specialized Cardiology Clinical Trial Agent. 
When parsing criteria, pay strict attention to:
- Left Ventricular Ejection Fraction (LVEF) boundaries.
- New York Heart Association (NYHA) Classification markers.
- Recent ischemic events, revascularization intervals, or unstable angina exceptions."""

GENERAL_TEMPLATE = """You are an expert Clinical Trial Protocol Analyzer."""

def build_extraction_prompt(context_text: str, therapeutic_area: str = "General", historical_examples: str = "") -> str:
    """Generates a dynamic prompt containing domain instructions, context, and optional few-shots."""
    if therapeutic_area.lower() == "oncology":
        domain_instruction = ONCOLOGY_TEMPLATE
    elif therapeutic_area.lower() == "cardiology":
        domain_instruction = CARDIOLOGY_TEMPLATE
    else:
        domain_instruction = GENERAL_TEMPLATE

    example_section = ""
    if historical_examples:
        example_section = f"\n### Historical Exemplars for Reference:\n{historical_examples}\n"

    prompt = f"""{domain_instruction}

Extract the patient inclusion and exclusion criteria strictly from the following clinical trial protocol text. 
Ensure the extracted points are atomic, factual, and accurate. Do not hallucinate fields or rules absent in the raw text.
{example_section}
### Protocol Context Text:
{context_text}

Provide output matching the requested schema.
"""
    return prompt

def build_correction_prompt(current_text: str, errors: str, feedback: str, criteria: dict) -> str:
    """Generates a dynamic self-correction prompt based on failing validation/telemetry state."""
    return f"""Your previous structural extraction failed internal clinical validation guardrails or faithfulness criteria.

### Previous Extraction Results:
Inclusion Criteria: {criteria.get('inclusion_criteria', [])}
Exclusion Criteria: {criteria.get('exclusion_criteria', [])}

### Discovered Failures & Feedback:
- Guardrail Flags: {errors}
- Evaluation/Faithfulness Feedback: {feedback}

### Underlying Protocol Context:
{current_text}

Please review the context text carefully. Regenerate the inclusion and exclusion criteria to fix these issues. Ensure all facts map 100% truthfully back to the context without breaking clinical boundary rules.
"""
