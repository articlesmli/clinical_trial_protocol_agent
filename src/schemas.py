import operator
from typing import Annotated, List, TypedDict
from pydantic import BaseModel, Field
from langchain_core.messages import BaseMessage

class ProtocolState(TypedDict):
    pdf_path: str
    extracted_criteria: List[str]
    search_results: List[str]
    compliance_report: str
    messages: Annotated[List[BaseMessage], operator.add]
    # --- New fields for Human Review ---
    compliance_status: str
    validation_errors: str
    review_status: str  # e.g., "PENDING_HUMAN_REVIEW", "APPROVED", "REJECTED"
    reviewer_notes: str

class CriteriaExtractionOutput(BaseModel):
    inclusion_criteria: List[str] = Field(description="List of patient inclusion criteria.")
    exclusion_criteria: List[str] = Field(description="List of patient exclusion criteria.")