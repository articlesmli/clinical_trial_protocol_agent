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

class CriteriaExtractionOutput(BaseModel):
    inclusion_criteria: List[str] = Field(description="List of patient inclusion criteria.")
    exclusion_criteria: List[str] = Field(description="List of patient exclusion criteria.")
