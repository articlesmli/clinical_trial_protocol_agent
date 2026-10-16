import re
from typing import List, Dict, Any

class ProtocolGuardrails:
    """Implements safety, validation, and sanitization guardrails for clinical trial data."""

    @staticmethod
    def sanitize_input_text(text: str) -> str:
        """Removes potential injection patterns or malformed control characters from extracted text."""
        # Remove null bytes or weird control sequences
        sanitized = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', '', text)
        # Strip excessive whitespace
        sanitized = re.sub(r'\s+', ' ', sanitized).strip()
        return sanitized

    @staticmethod
    def validate_clinical_boundaries(criteria_list: List[str]) -> Dict[str, Any]:
        """Ensures extracted criteria do not contain forbidden terms or violate basic clinical constraints."""
        flagged_items = []
        for criterion in criteria_list:
            cleaned = ProtocolGuardrails.sanitize_input_text(criterion)
            # Example rule: Check for ungrounded or hallucinatory wildcards
            if "cure-all" in cleaned.lower() or "guaranteed" in cleaned.lower():
                flagged_items.append(criterion)
        
        return {
            "passed": len(flagged_items) == 0,
            "flagged_items": flagged_items
        }
