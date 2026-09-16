"""
Standardized reason codes for escalation/auto-handle decisions.

Each decision must include reason codes explaining why it was made.
Multiple codes can apply to a single case.
"""

from enum import Enum
from typing import Set

class ReasonCode(Enum):
    """Standardized codes for explaining decisions."""
    
    # Intent-related
    LOW_INTENT_CONFIDENCE = "LOW_INTENT_CONFIDENCE"
    AMBIGUOUS_INTENT = "AMBIGUOUS_INTENT"
    HIGH_INTENT_DISAGREEMENT = "HIGH_INTENT_DISAGREEMENT"
    
    # Retrieval-related
    WEAK_RETRIEVAL = "WEAK_RETRIEVAL"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
    CONFLICTING_EVIDENCE = "CONFLICTING_EVIDENCE"
    LOW_INTENT_ALIGNMENT = "LOW_INTENT_ALIGNMENT"
    
    # Generation-related
    LOW_GROUNDING_CONFIDENCE = "LOW_GROUNDING_CONFIDENCE"
    UNSUPPORTED_CLAIM_RISK = "UNSUPPORTED_CLAIM_RISK"
    GENERIC_HISTORICAL_RESPONSE = "GENERIC_HISTORICAL_RESPONSE"
    
    # Query-related
    MULTI_ISSUE_QUERY = "MULTI_ISSUE_QUERY"
    OUT_OF_DISTRIBUTION = "OUT_OF_DISTRIBUTION"
    
    # Risk-related
    HIGH_RISK_ACTION = "HIGH_RISK_ACTION"  # Refunds, account access, etc.
    ACCOUNT_SPECIFIC_ACTION = "ACCOUNT_SPECIFIC_ACTION"
    
    # Data quality
    INSUFFICIENT_HISTORICAL_EXAMPLES = "INSUFFICIENT_HISTORICAL_EXAMPLES"
    
    # Other
    OTHER = "OTHER"


# Risk levels for different action types
HIGH_RISK_INTENT_KEYWORDS = {
    "account_signin": "Account access - requires verification",
    "account_deletion": "Account deletion - sensitive action",
    "payment_refund": "Refund - requires authorization",
    "warranty_claim": "Warranty - requires documentation",
    "data_backup": "Data access - privacy sensitive",
}

MEDIUM_RISK_KEYWORDS = {
    "technical_issue": False,  # Usually safe if evidence is strong
    "complaint_feedback": True,  # Requires care with tone
}


def categorize_by_risk(predicted_intent: str) -> str:
    """Categorize intent by risk level."""
    intent_lower = predicted_intent.lower()
    
    if intent_lower in HIGH_RISK_INTENT_KEYWORDS:
        return "HIGH"
    elif intent_lower in MEDIUM_RISK_KEYWORDS:
        return "MEDIUM"
    else:
        return "LOW"


def explain_reason_code(code: ReasonCode) -> str:
    """Human-readable explanation of reason code."""
    
    explanations = {
        ReasonCode.LOW_INTENT_CONFIDENCE: 
            "Intent prediction confidence below threshold",
        ReasonCode.AMBIGUOUS_INTENT: 
            "Multiple intents appear equally likely",
        ReasonCode.HIGH_INTENT_DISAGREEMENT: 
            "Retrieved evidence intents disagree with predicted intent",
        ReasonCode.WEAK_RETRIEVAL: 
            "Retrieved evidence has low similarity to query",
        ReasonCode.INSUFFICIENT_EVIDENCE: 
            "Not enough historical evidence to ground reply",
        ReasonCode.CONFLICTING_EVIDENCE: 
            "Retrieved evidence suggests conflicting solutions",
        ReasonCode.LOW_INTENT_ALIGNMENT: 
            "Retrieved evidence intents misaligned",
        ReasonCode.LOW_GROUNDING_CONFIDENCE: 
            "Generated reply not well grounded in evidence",
        ReasonCode.UNSUPPORTED_CLAIM_RISK: 
            "Reply contains unsupported factual claims",
        ReasonCode.GENERIC_HISTORICAL_RESPONSE: 
            "Historical evidence response is generic/non-actionable",
        ReasonCode.MULTI_ISSUE_QUERY: 
            "Query involves multiple unrelated issues",
        ReasonCode.OUT_OF_DISTRIBUTION: 
            "Query is unusual/outside historical distribution",
        ReasonCode.HIGH_RISK_ACTION: 
            "Request involves high-risk action (refund, account, etc.)",
        ReasonCode.ACCOUNT_SPECIFIC_ACTION: 
            "Request requires account-specific information",
        ReasonCode.INSUFFICIENT_HISTORICAL_EXAMPLES: 
            "Few historical examples for this intent/issue",
        ReasonCode.OTHER: 
            "See decision metadata for details",
    }
    
    return explanations.get(code, "Unknown reason")


def format_reason_codes(codes: Set[ReasonCode]) -> dict:
    """Format reason codes for output."""
    
    return {
        "codes": [c.value for c in codes],
        "explanations": [explain_reason_code(c) for c in codes],
        "count": len(codes)
    }
