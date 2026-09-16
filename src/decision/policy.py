"""
Decision policies: Baseline and calibrated.

Policies determine: AUTO_HANDLE vs ASSIST_AND_ESCALATE vs HUMAN_ESCALATION
"""

from typing import Dict, Any, Set
from src.decision import Decision
from src.decision.reason_codes import ReasonCode
from src.decision.risk import RiskAssessor


class BaselinePolicy:
    """
    Simple deterministic baseline policy based on evidence quality rules.
    
    Rules:
    1. If grounding_status == insufficient → HUMAN_ESCALATION
    2. Else if risk_score >= 0.75 → HUMAN_ESCALATION
    3. Else if risk_score >= 0.50 → ASSIST_AND_ESCALATE
    4. Else → AUTO_HANDLE
    """
    
    def __init__(self):
        self.risk_assessor = RiskAssessor()
    
    def decide(self, signals: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make decision based on baseline rules.
        
        Returns:
        {
            "decision": Decision.AUTO_HANDLE | ASSIST_AND_ESCALATE | HUMAN_ESCALATION,
            "confidence": float (0-1),
            "reason_codes": list,
            "policy": "baseline"
        }
        """
        
        # Assess risk
        risk_assessment = self.risk_assessor.assess(signals)
        risk_score = risk_assessment["risk_score"]
        reason_codes = set(ReasonCode(code) for code in risk_assessment["reason_codes"])
        
        # Apply baseline rules
        grounding_status = signals.get("grounding_status", "unknown")
        
        if grounding_status == "insufficient":
            decision = Decision.HUMAN_ESCALATION
            confidence = 0.95
            reason_codes.add(ReasonCode.INSUFFICIENT_EVIDENCE)
        
        elif risk_score >= 0.75:
            decision = Decision.HUMAN_ESCALATION
            confidence = 0.80
        
        elif risk_score >= 0.50:
            decision = Decision.ASSIST_AND_ESCALATE
            confidence = 0.70
        
        else:
            decision = Decision.AUTO_HANDLE
            # Confidence increases as risk decreases
            confidence = max(0.60, 1.0 - risk_score)
        
        return {
            "decision": decision.value,
            "confidence": confidence,
            "reason_codes": [code.value for code in reason_codes],
            "risk_score": risk_score,
            "policy": "baseline",
            "signals": risk_assessment["signals"]
        }


class CalibratedPolicy:
    """
    Calibrated policy using logistic regression or decision rules.
    
    This policy is learned from evaluation data to optimize for safety
    while maintaining reasonable coverage.
    """
    
    def __init__(self, threshold_auto: float = 0.75, threshold_assist: float = 0.50):
        """
        Initialize with calibrated thresholds.
        
        Args:
            threshold_auto: Risk score below this → AUTO_HANDLE
            threshold_assist: Risk score below this (but above auto) → ASSIST_AND_ESCALATE
        """
        self.threshold_auto = threshold_auto
        self.threshold_assist = threshold_assist
        self.risk_assessor = RiskAssessor()
    
    def decide(self, signals: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make decision using calibrated thresholds.
        
        Returns structured decision with reasoning.
        """
        
        # Assess risk
        risk_assessment = self.risk_assessor.assess(signals)
        risk_score = risk_assessment["risk_score"]
        reason_codes = set(ReasonCode(code) for code in risk_assessment["reason_codes"])
        
        # Apply calibrated thresholds
        grounding_status = signals.get("grounding_status", "unknown")
        
        # Hard constraint: insufficient evidence always escalates
        if grounding_status == "insufficient":
            decision = Decision.HUMAN_ESCALATION
            confidence = 0.95
            reason_codes.add(ReasonCode.INSUFFICIENT_EVIDENCE)
        
        # Thresholds
        elif risk_score < self.threshold_auto:
            decision = Decision.AUTO_HANDLE
            confidence = max(0.70, 1.0 - risk_score)
        
        elif risk_score < self.threshold_assist:
            decision = Decision.ASSIST_AND_ESCALATE
            confidence = 0.65
        
        else:
            decision = Decision.HUMAN_ESCALATION
            confidence = min(0.95, 0.50 + risk_score)
        
        return {
            "decision": decision.value,
            "confidence": confidence,
            "reason_codes": [code.value for code in reason_codes],
            "risk_score": risk_score,
            "policy": "calibrated",
            "thresholds": {
                "auto_handle_max_risk": self.threshold_auto,
                "assist_max_risk": self.threshold_assist
            },
            "signals": risk_assessment["signals"]
        }


class ConservativePolicy:
    """
    Very conservative policy: minimize false automation at cost of coverage.
    
    Only auto-handles when risk is very low AND evidence is strong.
    """
    
    def __init__(self):
        self.risk_assessor = RiskAssessor()
    
    def decide(self, signals: Dict[str, Any]) -> Dict[str, Any]:
        """Make conservative decision (prefer escalation)."""
        
        # Assess risk
        risk_assessment = self.risk_assessor.assess(signals)
        risk_score = risk_assessment["risk_score"]
        reason_codes = set(ReasonCode(code) for code in risk_assessment["reason_codes"])
        
        grounding_status = signals.get("grounding_status", "unknown")
        intent_conf = signals.get("intent_confidence", 0.0)
        top1_score = signals.get("retrieval_top1_score", 0.0)
        
        # Very strict conditions for auto-handle
        if (grounding_status == "supported" and
            risk_score < 0.30 and
            intent_conf >= 0.85 and
            top1_score >= 0.75):
            decision = Decision.AUTO_HANDLE
            confidence = 0.85
        
        elif risk_score < 0.50 and grounding_status != "insufficient":
            decision = Decision.ASSIST_AND_ESCALATE
            confidence = 0.70
        
        else:
            decision = Decision.HUMAN_ESCALATION
            confidence = 0.90
        
        return {
            "decision": decision.value,
            "confidence": confidence,
            "reason_codes": [code.value for code in reason_codes],
            "risk_score": risk_score,
            "policy": "conservative",
            "signals": risk_assessment["signals"]
        }


class AggressivePolicy:
    """
    Aggressive policy: maximize coverage at some risk to safety.
    
    WARNING: Not recommended for production.
    Included for comparison only.
    """
    
    def __init__(self):
        self.risk_assessor = RiskAssessor()
    
    def decide(self, signals: Dict[str, Any]) -> Dict[str, Any]:
        """Make aggressive decision (prefer automation)."""
        
        # Assess risk
        risk_assessment = self.risk_assessor.assess(signals)
        risk_score = risk_assessment["risk_score"]
        reason_codes = set(ReasonCode(code) for code in risk_assessment["reason_codes"])
        
        grounding_status = signals.get("grounding_status", "unknown")
        
        # Only escalate if serious issues
        if grounding_status == "insufficient" or risk_score >= 0.85:
            decision = Decision.HUMAN_ESCALATION
            confidence = 0.80
        
        elif risk_score >= 0.60:
            decision = Decision.ASSIST_AND_ESCALATE
            confidence = 0.60
        
        else:
            decision = Decision.AUTO_HANDLE
            confidence = max(0.50, 1.0 - risk_score * 0.5)
        
        return {
            "decision": decision.value,
            "confidence": confidence,
            "reason_codes": [code.value for code in reason_codes],
            "risk_score": risk_score,
            "policy": "aggressive",
            "warning": "Aggressive policy not recommended for production",
            "signals": risk_assessment["signals"]
        }


def get_policy(policy_name: str) -> object:
    """Get policy by name."""
    
    policies = {
        "baseline": BaselinePolicy,
        "calibrated": CalibratedPolicy,
        "conservative": ConservativePolicy,
        "aggressive": AggressivePolicy,
    }
    
    if policy_name not in policies:
        raise ValueError(f"Unknown policy: {policy_name}. Available: {list(policies.keys())}")
    
    return policies[policy_name]()
