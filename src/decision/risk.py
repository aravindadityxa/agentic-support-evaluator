"""
Risk assessment for support automation.

Evaluates signals from the pipeline to determine automation safety.
"""

from typing import Dict, Any, Set, Tuple
from src.decision.reason_codes import ReasonCode

class RiskAssessor:
    """Assesses risk signals from classification, retrieval, and generation."""
    
    def __init__(self):
        self.reason_codes: Set[ReasonCode] = set()
        self.risk_signals = {}
    
    def assess(self, signals: Dict[str, Any]) -> Dict[str, Any]:
        """
        Assess all risk signals.
        
        Signals expected:
        - intent_confidence: float (0-1)
        - intent_top_k: list of (intent, prob) tuples
        - retrieval_top1_score: float (0-1)
        - retrieval_top3_mean: float (0-1)
        - retrieval_intent_agreement: float (0-1)
        - grounding_status: str (supported/partially_supported/insufficient)
        - unsupported_claim_risk: float (0-1)
        - evidence_quality_score: float (0-1)
        - generated_reply: str
        - predicted_intent: str
        
        Returns:
        {
            "risk_score": float (0-1),
            "reason_codes": list,
            "signals": dict,
            "escalation_recommended": bool
        }
        """
        
        self.reason_codes = set()
        self.risk_signals = {}
        
        # Assess each signal
        self._assess_intent_confidence(signals)
        self._assess_intent_ambiguity(signals)
        self._assess_retrieval_quality(signals)
        self._assess_grounding(signals)
        self._assess_unsupported_claims(signals)
        self._assess_query_characteristics(signals)
        self._assess_risk_action(signals)
        
        # Compute aggregate risk score
        risk_score = self._compute_risk_score()
        
        return {
            "risk_score": risk_score,
            "reason_codes": list(self.reason_codes),
            "signals": self.risk_signals,
            "escalation_recommended": risk_score >= 0.5
        }
    
    def _assess_intent_confidence(self, signals: Dict[str, Any]):
        """Assess intent classifier confidence."""
        
        intent_conf = signals.get("intent_confidence", 0.0)
        
        self.risk_signals["intent_confidence"] = intent_conf
        
        if intent_conf < 0.60:
            self.reason_codes.add(ReasonCode.LOW_INTENT_CONFIDENCE)
            self.risk_signals["intent_confidence_risk"] = "high"
        elif intent_conf < 0.75:
            self.risk_signals["intent_confidence_risk"] = "medium"
        else:
            self.risk_signals["intent_confidence_risk"] = "low"
    
    def _assess_intent_ambiguity(self, signals: Dict[str, Any]):
        """Assess whether intent is ambiguous."""
        
        intent_top_k = signals.get("intent_top_k", [])
        
        if len(intent_top_k) < 2:
            return
        
        top_prob = intent_top_k[0][1] if intent_top_k else 0.0
        second_prob = intent_top_k[1][1] if len(intent_top_k) > 1 else 0.0
        
        gap = top_prob - second_prob
        self.risk_signals["intent_probability_gap"] = gap
        
        if gap < 0.15:  # Close race between top 2
            self.reason_codes.add(ReasonCode.AMBIGUOUS_INTENT)
            self.risk_signals["intent_ambiguity_risk"] = "high"
        elif gap < 0.25:
            self.risk_signals["intent_ambiguity_risk"] = "medium"
        else:
            self.risk_signals["intent_ambiguity_risk"] = "low"
    
    def _assess_retrieval_quality(self, signals: Dict[str, Any]):
        """Assess retrieval evidence quality."""
        
        top1_score = signals.get("retrieval_top1_score", 0.0)
        top3_mean = signals.get("retrieval_top3_mean", 0.0)
        intent_agreement = signals.get("retrieval_intent_agreement", 0.0)
        
        self.risk_signals["retrieval_top1_score"] = top1_score
        self.risk_signals["retrieval_top3_mean"] = top3_mean
        self.risk_signals["retrieval_intent_agreement"] = intent_agreement
        
        # Top-1 similarity is primary signal
        if top1_score < 0.50:
            self.reason_codes.add(ReasonCode.INSUFFICIENT_EVIDENCE)
            self.reason_codes.add(ReasonCode.WEAK_RETRIEVAL)
            self.risk_signals["retrieval_quality_risk"] = "high"
        elif top1_score < 0.65:
            self.reason_codes.add(ReasonCode.WEAK_RETRIEVAL)
            self.risk_signals["retrieval_quality_risk"] = "medium"
        else:
            self.risk_signals["retrieval_quality_risk"] = "low"
        
        # Intent agreement matters
        if intent_agreement < 0.40:
            self.reason_codes.add(ReasonCode.LOW_INTENT_ALIGNMENT)
            self.risk_signals["intent_alignment_risk"] = "high"
        elif intent_agreement < 0.60:
            self.risk_signals["intent_alignment_risk"] = "medium"
        else:
            self.risk_signals["intent_alignment_risk"] = "low"
    
    def _assess_grounding(self, signals: Dict[str, Any]):
        """Assess whether reply is grounded in evidence."""
        
        grounding_status = signals.get("grounding_status", "unknown")
        evidence_quality = signals.get("evidence_quality_score", 0.0)
        
        self.risk_signals["grounding_status"] = grounding_status
        self.risk_signals["evidence_quality_score"] = evidence_quality
        
        if grounding_status == "insufficient":
            self.reason_codes.add(ReasonCode.INSUFFICIENT_EVIDENCE)
            self.reason_codes.add(ReasonCode.LOW_GROUNDING_CONFIDENCE)
            self.risk_signals["grounding_risk"] = "high"
        elif grounding_status == "partially_supported":
            self.reason_codes.add(ReasonCode.LOW_GROUNDING_CONFIDENCE)
            self.risk_signals["grounding_risk"] = "medium"
        else:
            self.risk_signals["grounding_risk"] = "low"
    
    def _assess_unsupported_claims(self, signals: Dict[str, Any]):
        """Assess risk of unsupported claims in reply."""
        
        unsupported_risk = signals.get("unsupported_claim_risk", 0.0)
        
        self.risk_signals["unsupported_claim_risk"] = unsupported_risk
        
        if unsupported_risk > 0.30:
            self.reason_codes.add(ReasonCode.UNSUPPORTED_CLAIM_RISK)
            self.risk_signals["hallucination_risk"] = "high"
        elif unsupported_risk > 0.15:
            self.risk_signals["hallucination_risk"] = "medium"
        else:
            self.risk_signals["hallucination_risk"] = "low"
    
    def _assess_query_characteristics(self, signals: Dict[str, Any]):
        """Assess query-level risk factors."""
        
        # Out-of-distribution detection
        ood_score = signals.get("ood_score", 0.0)
        
        self.risk_signals["ood_score"] = ood_score
        
        if ood_score > 0.70:
            self.reason_codes.add(ReasonCode.OUT_OF_DISTRIBUTION)
            self.risk_signals["ood_risk"] = "high"
        elif ood_score > 0.50:
            self.risk_signals["ood_risk"] = "medium"
        else:
            self.risk_signals["ood_risk"] = "low"
        
        # Multi-issue detection (heuristic)
        customer_text = signals.get("customer_text", "")
        if self._is_multi_issue(customer_text):
            self.reason_codes.add(ReasonCode.MULTI_ISSUE_QUERY)
            self.risk_signals["multi_issue"] = True
    
    def _assess_risk_action(self, signals: Dict[str, Any]):
        """Assess whether request involves high-risk actions."""
        
        predicted_intent = signals.get("predicted_intent", "").lower()
        
        high_risk_keywords = ["refund", "account", "payment", "delete", "warranty"]
        
        for keyword in high_risk_keywords:
            if keyword in predicted_intent:
                self.reason_codes.add(ReasonCode.HIGH_RISK_ACTION)
                self.risk_signals["high_risk_action"] = keyword
                break
    
    def _is_multi_issue(self, text: str) -> bool:
        """Heuristic: detect if query involves multiple issues."""
        
        # Simple heuristics (could be improved)
        issue_markers = ["also", "plus", "and my", "another", "in addition"]
        problem_words = ["not working", "won't", "issue", "problem", "crash", "slow"]
        
        text_lower = text.lower()
        issue_count = sum(1 for word in problem_words if word in text_lower)
        has_conjunction = any(marker in text_lower for marker in issue_markers)
        
        return issue_count >= 2 or (has_conjunction and issue_count >= 1)
    
    def _compute_risk_score(self) -> float:
        """Compute aggregate risk score (0-1)."""
        
        # Map risk levels to scores
        risk_level_scores = {"low": 0.1, "medium": 0.5, "high": 0.9}
        
        # Collect all risk signals
        risk_levels = [
            self.risk_signals.get("intent_confidence_risk", "low"),
            self.risk_signals.get("intent_ambiguity_risk", "low"),
            self.risk_signals.get("retrieval_quality_risk", "low"),
            self.risk_signals.get("intent_alignment_risk", "low"),
            self.risk_signals.get("grounding_risk", "low"),
            self.risk_signals.get("hallucination_risk", "low"),
            self.risk_signals.get("ood_risk", "low"),
        ]
        
        # Weight: retrieval and grounding are most important
        weights = {
            "retrieval_quality_risk": 0.25,
            "grounding_risk": 0.25,
            "unsupported_claim_risk": 0.15,
            "intent_confidence_risk": 0.15,
            "intent_ambiguity_risk": 0.10,
            "ood_risk": 0.10,
        }
        
        weighted_risk = 0.0
        total_weight = 0.0
        
        for signal_name, weight in weights.items():
            if signal_name in self.risk_signals:
                level = self.risk_signals.get(signal_name, "low")
                if isinstance(level, str):
                    score = risk_level_scores.get(level, 0.5)
                else:
                    score = level
                weighted_risk += score * weight
                total_weight += weight
        
        if total_weight > 0:
            return min(weighted_risk / total_weight, 1.0)
        
        return 0.0
