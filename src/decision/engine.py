"""
Decision Engine: End-to-End Decision Making

Orchestrates decision-making by:
1. Collecting signals from the pipeline
2. Assessing out-of-distribution risk
3. Computing risk scores
4. Making three-way decision
5. Logging decision for audit
"""

from typing import Dict, Any, List
from datetime import datetime
import json

from src.decision import Decision
from src.decision.policy import get_policy, BaselinePolicy, CalibratedPolicy, ConservativePolicy, AggressivePolicy
from src.decision.ood import SimpleOODDetector
from src.decision.reason_codes import ReasonCode


class DecisionEngine:
    """End-to-end decision making for support queries."""
    
    def __init__(self, policy_name: str = "calibrated"):
        """
        Initialize decision engine.
        
        Args:
            policy_name: "baseline", "calibrated", "conservative", or "aggressive"
        """
        
        self.policy = get_policy(policy_name)
        self.ood_detector = SimpleOODDetector()
        self.policy_name = policy_name
        self.decision_log = []
    
    def decide(self, pipeline_output: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make end-to-end decision from pipeline outputs.
        
        Input (from the pipeline):
        - customer_text
        - predicted_intent
        - intent_confidence
        - intent_top_k (optional)
        - retrieved_evidence (list of evidence dicts)
        - retrieval_top1_score
        - retrieval_top3_mean
        - retrieval_intent_agreement
        - grounding_status
        - unsupported_claim_risk (0-1)
        - evidence_quality_score (0-1)
        - generated_reply
        
        Returns:
        {
            "decision": "AUTO_HANDLE" | "ASSIST_AND_ESCALATE" | "HUMAN_ESCALATION",
            "confidence": float,
            "reason_codes": list,
            "ood_score": float,
            "risk_score": float,
            "decision_metadata": dict
        }
        """
        
        # Prepare signals
        signals = self._prepare_signals(pipeline_output)
        
        # Detect OOD
        ood_result = self.ood_detector.detect(signals)
        ood_score = ood_result["ood_score"]
        
        # Add OOD score to signals
        signals["ood_score"] = ood_score
        
        # Hard constraint: High OOD → escalate
        if ood_result["is_ood"]:
            decision = {
                "decision": Decision.HUMAN_ESCALATION.value,
                "confidence": 0.85,
                "reason_codes": [ReasonCode.OUT_OF_DISTRIBUTION.value],
                "ood_score": ood_score,
                "risk_score": 0.0,  # Not computed due to OOD
                "ood_reasons": ood_result["indicators"],
                "policy": self.policy_name
            }
        
        else:
            # Use policy to make decision
            decision = self.policy.decide(signals)
            decision["ood_score"] = ood_score
            decision["decision_timestamp"] = datetime.now().isoformat()
        
        # Log decision
        self._log_decision(pipeline_output, decision)
        
        return decision
    
    def _prepare_signals(self, pipeline_output: Dict[str, Any]) -> Dict[str, Any]:
        """Extract and prepare signals from pipeline output."""
        
        signals = {
            "customer_text": pipeline_output.get("customer_text", ""),
            "predicted_intent": pipeline_output.get("predicted_intent", "unknown"),
            "intent_confidence": pipeline_output.get("intent_confidence", 0.0),
            "intent_top_k": pipeline_output.get("intent_top_k", []),
            
            # Retrieval signals
            "retrieval_top1_score": pipeline_output.get("retrieval_top1_score", 0.0),
            "retrieval_top3_mean": pipeline_output.get("retrieval_top3_mean", 0.0),
            "retrieval_intent_agreement": pipeline_output.get("retrieval_intent_agreement", 0.0),
            
            # Generation signals
            "grounding_status": pipeline_output.get("grounding_status", "unknown"),
            "evidence_quality_score": pipeline_output.get("evidence_quality_score", 0.0),
            "unsupported_claim_risk": pipeline_output.get("unsupported_claim_risk", 0.0),
            "generated_reply": pipeline_output.get("generated_reply", ""),
        }
        
        return signals
    
    def _log_decision(self, pipeline_output: Dict[str, Any], decision: Dict[str, Any]):
        """Log decision for audit trail."""
        
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "customer_text": pipeline_output.get("customer_text", "")[:100],  # Truncate for privacy
            "predicted_intent": pipeline_output.get("predicted_intent", ""),
            "decision": decision.get("decision", ""),
            "confidence": decision.get("confidence", 0.0),
            "reason_codes": decision.get("reason_codes", []),
            "risk_score": decision.get("risk_score", 0.0),
            "ood_score": decision.get("ood_score", 0.0),
            "policy": self.policy_name,
        }
        
        self.decision_log.append(log_entry)
    
    def get_decision_log(self) -> List[Dict[str, Any]]:
        """Get all logged decisions."""
        return self.decision_log
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics from decision log."""
        
        if not self.decision_log:
            return {}
        
        decisions = [entry["decision"] for entry in self.decision_log]
        
        stats = {
            "total_decisions": len(self.decision_log),
            "auto_handle_count": decisions.count("AUTO_HANDLE"),
            "assist_escalate_count": decisions.count("ASSIST_AND_ESCALATE"),
            "human_escalation_count": decisions.count("HUMAN_ESCALATION"),
            "avg_confidence": sum(e["confidence"] for e in self.decision_log) / len(self.decision_log),
            "avg_risk_score": sum(e.get("risk_score", 0.0) for e in self.decision_log) / len(self.decision_log),
            "avg_ood_score": sum(e.get("ood_score", 0.0) for e in self.decision_log) / len(self.decision_log),
        }
        
        return stats


class DecisionEvaluator:
    """Evaluates decision policy performance."""
    
    @staticmethod
    def evaluate_policy(decisions: List[Dict[str, Any]], 
                       gold_decisions: List[str]) -> Dict[str, Any]:
        """
        Evaluate policy against gold labels.
        
        Args:
            decisions: Predicted decisions
            gold_decisions: Gold-label decisions
        
        Returns:
            Evaluation metrics
        """
        
        if len(decisions) != len(gold_decisions):
            raise ValueError("Length mismatch")
        
        # Compute metrics per decision class
        metrics = {}
        
        for decision_class in ["AUTO_HANDLE", "ASSIST_AND_ESCALATE", "HUMAN_ESCALATION"]:
            pred_mask = [d["decision"] == decision_class for d in decisions]
            gold_mask = [g == decision_class for g in gold_decisions]
            
            tp = sum(1 for p, g in zip(pred_mask, gold_mask) if p and g)
            fp = sum(1 for p, g in zip(pred_mask, gold_mask) if p and not g)
            fn = sum(1 for p, g in zip(pred_mask, gold_mask) if not p and g)
            tn = sum(1 for p, g in zip(pred_mask, gold_mask) if not p and not g)
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
            
            metrics[decision_class] = {
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "tn": tn
            }
        
        # Overall accuracy
        correct = sum(1 for d, g in zip(decisions, gold_decisions) if d["decision"] == g)
        accuracy = correct / len(decisions)
        
        metrics["overall"] = {
            "accuracy": accuracy,
            "total": len(decisions),
            "correct": correct
        }
        
        return metrics
    
    @staticmethod
    def unsafe_auto_handle_rate(decisions: List[Dict[str, Any]], 
                               gold_decisions: List[str]) -> float:
        """
        Calculate unsafe auto-handle rate.
        
        = (incorrect AUTO_HANDLE decisions) / (total AUTO_HANDLE decisions)
        
        This is the false automation rate - most important safety metric.
        """
        
        auto_handle_indices = [i for i, d in enumerate(decisions) 
                              if d["decision"] == "AUTO_HANDLE"]
        
        if not auto_handle_indices:
            return 0.0
        
        incorrect = sum(1 for i in auto_handle_indices 
                       if gold_decisions[i] != "AUTO_HANDLE")
        
        return incorrect / len(auto_handle_indices)
    
    @staticmethod
    def compute_coverage_safety_curve(decisions: List[Dict[str, Any]], 
                                     gold_decisions: List[str]) -> List[Dict[str, Any]]:
        """
        Compute coverage vs safety tradeoff at different confidence thresholds.
        
        Returns:
        [
            {"confidence_threshold": 0.50, "coverage": 0.95, "precision": 0.92},
            {"confidence_threshold": 0.60, "coverage": 0.90, "precision": 0.95},
            ...
        ]
        """
        
        thresholds = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
        curve = []
        
        for threshold in thresholds:
            filtered_decisions = [d for d in decisions if d["confidence"] >= threshold]
            
            if not filtered_decisions:
                continue
            
            # Get indices of filtered decisions
            filtered_indices = [i for i, d in enumerate(decisions) 
                              if d["confidence"] >= threshold]
            
            # Coverage
            coverage = len(filtered_indices) / len(decisions)
            
            # Precision on filtered set
            correct = sum(1 for i in filtered_indices 
                         if decisions[i]["decision"] == gold_decisions[i])
            precision = correct / len(filtered_indices) if filtered_indices else 0.0
            
            curve.append({
                "confidence_threshold": threshold,
                "coverage": coverage,
                "precision": precision,
                "count": len(filtered_indices)
            })
        
        return curve
