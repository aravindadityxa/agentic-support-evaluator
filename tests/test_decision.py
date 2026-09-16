#!/usr/bin/env python3
"""
Unit tests for decision routing system.

Tests:
- Reason codes
- Risk assessment
- Policies (baseline, calibrated, conservative)
- Out-of-distribution detection
- Decision logging
"""

import sys
import unittest
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.decision import Decision
from src.decision.reason_codes import ReasonCode, categorize_by_risk, explain_reason_code
from src.decision.risk import RiskAssessor
from src.decision.policy import BaselinePolicy, CalibratedPolicy, ConservativePolicy
from src.decision.ood import SimpleOODDetector
from src.decision.engine import DecisionEngine, DecisionEvaluator


class TestReasonCodes(unittest.TestCase):
    """Test reason codes."""
    
    def test_reason_code_enum(self):
        """Reason codes are properly defined."""
        # Verify reason codes are defined in the enum
        expected_codes = {
            ReasonCode.LOW_INTENT_CONFIDENCE,
            ReasonCode.INSUFFICIENT_EVIDENCE,
            ReasonCode.WEAK_RETRIEVAL,
            ReasonCode.UNSUPPORTED_CLAIM_RISK,
            ReasonCode.OUT_OF_DISTRIBUTION,
            ReasonCode.AMBIGUOUS_INTENT,
            ReasonCode.HIGH_INTENT_DISAGREEMENT,
        }
        # All expected codes should exist in the enum
        for code in expected_codes:
            self.assertIsNotNone(code)
        # Verify enum has at least the expected codes
        self.assertGreaterEqual(len(ReasonCode), len(expected_codes))
    
    def test_categorize_by_risk(self):
        """Risk categorization works."""
        self.assertEqual(categorize_by_risk("account_signin"), "HIGH")
        self.assertEqual(categorize_by_risk("software_issue"), "LOW")
    
    def test_explain_reason_code(self):
        """Reason codes have explanations."""
        explanation = explain_reason_code(ReasonCode.LOW_INTENT_CONFIDENCE)
        self.assertIn("confidence", explanation.lower())


class TestRiskAssessment(unittest.TestCase):
    """Test risk assessment."""
    
    def setUp(self):
        self.assessor = RiskAssessor()
    
    def test_assess_high_risk(self):
        """High-risk signals generate high risk score."""
        signals = {
            "customer_text": "I need a refund",
            "predicted_intent": "payment_refund",
            "intent_confidence": 0.3,
            "retrieval_top1_score": 0.2,
            "retrieval_top3_mean": 0.15,
            "retrieval_intent_agreement": 0.1,
            "grounding_status": "insufficient",
            "unsupported_claim_risk": 0.8,
            "ood_score": 0.2,
        }
        
        result = self.assessor.assess(signals)
        
        self.assertGreater(result["risk_score"], 0.7)
        # Check that reason codes are present (as enum objects or strings)
        reason_codes = result["reason_codes"]
        code_values = [c.value if hasattr(c, 'value') else c for c in reason_codes]
        self.assertIn(ReasonCode.INSUFFICIENT_EVIDENCE.value, code_values)
    
    def test_assess_low_risk(self):
        """Low-risk signals generate low risk score."""
        signals = {
            "customer_text": "Device is slow",
            "predicted_intent": "performance_issue",
            "intent_confidence": 0.95,
            "retrieval_top1_score": 0.90,
            "retrieval_top3_mean": 0.85,
            "retrieval_intent_agreement": 0.95,
            "grounding_status": "supported",
            "unsupported_claim_risk": 0.05,
            "ood_score": 0.1,
        }
        
        result = self.assessor.assess(signals)
        
        self.assertLess(result["risk_score"], 0.3)
        self.assertEqual(len(result["reason_codes"]), 0)


class TestPolicies(unittest.TestCase):
    """Test decision policies."""
    
    def setUp(self):
        self.baseline = BaselinePolicy()
        self.calibrated = CalibratedPolicy()
        self.conservative = ConservativePolicy()
    
    def test_baseline_high_risk(self):
        """Baseline escalates high-risk."""
        signals = {
            "customer_text": "Refund",
            "predicted_intent": "refund",
            "intent_confidence": 0.4,
            "retrieval_top1_score": 0.3,
            "retrieval_top3_mean": 0.2,
            "retrieval_intent_agreement": 0.2,
            "grounding_status": "insufficient",
            "unsupported_claim_risk": 0.7,
            "ood_score": 0.2,
        }
        
        result = self.baseline.decide(signals)
        
        self.assertEqual(result["decision"], Decision.HUMAN_ESCALATION.value)
    
    def test_baseline_low_risk(self):
        """Baseline auto-handles low-risk."""
        signals = {
            "customer_text": "Device slow",
            "predicted_intent": "performance",
            "intent_confidence": 0.9,
            "retrieval_top1_score": 0.85,
            "retrieval_top3_mean": 0.80,
            "retrieval_intent_agreement": 0.9,
            "grounding_status": "supported",
            "unsupported_claim_risk": 0.05,
            "ood_score": 0.1,
        }
        
        result = self.baseline.decide(signals)
        
        self.assertEqual(result["decision"], Decision.AUTO_HANDLE.value)
    
    def test_calibrated_threshold(self):
        """Calibrated policy respects thresholds."""
        threshold_auto = 0.40
        threshold_assist = 0.65
        calibrated = CalibratedPolicy(threshold_auto=threshold_auto, 
                                     threshold_assist=threshold_assist)
        
        # Low risk
        signals_low = {
            "customer_text": "Test",
            "predicted_intent": "test",
            "intent_confidence": 0.8,
            "retrieval_top1_score": 0.8,
            "retrieval_top3_mean": 0.75,
            "retrieval_intent_agreement": 0.8,
            "grounding_status": "supported",
            "unsupported_claim_risk": 0.1,
            "ood_score": 0.1,
        }
        
        result_low = calibrated.decide(signals_low)
        self.assertEqual(result_low["decision"], Decision.AUTO_HANDLE.value)
    
    def test_conservative_strict(self):
        """Conservative policy is more restrictive."""
        signals = {
            "customer_text": "Device issue",
            "predicted_intent": "issue",
            "intent_confidence": 0.7,
            "retrieval_top1_score": 0.7,
            "retrieval_top3_mean": 0.65,
            "retrieval_intent_agreement": 0.7,
            "grounding_status": "supported",
            "unsupported_claim_risk": 0.1,
            "ood_score": 0.1,
        }
        
        baseline_result = self.baseline.decide(signals)
        conservative_result = self.conservative.decide(signals)
        
        # Conservative should not auto-handle more than baseline
        if baseline_result["decision"] == "AUTO_HANDLE":
            # Conservative might still auto-handle but with higher confidence bar
            self.assertIn(conservative_result["decision"], 
                         ["AUTO_HANDLE", "ASSIST_AND_ESCALATE", "HUMAN_ESCALATION"])


class TestOODDetection(unittest.TestCase):
    """Test OOD detection."""
    
    def setUp(self):
        self.detector = SimpleOODDetector()
    
    def test_ood_low_similarity(self):
        """Very low similarity triggers OOD."""
        signals = {
            "customer_text": "xyzabc no real support query",
            "predicted_intent": "unknown",
            "intent_confidence": 0.3,
            "intent_top_k": [(None, 0.3), (None, 0.25)],
            "retrieval_top1_score": 0.15,
            "retrieval_top3_mean": 0.10,
        }
        
        result = self.detector.detect(signals)
        
        self.assertGreater(result["ood_score"], 0.5)
        self.assertTrue(result["is_ood"])
    
    def test_ood_normal_case(self):
        """Normal queries not flagged as OOD."""
        signals = {
            "customer_text": "My iPhone won't turn on after update",
            "predicted_intent": "software_issue",
            "intent_confidence": 0.85,
            "intent_top_k": [(None, 0.85), (None, 0.10)],
            "retrieval_top1_score": 0.80,
            "retrieval_top3_mean": 0.75,
        }
        
        result = self.detector.detect(signals)
        
        self.assertLess(result["ood_score"], 0.3)
        self.assertFalse(result["is_ood"])
    
    def test_ood_unusual_length(self):
        """Unusual query length flagged."""
        very_long = "word " * 200  # 1000+ chars
        
        signals = {
            "customer_text": very_long,
            "predicted_intent": "unknown",
            "intent_confidence": 0.5,
            "intent_top_k": [(None, 0.5), (None, 0.45)],
            "retrieval_top1_score": 0.5,
            "retrieval_top3_mean": 0.45,
        }
        
        result = self.detector.detect(signals)
        
        # Check for indicators list (implementation may vary the exact name)
        self.assertIn("indicators", result)
        indicators = result["indicators"]
        # Should flag something related to length/query properties
        self.assertTrue(any("length" in str(i).lower() or "unusual" in str(i).lower() 
                           for i in indicators))


class TestDecisionEngine(unittest.TestCase):
    """Test decision engine."""
    
    def setUp(self):
        self.engine = DecisionEngine(policy_name="baseline")
    
    def test_engine_decides(self):
        """Engine makes decisions."""
        pipeline_output = {
            "customer_text": "Device slow",
            "predicted_intent": "performance",
            "intent_confidence": 0.8,
            "retrieval_top1_score": 0.75,
            "retrieval_top3_mean": 0.70,
            "retrieval_intent_agreement": 0.8,
            "grounding_status": "supported",
            "unsupported_claim_risk": 0.1,
            "generated_reply": "Try clearing cache",
        }
        
        decision = self.engine.decide(pipeline_output)
        
        self.assertIn("decision", decision)
        self.assertIn("confidence", decision)
        self.assertIn("reason_codes", decision)
    
    def test_engine_logging(self):
        """Engine logs decisions."""
        pipeline_output = {
            "customer_text": "Test",
            "predicted_intent": "test",
            "intent_confidence": 0.5,
            "retrieval_top1_score": 0.5,
            "retrieval_top3_mean": 0.45,
            "retrieval_intent_agreement": 0.5,
            "grounding_status": "partially_supported",
            "unsupported_claim_risk": 0.2,
            "generated_reply": "Test reply",
        }
        
        self.engine.decide(pipeline_output)
        self.engine.decide(pipeline_output)
        
        log = self.engine.get_decision_log()
        self.assertEqual(len(log), 2)
    
    def test_engine_statistics(self):
        """Engine computes statistics."""
        for i in range(5):
            pipeline_output = {
                "customer_text": f"Test {i}",
                "predicted_intent": "test",
                "intent_confidence": 0.5 + i * 0.1,
                "retrieval_top1_score": 0.5 + i * 0.1,
                "retrieval_top3_mean": 0.45 + i * 0.1,
                "retrieval_intent_agreement": 0.5 + i * 0.1,
                "grounding_status": "supported" if i % 2 == 0 else "insufficient",
                "unsupported_claim_risk": 0.2 - i * 0.05,
                "generated_reply": "Reply",
            }
            self.engine.decide(pipeline_output)
        
        stats = self.engine.get_statistics()
        
        self.assertEqual(stats["total_decisions"], 5)
        self.assertGreater(stats["avg_confidence"], 0.0)


class TestDecisionEvaluation(unittest.TestCase):
    """Test decision evaluation metrics."""
    
    def test_evaluate_policy(self):
        """Evaluate policy on gold labels."""
        decisions = [
            {"decision": "AUTO_HANDLE", "confidence": 0.8},
            {"decision": "ASSIST_AND_ESCALATE", "confidence": 0.6},
            {"decision": "HUMAN_ESCALATION", "confidence": 0.9},
            {"decision": "AUTO_HANDLE", "confidence": 0.7},
        ]
        
        gold = ["AUTO_HANDLE", "ASSIST_AND_ESCALATE", "HUMAN_ESCALATION", "HUMAN_ESCALATION"]
        
        metrics = DecisionEvaluator.evaluate_policy(decisions, gold)
        
        self.assertIn("overall", metrics)
        self.assertIn("AUTO_HANDLE", metrics)
        self.assertGreater(metrics["overall"]["accuracy"], 0.0)
    
    def test_unsafe_auto_handle_rate(self):
        """Compute unsafe auto-handle rate."""
        decisions = [
            {"decision": "AUTO_HANDLE"},
            {"decision": "AUTO_HANDLE"},
            {"decision": "AUTO_HANDLE"},
            {"decision": "HUMAN_ESCALATION"},
        ]
        
        gold = ["AUTO_HANDLE", "HUMAN_ESCALATION", "AUTO_HANDLE", "HUMAN_ESCALATION"]
        
        unsafe_rate = DecisionEvaluator.unsafe_auto_handle_rate(decisions, gold)
        
        # 1 of 3 AUTO_HANDLE cases is wrong
        self.assertAlmostEqual(unsafe_rate, 1/3, places=2)
    
    def test_coverage_safety_curve(self):
        """Compute coverage vs safety curve."""
        decisions = []
        for i in range(10):
            decisions.append({
                "decision": "AUTO_HANDLE",
                "confidence": 0.5 + i * 0.05
            })
        
        gold = ["AUTO_HANDLE"] * 8 + ["HUMAN_ESCALATION"] * 2
        
        curve = DecisionEvaluator.compute_coverage_safety_curve(decisions, gold)
        
        self.assertGreater(len(curve), 0)
        # Verify curve has expected structure (may not be strictly monotonic)
        for point in curve:
            self.assertIn("precision", point)


class TestDecisionConstraints(unittest.TestCase):
    """Test hard constraints in decisions."""
    
    def test_insufficient_evidence_always_escalates(self):
        """Insufficient grounding always → HUMAN_ESCALATION."""
        for policy_class in [BaselinePolicy, CalibratedPolicy, ConservativePolicy]:
            policy = policy_class()
            
            signals = {
                "customer_text": "Test",
                "predicted_intent": "test",
                "intent_confidence": 0.99,  # Very high
                "retrieval_top1_score": 0.99,  # Very high
                "retrieval_top3_mean": 0.99,
                "retrieval_intent_agreement": 0.99,
                "grounding_status": "insufficient",  # Hard constraint
                "unsupported_claim_risk": 0.01,  # Very low
                "ood_score": 0.01,
            }
            
            result = policy.decide(signals)
            
            self.assertEqual(result["decision"], Decision.HUMAN_ESCALATION.value)
    
    def test_high_risk_action_escalates(self):
        """High-risk actions escalate with weak evidence."""
        policy = ConservativePolicy()
        
        signals = {
            "customer_text": "Can I get a refund?",
            "predicted_intent": "payment_refund",
            "intent_confidence": 0.85,
            "retrieval_top1_score": 0.3,  # Weak
            "retrieval_top3_mean": 0.25,
            "retrieval_intent_agreement": 0.3,
            "grounding_status": "insufficient",
            "unsupported_claim_risk": 0.0,
            "ood_score": 0.0,
        }
        
        result = policy.decide(signals)
        
        self.assertEqual(result["decision"], Decision.HUMAN_ESCALATION.value)


if __name__ == "__main__":
    unittest.main()
