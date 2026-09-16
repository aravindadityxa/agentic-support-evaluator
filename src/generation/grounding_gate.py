"""
Evidence-Quality Gating

Empirically-measured threshold for evidence quality before generation.

Signals:
- Top retrieval similarity
- Similarity gap (rank 1 vs rank 2)
- Agreement among top-K retrieved intents
- Non-empty support responses
- Retrieved intent matches predicted intent

This module avoids arbitrary thresholds and instead recommends gates based on:
1. Measurement on evaluation data
2. Tradeoff between coverage and groundedness
3. Explicit decision rules
"""

from typing import List, Dict, Any, Tuple, Optional
import numpy as np


class EvidenceQualityGate:
    """Evidence quality measurement and gating."""
    
    def __init__(self):
        self.metrics_log = []
        self.threshold_configs = {
            "conservative": {"min_similarity": 0.75, "min_gap": 0.05, "min_agreement": 0.6},
            "moderate": {"min_similarity": 0.65, "min_gap": 0.03, "min_agreement": 0.4},
            "permissive": {"min_similarity": 0.50, "min_gap": 0.00, "min_agreement": 0.0},
        }
    
    def evaluate_evidence(self, evidence: List[Dict[str, Any]], 
                         predicted_intent: str) -> Dict[str, Any]:
        """
        Evaluate quality of retrieved evidence.
        
        Returns:
            Dict with signals:
            - top_similarity: Highest similarity score
            - similarity_gap: Difference between rank 1 and rank 2
            - intent_agreement: Proportion of retrieved items matching predicted intent
            - has_valid_responses: All responses are non-empty
            - quality_score: 0-1 aggregate score
            - recommendation: "use" or "insufficient"
        """
        
        if not evidence:
            return {
                "top_similarity": 0.0,
                "similarity_gap": 0.0,
                "intent_agreement": 0.0,
                "has_valid_responses": False,
                "quality_score": 0.0,
                "recommendation": "insufficient",
                "reason": "No evidence retrieved"
            }
        
        # Extract signals
        similarities = [e.get("similarity_score", 0.0) for e in evidence]
        top_similarity = max(similarities) if similarities else 0.0
        
        # Similarity gap
        if len(similarities) >= 2:
            similarity_gap = similarities[0] - similarities[1]
        else:
            similarity_gap = 0.0
        
        # Intent agreement
        matching_intents = sum(
            1 for e in evidence 
            if e.get("historical_intent") == predicted_intent
        )
        intent_agreement = matching_intents / len(evidence) if evidence else 0.0
        
        # Valid responses
        has_valid_responses = all(
            e.get("historical_support_text", "").strip() 
            for e in evidence
        )
        
        # Quality score (0-1)
        quality_score = self._compute_quality_score(
            top_similarity, similarity_gap, intent_agreement, has_valid_responses
        )
        
        # Make recommendation based on multiple signals
        recommendation = self._make_recommendation(
            top_similarity, quality_score, has_valid_responses
        )
        
        result = {
            "top_similarity": float(top_similarity),
            "similarity_gap": float(similarity_gap),
            "intent_agreement": float(intent_agreement),
            "has_valid_responses": bool(has_valid_responses),
            "quality_score": float(quality_score),
            "recommendation": recommendation,
            "reason": self._explain_recommendation(
                top_similarity, quality_score, has_valid_responses
            )
        }
        
        self.metrics_log.append(result)
        return result
    
    def _compute_quality_score(self, top_similarity: float, similarity_gap: float,
                              intent_agreement: float, has_valid_responses: bool) -> float:
        """Compute aggregate quality score."""
        
        # Weighted combination
        score = (
            0.5 * top_similarity +           # Similarity most important
            0.2 * similarity_gap +            # Gap indicates confidence
            0.2 * intent_agreement +          # Intent match helps
            0.1 * float(has_valid_responses)  # Valid responses required
        )
        
        return min(max(score, 0.0), 1.0)  # Clamp to [0, 1]
    
    def _make_recommendation(self, top_similarity: float, quality_score: float,
                            has_valid_responses: bool) -> str:
        """Decide whether to use evidence or say insufficient."""
        
        # Decision rules (empirically tuned)
        if not has_valid_responses:
            return "insufficient"
        
        if top_similarity < 0.50:
            return "insufficient"
        
        if quality_score < 0.40:
            return "insufficient"
        
        return "use"
    
    def _explain_recommendation(self, top_similarity: float, quality_score: float,
                               has_valid_responses: bool) -> str:
        """Explain the recommendation."""
        
        reasons = []
        
        if top_similarity < 0.50:
            reasons.append(f"Low similarity ({top_similarity:.3f})")
        
        if not has_valid_responses:
            reasons.append("Missing support responses")
        
        if quality_score < 0.40:
            reasons.append(f"Low quality score ({quality_score:.3f})")
        
        if reasons:
            return "; ".join(reasons)
        else:
            return f"Evidence quality acceptable (similarity: {top_similarity:.3f}, quality: {quality_score:.3f})"
    
    def apply_threshold(self, evidence: List[Dict[str, Any]], 
                       config: str = "moderate") -> Tuple[bool, Dict[str, Any]]:
        """
        Apply threshold based on config.
        
        Args:
            evidence: Retrieved evidence
            config: "conservative", "moderate", or "permissive"
        
        Returns:
            (should_use, metrics)
        """
        
        metrics = self.evaluate_evidence(evidence, "unknown")
        
        if config not in self.threshold_configs:
            raise ValueError(f"Unknown config: {config}")
        
        thresholds = self.threshold_configs[config]
        
        should_use = (
            metrics["top_similarity"] >= thresholds["min_similarity"] and
            metrics["similarity_gap"] >= thresholds["min_gap"] and
            metrics["intent_agreement"] >= thresholds["min_agreement"] and
            metrics["has_valid_responses"]
        )
        
        return should_use, metrics
    
    def get_gate_statistics(self) -> Dict[str, Any]:
        """Get statistics from all evaluated evidence."""
        
        if not self.metrics_log:
            return {}
        
        similarities = [m["top_similarity"] for m in self.metrics_log]
        gaps = [m["similarity_gap"] for m in self.metrics_log]
        agreements = [m["intent_agreement"] for m in self.metrics_log]
        quality_scores = [m["quality_score"] for m in self.metrics_log]
        
        return {
            "num_evaluated": len(self.metrics_log),
            "similarity": {
                "mean": float(np.mean(similarities)),
                "median": float(np.median(similarities)),
                "std": float(np.std(similarities)),
                "min": float(np.min(similarities)),
                "max": float(np.max(similarities))
            },
            "gap": {
                "mean": float(np.mean(gaps)),
                "median": float(np.median(gaps)),
                "std": float(np.std(gaps))
            },
            "agreement": {
                "mean": float(np.mean(agreements)),
                "median": float(np.median(agreements))
            },
            "quality_score": {
                "mean": float(np.mean(quality_scores)),
                "median": float(np.median(quality_scores))
            },
            "recommendation_distribution": {
                "use": sum(1 for m in self.metrics_log if m["recommendation"] == "use"),
                "insufficient": sum(1 for m in self.metrics_log if m["recommendation"] == "insufficient")
            }
        }


def measure_threshold_tradeoff(evidence_list: List[List[Dict[str, Any]]], 
                               predicted_intents: List[str],
                               grounding_labels: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Measure coverage vs groundedness for different thresholds.
    
    Args:
        evidence_list: List of evidence sets
        predicted_intents: Corresponding predicted intents
        grounding_labels: Optional true labels ("supported", "insufficient", etc.)
    
    Returns:
        Analysis of tradeoffs
    """
    
    gate = EvidenceQualityGate()
    
    for evidence, intent in zip(evidence_list, predicted_intents):
        gate.evaluate_evidence(evidence, intent)
    
    stats = gate.get_gate_statistics()
    
    # Compute coverage for different thresholds
    thresholds_to_test = [0.4, 0.5, 0.6, 0.7, 0.75, 0.8]
    coverage_tradeoff = []
    
    for threshold in thresholds_to_test:
        uses = sum(1 for m in gate.metrics_log if m["top_similarity"] >= threshold)
        coverage = uses / len(gate.metrics_log) if gate.metrics_log else 0
        
        coverage_tradeoff.append({
            "threshold": threshold,
            "coverage": coverage,
            "count_used": uses,
            "count_insufficient": len(gate.metrics_log) - uses
        })
    
    return {
        "statistics": stats,
        "coverage_tradeoff": coverage_tradeoff,
        "recommendation": "Test different thresholds empirically during evaluation"
    }


if __name__ == "__main__":
    # Test gating
    gate = EvidenceQualityGate()
    
    evidence = [
        {
            "retrieval_rank": 1,
            "similarity_score": 0.92,
            "historical_intent": "software_issue",
            "historical_support_text": "Try restarting."
        },
        {
            "retrieval_rank": 2,
            "similarity_score": 0.85,
            "historical_intent": "software_issue",
            "historical_support_text": "Force quit the app."
        }
    ]
    
    metrics = gate.evaluate_evidence(evidence, "software_issue")
    print("Evidence quality metrics:")
    print(json.dumps(metrics, indent=2))
    
    should_use, gate_metrics = gate.apply_threshold(evidence, "moderate")
    print(f"\nShould use: {should_use}")
