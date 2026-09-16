"""
Out-Of-Distribution (OOD) Detection

Simple heuristic OOD detection to identify queries that don't resemble
the historical AppleSupport corpus.

Not a formal OOD detector, but practical signals for routing.
"""

from typing import Dict, Any
import numpy as np


class SimpleOODDetector:
    """
    Heuristic out-of-distribution detection.
    
    Signals:
    - Low embedding similarity (nothing matches)
    - Low classifier confidence (can't decide intent)
    - Disagreement among predictions
    - Unusual query length/structure
    """
    
    def __init__(self, historical_stats: Dict[str, Any] = None):
        """
        Initialize with optional historical corpus statistics.
        
        Args:
            historical_stats: Stats from training corpus (optional)
        """
        
        # Default statistics (can be updated with real data)
        self.avg_query_length = 60
        self.std_query_length = 40
        self.avg_similarity = 0.45
        self.std_similarity = 0.25
        self.min_similarity_threshold = 0.20
        
        if historical_stats:
            self.avg_query_length = historical_stats.get("avg_query_length", 60)
            self.std_query_length = historical_stats.get("std_query_length", 40)
            self.avg_similarity = historical_stats.get("avg_similarity", 0.45)
            self.std_similarity = historical_stats.get("std_similarity", 0.25)
    
    def detect(self, signals: Dict[str, Any]) -> Dict[str, Any]:
        """
        Detect OOD indicators.
        
        Signals expected:
        - customer_text: str
        - predicted_intent: str
        - intent_confidence: float
        - intent_top_k: list of (intent, prob)
        - retrieval_top1_score: float
        - retrieval_top3_mean: float
        
        Returns:
        {
            "ood_score": float (0-1),
            "is_ood": bool,
            "signals": dict,
            "reasons": list
        }
        """
        
        ood_indicators = []
        signal_values = {}
        
        # Signal 1: Very low retrieval similarity
        top1_sim = signals.get("retrieval_top1_score", 0.0)
        signal_values["top1_similarity"] = top1_sim
        
        if top1_sim < 0.30:
            ood_indicators.append("very_low_retrieval_similarity")
            signal_values["top1_sim_score"] = 0.8
        elif top1_sim < 0.45:
            ood_indicators.append("low_retrieval_similarity")
            signal_values["top1_sim_score"] = 0.5
        else:
            signal_values["top1_sim_score"] = 0.1
        
        # Signal 2: Very low intent confidence
        intent_conf = signals.get("intent_confidence", 0.5)
        signal_values["intent_confidence"] = intent_conf
        
        if intent_conf < 0.50:
            ood_indicators.append("very_low_intent_confidence")
            signal_values["intent_conf_score"] = 0.7
        elif intent_conf < 0.65:
            ood_indicators.append("low_intent_confidence")
            signal_values["intent_conf_score"] = 0.4
        else:
            signal_values["intent_conf_score"] = 0.1
        
        # Signal 3: High disagreement among top predictions
        intent_top_k = signals.get("intent_top_k", [])
        if len(intent_top_k) >= 2:
            top_prob = intent_top_k[0][1]
            second_prob = intent_top_k[1][1]
            gap = top_prob - second_prob
            
            signal_values["intent_probability_gap"] = gap
            
            if gap < 0.10:
                ood_indicators.append("high_intent_disagreement")
                signal_values["disagreement_score"] = 0.6
            elif gap < 0.20:
                signal_values["disagreement_score"] = 0.3
            else:
                signal_values["disagreement_score"] = 0.05
        else:
            signal_values["disagreement_score"] = 0.0
        
        # Signal 4: Unusual query length
        customer_text = signals.get("customer_text", "")
        query_length = len(customer_text)
        signal_values["query_length"] = query_length
        
        # Standardize
        if self.std_query_length > 0:
            z_score = abs((query_length - self.avg_query_length) / self.std_query_length)
        else:
            z_score = 0.0
        
        if z_score > 3.0:  # Very unusual
            ood_indicators.append("very_unusual_query_length")
            signal_values["length_score"] = 0.5
        elif z_score > 2.0:
            ood_indicators.append("unusual_query_length")
            signal_values["length_score"] = 0.25
        else:
            signal_values["length_score"] = 0.05
        
        # Signal 5: Unusual query characteristics (heuristics)
        unusual_markers = self._detect_unusual_markers(customer_text)
        if unusual_markers:
            ood_indicators.append("unusual_query_markers")
            signal_values["markers"] = unusual_markers
            signal_values["markers_score"] = 0.3
        else:
            signal_values["markers_score"] = 0.0
        
        # Compute aggregate OOD score
        scores = [
            signal_values.get("top1_sim_score", 0.0),
            signal_values.get("intent_conf_score", 0.0),
            signal_values.get("disagreement_score", 0.0),
            signal_values.get("length_score", 0.0),
            signal_values.get("markers_score", 0.0),
        ]
        
        # Weighted average (retrieval similarity is most important)
        weights = [0.35, 0.25, 0.15, 0.15, 0.10]
        ood_score = sum(s * w for s, w in zip(scores, weights))
        
        # Threshold for is_ood
        is_ood = ood_score >= 0.50
        
        return {
            "ood_score": float(ood_score),
            "is_ood": is_ood,
            "indicators": ood_indicators,
            "signals": signal_values,
            "recommendation": "escalate" if is_ood else "continue"
        }
    
    def _detect_unusual_markers(self, text: str) -> list:
        """Detect unusual query characteristics."""
        
        markers = []
        text_lower = text.lower()
        
        # Extremely short or long
        if len(text) < 10:
            markers.append("very_short_query")
        elif len(text) > 500:
            markers.append("very_long_query")
        
        # No punctuation (suspicious)
        if len(text) > 20 and text.count(".") + text.count("?") + text.count("!") == 0:
            markers.append("no_punctuation")
        
        # Lots of special characters
        special_chars = sum(1 for c in text if not c.isalnum() and c != " ")
        if special_chars > len(text) * 0.3:
            markers.append("excessive_special_chars")
        
        # All caps
        if len(text) > 20 and text.isupper():
            markers.append("all_caps")
        
        # Repetitive text
        words = text_lower.split()
        if len(words) > 5:
            unique_ratio = len(set(words)) / len(words)
            if unique_ratio < 0.4:
                markers.append("repetitive_text")
        
        return markers
