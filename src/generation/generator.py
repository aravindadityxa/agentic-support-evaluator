"""
Generator Pipeline

Orchestrates:
1. Intent classification
2. Evidence retrieval
3. Evidence quality gating
4. Reply generation via LLM
5. Response parsing and validation
"""

import sys
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import json
import numpy as np

# Add parent directory to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.generation.llm_client import get_llm_client, LLMProvider
from src.generation.prompt import get_prompt_template, PromptTemplate
from src.generation.grounding_gate import EvidenceQualityGate


class ReplyGenerator:
    """Core reply generation pipeline."""
    
    def __init__(self, llm_provider: Optional[LLMProvider] = None,
                 prompt_template: Optional[PromptTemplate] = None,
                 enable_gating: bool = True):
        """
        Initialize generator.
        
        Args:
            llm_provider: LLM provider instance (uses default if None)
            prompt_template: Prompt template (uses default if None)
            enable_gating: Whether to use evidence quality gating
        """
        
        self.llm = llm_provider or get_llm_client()
        self.prompt = prompt_template or get_prompt_template("grounded_support_v1")
        self.gate = EvidenceQualityGate() if enable_gating else None
        self.call_log = []
    
    def generate(self, customer_text: str, predicted_intent: str,
                 evidence: List[Dict[str, Any]], configuration: str = "grounded",
                 k: Optional[int] = None) -> Dict[str, Any]:
        """
        Generate grounded support reply.
        
        Args:
            customer_text: Customer's message
            predicted_intent: Intent from the intent classifier
            evidence: Retrieved interactions from the retriever
            configuration: "grounded", "no_retrieval", etc.
            k: Number of evidence items to use (overrides evidence length)
        
        Returns:
            Dict with:
            - reply: Generated reply text
            - grounding_status: "supported", "partially_supported", "insufficient"
            - evidence_ranks: Which evidence was used
            - gate_decision: Whether gating approved evidence
            - llm_call_metadata: Cost, tokens, etc.
        """
        
        # Limit evidence to K items
        if k is not None:
            evidence = evidence[:k]
        
        # Apply evidence quality gate if enabled
        gate_decision = None
        gate_metrics = None
        
        if self.gate is not None and evidence:
            gate_decision, gate_metrics = self.gate.apply_threshold(
                evidence, config="moderate"
            )
        
        # Prepare prompt based on configuration
        if configuration == "no_retrieval":
            # No evidence configuration
            self.prompt = get_prompt_template("no_evidence")
            evidence_to_use = []
        else:
            # Grounded configuration (use evidence if available and gating passes)
            self.prompt = get_prompt_template("grounded_support_v1")
            evidence_to_use = evidence if (gate_decision is None or gate_decision) else []
        
        # Get prompts
        system_prompt = self.prompt.system_prompt()
        user_message = self.prompt.user_message(customer_text, predicted_intent, evidence_to_use)
        
        # Call LLM
        try:
            response_text = self.llm.generate(
                system_prompt=system_prompt,
                user_message=user_message,
                temperature=0.7,
                max_tokens=500
            )
        except Exception as e:
            response_text = json.dumps({
                "reply": "I encountered an error processing your request.",
                "grounding_status": "insufficient",
                "evidence_ranks": [],
                "error": str(e)
            })
        
        # Parse response
        parsed_response = self.prompt.parse_response(response_text)
        
        # Log call
        call_record = {
            "customer_text": customer_text,
            "predicted_intent": predicted_intent,
            "configuration": configuration,
            "k": k,
            "evidence_provided": len(evidence_to_use),
            "gate_decision": gate_decision,
            "gate_metrics": gate_metrics,
            "response": parsed_response,
            "raw_response_text": response_text
        }
        self.call_log.append(call_record)
        
        # Build result
        result = {
            "reply": parsed_response.get("reply", ""),
            "grounding_status": parsed_response.get("grounding_status", "unknown"),
            "evidence_ranks": parsed_response.get("evidence_ranks", []),
            "configuration": configuration,
            "k": k,
            "gate_decision": gate_decision,
            "gate_metrics": gate_metrics,
            "evidence_provided": len(evidence_to_use),
            "parse_error": parsed_response.get("parse_error"),
            "customer_text": customer_text,
            "predicted_intent": predicted_intent
        }
        
        return result
    
    def batch_generate(self, queries: List[Dict[str, Any]], 
                      retrieve_fn, configurations: Optional[List[str]] = None,
                      ks: Optional[List[int]] = None) -> List[Dict[str, Any]]:
        """
        Generate replies for multiple queries with different configs.
        
        Args:
            queries: List of {"customer_text", "predicted_intent"}
            retrieve_fn: Function to retrieve evidence given customer_text
            configurations: Configs to test ["grounded", "no_retrieval", ...]
            ks: K values to test [1, 3, 5]
        
        Returns:
            List of generation results
        """
        
        if configurations is None:
            configurations = ["grounded", "no_retrieval"]
        
        if ks is None:
            ks = [1, 3, 5]
        
        results = []
        
        for query_idx, query in enumerate(queries):
            customer_text = query["customer_text"]
            predicted_intent = query.get("predicted_intent", "unknown")
            
            # Retrieve evidence once
            evidence = retrieve_fn(customer_text)
            
            # Generate with each configuration
            for config in configurations:
                if config == "no_retrieval":
                    result = self.generate(
                        customer_text, predicted_intent, [], 
                        configuration=config
                    )
                    results.append(result)
                else:
                    # Test multiple K values
                    for k in ks:
                        result = self.generate(
                            customer_text, predicted_intent, evidence,
                            configuration=config,
                            k=k
                        )
                        results.append(result)
        
        return results
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics from generation calls."""
        
        if not self.call_log:
            return {}
        
        configurations = {}
        for call in self.call_log:
            config = call["configuration"]
            if config not in configurations:
                configurations[config] = []
            configurations[config].append(call)
        
        stats = {
            "total_calls": len(self.call_log),
            "by_configuration": {}
        }
        
        for config, calls in configurations.items():
            stats["by_configuration"][config] = {
                "count": len(calls),
                "avg_evidence_provided": np.mean([c["evidence_provided"] for c in calls]),
                "gate_approved": sum(1 for c in calls if c["gate_decision"] is not False),
                "gate_rejected": sum(1 for c in calls if c["gate_decision"] is False)
            }
        
        return stats


def create_retriever(train_df, tfidf_vectorizer, tfidf_matrix, 
                     embeddings, embedder, intent_classifier) -> callable:
    """
    Factory to create a retrieval function for batch generation.
    
    Returns a function: customer_text -> List[evidence_dict]
    """
    
    def retrieve_evidence(customer_text: str, k: int = 5) -> List[Dict[str, Any]]:
        """Retrieve top-K historical interactions."""
        from sklearn.metrics.pairwise import cosine_similarity
        
        # Get intent
        query_emb = embedder.encode([customer_text])[0]
        intent_probs = intent_classifier.predict_proba([customer_text])[0]
        intent_idx = np.argmax(intent_probs)
        
        # Get semantic similarities
        sim_scores = cosine_similarity([query_emb], embeddings)[0]
        top_indices = np.argsort(sim_scores)[::-1][:k]
        
        evidence = []
        for rank, idx in enumerate(top_indices, 1):
            row = train_df.iloc[idx]
            evidence.append({
                "retrieval_rank": rank,
                "similarity_score": float(sim_scores[idx]),
                "historical_customer_text": row.get("customer_text", ""),
                "historical_support_text": row.get("support_text", ""),
                "historical_timestamp": row.get("timestamp", ""),
                "historical_intent": row.get("candidate_intent", "unknown")
            })
        
        return evidence
    
    return retrieve_evidence


if __name__ == "__main__":
    # Test generator with mock LLM
    from src.generation.llm_client import MockLLMClient
    
    generator = ReplyGenerator(
        llm_provider=MockLLMClient(),
        prompt_template=get_prompt_template("grounded_support_v1"),
        enable_gating=True
    )
    
    # Test query
    customer_text = "My iPhone won't turn on after the iOS update"
    predicted_intent = "software_issue"
    evidence = [
        {
            "retrieval_rank": 1,
            "similarity_score": 0.92,
            "historical_customer_text": "iPhone stuck after update",
            "historical_support_text": "Try force restarting by holding power button.",
            "historical_intent": "software_issue"
        }
    ]
    
    result = generator.generate(customer_text, predicted_intent, evidence)
    print("Generated reply:")
    print(json.dumps(result, indent=2))
