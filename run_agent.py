#!/usr/bin/env python3
"""
MAIN ENTRY POINT: AI Support Agent

End-to-end pipeline for customer support query processing:
  1. Intent Classification
  2. Historical Retrieval
  3. Grounded Reply Generation
  4. Safety/Escalation Decision

Usage:
  python run_agent.py --text "customer message here"
  
Example:
  python run_agent.py --text "My iPhone keeps crashing after the update"
"""

import sys
import argparse
import json
from pathlib import Path

# Add src to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

# ============================================================================
# INTENT CLASSIFICATION
# ============================================================================

def classify_intent(customer_text: str) -> dict:
    """
    Classify customer message into one of 10 intent categories.
    
    Returns: {
        'predicted_intent': str,
        'intent_confidence': float (0-1),
        'intent_candidates': list of (intent, confidence) tuples
    }
    """
    # 10-intent taxonomy
    INTENTS = [
        "software_issue",
        "app_issue", 
        "device_hardware_issue",
        "performance_issue",
        "account_signin",
        "billing_payment",
        "connectivity",
        "data_backup",
        "complaint_feedback",
        "other_unclear"
    ]
    
    # In production: Load actual classifier
    # For demo: Use simple heuristic-based classification
    text_lower = customer_text.lower()
    
    intent_scores = {}
    
    # Heuristic scoring (demo only - real system uses TF-IDF + Logistic Regression)
    if any(word in text_lower for word in ["crash", "freeze", "lag", "slow"]):
        intent_scores["performance_issue"] = 0.85
    elif any(word in text_lower for word in ["update", "install", "version", "bug"]):
        intent_scores["software_issue"] = 0.80
    elif any(word in text_lower for word in ["wifi", "network", "connection", "signal"]):
        intent_scores["connectivity"] = 0.80
    elif any(word in text_lower for word in ["login", "password", "signin", "account"]):
        intent_scores["account_signin"] = 0.75
    elif any(word in text_lower for word in ["charge", "battery", "power", "drain"]):
        intent_scores["device_hardware_issue"] = 0.75
    elif any(word in text_lower for word in ["cost", "price", "bill", "payment", "charge"]):
        intent_scores["billing_payment"] = 0.75
    elif any(word in text_lower for word in ["backup", "restore", "icloud", "data"]):
        intent_scores["data_backup"] = 0.75
    else:
        intent_scores["other_unclear"] = 0.50
    
    # Get top intent
    sorted_intents = sorted(intent_scores.items(), key=lambda x: x[1], reverse=True)
    predicted_intent = sorted_intents[0][0]
    intent_confidence = sorted_intents[0][1]
    
    return {
        "predicted_intent": predicted_intent,
        "intent_confidence": float(intent_confidence),
        "intent_candidates": [(intent, float(conf)) for intent, conf in sorted_intents[:3]]
    }

# ============================================================================
# HISTORICAL RETRIEVAL
# ============================================================================

def retrieve_evidence(customer_text: str, predicted_intent: str) -> dict:
    """
    Retrieve top-3 historical resolutions similar to customer query.
    
    Returns: {
        'retrieved_cases': list of {
            'similarity_score': float,
            'historical_query': str,
            'historical_resolution': str
        },
        'retrieval_quality': str (good/moderate/weak)
    }
    """
    # In production: Use semantic embeddings (all-MiniLM-L6-v2)
    # For demo: Simple similarity heuristics
    
    # Mock historical case database
    historical_cases = {
        "performance_issue": [
            {
                "query": "iPhone keeps freezing after update",
                "resolution": "Try force restarting your device: Hold both volume buttons and side button until power off screen appears, then restart.",
                "score": 0.78
            },
            {
                "query": "Apps running very slowly",
                "resolution": "Clear cache by going to Settings > General > iPhone Storage, then offload unused apps.",
                "score": 0.65
            },
            {
                "query": "Phone lag and stuttering",
                "resolution": "Close background apps and restart your device. If persists, check Settings > Battery Health.",
                "score": 0.58
            }
        ],
        "connectivity": [
            {
                "query": "WiFi not connecting",
                "resolution": "Forget the WiFi network in Settings > WiFi, then reconnect. If still not working, restart your WiFi router.",
                "score": 0.82
            },
            {
                "query": "Weak WiFi signal",
                "resolution": "Move closer to the router or restart both your device and router.",
                "score": 0.70
            }
        ],
        "software_issue": [
            {
                "query": "App crashes after update",
                "resolution": "Update the app to the latest version from App Store, or force quit and reopen.",
                "score": 0.75
            },
            {
                "query": "iOS update failed",
                "resolution": "Connect to WiFi, ensure 50% battery, then try Settings > General > Software Update again.",
                "score": 0.72
            }
        ]
    }
    
    # Get cases for predicted intent, or use generic fallback
    cases = historical_cases.get(predicted_intent, [])
    if not cases:
        cases = [
            {
                "query": "General device issue",
                "resolution": "Restart your device first. If the issue persists, please contact Apple Support.",
                "score": 0.40
            }
        ]
    
    # Take top 3
    retrieved_cases = cases[:3]
    
    # Determine overall quality
    if retrieved_cases[0]["score"] > 0.75:
        quality = "good"
    elif retrieved_cases[0]["score"] > 0.60:
        quality = "moderate"
    else:
        quality = "weak"
    
    return {
        "retrieved_cases": [
            {
                "similarity_score": float(case["score"]),
                "historical_query": case["query"],
                "historical_resolution": case["resolution"]
            }
            for case in retrieved_cases
        ],
        "retrieval_quality": quality,
        "mean_similarity": float(sum(c["score"] for c in retrieved_cases) / len(retrieved_cases))
    }

# ============================================================================
# GROUNDED REPLY GENERATION
# ============================================================================

def generate_reply(customer_text: str, predicted_intent: str, retrieved_evidence: list) -> dict:
    """
    Generate grounded customer reply using LLM + retrieval evidence.
    
    Returns: {
        'generated_reply': str,
        'grounding_status': str (supported/partially_supported/insufficient),
        'unsupported_claim_risk': float (0-1),
        'generation_quality_score': float (1-5)
    }
    """
    # In production: Call LLM API with evidence-based prompting
    # For demo: Generate template-based replies
    
    if not retrieved_evidence:
        return {
            "generated_reply": "Thank you for contacting Apple Support. I need more information to assist you further. Please contact our support team for personalized help.",
            "grounding_status": "insufficient",
            "unsupported_claim_risk": 0.9,
            "generation_quality_score": 1.5
        }
    
    # Extract primary resolution from top retrieved case
    primary_resolution = retrieved_evidence[0]["historical_resolution"] if retrieved_evidence else ""
    
    # Generate contextualized reply
    template_replies = {
        "performance_issue": f"Thanks for reaching out! I understand your device is running slowly. {primary_resolution} This should improve performance in most cases.",
        "connectivity": f"I'm sorry you're having connectivity issues. {primary_resolution} Let me know if this helps!",
        "software_issue": f"We appreciate your patience. {primary_resolution} Please try this and let us know if the issue is resolved.",
        "device_hardware_issue": f"I'm happy to help with your device issue. {primary_resolution} If you continue experiencing problems, we may need to arrange service.",
        "account_signin": f"Help with your account: {primary_resolution}",
        "billing_payment": f"Regarding your billing question: {primary_resolution}",
    }
    
    reply = template_replies.get(predicted_intent, f"Thank you for your message. {primary_resolution}")
    
    # Assess grounding status
    if retrieved_evidence and retrieved_evidence[0]["similarity_score"] > 0.70:
        grounding = "supported"
        quality = 4.2
        risk = 0.05
    elif retrieved_evidence and retrieved_evidence[0]["similarity_score"] > 0.50:
        grounding = "partially_supported"
        quality = 3.5
        risk = 0.15
    else:
        grounding = "insufficient"
        quality = 2.0
        risk = 0.35
    
    return {
        "generated_reply": reply,
        "grounding_status": grounding,
        "unsupported_claim_risk": float(risk),
        "generation_quality_score": float(quality)
    }

# ============================================================================
# DECISION & ESCALATION ROUTING
# ============================================================================

def decide_routing(intent_confidence: float, retrieval_quality: str, 
                  grounding_status: str, unsupported_risk: float) -> dict:
    """
    Decide whether to auto-handle, assist+escalate, or escalate to human.
    
    Returns: {
        'decision': str (AUTO_HANDLE / ASSIST_AND_ESCALATE / HUMAN_ESCALATION),
        'decision_confidence': float (0-1),
        'reason_codes': list of str,
        'recommendation': str
    }
    """
    reason_codes = []
    risk_score = 0.0
    
    # Risk scoring
    if intent_confidence < 0.6:
        reason_codes.append("intent_uncertain")
        risk_score += 0.3
    
    if retrieval_quality == "weak":
        reason_codes.append("weak_retrieval")
        risk_score += 0.4
    elif retrieval_quality == "moderate":
        reason_codes.append("moderate_retrieval")
        risk_score += 0.2
    
    if grounding_status == "insufficient":
        reason_codes.append("grounding_risk")
        risk_score += 0.5
    elif grounding_status == "partially_supported":
        reason_codes.append("partial_grounding")
        risk_score += 0.2
    
    if unsupported_risk > 0.2:
        reason_codes.append("unsupported_claims")
        risk_score += 0.3
    
    risk_score = risk_score / 4.0  # Normalize
    
    # Decision logic
    if risk_score < 0.2:
        decision = "AUTO_HANDLE"
        confidence = 0.90
        recommendation = "✓ Safe to send reply automatically. Customer issue clear and evidence strong."
    elif risk_score < 0.4:
        decision = "ASSIST_AND_ESCALATE"
        confidence = 0.75
        recommendation = "⚠ Show reply to customer with escalation option available. Some uncertainty in evidence."
    else:
        decision = "HUMAN_ESCALATION"
        confidence = 0.85
        recommendation = "→ Route to human support. Evidence insufficient or issue too complex for automation."
    
    return {
        "decision": decision,
        "decision_confidence": float(confidence),
        "reason_codes": reason_codes,
        "risk_score": float(risk_score),
        "recommendation": recommendation
    }

# ============================================================================
# MAIN PIPELINE
# ============================================================================

def run_agent(customer_text: str) -> dict:
    """
    Complete end-to-end pipeline.
    
    Args:
        customer_text: Customer support message
    
    Returns:
        dict with complete pipeline output
    """
    print("\n" + "="*80)
    print("HIVER AGENTIC SUPPORT EVALUATOR - COMPLETE PIPELINE")
    print("="*80)
    
    # Intent Classification
    print("\n[Intent Classification]...")
    intent_result = classify_intent(customer_text)
    print(f"  → Predicted Intent: {intent_result['predicted_intent']}")
    print(f"  → Confidence: {intent_result['intent_confidence']:.1%}")
    
    # Historical Retrieval
    print("\n[Historical Retrieval]...")
    retrieval_result = retrieve_evidence(customer_text, intent_result["predicted_intent"])
    print(f"  → Retrieval Quality: {retrieval_result['retrieval_quality']}")
    print(f"  → Mean Similarity: {retrieval_result['mean_similarity']:.3f}")
    print(f"  → Top Case Similarity: {retrieval_result['retrieved_cases'][0]['similarity_score']:.3f}")
    
    # Grounded Generation
    print("\n[Grounded Reply Generation]...")
    generation_result = generate_reply(
        customer_text,
        intent_result["predicted_intent"],
        retrieval_result["retrieved_cases"]
    )
    print(f"  → Grounding Status: {generation_result['grounding_status']}")
    print(f"  → Quality Score: {generation_result['generation_quality_score']:.2f}/5.0")
    print(f"  → Unsupported Claim Risk: {generation_result['unsupported_claim_risk']:.1%}")
    
    # Decision Routing
    print("\n[Decision & Escalation Routing]...")
    decision_result = decide_routing(
        intent_result["intent_confidence"],
        retrieval_result["retrieval_quality"],
        generation_result["grounding_status"],
        generation_result["unsupported_claim_risk"]
    )
    print(f"  → Decision: {decision_result['decision']}")
    print(f"  → Confidence: {decision_result['decision_confidence']:.1%}")
    print(f"  → Risk Score: {decision_result['risk_score']:.3f}")
    
    # ========================================================================
    # OUTPUT
    # ========================================================================
    print("\n" + "="*80)
    print("FINAL OUTPUT")
    print("="*80)
    
    output = {
        "input": {
            "customer_text": customer_text
        },
        "stage_1_intent": {
            "predicted_intent": intent_result["predicted_intent"],
            "intent_confidence": intent_result["intent_confidence"],
            "candidates": intent_result["intent_candidates"]
        },
        "stage_2_retrieval": {
            "top_case": {
                "similarity": retrieval_result["retrieved_cases"][0]["similarity_score"],
                "query": retrieval_result["retrieved_cases"][0]["historical_query"],
                "resolution": retrieval_result["retrieved_cases"][0]["historical_resolution"]
            },
            "retrieval_quality": retrieval_result["retrieval_quality"],
            "mean_similarity": retrieval_result["mean_similarity"]
        },
        "stage_3_generation": {
            "generated_reply": generation_result["generated_reply"],
            "grounding_status": generation_result["grounding_status"],
            "quality_score": generation_result["generation_quality_score"],
            "unsupported_claim_risk": generation_result["unsupported_claim_risk"]
        },
        "stage_4_decision": {
            "decision": decision_result["decision"],
            "decision_confidence": decision_result["decision_confidence"],
            "reason_codes": decision_result["reason_codes"],
            "risk_score": decision_result["risk_score"],
            "recommendation": decision_result["recommendation"]
        }
    }
    
    # Print formatted output
    print(f"\n📤 CUSTOMER REQUEST:")
    print(f"  {customer_text}\n")
    
    print(f"🎯 INTENT: {output['stage_1_intent']['predicted_intent']} ({output['stage_1_intent']['intent_confidence']:.0%})")
    print(f"📚 RETRIEVAL: {output['stage_2_retrieval']['retrieval_quality']} (similarity: {output['stage_2_retrieval']['mean_similarity']:.3f})")
    print(f"💬 REPLY: {generation_result['grounding_status']} grounding ({generation_result['generation_quality_score']:.1f}/5)")
    
    print(f"\n📋 GENERATED REPLY:")
    print(f"  \"{output['stage_3_generation']['generated_reply']}\"\n")
    
    print(f"🔴 DECISION: {output['stage_4_decision']['decision']}")
    print(f"   Confidence: {output['stage_4_decision']['decision_confidence']:.0%}")
    print(f"   Risk Score: {output['stage_4_decision']['risk_score']:.3f}")
    print(f"   Reason Codes: {', '.join(output['stage_4_decision']['reason_codes']) if output['stage_4_decision']['reason_codes'] else 'none'}")
    print(f"\n   → {output['stage_4_decision']['recommendation']}")
    
    print("\n" + "="*80)
    
    return output

# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Hiver Agentic Support Evaluator - End-to-End Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_agent.py --text "My iPhone keeps crashing after the update"
  python run_agent.py --text "WiFi not connecting"
  python run_agent.py --text "Can I reset my password?"
        """
    )
    
    parser.add_argument(
        "--text",
        type=str,
        required=True,
        help="Customer support message"
    )
    
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON"
    )
    
    args = parser.parse_args()
    
    # Run pipeline
    result = run_agent(args.text)
    
    # Optional JSON output
    if args.json:
        print("\nJSON Output:")
        print(json.dumps(result, indent=2))
    
    return result

if __name__ == "__main__":
    main()
