#!/usr/bin/env python3
"""Create final golden evaluation set."""

import pandas as pd
import numpy as np
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "processed"
EVAL_DIR = PROJECT_ROOT / "evaluation"
EVAL_DIR.mkdir(exist_ok=True)

print("="*80)
print("Creating Final Golden Set (249 Cases)")
print("="*80)

# Load validation data
val_df = pd.read_csv(DATA_DIR / "validation_labeled.csv")
print(f"\n✓ Loaded validation data: {len(val_df)} interactions")

# 10-intent taxonomy
INTENT_TAXONOMY = [
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

def assess_difficulty(text, intent):
    """Assess query difficulty."""
    text_lower = str(text).lower() if pd.notna(text) else ""
    text_len = len(text_lower)
    
    if intent in ["other_unclear", "complaint_feedback"]:
        return "hard"
    if text_len < 15:
        return "hard"
    if intent in ["software_issue", "app_issue"] and text_len > 40:
        if any(w in text_lower for w in ["error", "update", "crash", "freeze"]):
            return "easy"
    return "medium"

# Create golden set
print("\nCreating stratified sample (25 cases per intent)...")
np.random.seed(42)
golden_set_list = []

for intent in INTENT_TAXONOMY:
    intent_df = val_df[val_df.get("candidate_intent", "") == intent]
    if len(intent_df) == 0:
        print(f"  ⚠ {intent}: NO CASES")
        continue
    
    take = min(25, len(intent_df))
    sampled = intent_df.sample(n=take, random_state=42)
    print(f"  {intent:<25} {take:2d} cases")
    
    for local_idx, (_, row) in enumerate(sampled.iterrows()):
        customer_text = str(row.get("customer_text", "")).strip()
        support_text = str(row.get("support_text", "")).strip()
        
        if pd.notna(support_text) and len(support_text) > 10:
            expected_resolution = support_text[:200]
            resolution_confidence = 0.85
        else:
            expected_resolution = "INSUFFICIENT_EVIDENCE"
            resolution_confidence = 0.0
        
        golden_set_list.append({
            "query_id": f"{intent}_{local_idx:03d}",
            "customer_text": customer_text,
            "source_timestamp": row.get("created_at", ""),
            "source_split": "validation",
            "gold_intent": intent,
            "intent_annotation_method": "candidate_label",
            "expected_resolution": expected_resolution,
            "resolution_source": "historical_response" if expected_resolution != "INSUFFICIENT_EVIDENCE" else "missing",
            "resolution_confidence": resolution_confidence,
            "gold_retrieval_relevance": None,
            "gold_auto_handle_decision": None,
            "decision_reason": None,
            "difficulty": assess_difficulty(customer_text, intent),
            "annotation_source": "automated",
            "human_reviewed": False,
        })

golden_set_df = pd.DataFrame(golden_set_list)

print(f"\n✓ Created golden set: {len(golden_set_df)} cases")

# Verify schema
required_fields = [
    "query_id", "customer_text", "source_timestamp", "source_split",
    "gold_intent", "intent_annotation_method", "expected_resolution",
    "resolution_source", "resolution_confidence", "difficulty",
    "annotation_source", "human_reviewed"
]

for field in required_fields:
    if field not in golden_set_df.columns:
        print(f"ERROR: Missing {field}")
        exit(1)

print(f"✓ Schema verified")

# Distribution analysis
print(f"\nIntent distribution:")
for intent in INTENT_TAXONOMY:
    count = len(golden_set_df[golden_set_df["gold_intent"] == intent])
    pct = 100 * count / len(golden_set_df)
    print(f"  {intent:<25} {count:3d} ({pct:5.1f}%)")

print(f"\nDifficulty distribution:")
for diff in ["easy", "medium", "hard"]:
    count = len(golden_set_df[golden_set_df["difficulty"] == diff])
    pct = 100 * count / len(golden_set_df)
    print(f"  {diff:<25} {count:3d} ({pct:5.1f}%)")

# Save
golden_set_df.to_csv(EVAL_DIR / "final_golden_set.csv", index=False)
print(f"\n✓ Saved: {EVAL_DIR / 'final_golden_set.csv'}")

metadata = {
    "total_cases": len(golden_set_df),
    "intents_covered": len(INTENT_TAXONOMY),
    "cases_per_intent": 25,
    "source": "validation_labeled.csv",
    "stratification": "equal_by_intent",
    "annotation_provenance": "automated (no human review)",
}

with open(EVAL_DIR / "final_golden_set_metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)
print(f"✓ Saved: {EVAL_DIR / 'final_golden_set_metadata.json'}")
print("\n" + "="*80)
print("Golden set creation complete")
print("="*80)
