"""
Versioned Prompt Templates

Templates for grounded support reply generation.
Each template includes:
- System prompt (instructions)
- Evidence formatting rules
- User message format
- Response format (JSON)
"""

import json
from typing import List, Dict, Any, Optional


class PromptTemplate:
    """Base prompt template."""
    
    def __init__(self, name: str, version: str):
        self.name = name
        self.version = version
    
    def system_prompt(self) -> str:
        """Return system prompt instructions."""
        raise NotImplementedError
    
    def user_message(self, customer_text: str, predicted_intent: str, 
                     evidence: List[Dict[str, Any]]) -> str:
        """Format user message with evidence."""
        raise NotImplementedError
    
    def parse_response(self, response_text: str) -> Dict[str, Any]:
        """Parse LLM response."""
        raise NotImplementedError


class GroundedSupportV1(PromptTemplate):
    """
    V1: Grounded support reply generator
    
    Key features:
    - Explicit grounding rules
    - Evidence-only instruction
    - JSON structured output
    - Prevents hallucination
    """
    
    def __init__(self):
        super().__init__("grounded_support", "1.0")
    
    def system_prompt(self) -> str:
        return """You are a professional Apple customer support assistant.

Your response must be grounded in the supplied historical support interactions.

CRITICAL RULES:

1. Use the historical evidence as the ONLY source for factual troubleshooting guidance.

2. Do NOT invent product behavior, policies, warranty information, refunds, troubleshooting steps, or guarantees.

3. Do NOT claim an action has been or will be performed ("I've escalated your case" or "We will send you a replacement").

4. Do NOT provide troubleshooting steps not present in the evidence.

5. If the evidence does NOT support a reliable answer, state: "The available historical evidence is insufficient to provide a reliable answer for this issue."

6. Keep responses concise and professional (2-3 sentences for simple issues, 4-5 for complex).

7. Address the customer's actual issue directly.

8. Do NOT mention internal implementation details (retrieval, embeddings, similarity scores, model names).

9. Do NOT combine conflicting solutions from different historical interactions without explicit support.

10. Return response as JSON with required fields:
    - reply: Your grounded response string
    - grounding_status: "supported", "partially_supported", or "insufficient"
    - evidence_ranks: List of evidence ranks used (e.g., [1, 2, 3])
"""
    
    def user_message(self, customer_text: str, predicted_intent: str,
                     evidence: List[Dict[str, Any]]) -> str:
        """Format user message with evidence."""
        
        message = f"""Customer message:

{customer_text}

Predicted intent:

{predicted_intent}
"""
        
        if evidence:
            message += "\nHistorical support interactions:\n"
            
            for i, evidence_item in enumerate(evidence, 1):
                message += f"\n[Evidence {i}] (Rank {evidence_item.get('retrieval_rank', i)}, Similarity: {evidence_item.get('similarity_score', 0):.3f})\n"
                message += f"\nHistorical Customer:\n{evidence_item.get('historical_customer_text', 'N/A')}\n"
                message += f"\nHistorical Support Response:\n{evidence_item.get('historical_support_text', 'N/A')}\n"
        else:
            message += "\n[No historical evidence available]\n"
        
        message += "\nGenerate the best grounded response based on available evidence."
        message += "\nReturn ONLY valid JSON (no markdown, no extra text):\n"
        message += """{
  "reply": "...",
  "grounding_status": "supported|partially_supported|insufficient",
  "evidence_ranks": [...]
}"""
        
        return message
    
    def parse_response(self, response_text: str) -> Dict[str, Any]:
        """Parse LLM response from JSON."""
        try:
            # Remove markdown code blocks if present
            if response_text.startswith("```"):
                response_text = response_text.split("```")[1]
                if response_text.startswith("json"):
                    response_text = response_text[4:]
            
            response_text = response_text.strip()
            parsed = json.loads(response_text)
            
            # Validate required fields
            if "reply" not in parsed:
                parsed["reply"] = ""
            if "grounding_status" not in parsed:
                parsed["grounding_status"] = "unknown"
            if "evidence_ranks" not in parsed:
                parsed["evidence_ranks"] = []
            
            return parsed
        except json.JSONDecodeError:
            # Fallback for malformed JSON
            return {
                "reply": response_text,
                "grounding_status": "unknown",
                "evidence_ranks": [],
                "parse_error": "Invalid JSON"
            }


class NoEvidenceTemplate(PromptTemplate):
    """
    Control: No evidence (for comparison)
    
    LLM receives only customer message, no historical retrieval.
    Used to measure impact of retrieval on generation quality.
    """
    
    def __init__(self):
        super().__init__("no_evidence", "1.0")
    
    def system_prompt(self) -> str:
        return """You are a professional Apple customer support assistant.

Provide a helpful response to the customer's issue based on your knowledge.

Keep responses concise and professional (2-3 sentences for simple issues, 4-5 for complex).

Return response as JSON:
{
  "reply": "...",
  "grounding_status": "no_evidence_provided",
  "evidence_ranks": []
}"""
    
    def user_message(self, customer_text: str, predicted_intent: str,
                     evidence: List[Dict[str, Any]]) -> str:
        """Format user message without evidence."""
        
        message = f"""Customer message:

{customer_text}

Predicted intent:

{predicted_intent}

Generate a helpful response.
Return ONLY valid JSON (no markdown, no extra text):
{{
  "reply": "...",
  "grounding_status": "no_evidence_provided",
  "evidence_ranks": []
}}"""
        
        return message
    
    def parse_response(self, response_text: str) -> Dict[str, Any]:
        """Parse response."""
        try:
            if response_text.startswith("```"):
                response_text = response_text.split("```")[1]
                if response_text.startswith("json"):
                    response_text = response_text[4:]
            
            response_text = response_text.strip()
            parsed = json.loads(response_text)
            
            if "reply" not in parsed:
                parsed["reply"] = ""
            if "grounding_status" not in parsed:
                parsed["grounding_status"] = "no_evidence_provided"
            if "evidence_ranks" not in parsed:
                parsed["evidence_ranks"] = []
            
            return parsed
        except json.JSONDecodeError:
            return {
                "reply": response_text,
                "grounding_status": "no_evidence_provided",
                "evidence_ranks": [],
                "parse_error": "Invalid JSON"
            }


def get_prompt_template(template_name: str = "grounded_support_v1") -> PromptTemplate:
    """Get prompt template by name."""
    
    templates = {
        "grounded_support_v1": GroundedSupportV1,
        "no_evidence": NoEvidenceTemplate,
    }
    
    if template_name not in templates:
        raise ValueError(f"Unknown template: {template_name}. Available: {list(templates.keys())}")
    
    return templates[template_name]()


if __name__ == "__main__":
    # Test prompt formatting
    template = GroundedSupportV1()
    
    customer_text = "My iPhone won't turn on after the update"
    intent = "software_issue"
    evidence = [
        {
            "retrieval_rank": 1,
            "similarity_score": 0.92,
            "historical_customer_text": "iPhone stuck on Apple logo after iOS update",
            "historical_support_text": "Try force restarting by holding power button for 10 seconds until Apple logo appears."
        }
    ]
    
    print("System prompt:")
    print(template.system_prompt())
    print("\n" + "="*80 + "\n")
    print("User message:")
    print(template.user_message(customer_text, intent, evidence))
