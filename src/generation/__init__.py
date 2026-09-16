"""
Grounded Support Reply Generation

Submodules:
- llm_client: LLM provider abstraction (OpenAI, mock mode)
- prompt: Versioned prompt templates with evidence formatting
- generator: Core generation pipeline (intent → retrieve → format → generate)
- grounding_gate: Evidence-quality gating and threshold measurement
"""

__version__ = "1.0.0"
