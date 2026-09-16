"""
LLM Client Abstraction

Configurable LLM provider:
- OpenAI (if API key available)
- Mock mode (deterministic for testing)

Environment variables:
- LLM_PROVIDER: 'openai' or 'mock' (default: auto-detect)
- OPENAI_API_KEY: OpenAI API key (optional, falls back to mock)
- LLM_MODEL: Model name (default: gpt-4-turbo for OpenAI, mock for mock)
"""

import json
import os
import hashlib
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import warnings

class LLMProvider(ABC):
    """Abstract base for LLM providers."""
    
    @abstractmethod
    def generate(self, system_prompt: str, user_message: str, 
                 temperature: float = 0.7, max_tokens: int = 500) -> str:
        """Generate text response."""
        pass


class OpenAIClient(LLMProvider):
    """OpenAI API client."""
    
    def __init__(self, api_key: str, model: str = "gpt-4-turbo-preview"):
        self.api_key = api_key
        self.model = model
        self.client = None
        self._init_client()
    
    def _init_client(self):
        """Initialize OpenAI client."""
        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=self.api_key)
        except ImportError:
            raise ImportError("openai package required: pip install openai")
    
    def generate(self, system_prompt: str, user_message: str, 
                 temperature: float = 0.7, max_tokens: int = 500) -> str:
        """Generate reply using OpenAI API."""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message}
                ],
                temperature=temperature,
                max_tokens=max_tokens
            )
            return response.choices[0].message.content
        except Exception as e:
            raise RuntimeError(f"OpenAI API error: {str(e)}")


class MockLLMClient(LLMProvider):
    """Deterministic mock LLM for testing without API keys."""
    
    def __init__(self):
        self.call_count = 0
        self.generated_responses = {}
    
    def generate(self, system_prompt: str, user_message: str,
                 temperature: float = 0.7, max_tokens: int = 500) -> str:
        """Generate deterministic mock response based on input hash."""
        self.call_count += 1
        
        # Create deterministic response based on input
        input_key = f"{system_prompt}|{user_message}".encode('utf-8')
        input_hash = hashlib.md5(input_key).hexdigest()[:8]
        
        # Check cache
        if input_hash in self.generated_responses:
            return self.generated_responses[input_hash]
        
        # Generate deterministic mock response
        response = self._generate_mock_response(system_prompt, user_message, input_hash)
        self.generated_responses[input_hash] = response
        return response
    
    def _generate_mock_response(self, system_prompt: str, user_message: str, 
                               hash_suffix: str) -> str:
        """Generate deterministic response for testing."""
        
        # Parse user message for customer text and evidence
        try:
            # Extract customer message from structured prompt
            if "Customer message:" in user_message:
                customer_start = user_message.find("Customer message:") + len("Customer message:")
                customer_end = user_message.find("Predicted intent:")
                customer_text = user_message[customer_start:customer_end].strip()
            else:
                customer_text = user_message[:50]
            
            # Check if evidence exists
            has_evidence = "Historical support interactions:" in user_message
            
            # Generate mock reply based on content
            if has_evidence:
                grounding = "supported"
                reply = self._mock_grounded_reply(customer_text, hash_suffix)
            else:
                grounding = "insufficient"
                reply = "I don't have historical evidence to provide a reliable answer."
            
            # Return structured JSON
            result = {
                "reply": reply,
                "grounding_status": grounding,
                "evidence_ranks": self._extract_evidence_ranks(user_message),
                "mock_mode": True
            }
            return json.dumps(result)
        
        except Exception as e:
            # Fallback response
            result = {
                "reply": "I encountered an error processing your request.",
                "grounding_status": "insufficient",
                "evidence_ranks": [],
                "mock_mode": True,
                "error": str(e)
            }
            return json.dumps(result)
    
    def _mock_grounded_reply(self, customer_text: str, hash_suffix: str) -> str:
        """Generate a plausible mock grounded reply."""
        
        # Map common keywords to mock responses
        keywords = customer_text.lower()
        
        if "won't" in keywords and "turn" in keywords:
            return "Try force restarting your device by holding down the power button for 10 seconds until the Apple logo appears. This resolves most power-related issues based on historical support data."
        elif "slow" in keywords or "lag" in keywords:
            return "Background app refresh and cached data can cause slowness. Try going to Settings > General > iPhone Storage and review large apps. Clearing cache often helps."
        elif "won't" in keywords and "open" in keywords:
            return "Try updating the app from the App Store or reinstalling it. Remove it completely, restart your device, then reinstall. This resolves app launch issues in most cases."
        elif "crash" in keywords or "crash" in keywords:
            return "App crashes often resolve by force closing the app (swipe up in App Switcher) and restarting. If it persists, try updating iOS to the latest version."
        elif "connect" in keywords or "wifi" in keywords or "connection" in keywords:
            return "Try forgetting the WiFi network (Settings > WiFi > network name > Forget) and reconnecting. Also try restarting your router. This resolves most connectivity issues."
        elif "battery" in keywords or "drain" in keywords:
            return "Check Settings > Battery > Battery Health. Excessive battery drain often comes from background app refresh or location services. Disable for apps that don't need it."
        else:
            return f"Based on historical support data, I recommend checking your device storage, ensuring you have the latest iOS version, and force restarting the device. If the issue persists, please provide more details."
    
    def _extract_evidence_ranks(self, user_message: str) -> list:
        """Extract which evidence ranks were used."""
        try:
            if "Evidence 1" in user_message:
                ranks = [1]
                if "Evidence 2" in user_message:
                    ranks.append(2)
                if "Evidence 3" in user_message:
                    ranks.append(3)
                if "Evidence 4" in user_message:
                    ranks.append(4)
                if "Evidence 5" in user_message:
                    ranks.append(5)
                return ranks
            return []
        except:
            return []


def get_llm_client(provider: Optional[str] = None, 
                   model: Optional[str] = None) -> LLMProvider:
    """
    Get configured LLM client.
    
    Priority:
    1. Environment variable LLM_PROVIDER
    2. Function parameter
    3. Auto-detect (try OpenAI, fall back to mock)
    
    Args:
        provider: 'openai' or 'mock'
        model: Model name (provider-specific)
    
    Returns:
        Configured LLM provider instance
    """
    
    # Determine provider
    if provider is None:
        provider = os.environ.get('LLM_PROVIDER', 'auto')
    
    provider = provider.lower().strip()
    
    # Handle mock mode
    if provider == 'mock':
        return MockLLMClient()
    
    # Handle OpenAI
    if provider == 'openai':
        api_key = os.environ.get('OPENAI_API_KEY')
        if not api_key:
            warnings.warn(
                "OPENAI_API_KEY not found. Falling back to mock mode. "
                "Set OPENAI_API_KEY to use real OpenAI API.",
                RuntimeWarning
            )
            return MockLLMClient()
        
        model = model or os.environ.get('LLM_MODEL', 'gpt-4-turbo-preview')
        return OpenAIClient(api_key, model)
    
    # Auto-detect: try OpenAI, fall back to mock
    if provider == 'auto':
        api_key = os.environ.get('OPENAI_API_KEY')
        if api_key:
            model = model or os.environ.get('LLM_MODEL', 'gpt-4-turbo-preview')
            try:
                return OpenAIClient(api_key, model)
            except Exception:
                pass
        return MockLLMClient()
    
    raise ValueError(f"Unknown LLM provider: {provider}")


if __name__ == "__main__":
    # Test mock mode
    client = get_llm_client('mock')
    
    system_prompt = "You are a support assistant. Use only provided evidence."
    user_message = "Customer has a slow device. Historical evidence: restart device."
    
    response = client.generate(system_prompt, user_message)
    print("Mock response:")
    print(response)
