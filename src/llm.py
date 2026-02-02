import os
from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic

def get_llm(temperature: float = 0) -> BaseChatModel:
    """
    Get the LLM instance based on environment configuration.
    
    Supported Providers:
    - openai (default)
    - anthropic
    
    Env Vars:
    - LLM_PROVIDER: "openai" or "anthropic"
    - OPENAI_API_KEY: Required for OpenAI
    - ANTHROPIC_API_KEY: Required for Anthropic
    """
    provider = os.getenv("LLM_PROVIDER", "openai").lower()
    
    if provider == "anthropic":
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY not found in environment variables.")
        
        # Anthropic models: claude-3-opus-20240229, claude-3-sonnet-20240229, claude-3-haiku-20240307
        model = os.getenv("LLM_MODEL", "claude-3-haiku-20240307")
        return ChatAnthropic(
            model=model,
            temperature=temperature,
            anthropic_api_key=api_key
        )
        
    else:
        # Default to OpenAI
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key and provider == "openai":
            # Only raise if explicitly set to openai or defaulting
            # If provider is unknown, we might fall through, but let's strict validate
            pass
            
        if not api_key:
             # If defaulting to openai but no key, maybe they meant another provider?
             # But here we just assume they want openai.
             pass 

        model = os.getenv("LLM_MODEL", "gpt-4o-mini")
        return ChatOpenAI(
            model=model,
            temperature=temperature,
            api_key=api_key # type: ignore
        )
