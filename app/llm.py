import os
from dotenv import load_dotenv

load_dotenv()

def get_llm():
    """
    Instantiates and returns the configured LangChain Chat model (ChatOllama or ChatGroq).
    """
    provider = os.getenv("LLM_PROVIDER", "ollama").lower()
    
    if provider == "ollama":
        try:
            from langchain_ollama import ChatOllama
            base_url = os.getenv("LLM_BASE_URL", "http://localhost:11434")
            model_name = os.getenv("OLLAMA_MODEL", "llama3.1:8b-instruct-q4_K_M")
            print(f"Initializing ChatOllama model '{model_name}' at {base_url}")
            return ChatOllama(
                base_url=base_url,
                model=model_name,
                temperature=0
            )
        except Exception as e:
            print(f"ChatOllama initialization failed: {e}.")
            groq_key = os.getenv("GROQ_API_KEY")
            if groq_key:
                print("Falling back to ChatGroq...")
                from langchain_groq import ChatGroq
                return ChatGroq(
                    api_key=groq_key,
                    model_name="llama-3.1-8b-instant",
                    temperature=0
                )
            raise e
    elif provider == "groq":
        groq_key = os.getenv("GROQ_API_KEY")
        if not groq_key:
            raise ValueError("GROQ_API_KEY environment variable is not set.")
        from langchain_groq import ChatGroq
        return ChatGroq(
            api_key=groq_key,
            model_name="llama-3.1-8b-instant",
            temperature=0
        )
    else:
        raise ValueError(f"Unsupported LLM_PROVIDER: {provider}")
