import os
import json
import re
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage

load_dotenv()

def evaluate_faithfulness(query: str, context: str, answer: str) -> dict:
    """
    Evaluates the faithfulness (groundedness) of the generated answer against the retrieved context
    using an LLM-as-a-judge approach. Returns a score between 0.0 and 1.0, a list of claims evaluated,
    and a summary explanation.
    """
    # Quick bypass for refusal and conversational responses
    refusal_phrases = [
        "couldn't find sufficient evidence",
        "insufficient evidence",
        "do not have enough information",
        "not found in the provided document",
        "not mention in the provided document"
    ]
    
    is_refusal = any(phrase in answer.lower() for phrase in refusal_phrases)
    if is_refusal:
        return {
            "score": 1.0,
            "claims": [],
            "explanation": "The assistant correctly refused to answer because no evidence was found. Refusal is 100% faithful."
        }
        
    # Standard greetings bypass
    greetings = ["hello", "hi ", "hey there", "greetings", "good morning", "good afternoon"]
    is_greeting_only = len(answer.strip()) < 80 and any(g in answer.lower() for g in greetings)
    if is_greeting_only:
        return {
            "score": 1.0,
            "claims": [],
            "explanation": "Conversational response is 100% faithful."
        }

    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        raise ValueError("GROQ_API_KEY environment variable is not set.")
        
    model = ChatGroq(
        api_key=groq_api_key,
        model_name="llama-3.1-8b-instant",
        temperature=0,
    )
    
    system_prompt = """You are an expert evaluator assessing the faithfulness (groundedness) of a generated answer.
A faithful answer is one where every claim or fact stated in the answer is directly supported by the provided context. No claims should be made that cannot be found in the context (i.e. no assumptions, hallucinations, or external knowledge).

Your task is to:
1. Break down the Generated Answer into a list of independent factual claims.
2. For each claim, check if it is directly and fully supported by the Context.
3. Compute the overall Faithfulness Score: (number of supported claims) / (total number of claims). If there are no claims, the score is 1.0.
4. Provide a brief explanation for your evaluation.

You MUST respond strictly in the following JSON format:
{
  "claims": [
    {
      "statement": "the claim statement",
      "supported": true,
      "reason": "why it is or is not supported by the context"
    }
  ],
  "score": 0.85,
  "explanation": "overall summary explaining the score"
}
"""

    human_content = f"""Inputs:
- Context:
{context}

- Generated Answer:
{answer}
"""

    try:
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_content)
        ]
        
        response = model.invoke(messages)
        content_str = response.content.strip()
        
        # Clean markdown code block wraps if present
        if content_str.startswith("```"):
            content_str = re.sub(r"^```(?:json)?", "", content_str)
            content_str = re.sub(r"```$", "", content_str).strip()
            
        try:
            result = json.loads(content_str)
        except json.JSONDecodeError:
            # Try parsing by extracting JSON substring
            match = re.search(r"\{.*\}", content_str, re.DOTALL)
            if match:
                result = json.loads(match.group(0))
            else:
                raise ValueError("Could not parse LLM output as JSON.")
                
        # Validate structure
        score = result.get("score", 1.0)
        claims = result.get("claims", [])
        explanation = result.get("explanation", "Evaluation completed.")
        
        # Clamp score between 0.0 and 1.0
        score = max(0.0, min(1.0, float(score)))
        
        return {
            "score": score,
            "claims": claims,
            "explanation": explanation
        }
        
    except Exception as e:
        print(f"Error during faithfulness evaluation: {e}")
        # Default safety return in case of failure
        return {
            "score": 0.5,
            "claims": [],
            "explanation": f"Evaluation system error: {str(e)}"
        }
