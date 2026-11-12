import os
import json
import re
from dotenv import load_dotenv
from langchain_core.messages import SystemMessage, HumanMessage
from app.llm import get_llm, message_text

load_dotenv()

def evaluate_faithfulness(query: str, context: str, answer: str) -> dict:
    """
    Evaluates the faithfulness (groundedness) of the generated answer against the retrieved context
    using an LLM-as-a-judge approach. Returns a score between 0.0 and 1.0 (supported claims / total claims, or None if the judge failed), a list of claims evaluated,
    and a summary explanation.
    """
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
        
    greetings = ["hello", "hi ", "hey there", "greetings", "good morning", "good afternoon"]
    is_greeting_only = len(answer.strip()) < 80 and any(g in answer.lower() for g in greetings)
    if is_greeting_only:
        return {
            "score": 1.0,
            "claims": [],
            "explanation": "Conversational response is 100% faithful."
        }

    model = get_llm()
    
    system_prompt = """You are an expert evaluator assessing the faithfulness (groundedness) of a generated answer.
A faithful answer is one where every claim or fact stated in the answer is directly supported by the provided context. No claims should be made that cannot be found in the context (i.e. no assumptions, hallucinations, or external knowledge).

Your task is to:
1. Break down the Generated Answer into a list of independent factual claims.
2. For each claim, check if it is directly and fully supported by the Context.
3. Provide a brief explanation for your evaluation.

You MUST respond strictly in the following JSON format:
{
  "claims": [
    {
      "statement": "the claim statement",
      "supported": true,
      "reason": "why it is or is not supported by the context"
    }
  ],
  "explanation": "overall summary of the evaluation"
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
        content_str = message_text(response)
        
        if content_str.startswith("```"):
            content_str = re.sub(r"^```(?:json)?", "", content_str)
            content_str = re.sub(r"```$", "", content_str).strip()
            
        try:
            result = json.loads(content_str)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", content_str, re.DOTALL)
            if match:
                result = json.loads(match.group(0))
            else:
                raise ValueError("Could not parse LLM output as JSON.")
                
        claims = result.get("claims", [])
        explanation = result.get("explanation", "Evaluation completed.")

        # Score is computed here from the judge's per-claim verdicts, not taken from the model.
        verdicts = [c.get("supported") for c in claims if isinstance(c, dict) and isinstance(c.get("supported"), bool)]
        if not verdicts:
            return {"score": None, "claims": claims, "explanation": "Judge returned no usable claim verdicts."}
        score = sum(verdicts) / len(verdicts)

        return {
            "score": score,
            "claims": claims,
            "explanation": explanation
        }

    except Exception as e:
        print(f"Error during faithfulness evaluation: {e}")
        return {
            "score": None,
            "claims": [],
            "explanation": f"Evaluation system error: {str(e)}"
        }
