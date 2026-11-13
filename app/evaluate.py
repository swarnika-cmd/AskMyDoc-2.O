import os
import json
import re
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.messages import SystemMessage, HumanMessage
from app.database import get_vector_store, namespace_exists
from app.ingest import get_embeddings
from app.llm import get_llm, message_text
from app.retrieve import format_context

load_dotenv()

MAX_CHUNK_IDS = 10
MAX_NAMESPACES = 10
# Namespaces are created by the web page as doc_<name>_<timestamp>; the default is DefaultCollection.
NAMESPACE_RE = re.compile(r"^[A-Za-z0-9_-]{3,100}$")


class ChunkLookupError(ValueError):
    """Raised when chunk ids or namespaces are missing, invalid, too many, unknown, or not found in the store."""


def _validated_namespaces(namespaces) -> list:
    names = namespaces or ["DefaultCollection"]
    if not isinstance(names, list) or not all(isinstance(n, str) for n in names):
        raise ChunkLookupError("namespaces must be a list of strings.")
    if len(names) > MAX_NAMESPACES:
        raise ChunkLookupError(f"At most {MAX_NAMESPACES} namespaces are accepted.")
    for n in names:
        if not NAMESPACE_RE.match(n):
            raise ChunkLookupError(f"Invalid namespace name: {n[:40]!r}")
    # Check existence first: opening a store for an unknown name would create an empty collection.
    for n in names:
        if not namespace_exists(n):
            raise ChunkLookupError(f"Unknown namespace: {n}")
    return names


def context_from_chunk_ids(chunk_ids: list, namespaces: list) -> str:
    """
    The context comes from the store for the given chunk ids, in the given order. Nothing a client sends as
    text is used. chunk_ids are the `chunkIds` of a /api/chat response.
    """
    ids = [i for i in (chunk_ids or []) if isinstance(i, str) and i]
    if not ids:
        raise ChunkLookupError("chunkIds is required. Re-upload the document if it was ingested before chunk ids existed.")
    if len(ids) > MAX_CHUNK_IDS:
        raise ChunkLookupError(f"At most {MAX_CHUNK_IDS} chunk ids are accepted.")
    names = _validated_namespaces(namespaces)

    found = {}
    # The store is cached per namespace and shared with chat retrieval, so it must be built with the real embeddings.
    embeddings = get_embeddings()
    for ns in names:
        missing = [i for i in ids if i not in found]
        if not missing:
            break
        store = get_vector_store(embeddings, ns)
        got = store.get(ids=missing)
        for cid, text, meta in zip(got.get("ids", []), got.get("documents", []), got.get("metadatas", [])):
            found[cid] = Document(page_content=text or "", metadata=meta or {})

    unknown = [i for i in ids if i not in found]
    if unknown:
        raise ChunkLookupError(f"Unknown chunk id(s): {', '.join(unknown)}")
    return format_context([found[i] for i in ids])


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
