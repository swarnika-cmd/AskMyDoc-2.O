"""Prompt rendering and output parsing shared by run_llm.py and aggregate_llm.py."""
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

GENERATOR_MODEL = "openai/gpt-oss-20b"
JUDGE_MODEL = "openai/gpt-oss-120b"
GENERATOR_MAX_TOKENS = 1024
JUDGE_MAX_TOKENS = 2048
REFUSAL_TEXT = "I couldn't find sufficient evidence in the provided document to answer this question."
CORRECTNESS = ("correct", "partial", "incorrect")


def _read(name):
    with open(os.path.join(HERE, "prompts", name), encoding="utf-8") as f:
        return f.read()


def render_context(units):
    """The context block the app sends: parent text per unit in rank order, labelled [Source i - <file>]."""
    from langchain_core.documents import Document
    from app.retrieve import format_context

    docs = [Document(page_content=u["text"], metadata={"source": u["doc"] + ".pdf"}) for u in units]
    return format_context(docs)


def generator_messages(question, context):
    """System prompt exactly as app/retrieve.py builds it, then the question as the user message (no history)."""
    system = _read("generator.txt").replace("{context}", context)
    return [{"role": "system", "content": system}, {"role": "user", "content": question}]


def judge_messages(item, system_answer, context):
    text = _read("judge.txt")
    system_part, user_part = text.split("### USER\n", 1)
    system = system_part.replace("### SYSTEM\n", "", 1).strip()
    note = ("UNANSWERABLE (the information is not in the documents; the correct behaviour is to decline)"
            if item["type"] == "unanswerable" else "answerable (%s)" % item["type"])
    values = {
        "question_type_note": note, "question": item["question"], "reference_answer": item["answer"],
        "system_answer": system_answer, "context": context,
    }
    user = re.sub(r"\{(question_type_note|question|reference_answer|system_answer|context)\}",
                  lambda m: values[m.group(1)], user_part).strip()
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def _loads_lenient(content):
    s = (content or "").strip()
    if s.startswith("```"):
        s = re.sub(r"^```(?:json)?", "", s)
        s = re.sub(r"```$", "", s).strip()
    try:
        return json.loads(s)
    except ValueError:
        m = re.search(r"\{.*\}", s, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except ValueError:
                return None
    return None


def parse_generator(content):
    """Mirror app/retrieve.py: JSON with answer/noEvidence, else the raw text with the app's refusal-phrase rule."""
    obj = _loads_lenient(content)
    if isinstance(obj, dict):
        return {"parse_ok": True, "answer": str(obj.get("answer", "")), "no_evidence": bool(obj.get("noEvidence")),
                "is_conversational": bool(obj.get("isConversational"))}
    text = (content or "").strip()
    refusal = any(t in text for t in ("sufficient evidence", "information is not provided", "I couldn't find"))
    return {"parse_ok": False, "answer": text, "no_evidence": refusal, "is_conversational": False}


def parse_judge(content):
    obj = _loads_lenient(content)
    if not isinstance(obj, dict) or obj.get("correctness") not in CORRECTNESS or not isinstance(obj.get("is_refusal"), bool):
        return {"parse_ok": False}
    claims = []
    for c in obj.get("claims") or []:
        if isinstance(c, dict) and isinstance(c.get("supported"), bool):
            claims.append({"claim": str(c.get("claim", "")), "supported": c["supported"]})
    return {"parse_ok": True, "correctness": obj["correctness"], "is_refusal": obj["is_refusal"],
            "reason": str(obj.get("correctness_reason", "")), "claims": claims}


def faithfulness(claims):
    """Supported claims / total claims; None when there are no claims."""
    if not claims:
        return None
    return sum(1 for c in claims if c["supported"]) / float(len(claims))


def cost_inr(prices, model, usage):
    """USD per 1M tokens x tokens, converted at the recorded rate. usage: prompt_tokens, completion_tokens."""
    p = prices["prices_usd_per_1m_tokens"][model]
    usd = (usage.get("prompt_tokens", 0) * p["input"] + usage.get("completion_tokens", 0) * p["output"]) / 1e6
    return usd * prices["usd_to_inr"]["rate"]
