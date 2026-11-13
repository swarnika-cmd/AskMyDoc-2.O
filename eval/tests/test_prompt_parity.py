"""The generator prompt in eval/prompts/generator.txt must equal what app/retrieve.py sends."""
import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))

from langchain_core.documents import Document  # noqa: E402

import llm_eval as le  # noqa: E402
from app import retrieve  # noqa: E402


class FakeStore:
    def __init__(self, docs):
        self.docs = docs

    def similarity_search_with_score(self, query, k):
        return [(d, 0.5) for d in self.docs[:k]]


class FakeLLM:
    def __init__(self):
        self.messages = None

    def invoke(self, messages):
        self.messages = messages

        class R:
            content = '{"isConversational": false, "noEvidence": false, "answer": "ok"}'
        return R()


class Parity(unittest.TestCase):
    def test_generator_messages_match_the_app(self):
        children = [
            Document(page_content="child one", metadata={"source": "docA.pdf", "parentId": "p1", "parentContent": "PARENT ONE {braces} text"}),
            Document(page_content="child two", metadata={"source": "docB.pdf", "parentId": "p2", "parentContent": "PARENT TWO\nline"}),
        ]
        llm = FakeLLM()
        saved = (retrieve.get_vector_store, retrieve.get_embeddings, retrieve.get_llm)
        retrieve.get_vector_store = lambda emb, ns: FakeStore(children)
        retrieve.get_embeddings = lambda: None
        retrieve.get_llm = lambda: llm
        try:
            retrieve.ask_question("What is the rule?", ["ns"], [])
        finally:
            retrieve.get_vector_store, retrieve.get_embeddings, retrieve.get_llm = saved

        units = [{"doc": "docA", "text": "PARENT ONE {braces} text"}, {"doc": "docB", "text": "PARENT TWO\nline"}]
        mine = le.generator_messages("What is the rule?", le.render_context(units))
        self.assertEqual(llm.messages[0].content, mine[0]["content"])
        self.assertEqual(llm.messages[-1].content, mine[1]["content"])
        self.assertEqual(len(llm.messages), 2)


if __name__ == "__main__":
    unittest.main()
