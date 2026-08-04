import os
import sys
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, os.getcwd())

from app.ingest import ingest_document

def test_ingest_flow():
    print("Testing ingest_document on text file...")
    txt_path = "scratch/test.txt"
    res = ingest_document(txt_path, "test_ingest_collection", "test.txt")
    print("Ingest result:", res)
    assert res.get("success") is True
    assert res.get("chunksCount") > 0
    print("[PASS] Ingestion test passed!")

if __name__ == "__main__":
    test_ingest_flow()
