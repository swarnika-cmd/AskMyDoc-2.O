import os
import threading
import time
from dotenv import load_dotenv

load_dotenv()

vector_store_type = os.getenv("VECTOR_STORE_TYPE", "chromadb").lower()

_store_cache = {}
_store_lock = threading.Lock()


def _evict(namespace: str):
    with _store_lock:
        for key in [k for k in _store_cache if k[1] == namespace]:
            del _store_cache[key]


def get_vector_store(embeddings, namespace: str):
    """
    Returns a LangChain Chroma store for the given namespace, cached per (path, namespace).
    """
    from langchain_community.vectorstores import Chroma
    persist_directory = os.getenv("CHROMA_DB_PATH", "./chroma_db")
    key = (persist_directory, namespace)
    with _store_lock:
        store = _store_cache.get(key)
        if store is None:
            os.makedirs(persist_directory, exist_ok=True)
            store = Chroma(
                persist_directory=persist_directory,
                embedding_function=embeddings,
                collection_name=namespace
            )
            _store_cache[key] = store
        return store

def save_documents_to_store(documents, embeddings, namespace: str):
    """
    Saves a list of LangChain Document objects to the ChromaDB vector store under the given namespace.
    """
    from langchain_community.vectorstores import Chroma
    persist_directory = os.getenv("CHROMA_DB_PATH", "./chroma_db")
    os.makedirs(persist_directory, exist_ok=True)
    
    _evict(namespace)
    start_time = time.time()
    Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=persist_directory,
        collection_name=namespace
    )
    elapsed_ms = (time.time() - start_time) * 1000
    print(f"Saved {len(documents)} documents to ChromaDB collection '{namespace}' in {elapsed_ms:.2f}ms")
    return True

def delete_namespace(namespace: str):
    """
    Deletes the ChromaDB vector store collection.
    """
    _evict(namespace)
    try:
        import chromadb
        persist_directory = os.getenv("CHROMA_DB_PATH", "./chroma_db")
        os.makedirs(persist_directory, exist_ok=True)
        client = chromadb.PersistentClient(path=persist_directory)
        
        collections = [c.name for c in client.list_collections()]
        if namespace in collections:
            client.delete_collection(name=namespace)
            print(f"Deleted Chroma collection: {namespace}")
            return True
        print(f"Chroma collection not found: {namespace}")
        return False
    except Exception as e:
        print(f"Error deleting Chroma collection {namespace}: {e}")
        return False
