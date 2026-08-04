import os
import time
from dotenv import load_dotenv

load_dotenv()

vector_store_type = os.getenv("VECTOR_STORE_TYPE", "chromadb").lower()

def get_vector_store(embeddings, namespace: str):
    """
    Returns a LangChain VectorStore instance (ChromaDB) configured for the given namespace.
    """
    if vector_store_type == "chromadb":
        from langchain_community.vectorstores import Chroma
        persist_directory = os.getenv("CHROMA_DB_PATH", "./chroma_db")
        return Chroma(
            persist_directory=persist_directory,
            embedding_function=embeddings,
            collection_name=namespace
        )
    else:
        # Default fallback to ChromaDB for localized self-hosting
        from langchain_community.vectorstores import Chroma
        persist_directory = os.getenv("CHROMA_DB_PATH", "./chroma_db")
        return Chroma(
            persist_directory=persist_directory,
            embedding_function=embeddings,
            collection_name=namespace
        )

def save_documents_to_store(documents, embeddings, namespace: str):
    """
    Saves a list of LangChain Document objects to the ChromaDB vector store under the given namespace.
    """
    from langchain_community.vectorstores import Chroma
    persist_directory = os.getenv("CHROMA_DB_PATH", "./chroma_db")
    
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
    try:
        import chromadb
        persist_directory = os.getenv("CHROMA_DB_PATH", "./chroma_db")
        client = chromadb.PersistentClient(path=persist_directory)
        
        # Check if collection exists before deleting to avoid errors
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
