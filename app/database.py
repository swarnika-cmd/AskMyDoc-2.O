import os
from dotenv import load_dotenv

load_dotenv()

vector_store_type = os.getenv("VECTOR_STORE_TYPE", "chromadb").lower()

def get_vector_store(embeddings, namespace: str):
    """
    Returns a LangChain VectorStore instance (Chroma or PineconeStore) 
    configured for the given namespace.
    """
    if vector_store_type == "chromadb":
        from langchain_community.vectorstores import Chroma
        persist_directory = os.getenv("CHROMA_DB_PATH", "./chroma_db")
        return Chroma(
            persist_directory=persist_directory,
            embedding_function=embeddings,
            collection_name=namespace
        )
    elif vector_store_type == "pinecone":
        from langchain_pinecone import PineconeStore
        from pinecone import Pinecone
        
        api_key = os.getenv("PINECONE_API_KEY")
        index_name = os.getenv("PINECONE_INDEX")
        if not api_key:
            raise ValueError("PINECONE_API_KEY environment variable is not set.")
        if not index_name:
            raise ValueError("PINECONE_INDEX environment variable is not set.")
            
        pc = Pinecone(api_key=api_key)
        index = pc.Index(index_name)
        return PineconeStore(
            index=index,
            embedding=embeddings,
            namespace=namespace
        )
    else:
        raise ValueError(f"Unsupported VECTOR_STORE_TYPE: {vector_store_type}")

def save_documents_to_store(documents, embeddings, namespace: str):
    """
    Saves a list of LangChain Document objects to the vector store under the given namespace.
    """
    if vector_store_type == "chromadb":
        from langchain_community.vectorstores import Chroma
        persist_directory = os.getenv("CHROMA_DB_PATH", "./chroma_db")
        # Initialize and save
        Chroma.from_documents(
            documents=documents,
            embedding=embeddings,
            persist_directory=persist_directory,
            collection_name=namespace
        )
        return True
    elif vector_store_type == "pinecone":
        from langchain_pinecone import PineconeStore
        from pinecone import Pinecone
        
        api_key = os.getenv("PINECONE_API_KEY")
        index_name = os.getenv("PINECONE_INDEX")
        if not api_key or not index_name:
            raise ValueError("Pinecone API Key and Index name must be set to save documents.")
            
        pc = Pinecone(api_key=api_key)
        index = pc.Index(index_name)
        PineconeStore.from_documents(
            documents=documents,
            embedding=embeddings,
            index_name=index_name,
            namespace=namespace
        )
        return True
    else:
        raise ValueError(f"Unsupported VECTOR_STORE_TYPE: {vector_store_type}")

def delete_namespace(namespace: str):
    """
    Deletes the vector store collection or namespace.
    """
    if vector_store_type == "chromadb":
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
    elif vector_store_type == "pinecone":
        from pinecone import Pinecone
        api_key = os.getenv("PINECONE_API_KEY")
        index_name = os.getenv("PINECONE_INDEX")
        if not api_key or not index_name:
            raise ValueError("Pinecone API Key and Index name must be set to delete a namespace.")
        pc = Pinecone(api_key=api_key)
        index = pc.Index(index_name)
        try:
            # Check if there are vectors or just attempt delete
            index.delete(delete_all=True, namespace=namespace)
            print(f"Deleted Pinecone namespace: {namespace}")
            return True
        except Exception as e:
            print(f"Error deleting Pinecone namespace {namespace}: {e}")
            return False
    return False
