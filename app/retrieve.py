import os
import json
import re
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from app.database import get_vector_store, vector_store_type
from app.ingest import get_local_embeddings

load_dotenv()

def ask_question(query: str, namespaces: list, history: list = None) -> dict:
    """
    Searches the vector store across multiple namespaces, reformulates the query if history is present,
    asks Groq for a grounded answer, and returns the answer, sources, score, and confidence.
    """
    if history is None:
        history = []
        
    print(f"Searching for context to answer: '{query}' across {len(namespaces)} documents")
    
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        raise ValueError("GROQ_API_KEY environment variable is not set.")
        
    # Initialize Groq LLM
    model = ChatGroq(
        api_key=groq_api_key,
        model_name="llama-3.1-8b-instant",
        temperature=0,
    )
    
        standalone_query = query

# Initialize Embeddings function
    hf_token = os.getenv("HF_TOKEN")
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    if hf_token:
        try:
            from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
            embeddings = HuggingFaceInferenceAPIEmbeddings(
                api_key=hf_token,
                model_name=model_name
            )
            # Test connection
            embeddings.embed_query("test")
        except Exception as e:
            print(f"Embeddings API connection failed in retriever: {e}. Falling back to local Embeddings.")
            embeddings = get_local_embeddings(model_name)
    else:
        embeddings = get_local_embeddings(model_name)

    # Search across all namespaces
    all_chunks_with_score = []
    for ns in namespaces:
        try:
            vector_store = get_vector_store(embeddings, ns)
            # Fetch top 5 from each namespace with score
            chunks = vector_store.similarity_search_with_score(standalone_query, 5)
            
            # Map distance/score to cosine similarity (0 to 1) for uniform comparison
            for doc, raw_score in chunks:
                if vector_store_type == "chromadb":
                    # Chroma returns squared L2 distance (smaller is better).
                    # Map distance to similarity score: similarity = 1 - distance / 2
                    score = max(0.0, min(1.0, 1.0 - raw_score / 2.0))
                else:
                    # Pinecone returns cosine similarity directly
                    score = raw_score
                all_chunks_with_score.append((doc, score))
        except Exception as e:
            print(f"Error searching namespace {ns}: {e}")

    # Sort all chunks by score descending
    all_chunks_with_score.sort(key=lambda x: x[1], reverse=True)
    
    max_score = all_chunks_with_score[0][1] if all_chunks_with_score else 0.0
    print(f"Maximum similarity score: {max_score}")
    
    # Take absolute top 5 most relevant child chunks
    top_child_chunks = [c[0] for c in all_chunks_with_score[:5]]
    
    # Deduplicate parent chunks using parentId or source
    unique_parents = {}
    for chunk in top_child_chunks:
        parent_id = chunk.metadata.get("parentId") or chunk.metadata.get("source") or str(hash(chunk.page_content))
        if parent_id not in unique_parents:
            unique_parents[parent_id] = chunk
            
    top_chunks = list(unique_parents.values())

    # Build the strictly grounded System Prompt requesting JSON output
    contexts = []
    for i, chunk in enumerate(top_chunks):
        source = chunk.metadata.get("source", "Document")
        content = chunk.metadata.get("parentContent") or chunk.page_content
        contexts.append(f"[Source {i + 1} - {source}]:\n{content}")
        
    context_text = "\n\n".join(contexts)
    
    system_prompt = f"""You are a helpful AI Assistant.

Rules:
1. If the user asks a conversational question (e.g., greetings, asking how you are, thanking you), respond politely and naturally. Set isConversational to true, noEvidence to false, and provide your response in the answer field.
2. For all other questions, answer based ONLY on the provided document context below. Do not use prior knowledge. Set isConversational to false.
3. If the answer to a document-related question is not found in the context, set noEvidence to true and set the answer field exactly to "I couldn't find sufficient evidence in the provided document to answer this question."
4. Be clear, concise, and helpful.

You MUST respond in the following JSON format:
{{
  "isConversational": boolean,
  "noEvidence": boolean,
  "answer": "your answer here"
}}

Document Context:
{context_text}
"""

    messages = [
        SystemMessage(content=system_prompt),
    ]
    
    # Append conversation history
    for msg in history:
        role = msg.get("role")
        content = msg.get("content")
        if role == "user":
            messages.append(HumanMessage(content=content))
        elif role == "assistant":
            # Format assistant messages as JSON to stay consistent with the SystemMessage instructions
            from langchain_core.messages import AIMessage
            is_conv = any(greet in content.lower() for greet in ["hello", "hi ", "hey", "thanks", "thank you"])
            assistant_json = json.dumps({
                "isConversational": is_conv,
                "noEvidence": False,
                "answer": content
            })
            messages.append(AIMessage(content=assistant_json))
            
    # Add final user query
    messages.append(HumanMessage(content=query))
    
    try:
        response = model.invoke(messages)
        content_str = response.content.strip()
        
        # Clean JSON wrappers if LLM returned markdown codeblock
        if content_str.startswith("```"):
            content_str = re.sub(r"^```(?:json)?", "", content_str)
            content_str = re.sub(r"```$", "", content_str).strip()
            
        try:
            result_json = json.loads(content_str)
        except json.JSONDecodeError:
            # Fallback regex parsing if JSON parsing fails
            match = re.search(r"\{.*\}", content_str, re.DOTALL)
            if match:
                try:
                    result_json = json.loads(match.group(0))
                except json.JSONDecodeError:
                    result_json = None
            else:
                result_json = None
                
            if not result_json:
                print("Failed to parse JSON response, using fallback text rules")
                is_refusal = any(term in content_str for term in ["sufficient evidence", "information is not provided", "I couldn't find"])
                result_json = {
                    "isConversational": not is_refusal and max_score < 0.4 and (len(query) < 15 or "hello" in query.lower() or "hi" in query.lower()),
                    "noEvidence": is_refusal,
                    "answer": content_str
                }
    except Exception as e:
        print(f"Error during LLM inference: {e}")
        raise RuntimeError(f"LLM inference error: {e}")

    answer = result_json.get("answer", "No response generated.")
    confidence = 'HIGH'
    
    if result_json.get("isConversational"):
        confidence = 'HIGH'
    else:
        # Check if query is a summary or general document query
        q_clean = query.lower().strip()
        is_summary_query = any(k in q_clean for k in ["summar", "overview", "about", "outline", "main topic", "key point", "key finding", "conclus", "gist", "synopsis"])
        
        # Confidence logic based on score threshold
        if max_score >= 0.7:
            confidence = 'HIGH'
        elif max_score >= 0.4:
            confidence = 'LOW'
        elif is_summary_query and max_score >= 0.1:
            # Relax threshold for general/summary requests as they lack specific term-level matches
            confidence = 'LOW'
        elif not result_json.get("noEvidence") and max_score >= 0.2:
            # If the LLM asserts it has evidence and the score is reasonable, allow the answer
            confidence = 'LOW'
        else:
            confidence = 'NOT_FOUND'
            
        if result_json.get("noEvidence"):
            confidence = 'NOT_FOUND'
            
        if confidence == 'NOT_FOUND':
            answer = "I couldn't find sufficient evidence in the provided document to answer this question."

    # Return serialized version of documents for API response
    serialized_sources = []
    for chunk in top_chunks:
        serialized_sources.append({
            "pageContent": chunk.page_content,
            "metadata": chunk.metadata
        })

    return {
        "answer": answer,
        "sources": serialized_sources,
        "score": max_score,
        "confidence": confidence
    }
