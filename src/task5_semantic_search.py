"""
Task 5 — Semantic Search Module.

Viết module tìm kiếm ngữ nghĩa (dense retrieval) trên vector store.

Yêu cầu:
    - Input: query string + top_k
    - Output: danh sách chunks có score, sorted descending
    - Phải tương thích với embedding model và vector store ở Task 4
"""


import chromadb
from sentence_transformers import SentenceTransformer
from task4_chunking_indexing import CHROMA_DIR, COLLECTION_NAME, EMBEDDING_MODEL

_model = None
_collection = None

def get_embedding_model():
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBEDDING_MODEL)
    return _model

def get_collection():
    global _collection
    if _collection is None:
        client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        _collection = client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
    return _collection

def semantic_search(query: str, top_k: int = 10) -> list[dict]:
    """
    Tìm kiếm ngữ nghĩa sử dụng vector similarity.

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả tối đa

    Returns:
        List of {
            'content': str,      # Nội dung chunk
            'score': float,      # Cosine similarity score
            'metadata': dict     # source, doc_type, chunk_index
        }
        Sorted by score descending.
    """
    model = get_embedding_model()
    query_vector = model.encode(query).tolist()
    
    collection = get_collection()
    results = collection.query(
        query_embeddings=[query_vector],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )
    
    output = []
    if not results["documents"] or not results["documents"][0]:
        return output

    for doc, meta, dist in zip(
        results["documents"][0], results["metadatas"][0], results["distances"][0]
    ):
        score = max(0.0, 1.0 - dist)  # cosine distance → similarity
        output.append({"content": doc, "score": round(score, 4), "metadata": meta})
    
    output.sort(key=lambda x: x["score"], reverse=True)
    return output[:top_k]


def hyde_search(query: str, top_k: int = 10) -> list[dict]:
    """
    Tìm kiếm ngữ nghĩa sử dụng kỹ thuật HyDE (Hypothetical Document Embeddings).
    Tạo ra một tài liệu giả định (hypothetical document) bằng LLM, sau đó dùng tài liệu này
    làm vector truy vấn thay vì dùng trực tiếp câu hỏi gốc.
    """
    import os
    from openai import OpenAI

    api_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")
    hypothetical_doc = query  # Mặc định fallback về câu hỏi gốc nếu không gọi được LLM

    if api_key:
        try:
            client = OpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1")
            prompt = f"Please write a short informative passage that answers the following question: {query}"
            
            response = client.chat.completions.create(
                model="poolside/laguna-s-2.1:free", # Hoặc model :free
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
            )
            hypothetical_doc = response.choices[0].message.content
            print(f"\n[HyDE] Hypothetical Document:\n{hypothetical_doc}\n")
        except Exception as e:
            print(f"\n[HyDE Error] LLM generation failed: {e}. Fallback to original query.")

    model = get_embedding_model()
    query_vector = model.encode(hypothetical_doc).tolist()
    
    collection = get_collection()
    results = collection.query(
        query_embeddings=[query_vector],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )
    
    output = []
    if not results["documents"] or not results["documents"][0]:
        return output

    for doc, meta, dist in zip(
        results["documents"][0], results["metadatas"][0], results["distances"][0]
    ):
        score = max(0.0, 1.0 - dist)
        output.append({"content": doc, "score": round(score, 4), "metadata": meta})
    
    output.sort(key=lambda x: x["score"], reverse=True)
    return output[:top_k]


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()

    query = "what is the tuition fee"
    print("=== Normal Semantic Search ===")
    results = semantic_search(query, top_k=3)
    for r in results:
        print(f"[{r['score']:.3f}] {r['content'][:100]}...")

    print("\n=== HyDE Search ===")
    hyde_results = hyde_search(query, top_k=3)
    for r in hyde_results:
        print(f"[{r['score']:.3f}] {r['content'][:100]}...")
