"""
Task 6 — Lexical Search Module (BM25).

Mặc định sử dụng BM25. Nếu dùng phương pháp khác (TF-IDF, Elasticsearch,
Weaviate BM25 built-in), hãy giải thích cơ chế trong buổi demo → +5 bonus.

Cài đặt:
    pip install rank-bm25

BM25 hoạt động thế nào:
    - Term Frequency (TF): từ xuất hiện nhiều trong document → điểm cao
    - Inverse Document Frequency (IDF): từ hiếm → quan trọng hơn
    - Document length normalization: document dài không bị ưu tiên quá mức
    - Formula: score(q,d) = Σ IDF(qi) * (tf(qi,d) * (k1+1)) / (tf(qi,d) + k1*(1-b+b*|d|/avgdl))
    - k1=1.5 (term saturation), b=0.75 (length normalization)
"""

from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi
from task4_chunking_indexing import load_documents, chunk_documents

# Load chunked corpus via task4 to ensure compatibility with semantic search
CORPUS = chunk_documents(load_documents())
_bm25_index = None

def build_bm25_index(corpus: list[dict]):
    """
    Xây dựng BM25 index từ corpus.

    Args:
        corpus: List of {'content': str, 'metadata': dict}
    """
    tokenized_corpus = [doc["content"].lower().split() for doc in corpus]
    bm25 = BM25Okapi(tokenized_corpus)
    return bm25

def get_bm25_index():
    global _bm25_index
    if _bm25_index is None:
        _bm25_index = build_bm25_index(CORPUS)
    return _bm25_index

def lexical_search(query: str, top_k: int = 10) -> list[dict]:
    """
    Tìm kiếm từ khóa sử dụng BM25.

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả tối đa

    Returns:
        List of {
            'content': str,
            'score': float,      # BM25 score
            'metadata': dict
        }
        Sorted by score descending.
    """
    bm25 = get_bm25_index()
    tokenized_query = query.lower().split()
    scores = bm25.get_scores(tokenized_query)
    
    # Get top_k indices
    top_indices = np.argsort(scores)[::-1][:top_k]
    
    results = []
    for idx in top_indices:
        if scores[idx] > 0:
            results.append({
                "content": CORPUS[idx]["content"],
                "score": float(scores[idx]),
                "metadata": CORPUS[idx]["metadata"]
            })
    return results


_tfidf_vectorizer = None
_tfidf_matrix = None

def build_tfidf_index(corpus: list[dict]):
    """
    Xây dựng TF-IDF index từ corpus sử dụng scikit-learn.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    vectorizer = TfidfVectorizer()
    texts = [doc["content"] for doc in corpus]
    matrix = vectorizer.fit_transform(texts)
    return vectorizer, matrix

def get_tfidf_index():
    global _tfidf_vectorizer, _tfidf_matrix
    if _tfidf_vectorizer is None or _tfidf_matrix is None:
        _tfidf_vectorizer, _tfidf_matrix = build_tfidf_index(CORPUS)
    return _tfidf_vectorizer, _tfidf_matrix

def tfidf_search(query: str, top_k: int = 10) -> list[dict]:
    """
    Tìm kiếm từ khóa sử dụng TF-IDF và cosine similarity.
    """
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np
    
    vectorizer, matrix = get_tfidf_index()
    query_vec = vectorizer.transform([query])
    
    # Tính cosine similarity giữa query và tất cả document
    scores = cosine_similarity(query_vec, matrix).flatten()
    
    # Lấy top_k indices
    top_indices = np.argsort(scores)[::-1][:top_k]
    
    results = []
    for idx in top_indices:
        if scores[idx] > 0:
            results.append({
                "content": CORPUS[idx]["content"],
                "score": float(scores[idx]),
                "metadata": CORPUS[idx]["metadata"]
            })
    return results


if __name__ == "__main__":
    query = "tuition fee payment methods"
    
    print("=== BM25 Search ===")
    results = lexical_search(query, top_k=3)
    for r in results:
        print(f"[{r['score']:.3f}] {r['content'][:100]}...")

    print("\n=== TF-IDF Search ===")
    tfidf_res = tfidf_search(query, top_k=3)
    for r in tfidf_res:
        print(f"[{r['score']:.3f}] {r['content'][:100]}...")
