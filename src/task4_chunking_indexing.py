"""
Task 4 — Chunking & Indexing vào Vector Store.

Hướng dẫn:
    1. Đọc toàn bộ markdown files từ data/standardized/
    2. Chọn 1 chunking strategy (giải thích lý do)
    3. Chọn 1 embedding model (giải thích lý do)
    4. Index vào vector store (ChromaDB khuyến cáo — đơn giản, local, không cần Docker)

Chunking options (langchain-text-splitters):
    - RecursiveCharacterTextSplitter: an toàn, phổ biến
    - MarkdownHeaderTextSplitter: tốt cho file có heading
    - SemanticChunker: dùng embedding để tách (nâng cao)

Embedding model options:
    - sentence-transformers/all-MiniLM-L6-v2 (384 dim, nhẹ)
    - BAAI/bge-m3 (1024 dim, multilingual, tốt cho cả tiếng Việt lẫn tiếng Anh)
    - OpenAI text-embedding-3-small (1536 dim, API)

Vector store options:
    - ChromaDB (khuyến cáo: đơn giản, local persistent, không cần Docker)
    - Weaviate (hỗ trợ hybrid search built-in, cần Docker/Cloud)
    - FAISS (chỉ dense search)

Cài đặt:
    pip install langchain-text-splitters sentence-transformers chromadb

Lưu ý quan trọng: nếu sau này đổi corpus (đổi chủ đề, thêm/bớt tài liệu), phải XÓA
chroma_db/ cũ trước khi reindex — nếu không, chunk cũ và mới sẽ tồn tại lẫn lộn
trong cùng collection, retrieval sẽ trả về kết quả rác từ dữ liệu cũ.
"""

from pathlib import Path

STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"
CHROMA_DIR = Path(__file__).parent.parent / "chroma_db"


# =============================================================================
# CONFIGURATION — Giải thích lựa chọn của bạn trong comment
# =============================================================================

# Chunking strategy: RecursiveCharacterTextSplitter
# Lý do chọn "recursive":
#   - Corpus gồm cả văn bản pháp lý (legal) và tin tức (news) tiếng Việt, cấu trúc
#     heading không đồng nhất giữa các nguồn -> MarkdownHeaderTextSplitter dễ bỏ sót
#     chunk khi file thiếu heading chuẩn.
#   - SemanticChunker cần gọi embedding cho từng câu để xác định điểm cắt -> chậm,
#     tốn chi phí hơn nhiều so với lợi ích mang lại ở quy mô dữ liệu này.
#   - RecursiveCharacterTextSplitter dùng danh sách separators phân cấp
#     (đoạn -> dòng -> câu -> từ) nên vẫn tôn trọng ranh giới ngữ nghĩa tự nhiên
#     của văn bản, đồng thời đơn giản, ổn định, dễ debug.
CHUNK_SIZE = 500        # ~500 ký tự cho phép mỗi chunk chứa trọn 1-2 đoạn văn/điều
                        # luật ngắn, đủ ngữ cảnh cho retrieval mà không quá dài
                        # khiến embedding bị loãng thông tin.
CHUNK_OVERLAP = 50      # 10% của chunk_size — đủ để không cắt đứt câu/ý ở ranh giới
                        # chunk (ví dụ 1 điều luật bị chia làm 2), mà không tạo quá
                        # nhiều trùng lặp gây phình index.
CHUNKING_METHOD = "recursive"  # "recursive" | "markdown_header" | "semantic"

# Embedding model: BAAI/bge-m3
# Lý do:
#   - Corpus có cả tiếng Việt (văn bản pháp luật, tin tức nội bộ) lẫn tiếng Anh
#     (thuật ngữ, tài liệu quốc tế) -> cần model multilingual chất lượng cao.
#     all-MiniLM-L6-v2 chỉ train chủ yếu trên tiếng Anh, kém với tiếng Việt.
#   - bge-m3 hỗ trợ >100 ngôn ngữ, hiệu năng tốt trên các benchmark retrieval
#     đa ngôn ngữ (MIRACL, MKQA), và có thể chạy local (không cần API key/OpenAI).
#   - Đánh đổi: 1024-dim nặng hơn MiniLM (384-dim) nên index lớn hơn, nhưng chấp
#     nhận được vì ưu tiên độ chính xác retrieval hơn là tốc độ/kích thước ở giai
#     đoạn này.
EMBEDDING_MODEL = "intfloat/multilingual-e5-small"
EMBEDDING_DIM = 384

# Vector store: ChromaDB
# Lý do: local, persistent, không cần Docker/server riêng, phù hợp cho pipeline
# offline/dev. Nếu sau này cần hybrid search (dense + BM25/keyword) built-in thì
# nên cân nhắc chuyển sang Weaviate.
VECTOR_STORE = "chromadb"  # "chromadb" | "weaviate" | "faiss"
COLLECTION_NAME = "university_services_docs"


# =============================================================================
# IMPLEMENTATION
# =============================================================================

def load_documents() -> list[dict]:
    """
    Đọc toàn bộ markdown files từ data/standardized/.

    Returns:
        List of {'content': str, 'metadata': {'source': str, 'type': str}}
    """
    documents = []

    if not STANDARDIZED_DIR.exists():
        raise FileNotFoundError(
            f"Không tìm thấy thư mục: {STANDARDIZED_DIR}. "
            "Hãy chắc chắn đã chạy bước standardize dữ liệu trước (data/standardized/)."
        )

    for md_file in sorted(STANDARDIZED_DIR.rglob("*.md")):
        content = md_file.read_text(encoding="utf-8").strip()
        if not content:
            # Bỏ qua file rỗng, tránh tạo chunk vô nghĩa
            continue

        doc_type = "legal" if "legal" in str(md_file).lower() else "news"
        documents.append({
            "content": content,
            "metadata": {
                "source": md_file.name,
                "type": doc_type,
                "path": str(md_file.relative_to(STANDARDIZED_DIR)),
            },
        })

    if not documents:
        raise ValueError(f"Không tìm thấy file .md nào trong {STANDARDIZED_DIR}")

    return documents


def chunk_documents(documents: list[dict]) -> list[dict]:
    """
    Chunk documents theo strategy đã chọn (RecursiveCharacterTextSplitter).

    Returns:
        List of {'content': str, 'metadata': dict} — mỗi item là 1 chunk
    """
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = []
    for doc in documents:
        splits = splitter.split_text(doc["content"])
        for i, chunk_text in enumerate(splits):
            chunk_text = chunk_text.strip()
            if not chunk_text:
                continue
            chunks.append({
                "content": chunk_text,
                "metadata": {**doc["metadata"], "chunk_index": i},
            })

    return chunks


def embed_chunks(chunks: list[dict]) -> list[dict]:
    """
    Embed toàn bộ chunks bằng model đã chọn.

    Returns:
        Mỗi chunk dict được thêm key 'embedding': list[float]
    """
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(EMBEDDING_MODEL)
    
    # E5 models require "passage: " prefix for document chunks
    if "e5" in EMBEDDING_MODEL.lower():
        texts = [f"passage: {c['content']}" for c in chunks]
    else:
        texts = [c["content"] for c in chunks]

    embeddings = model.encode(
        texts,
        show_progress_bar=True,
        normalize_embeddings=True,  # chuẩn hóa để dùng cosine similarity ổn định
        batch_size=32,
    )

    for chunk, emb in zip(chunks, embeddings):
        chunk["embedding"] = emb.tolist()

    return chunks


def index_to_vectorstore(chunks: list[dict]):
    """
    Lưu chunks vào ChromaDB (persistent, local).
    """
    import chromadb

    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    ids = [
        f"{c['metadata']['source']}_chunk_{c['metadata']['chunk_index']}"
        for c in chunks
    ]

    # Upsert theo batch để tránh vượt giới hạn payload của ChromaDB khi corpus lớn
    BATCH_SIZE = 500
    for start in range(0, len(chunks), BATCH_SIZE):
        end = start + BATCH_SIZE
        batch = chunks[start:end]
        collection.upsert(
            ids=ids[start:end],
            documents=[c["content"] for c in batch],
            embeddings=[c["embedding"] for c in batch],
            metadatas=[c["metadata"] for c in batch],
        )


def run_pipeline():
    """Chạy toàn bộ pipeline: load → chunk → embed → index."""
    print("=" * 50)
    print("Task 4: Chunking & Indexing")
    print(f"  Chunking: {CHUNKING_METHOD} (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")
    print(f"  Embedding: {EMBEDDING_MODEL} (dim={EMBEDDING_DIM})")
    print(f"  Vector Store: {VECTOR_STORE}")
    print("=" * 50)

    docs = load_documents()
    print(f"\n✓ Loaded {len(docs)} documents")

    chunks = chunk_documents(docs)
    print(f"✓ Created {len(chunks)} chunks")

    chunks = embed_chunks(chunks)
    print(f"✓ Embedded {len(chunks)} chunks")

    index_to_vectorstore(chunks)
    print("✓ Indexed to vector store")


if __name__ == "__main__":
    run_pipeline()