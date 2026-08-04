"""
Task 8 — PageIndex Vectorless RAG.

Đăng ký tài khoản tại: https://pageindex.ai/
SDK & sample code: https://github.com/VectifyAI/PageIndex

PageIndex cho phép RAG mà không cần vector store — sử dụng
structural understanding của document thay vì embedding.

Cài đặt:
    pip install pageindex

Hướng dẫn:
    1. Đăng ký account tại pageindex.ai
    2. Lấy API key
    3. Upload documents
    4. Query sử dụng PageIndex API

Lưu ý: API `/retrieval` của PageIndex hiện đã deprecated (vẫn hoạt động, nhưng response
có field "deprecation" cảnh báo) và trả kết quả trong "retrieved_nodes" — mỗi node có
"relevant_contents": list[list[{section_title, relevant_content}]]. In response thật ra
(json.dumps(...)) trước khi viết logic parse, đừng đoán schema từ ví dụ code cũ.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

PAGEINDEX_API_KEY = os.getenv("PAGEINDEX_API_KEY", "")
STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"


def upload_documents():
    """
    Upload toàn bộ markdown documents lên PageIndex.
    """
    from pageindex.client import PageIndexClient
    from fpdf import FPDF
    import json

    client = PageIndexClient(api_key=PAGEINDEX_API_KEY)
    doc_ids = []

    # Gom tất cả nội dung vào 1 file PDF để tránh lỗi "LimitReached" của bản miễn phí
    pdf_path = STANDARDIZED_DIR / "all_documents_combined.pdf"
    
    if not pdf_path.exists():
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Helvetica", size=12)
        
        combined_content = ""
        for md_file in STANDARDIZED_DIR.rglob("*.md"):
            combined_content += f"--- {md_file.name} ---\n\n"
            combined_content += md_file.read_text(encoding="utf-8") + "\n\n"
            
        # Replace unsupported characters for basic Helvetica
        combined_content = combined_content.encode("latin-1", "replace").decode("latin-1")
        pdf.multi_cell(0, 10, txt=combined_content)
        pdf.output(str(pdf_path))

    print(f"Uploading combined PDF: {pdf_path.name}...")
    resp = client.submit_document(str(pdf_path))
    doc_id = resp.get("doc_id") or resp.get("id")
    if doc_id:
        print(f"  ✓ Uploaded combined document -> {doc_id}")
        doc_ids.append(doc_id)
            
    # Save doc_ids for retrieval
    doc_ids_path = STANDARDIZED_DIR / "pageindex_doc_ids.json"
    with open(doc_ids_path, "w") as f:
        json.dump(doc_ids, f)


def pageindex_search(query: str, top_k: int = 5) -> list[dict]:
    """
    Vectorless retrieval sử dụng PageIndex.
    Dùng làm fallback khi hybrid search không có kết quả tốt.

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả tối đa

    Returns:
        List of {
            'content': str,
            'score': float,
            'metadata': dict,
            'source': 'pageindex'   # Đánh dấu nguồn retrieval
        }
    """
    from pageindex.client import PageIndexClient
    import json
    import time

    client = PageIndexClient(api_key=PAGEINDEX_API_KEY)
    
    doc_ids_path = STANDARDIZED_DIR / "pageindex_doc_ids.json"
    if not doc_ids_path.exists():
        return []
        
    with open(doc_ids_path, "r") as f:
        doc_ids = json.load(f)
        
    results = []
    
    # Search across all uploaded docs
    for doc_id in doc_ids:
        resp = client.submit_query(doc_id=doc_id, query=query)
        retrieval_id = resp.get("retrieval_id") or resp.get("id")
        
        # Poll cho đến khi status == "completed"
        for _ in range(15):
            retrieval = client.get_retrieval(retrieval_id)
            if retrieval.get("status") == "completed":
                break
            time.sleep(2)
        
        # Parse retrieval["retrieved_nodes"]
        for node in retrieval.get("retrieved_nodes", [])[:2]:
            for group in node.get("relevant_contents", []):
                for item in group:
                    # PageIndex không trả score trực tiếp — tự gán theo rank (score giảm dần)
                    score = 1.0 / (len(results) + 1)
                    results.append({
                        "content": item.get("relevant_content", ""),
                        "score": score,
                        "metadata": {"section": item.get("section_title"), "doc_id": doc_id},
                        "source": "pageindex",
                    })
                    
    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:top_k]


if __name__ == "__main__":
    if not PAGEINDEX_API_KEY:
        print("⚠ Hãy set PAGEINDEX_API_KEY trong file .env")
        print("  Đăng ký tại: https://pageindex.ai/")
    else:
        print("Uploading documents...")
        upload_documents()

        print("\nTest query:")
        results = pageindex_search("tuition fee payment methods", top_k=3)
        for r in results:
            print(f"[{r['score']:.3f}] {r['content'][:100]}...")
