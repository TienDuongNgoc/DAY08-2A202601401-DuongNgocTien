---
title: University Services RAG Chatbot
emoji: 🎓
colorFrom: blue
colorTo: indigo
sdk: streamlit
sdk_version: "1.35.0"
app_file: app.py
pinned: false
---

# University Services RAG Chatbot

Chatbot hỏi đáp về dịch vụ, học phí và chính sách đại học từ dữ liệu công khai của RMIT Việt Nam. Hệ thống triển khai đầy đủ pipeline Retrieval-Augmented Generation (RAG), từ thu thập và chuẩn hóa dữ liệu đến hybrid retrieval, fallback, sinh câu trả lời có trích dẫn và đánh giá A/B.

## Tính năng chính

- Thu thập tài liệu PDF và crawl bài viết, sau đó chuẩn hóa sang Markdown.
- Chia đoạn bằng `RecursiveCharacterTextSplitter` và lập chỉ mục trong ChromaDB.
- Kết hợp dense retrieval (`BAAI/bge-m3`) với BM25 bằng Reciprocal Rank Fusion (RRF).
- Dùng điểm cosine gốc để quyết định PageIndex fallback.
- Sắp xếp lại context để giảm hiện tượng *lost in the middle*.
- Sinh câu trả lời có citation qua endpoint tương thích OpenAI của OpenRouter.
- Giao diện chat Streamlit có lịch sử phiên và danh sách nguồn đã dùng.
- Pipeline đánh giá A/B giữa Hybrid + rerank và Dense-only trên 20 câu hỏi.

## Kiến trúc

```text
Người dùng
    |
    v
Streamlit UI (app.py)
    |
    v
Retrieval Pipeline (Task 9)
    |
    +-------------------------+-------------------------+
    |                                                   |
    v                                                   v
Dense Retrieval                                    BM25 Retrieval
BGE-M3 + ChromaDB                                      |
    |                                                   |
    +-------------------------+-------------------------+
                              |
                              v
                         RRF Reranking
                              |
                    cosine score < threshold?
                         /               \
                       có               không
                       /                   \
                      v                     v
             PageIndex Fallback       Top-k context
                       \                   /
                        +--------+--------+
                                 |
                                 v
                    Reorder + LLM Generation
                                 |
                                 v
                    Câu trả lời + citations
```

### Cấu hình kỹ thuật

| Thành phần | Cấu hình hiện tại |
|---|---|
| Chunking | Recursive, `chunk_size=500`, `chunk_overlap=50` |
| Embedding | `BAAI/bge-m3`, 1024 chiều |
| Vector store | ChromaDB, collection `university_services_docs` |
| Sparse retrieval | BM25 |
| Reranking mặc định | RRF |
| Fallback threshold | Cosine similarity `< 0.3` |
| Số context mặc định | `top_k=5` |
| Generation | `inclusionai/ling-3.0-flash:free`, `temperature=0.3`, `top_p=0.9` |

## Cấu trúc project

```text
.
├── app.py                              # Giao diện Streamlit
├── data/
│   ├── landing/
│   │   ├── legal/                      # 5 tài liệu PDF gốc
│   │   └── news/                       # 5 bài viết JSON gốc
│   └── standardized/
│       ├── legal/                      # 5 tài liệu Markdown
│       └── news/                       # 5 bài viết Markdown
├── chroma_db/                          # Vector database đã lập chỉ mục
├── src/
│   ├── task1_collect_legal_docs.py
│   ├── task2_crawl_news.py
│   ├── task3_convert_markdown.py
│   ├── task4_chunking_indexing.py
│   ├── task5_semantic_search.py
│   ├── task6_lexical_search.py
│   ├── task7_reranking.py
│   ├── task8_pageindex_vectorless.py
│   ├── task9_retrieval_pipeline.py
│   └── task10_generation.py
├── group_project/
│   ├── README.md                       # Báo cáo kỹ thuật chi tiết
│   └── evaluation/
│       ├── golden_dataset.json         # 20 test case
│       ├── eval_pipeline.py            # Offline và RAGAS evaluation
│       └── results.md                  # Báo cáo A/B
├── tests/test_individual.py
├── requirements.txt                    # Môi trường RAG chính
├── requirements-crawl.txt              # Môi trường Crawl4AI riêng
└── .env.example
```

## Cài đặt

Project được kiểm thử với Python 3.11. Trên PowerShell:

```powershell
conda create -n rag-lab python=3.11 -y
conda activate rag-lab
pip install -r requirements.txt
Copy-Item .env.example .env
```

Điền `OPENROUTER_API_KEY` để chatbot có thể sinh câu trả lời. Các khóa còn lại chỉ cần khi dùng tính năng tương ứng:

```env
OPENROUTER_API_KEY=your_openrouter_key
# OPENAI_API_KEY=your_openai_key        # RAGAS evaluation
# PAGEINDEX_API_KEY=your_pageindex_key
# JINA_API_KEY=your_jina_key
```

Không commit `.env` hoặc API key thật. Nếu chỉ chạy đánh giá offline thì không cần API key.

### Môi trường crawler

Crawl4AI dùng OpenAI SDK 2.x, trong khi stack RAGAS hiện tại dùng OpenAI SDK 1.x. Cài crawler trong môi trường riêng để tránh xung đột dependency:

```powershell
conda create -n rag-crawl python=3.11 -y
conda activate rag-crawl
pip install -r requirements-crawl.txt
playwright install chromium
```

## Hướng dẫn chạy

Các lệnh dưới đây được chạy từ thư mục gốc của project.

### Chatbot

```powershell
conda run --no-capture-output -n rag-lab streamlit run app.py
```

Mở `http://localhost:8501` nếu trình duyệt không tự mở.

### Kiểm thử

```powershell
conda run --no-capture-output -n rag-lab python -X utf8 -m pytest -q
```

Kết quả kiểm tra gần nhất: `34 passed, 1 skipped`.

### Đánh giá A/B offline

```powershell
conda run --no-capture-output -n rag-lab python -u -X utf8 -m group_project.evaluation.eval_pipeline
```

Chế độ offline chạy đủ 20 câu, không gọi API và ghi báo cáo vào `group_project/evaluation/results.md`. Có thể chạy nhanh một phần dữ liệu hoặc đổi số context:

```powershell
python -m group_project.evaluation.eval_pipeline --limit 5 --top-k 3
```

### Đánh giá bằng RAGAS

Chế độ này cần `OPENROUTER_API_KEY` hoặc `OPENAI_API_KEY`:

```powershell
python -m group_project.evaluation.eval_pipeline --mode ragas
```

## Kết quả đánh giá hiện tại

Đánh giá offline dùng bốn metric xác định dựa trên token overlap và F1. Kết quả trên 20 câu hỏi, với `top_k=5`:

| Metric | Config A: Hybrid + rerank | Config B: Dense-only | Δ (A − B) |
|---|---:|---:|---:|
| Faithfulness | 1.000 | 1.000 | +0.000 |
| Answer Relevance | 0.394 | 0.384 | +0.010 |
| Context Recall | 0.834 | 0.801 | +0.033 |
| Context Precision | 0.562 | 0.527 | +0.035 |
| **Trung bình** | **0.697** | **0.678** | **+0.019** |

Config A đạt điểm trung bình cao hơn, với cải thiện rõ nhất ở Context Precision và Context Recall. Xem [báo cáo kết quả đầy đủ](group_project/evaluation/results.md) để biết các trường hợp có điểm thấp và đề xuất cải tiến.

## Thành viên và phân công

| Thành viên | MSSV | Vai trò |
|---|---|---|
| Dương Ngọc Tiến | 2A202601401 | Team Leader & RAG Architect |
| Nguyễn Minh Huy | 2A202601303 | Data Engineering & Scraping Developer |
| Ngô Phương Nam | 2A202601231 | Vector Database & Dense Search Developer |
| Nguyễn Mạnh Hiệp | 2A202601391 | Sparse Retrieval & Fallback Developer |
| Đặng Hoàng Hải | 2A202601303 | Frontend UI & App Integration Developer |
| Thiều Văn Long | 2A202601489 | Evaluation & Benchmark QA Developer |

## Tài liệu

- [Báo cáo kỹ thuật và phân tích chi tiết](group_project/README.md)
- [Golden dataset](group_project/evaluation/golden_dataset.json)
- [Kết quả evaluation](group_project/evaluation/results.md)
- [Hướng dẫn lab](LAB_GUIDE.md)

## Hạn chế hiện tại

- Lịch sử hội thoại được hiển thị trong UI nhưng chưa được dùng để viết lại follow-up query.
- Dữ liệu crawl vẫn có thể chứa menu, footer hoặc nội dung lặp.
- Offline metrics không đánh giá đầy đủ cách diễn đạt đồng nghĩa hay suy luận nhiều bước.
- PageIndex fallback và generation end-to-end cần API key tương ứng.
