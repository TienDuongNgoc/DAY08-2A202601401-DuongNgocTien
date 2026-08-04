# Báo cáo nhóm — University Services RAG Chatbot

## 1. Tổng quan

Dự án xây dựng chatbot hỏi đáp về dịch vụ, học phí và chính sách đại học bằng kiến trúc Retrieval-Augmented Generation (RAG). Hệ thống truy xuất thông tin từ dữ liệu RMIT Việt Nam đã chuẩn hóa, kết hợp nhiều phương pháp tìm kiếm và sử dụng LLM để tạo câu trả lời kèm nguồn tham khảo.

Dự án bao gồm hai sản phẩm chính:

1. Chatbot RAG có giao diện Streamlit.
2. Pipeline đánh giá A/B giữa Hybrid Retrieval và Dense-only bằng bốn metric chất lượng RAG.

## 2. Thành viên và phân công

Nhóm áp dụng phương án phân vai dành cho 6 thành viên trong đề bài.

| STT | Thành viên | MSSV | Vai trò | Nhiệm vụ chính | Trạng thái |
|---:|---|---|---|---|---|
| 1 | Dương Ngọc Tiến | 2A202601401 | Role 1 — Team Leader & RAG Architect | Quản lý nhóm, thiết kế kiến trúc tổng thể, tích hợp pipeline và điều phối demo | Đã phân công |
| 2 | Nguyễn Minh Huy | 2A202601303 | Role 2 — Data Engineering & Scraping Developer | Task 1–3: thu thập tài liệu, crawl tin tức và chuẩn hóa dữ liệu sang Markdown | Đã phân công |
| 3 | Ngô Phương Nam | 2A202601231 | Role 3 — Vector Database & Dense Search Developer | Task 4–5: chunking, ChromaDB, Semantic Search và HyDE | Đã phân công |
| 4 | Nguyễn Mạnh Hiệp | 2A202601391 | Role 4 — Sparse Retrieval & Fallback Developer | Task 6–8: BM25/TF-IDF, RRF reranking và PageIndex fallback | Đã phân công |
| 5 | Đặng Hoàng Hải | 2A202601303 | Role 5 — Frontend UI & App Integration Developer | Thiết kế Streamlit UI và tích hợp sinh câu trả lời có citation | Đã phân công |
| 6 | Thiều Văn Long | 2A202601489 | Role 6 — Evaluation & Benchmark QA Developer | Xây dựng golden dataset 20 câu, chạy benchmark A/B và viết báo cáo | Đã phân công |

## 3. Kiến trúc hệ thống

```text
Người dùng
    |
    v
Streamlit UI (app.py)
    |
    v
Retrieval Pipeline (Task 9)
    |
    +----------------------+----------------------+
    |                                             |
    v                                             v
Dense Retrieval                              Sparse Retrieval
BGE-M3 + ChromaDB                            BM25 / TF-IDF
    |                                             |
    +----------------------+----------------------+
                           |
                           v
                    RRF Reranking
                           |
                  cosine score < threshold?
                     /                 \
                   có                 không
                   /                     \
                  v                       v
         PageIndex Fallback       Top-k RAG Context
                   \                     /
                    +---------+---------+
                              |
                              v
                Reorder + Context Formatting
                              |
                              v
                  OpenRouter/OpenAI LLM
                              |
                              v
                Câu trả lời + nguồn trích dẫn
```

### Luồng đánh giá

```text
golden_dataset.json
        |
        +--> Config A: Hybrid + RRF/rerank --+
        |                                     |
        +--> Config B: Dense-only ------------+--> 4 metrics --> results.md
```

## 4. Cấu trúc thư mục chính

```text
.
├── app.py                              # Giao diện Streamlit
├── data/
│   ├── landing/                        # Dữ liệu thô
│   └── standardized/
│       ├── legal/                      # Văn bản chính sách đã chuẩn hóa
│       └── news/                       # Bài viết đã chuẩn hóa
├── chroma_db/                          # Vector database
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
│   └── evaluation/
│       ├── golden_dataset.json
│       ├── eval_pipeline.py
│       └── results.md
├── tests/
└── requirements.txt
```

## 5. Dữ liệu và Golden Dataset

Corpus hiện có 5 tài liệu thuộc nhóm `data/standardized/legal` và 5 tài liệu thuộc nhóm `data/standardized/news`.

Golden dataset được xây dựng trực tiếp từ các tài liệu đã chuẩn hóa, không tạo câu hỏi nằm ngoài corpus. File [evaluation/golden_dataset.json](evaluation/golden_dataset.json) gồm 20 test case:

- 12 câu dựa trên tài liệu `news`.
- 8 câu dựa trên tài liệu `legal`.
- Mỗi test case có `question`, `expected_answer` và `expected_context`.
- Các chủ đề gồm học phí, thanh toán, hoàn phí, tuyển sinh, điều kiện môn học và Student Connect.

## 6. Các thành phần kỹ thuật

### 6.1. Thu thập và chuẩn hóa dữ liệu

- Task 1 thu thập tài liệu chính sách vào `data/landing/legal`.
- Task 2 crawl tối thiểu 5 bài viết vào `data/landing/news`.
- Task 3 chuyển dữ liệu sang Markdown và lưu trong `data/standardized`.

Dependency dành cho crawler được lưu riêng trong `requirements-crawl.txt` để dễ tái lập và tránh ảnh hưởng đến môi trường chạy chatbot, evaluation.

### 6.2. Retrieval

- Dense retrieval sử dụng embedding model `BAAI/bge-m3` và ChromaDB.
- Sparse retrieval sử dụng BM25/TF-IDF.
- Kết quả dense và sparse được hợp nhất bằng Reciprocal Rank Fusion (RRF).
- Fallback được quyết định bằng cosine score gốc của dense retrieval, không dùng RRF score vì hai loại điểm có thang đo khác nhau.
- PageIndex được sử dụng khi kết quả semantic thấp hơn ngưỡng cấu hình.

### 6.3. Generation và UI

- Các đoạn context được reorder trước khi gửi tới LLM.
- LLM được gọi qua OpenRouter hoặc endpoint tương thích OpenAI.
- Câu trả lời hiển thị danh sách nguồn, loại tài liệu và retrieval score.
- Streamlit lưu và hiển thị lịch sử trong phiên làm việc. Lịch sử hiện chưa được đưa trở lại retrieval prompt, vì vậy câu hỏi follow-up nên chứa đủ ngữ cảnh.

## 7. Thiết kế đánh giá A/B

### Cấu hình

| Cấu hình | Retrieval | Mục đích |
|---|---|---|
| Config A | Dense + BM25 + RRF/rerank | Đánh giá lợi ích của hybrid retrieval |
| Config B | Dense-only | Làm đường cơ sở để so sánh |

Mỗi cấu hình lấy tối đa `top_k=5` đoạn ngữ cảnh cho một câu hỏi.

### Metric

| Metric | Ý nghĩa |
|---|---|
| Faithfulness | Câu trả lời có được hỗ trợ bởi context hay không |
| Answer Relevance | Câu trả lời có trực tiếp giải quyết câu hỏi hay không |
| Context Recall | Retriever có lấy đủ bằng chứng cần thiết hay không |
| Context Precision | Các context xếp hạng cao có thực sự hữu ích hay không |

Các chỉ số được tính theo phương pháp xác định dựa trên token overlap và F1: Faithfulness đo tỉ lệ nội dung câu trả lời xuất hiện trong context; Answer Relevance kết hợp F1 với đáp án tham chiếu và độ bao phủ câu hỏi; Context Recall đo độ bao phủ đáp án trong toàn bộ context; Context Precision tính độ liên quan có trọng số theo thứ hạng.

## 8. Kết quả thực nghiệm

### Kết quả trên toàn bộ 20 câu

| Cấu hình | Faithfulness | Answer Relevance | Context Recall | Context Precision | Trung bình |
|---|---:|---:|---:|---:|---:|
| Config A — Hybrid + rerank | **1.000** | **0.394** | **0.834** | **0.562** | **0.697** |
| Config B — Dense-only | 1.000 | 0.384 | 0.801 | 0.527 | 0.678 |
| Chênh lệch A − B | 0.000 | +0.010 | +0.033 | +0.035 | +0.019 |

Trên toàn bộ 20 câu, Config A cao hơn Config B 0.019 điểm trung bình. Cải thiện rõ nhất nằm ở Context Precision (+0.035) và Context Recall (+0.033), cho thấy việc kết hợp dense retrieval với BM25 giúp tăng độ bao phủ và giảm bớt context kém liên quan.

Bảng điểm chi tiết, các trường hợp có kết quả thấp và đề xuất cải tiến được lưu tại [evaluation/results.md](evaluation/results.md).

## 9. Phân tích lỗi

Các trường hợp có điểm thấp nhất tập trung ở câu hỏi đặt lịch/liên hệ Student Connect và điều kiện tiên quyết môn Toán của ngành Software Engineering.

Nguyên nhân chính:

1. Context còn chứa menu, footer, URL hoặc nội dung điều hướng.
2. Một số đoạn gần chủ đề nhưng không trực tiếp chứa câu trả lời vẫn được xếp hạng cao.
3. Câu trả lời sinh ra đôi lúc dài hoặc chưa trả lời trực tiếp trọng tâm câu hỏi.
4. RRF chỉ hợp nhất thứ hạng, không đánh giá sâu quan hệ ngữ nghĩa như cross-encoder.

Hướng cải thiện:

1. Làm sạch nội dung crawl trước khi chunking.
2. Thêm cross-encoder reranker sau RRF.
3. Hiệu chỉnh ngưỡng fallback bằng tập câu hỏi in-domain và out-of-domain.
4. Đặt temperature bằng 0 và yêu cầu LLM trả lời ngắn gọn, trực tiếp.
5. Bổ sung câu hỏi nhiều bước và câu hỏi không có đáp án vào golden dataset.
6. Mở rộng golden dataset và theo dõi điểm số sau mỗi lần thay đổi retrieval.

## 10. Hạn chế

- Phương pháp token overlap chưa đánh giá đầy đủ các trường hợp diễn đạt đồng nghĩa hoặc suy luận nhiều bước.
- Golden dataset hiện có 20 câu và mới bao phủ một phần câu hỏi thực tế của người dùng.
- Dữ liệu crawl vẫn có thể chứa menu, footer hoặc nội dung lặp làm giảm Context Precision.
- Conversation history mới được hiển thị trên UI, chưa được dùng để viết lại follow-up query.
- PageIndex và generation cần API key tương ứng để kiểm thử end-to-end đầy đủ.

## 11. Cài đặt

### 11.1. Tạo môi trường chính

```powershell
conda create -n rag-lab python=3.11 -y
conda activate rag-lab
pip install -r requirements.txt
```

### 11.2. Cấu hình API key

Sao chép `.env.example` thành `.env`, sau đó điền các khóa cần dùng:

```env
OPENROUTER_API_KEY=your_openrouter_key
# OPENAI_API_KEY=your_openai_key
# PAGEINDEX_API_KEY=your_pageindex_key
# JINA_API_KEY=your_jina_key
```

Không đưa `.env` hoặc API key thật lên Git.

### 11.3. Môi trường crawler tùy chọn

```powershell
conda create -n rag-crawl python=3.11 -y
conda activate rag-crawl
pip install -r requirements-crawl.txt
playwright install chromium
```

## 12. Hướng dẫn chạy

### 12.1. Chạy chatbot

```powershell
conda run --no-capture-output -n rag-lab streamlit run app.py
```

Mở `http://localhost:8501` nếu trình duyệt không tự mở.

### 12.2. Chạy kiểm thử

Chạy bộ kiểm thử từ thư mục gốc của project:

```powershell
conda run --no-capture-output -n rag-lab python -X utf8 -m pytest -q
```

Kết quả kiểm tra gần nhất: `34 passed, 1 skipped`.

### 12.3. Chạy đánh giá

```powershell
conda run --no-capture-output -n rag-lab python -u -X utf8 -m group_project.evaluation.eval_pipeline
```

Lệnh trên chạy đủ 20 câu, so sánh hai cấu hình và ghi kết quả vào `group_project/evaluation/results.md`.

## 13. Deliverables

- [x] Giao diện Streamlit tại [../app.py](../app.py).
- [x] Pipeline retrieval và generation Task 1–10.
- [x] [Golden dataset gồm 20 test case](evaluation/golden_dataset.json).
- [x] [Evaluation pipeline](evaluation/eval_pipeline.py).
- [x] So sánh Config A và Config B.
- [x] [Báo cáo đánh giá đủ 20 câu](evaluation/results.md).

## 14. Kết luận

Project đã tích hợp đầy đủ các thành phần chính của một hệ thống RAG: thu thập và chuẩn hóa dữ liệu, dense retrieval, sparse retrieval, RRF, fallback, generation có citation, UI và pipeline đánh giá A/B. Kết quả trên 20 câu cho thấy Hybrid Retrieval đạt điểm trung bình cao hơn Dense-only, với lợi thế rõ nhất ở độ bao phủ và độ chính xác của context. Vì vậy, Config A được chọn làm cấu hình phù hợp hơn cho corpus hiện tại.
