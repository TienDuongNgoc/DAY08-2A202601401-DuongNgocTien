ôngd

# RAG Evaluation Results

- **Ngày chạy:** 2026-08-04 14:54 SE Asia Standard Time
- **Phương pháp chấm:** Deterministic lexical scoring
- **Golden dataset:** 20 câu hỏi
- **Số context mỗi câu:** top_k=5

Các chỉ số được tính minh bạch như sau: Faithfulness là tỉ lệ token của câu trả lời có trong context; Answer Relevance là F1 với đáp án chuẩn kết hợp độ bao phủ câu hỏi; Context Recall là độ bao phủ đáp án chuẩn trong toàn bộ context; Context Precision là độ liên quan có trọng số theo thứ hạng.

## Overall Scores

| Metric            | Config A (hybrid + rerank) | Config B (dense-only) |      Δ (A − B) |
| ----------------- | -------------------------: | --------------------: | ---------------: |
| Faithfulness      |                      1.000 |                 1.000 |           +0.000 |
| Answer Relevance  |                      0.394 |                 0.384 |           +0.010 |
| Context Recall    |                      0.834 |                 0.801 |           +0.033 |
| Context Precision |                      0.562 |                 0.527 |           +0.035 |
| **Average** |            **0.697** |       **0.678** | **+0.019** |

## A/B Comparison Analysis

**Config A:** Kết hợp dense retrieval và BM25 bằng Reciprocal Rank Fusion, sau đó rerank theo độ tương đồng và mức bao phủ từ khóa.

**Config B:** Chỉ sử dụng dense retrieval, không BM25 và không reranking.

**Kết luận:** Config A đạt điểm trung bình cao hơn (0.697 so với 0.678). Chênh lệch lớn nhất nằm ở Context Precision.

## Worst Performers (Bottom 3 của Config A)

| # | Question                                                                                                                                       | Faithfulness | Relevance | Recall | Precision | Failure Stage | Root Cause                                                                                        |
| -: | ---------------------------------------------------------------------------------------------------------------------------------------------- | -----------: | --------: | -----: | --------: | ------------- | ------------------------------------------------------------------------------------------------- |
| 1 | Làm thế nào để đặt lịch hẹn với Student Connect và có những hình thức hẹn nào?                                                |        1.000 |     0.109 |  0.581 |     0.206 | Generation    | Câu trả lời chưa bao phủ đủ ý của đáp án chuẩn hoặc thiếu diễn đạt trực tiếp. |
| 2 | Sau khi gửi yêu cầu trên Student Connect Portal, sinh viên theo dõi tiến độ ở đâu và liên hệ bằng số nào nếu cần hỗ trợ? |        1.000 |     0.266 |  0.483 |     0.411 | Generation    | Câu trả lời chưa bao phủ đủ ý của đáp án chuẩn hoặc thiếu diễn đạt trực tiếp. |
| 3 | Điều kiện học thuật và điều kiện tiên quyết môn Toán của chương trình Kỹ sư Kỹ thuật phần mềm là gì?                  |        1.000 |     0.145 |  0.808 |     0.287 | Generation    | Câu trả lời chưa bao phủ đủ ý của đáp án chuẩn hoặc thiếu diễn đạt trực tiếp. |

## Recommendations

### 1. Tăng context precision

**Action:** Dùng cross-encoder đa ngôn ngữ thay cho lượt RRF thứ hai và lọc các đoạn menu/điều hướng trước khi lập chỉ mục.

### 2. Cải thiện câu trả lời

**Action:** Giữ temperature=0, yêu cầu câu trả lời ngắn theo từng ý có citation và từ chối trả lời khi không có bằng chứng.

### 3. Làm sạch dữ liệu crawl

**Action:** Loại menu điều hướng, URL lặp và nội dung chân trang trước khi chunk/index để truy vấn không bị khớp nhầm với tên chương trình trong navigation.

## Reproduction

```powershell
python -m group_project.evaluation.eval_pipeline
```
