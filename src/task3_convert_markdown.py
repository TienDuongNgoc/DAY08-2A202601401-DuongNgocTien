"""
Task 3 — Convert toàn bộ file trong data/landing/ thành Markdown.

Sử dụng MarkItDown của Microsoft:
    https://github.com/microsoft/markitdown

Cài đặt:
    pip install "markitdown[pdf]"
    # Lưu ý: cần extra [pdf] để convert được file PDF. Chỉ "pip install markitdown"
    # (không có extra) sẽ báo MissingDependencyException khi convert PDF, dù JSON/DOCX
    # vẫn convert bình thường.

Hướng dẫn:
    1. Scan toàn bộ file trong data/landing/ (PDF, DOCX, JSON)
    2. Convert sang Markdown
    3. Lưu vào data/standardized/ giữ nguyên cấu trúc thư mục
"""

import json
from pathlib import Path

from markitdown import MarkItDown

LANDING_DIR = Path(__file__).parent.parent / "data" / "landing"
OUTPUT_DIR = Path(__file__).parent.parent / "data" / "standardized"


def convert_legal_docs():
    """Convert PDF/DOCX files trong data/landing/legal/ sang markdown."""
    legal_dir = LANDING_DIR / "legal"
    output_dir = OUTPUT_DIR / "legal"
    output_dir.mkdir(parents=True, exist_ok=True)

    md = MarkItDown()

    for filepath in legal_dir.iterdir():
        if filepath.suffix.lower() in (".pdf", ".docx", ".doc"):
            print(f"Converting: {filepath.name}")
            # Bước 1: Sử dụng MarkItDown để chuyển đổi file (PDF, DOCX) thành text dạng Markdown
            result = md.convert(str(filepath))
            
            # Bước 2: Tạo đường dẫn cho file output
            # filepath.stem lấy tên file bỏ đuôi (ví dụ: 'Luat_Doanh_Nghiep.pdf' -> 'Luat_Doanh_Nghiep')
            output_path = output_dir / f"{filepath.stem}.md"
            
            # Bước 3: Ghi nội dung đã chuyển đổi ra file .md mới với mã hóa UTF-8 để không bị lỗi tiếng Việt
            output_path.write_text(result.text_content, encoding="utf-8")
            
            # Thông báo đã lưu thành công
            print(f"  ✓ Saved: {output_path}")


def convert_news_articles():
    """Convert JSON crawled articles trong data/landing/news/ sang markdown."""
    news_dir = LANDING_DIR / "news"
    output_dir = OUTPUT_DIR / "news"
    output_dir.mkdir(parents=True, exist_ok=True)

    for filepath in news_dir.iterdir():
        if filepath.suffix.lower() == ".json":
            print(f"Converting: {filepath.name}")
            # Bước 1: Đọc nội dung file JSON và parse thành dạng Dictionary của Python
            # Cần chỉ định encoding="utf-8" để đọc đúng tiếng Việt
            data = json.loads(filepath.read_text(encoding="utf-8"))
            
            # Bước 2: Chuẩn bị đường dẫn cho file output (.md)
            output_path = output_dir / f"{filepath.stem}.md"

            # Bước 3: Trích xuất các trường thông tin (metadata) để tạo Header cho file Markdown
            # Sử dụng .get() để tránh lỗi nếu key không tồn tại
            header = f"# {data.get('title', 'Unknown')}\n\n"
            header += f"**Source:** {data.get('url', 'N/A')}\n"
            header += f"**Crawled:** {data.get('date_crawled', 'N/A')}\n\n---\n\n"

            # Bước 4: Ghép Header với phần nội dung Markdown đã được crawler bóc tách sẵn (content_markdown)
            content = header + data.get("content_markdown", "")
            
            # Bước 5: Ghi toàn bộ nội dung ra file .md
            output_path.write_text(content, encoding="utf-8")
            
            # Thông báo đã lưu thành công
            print(f"  ✓ Saved: {output_path}")


def convert_all():
    """Convert toàn bộ files."""
    print("=" * 50)
    print("Task 3: Convert to Markdown (MarkItDown)")
    print("=" * 50)

    print("\n--- Legal Documents ---")
    convert_legal_docs()

    print("\n--- News Articles ---")
    convert_news_articles()

    print("\n✓ Done! Output tại:", OUTPUT_DIR)


if __name__ == "__main__":
    convert_all()
