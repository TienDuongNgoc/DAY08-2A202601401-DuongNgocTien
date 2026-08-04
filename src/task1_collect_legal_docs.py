"""Task 1 — Tải văn bản chính sách/dịch vụ sinh viên của RMIT Vietnam.

Chạy:
    python -m src.task1_collect_legal_docs

Thêm ``--force`` để tải lại các file đã tồn tại. Các URL dưới đây là link PDF
trực tiếp trên website công khai của RMIT Vietnam.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import requests

DATA_DIR = Path(__file__).parent.parent / "data" / "landing" / "legal"

# filename: direct public source URL
DOCUMENTS: dict[str, str] = {
    "student-fees-and-charges-guide-06-2026.pdf": (
        "https://www.rmit.edu.vn/assets/vn/en/assets-for-production/documents/"
        "pdfs/study-at-rmit/tuition-fees/student-fees-and-charges-guide-06-2026.pdf"
    ),
    "block-course-tuition-fee-extension.pdf": (
        "https://www.rmit.edu.vn/content/dam/rmit/vn/en/assets-for-production/"
        "documents/pdfs/students/fees-and-payments/Block%20Course_Application%20"
        "for%20Extension%20of%20Tuition%20Fee%20Payment.pdf"
    ),
    "intensive-course-tuition-fee-extension.pdf": (
        "https://www.rmit.edu.vn/content/dam/rmit/vn/en/assets-for-production/"
        "documents/pdfs/students/fees-and-payments/Intensive%20Course_Application%20"
        "for%20Extension%20of%20Tuition%20Fee%20Payment.pdf"
    ),
    "student-account-appointment-guide.pdf": (
        "https://www.rmit.edu.vn/content/dam/rmit/vn/en/assets-for-production/"
        "documents/pdfs/students/advice-support/how-to-book-an-appointment-"
        "using-your-student-account-PDF.pdf"
    ),
    "student-account-enquiry-guide.pdf": (
        "https://www.rmit.edu.vn/assets/vn/en/assets-for-production/documents/"
        "pdfs/students/advice-support/how-to-submit-an-enquiry-with-student-account.pdf"
    ),
}

MIN_FILE_SIZE_BYTES = 1024
REQUEST_TIMEOUT_SECONDS = 30


def setup_directory():
    """Tạo thư mục data/landing/legal/ nếu chưa có."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[OK] Thu muc da san sang: {DATA_DIR}")


def is_valid_pdf(filepath: Path) -> bool:
    """Return True when *filepath* is a non-empty PDF file."""
    if not filepath.is_file() or filepath.stat().st_size <= MIN_FILE_SIZE_BYTES:
        return False
    with filepath.open("rb") as source:
        return source.read(5) == b"%PDF-"


def download_file(url: str, filename: str, *, force: bool = False) -> Path | None:
    """Download one public PDF safely and return its path, or None on failure."""
    destination = DATA_DIR / filename
    if not force and is_valid_pdf(destination):
        print(f"[SKIP] File hop le da ton tai: {filename}")
        return destination

    temporary_file = destination.with_suffix(f"{destination.suffix}.part")
    try:
        with requests.get(
            url,
            stream=True,
            timeout=REQUEST_TIMEOUT_SECONDS,
            headers={"User-Agent": "UniversityServicesRAG/1.0"},
        ) as response:
            response.raise_for_status()
            with temporary_file.open("wb") as output:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        output.write(chunk)

        if not is_valid_pdf(temporary_file):
            raise ValueError("Nội dung tải về không phải PDF hợp lệ hoặc quá nhỏ")

        temporary_file.replace(destination)
        print(f"[OK] Da tai: {filename}")
        return destination
    except (OSError, requests.RequestException, ValueError) as error:
        temporary_file.unlink(missing_ok=True)
        print(f"[ERROR] Khong the tai {filename}: {error}")
        return None


def download_documents(*, force: bool = False) -> int:
    """Download all Task 1 documents and return the number of available PDFs."""
    setup_directory()
    available = 0
    for filename, url in DOCUMENTS.items():
        if download_file(url, filename, force=force):
            available += 1

    print(f"\nHoan tat: {available}/{len(DOCUMENTS)} tai lieu PDF san sang tai {DATA_DIR}")
    return available


def main() -> None:
    parser = argparse.ArgumentParser(description="Tải tài liệu Task 1 từ RMIT Vietnam")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Tải lại cả những file PDF hợp lệ đã có.",
    )
    args = parser.parse_args()
    available = download_documents(force=args.force)
    if available < 3:
        raise SystemExit("Task 1 cần tối thiểu 3 tài liệu PDF/DOCX hợp lệ.")


if __name__ == "__main__":
    main()
