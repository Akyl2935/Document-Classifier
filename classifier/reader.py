from pathlib import Path
from typing import List


def read_txt(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def read_pdf(path: Path) -> str:
    import pdfplumber

    pages = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                pages.append(text)
    return "\n".join(pages)


def read_docx(path: Path) -> str:
    from docx import Document

    doc = Document(str(path))
    return "\n".join(p.text for p in doc.paragraphs if p.text.strip())


def read_document(path: Path) -> str:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    suffix = path.suffix.lower()
    readers = {
        ".txt": read_txt,
        ".pdf": read_pdf,
        ".docx": read_docx,
        ".doc": read_docx,
    }

    reader = readers.get(suffix)
    if reader is None:
        supported = ", ".join(get_supported_extensions())
        raise ValueError(f"Unsupported file type '{suffix}'. Supported: {supported}")

    return reader(path)


def get_supported_extensions() -> List[str]:
    return [".txt", ".pdf", ".docx", ".doc"]
