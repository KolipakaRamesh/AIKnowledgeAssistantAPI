import io
import fitz  # PyMuPDF


def extract_text(filename: str, file_bytes: bytes) -> str:
    """
    Extract raw text from a PDF, TXT, or Markdown file.

    Args:
        filename: Original filename (used to determine format).
        file_bytes: Raw file content as bytes.

    Returns:
        Extracted plain-text string.

    Raises:
        ValueError: If the file type is unsupported.
    """
    ext = filename.rsplit(".", 1)[-1].lower()

    if ext == "pdf":
        return _extract_pdf(file_bytes)
    elif ext in ("txt", "md"):
        return file_bytes.decode("utf-8", errors="ignore")
    else:
        raise ValueError(
            f"Unsupported file type: .{ext}. Allowed types: pdf, txt, md"
        )


def _extract_pdf(file_bytes: bytes) -> str:
    """Extract text from all pages of a PDF document."""
    text_parts: list[str] = []
    with fitz.open(stream=io.BytesIO(file_bytes), filetype="pdf") as doc:
        for page in doc:
            text_parts.append(page.get_text("text"))
    return "\n".join(text_parts).strip()
