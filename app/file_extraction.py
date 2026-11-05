"""
Resume file upload support -- extracts raw text from an uploaded PDF or DOCX
file, so it can be fed into the same resume_text pipeline every endpoint
already uses (parse_resume, screen_resume_against_job, score_candidate, etc.).
"""
import io

from fastapi import HTTPException
from pypdf import PdfReader
import docx


def extract_text_from_pdf(file_bytes: bytes) -> str:
    reader = PdfReader(io.BytesIO(file_bytes))
    pages_text = []
    for page in reader.pages:
        text = page.extract_text() or ""
        pages_text.append(text)
    full_text = "\n".join(pages_text).strip()

    if not full_text:
        raise HTTPException(
            status_code=422,
            detail="Could not extract any text from this PDF. It may be a scanned/image-only "
                   "PDF with no selectable text layer -- try a text-based PDF, or use the JSON "
                   "request body with resume_text instead."
        )
    return full_text


def extract_text_from_docx(file_bytes: bytes) -> str:
    document = docx.Document(io.BytesIO(file_bytes))
    paragraphs = [p.text for p in document.paragraphs]

    for table in document.tables:
        for row in table.rows:
            for cell in row.cells:
                if cell.text.strip():
                    paragraphs.append(cell.text)

    full_text = "\n".join(p for p in paragraphs if p.strip()).strip()

    if not full_text:
        raise HTTPException(
            status_code=422,
            detail="Could not extract any text from this DOCX file -- it appears to be empty."
        )
    return full_text


def extract_resume_text(filename: str, file_bytes: bytes) -> str:
    """Dispatches to the right extractor based on file extension."""
    name_lower = (filename or "").lower()

    if name_lower.endswith(".pdf"):
        return extract_text_from_pdf(file_bytes)
    elif name_lower.endswith(".docx"):
        return extract_text_from_docx(file_bytes)
    elif name_lower.endswith(".doc"):
        raise HTTPException(
            status_code=422,
            detail="Legacy .doc files (old binary Word format) are not supported -- "
                   "please save/export the resume as .docx or .pdf and upload that instead."
        )
    else:
        raise HTTPException(
            status_code=422,
            detail=f"Unsupported file type: '{filename}'. Please upload a .pdf or .docx file."
        )
