"""Bounded, separate-process extraction. No OCR, macros, remote relationships or logs."""

import io
import json
import resource
import sys
import zipfile
from pathlib import Path
from defusedxml import ElementTree

MAX_CHARS = 60000
MAX_FILE = 5 * 1024 * 1024


def extract(data, extension):
    if len(data) > MAX_FILE:
        raise ValueError("file_too_large")
    warnings = []
    if extension == "txt":
        if b"\x00" in data:
            raise ValueError("invalid_text_encoding")
        text = data.decode("utf-8-sig", errors="strict")
    elif extension == "docx":
        if not data.startswith(b"PK"):
            raise ValueError("invalid_docx")
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            entries = archive.infolist()
            if (
                len(entries) > 1000
                or sum(e.file_size for e in entries) > 20 * 1024 * 1024
            ):
                raise ValueError("document_expansion_limit")
            if any(
                e.file_size > max(1, e.compress_size) * 150 or e.flag_bits & 1
                for e in entries
            ):
                raise ValueError("document_expansion_limit")
            if any(
                "vbaproject" in e.filename.lower()
                or "embeddings/" in e.filename.lower()
                for e in entries
            ):
                raise ValueError("embedded_content_not_supported")
            info = archive.getinfo("word/document.xml")
            if info.file_size > 4 * 1024 * 1024:
                raise ValueError("document_expansion_limit")
            root = ElementTree.fromstring(archive.read(info))
        namespace = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
        body = root.find(namespace + "body")
        if body is None:
            raise ValueError("invalid_docx")
        paragraphs = []
        for element in body:
            if element.tag == namespace + "tbl":
                paragraphs.append("⟦таблица пропущена⟧")
                warnings.append("Таблица не включена в оценку авторской прозы.")
            elif element.tag == namespace + "p":
                paragraphs.append(
                    "".join(
                        (item.text or "")
                        if item.tag == namespace + "t"
                        else "\n"
                        if item.tag == namespace + "br"
                        else "\t"
                        if item.tag == namespace + "tab"
                        else ""
                        for item in element.iter()
                    ).strip()
                )
        text = "\n\n".join(p for p in paragraphs if p)
    elif extension == "pdf":
        if not data.startswith(b"%PDF-"):
            raise ValueError("invalid_pdf")
        from pypdf import PdfReader

        reader = PdfReader(io.BytesIO(data), strict=True)
        if reader.is_encrypted:
            raise ValueError("encrypted_pdf")
        if len(reader.pages) > 40:
            raise ValueError("pdf_page_limit")
        pages = []
        for page in reader.pages:
            if (
                page.get_contents()
                and len(page.get_contents().get_data()) > 10 * 1024 * 1024
            ):
                raise ValueError("document_expansion_limit")
            pages.append(page.extract_text() or "")
            if sum(map(len, pages)) > MAX_CHARS:
                raise ValueError("text_too_large")
        text = "\n\n".join(pages)
        warnings.append(
            "PDF: порядок блоков зависит от текстового слоя; OCR не выполняется."
        )
    else:
        raise ValueError("unsupported_format")
    text = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    if not text or len(text) < 100:
        raise ValueError("no_text_layer" if extension == "pdf" else "insufficient_text")
    if len(text) > MAX_CHARS:
        raise ValueError("text_too_large")
    if any(ord(c) < 32 and c not in "\n\t" for c in text):
        raise ValueError("invalid_control_characters")
    return {"text": text, "warnings": list(dict.fromkeys(warnings))}


if __name__ == "__main__":
    resource.setrlimit(resource.RLIMIT_AS, (1024 * 1024 * 1024,) * 2)
    resource.setrlimit(resource.RLIMIT_CPU, (8, 9))
    try:
        result = extract(Path(sys.argv[1]).read_bytes(), sys.argv[2])
        print(json.dumps(result, ensure_ascii=False))
    except Exception as error:
        print(
            json.dumps(
                {
                    "error": str(error)
                    if isinstance(error, ValueError)
                    else "invalid_document"
                }
            )
        )
        sys.exit(2)
