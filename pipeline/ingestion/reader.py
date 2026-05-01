from pathlib import Path

from llama_index.readers.file import PDFReader, DocxReader

from pipeline.custom_types import (
    SourceBase, RawDocument, SUPPORTED_DOCUMENT_TYPES
)
from pipeline.hash import hash_text

def load_from_sources(sources: list[SourceBase]):
    for source in sources:
        ingest_document(source)

def ingest_document(source: SourceBase):
    assert source.type.lower() in SUPPORTED_DOCUMENT_TYPES, f"Document type not supported {source.type}"

    if source.type.lower() == "pdf":
        ingest = ingest_pdf

    raw_pages = []
    for file in Path(source.path).glob(source.glob):
        pages = ingest(file)
        raw_pages.extend([(str(file), page) for page in pages])
    
    docs = []
    for path, page in raw_pages:
        docs.append(RawDocument(
            source_id=page.id_,
            source_uri=path,
            content=page.text,
            content_hash=hash_text(page.text),
        ))
    
    return docs

def ingest_pdf(path: Path):
    reader = PDFReader()
    return reader.load_data(path)