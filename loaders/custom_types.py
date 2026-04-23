from langchain_core.documents import Document
import pydantic

class RAGChunkAndSrc(pydantic.BaseModel):
    chunks: list[str]
    source_id: str

class RAGUpsertEntry(pydantic.BaseModel):
    id: str
    context: str
    source: str

class RAGSearchResult(pydantic.BaseModel):
    context: str
    source: str

    @classmethod
    def to_document(cls) -> Document:
        return Document(page_content=cls.context, metadata={"source": cls.source})
