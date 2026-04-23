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
