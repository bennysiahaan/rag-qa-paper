import uuid
from pathlib import Path

from langchain_ollama import OllamaEmbeddings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.readers.file import PDFReader

from .custom_types import RAGChunkAndSrc, RAGSearchResult
from .storage import QdrantStorage

class Retriever:
    def __init__(self,
                 model: str,
                 top_k: int = 5,
                 chunk_size: int = 1000,
                 chunk_overlap: int = 200,
                 storage_url: str = "http://localhost:6333",
                 collection_name: str = "docs",
                 embedding_dim: int = 1024,
                 timeout: int = 30,
                 preload_pdf_dirpath: str | None = None):
        self.model = model
        self.top_k = top_k
        self.collection_name = collection_name

        self.embeddings = OllamaEmbeddings(model=model)

        self.doc_splitter = SentenceSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        self.pdf_reader = PDFReader()

        self.storage = QdrantStorage(
            url=storage_url, collection=collection_name,
            embedding_dim=embedding_dim, timeout=timeout,
        )

        self._ingest_pdf_once(preload_pdf_dirpath)
    
    def _ingest_pdf_once(self, dirpath: str | None):
        if dirpath is None:
            return
        if self.storage.client.count(self.collection_name, exact=False).count > 0:
            return
        
        pdf_dirpath = Path(dirpath)
        entries: list[RAGChunkAndSrc] = []
        for file in pdf_dirpath.glob("*.pdf"):
            docs = self.pdf_reader.load_data(file=file)
            texts = [d.text for d in docs if getattr(d, "text", None)]
            chunks = []
            for t in texts:
                chunks.extend(self.doc_splitter.split_text(t))
            entries.append(RAGChunkAndSrc(chunks=chunks, source_id=file.name))
        result = self.upsert(entries)
        print(result)
    
    def upsert(self, entries: list[RAGChunkAndSrc]):
        ids = [str(uuid.uuid5(uuid.NAMESPACE_URL, f"{entry.source_id}:{i+j}"))
               for i, entry in enumerate(entries) for j in range(len(entry.chunks))]
        texts = []
        for entry in entries:
            texts.extend(entry.chunks)
        vectors = self.embeddings.embed_documents(texts)
        payloads = [{
            "source": entry.source_id,
            "text": chunk,
        } for entry in entries for chunk in entry.chunks]
        out = self.storage.upsert(ids, vectors, payloads)
        return out
    
    def search(self, query: str):
        query_vec = self.embeddings.embed_query(query)
        docs = self.storage.query_single(query_vec, self.top_k)
        result = [RAGSearchResult(context=doc["context"], source=doc["source"])
                  for doc in docs]
        return result