import regex as re
import uuid
from pathlib import Path

from fastembed import SparseTextEmbedding
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.readers.file import PDFReader
from qdrant_client.models import SparseVector

from .custom_types import RAGChunkAndSrc
from .storage import QdrantStorage

class Retriever:
    def __init__(self,
                 top_k: int = 60,
                 chunk_size: int = 1000,
                 chunk_overlap: int = 200,
                 storage_url: str = "http://localhost:6333",
                 collection_name: str = "docs",
                 embedding_dim: int = 1024,
                 timeout: int = 30,
                 preload_pdf_dirpath: str | None = None,
                 cleanup_on_init: bool = False):
        self.top_k = top_k
        self.collection_name = collection_name
        self.sparse_model = "Qdrant/bm25"
        self.dense_model = "bge-m3:latest"

        self.sparse_embed = SparseTextEmbedding(model_name=self.sparse_model)
        self.dense_embed = OllamaEmbeddings(model=self.dense_model)

        self.doc_splitter = SentenceSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        self.pdf_reader = PDFReader()

        self.storage = QdrantStorage(
            url=storage_url, collection=collection_name,
            embedding_dim=embedding_dim, timeout=timeout,
            cleanup_on_init=cleanup_on_init,
        )

        self.whitespace_regex = re.compile(r"\s+")

        self._ingest_pdf_once(preload_pdf_dirpath)
    
    def _ingest_pdf_once(self, dirpath: str | None):
        if dirpath is None:
            return
        if self.storage.count_points() > 0:
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

        # Filter PDF page number artefacts
        def _clean(text: str) -> str:
            return self.whitespace_regex.sub(" ", text).strip() if text else ""
        def _is_junk(text: str) -> bool:
            if len(text) < 20:
                return True
            words = [w for w in text.split() if sum(c.isalpha() for c in w) >= 3]
            if len(words) < 5:
                return True
            return False
        filtered_entries: list[RAGChunkAndSrc] = []
        for entry in entries:
            chunks = []
            for chunk in entry.chunks:
                cleaned = _clean(chunk)
                if _is_junk(cleaned):
                    continue
                chunks.append(cleaned)
            filtered_entries.append(RAGChunkAndSrc(chunks=chunks,
                                                   source_id=entry.source_id))
        
        result = self.upsert(filtered_entries)
        print(result)

    def upsert(self, entries: list[RAGChunkAndSrc]):
        items = [(entry.source_id, chunk) for entry in entries for chunk in entry.chunks]
        ids = [str(uuid.uuid5(uuid.NAMESPACE_URL, f"{src}:{i}"))
               for i, (src, _) in enumerate(items)]
        texts = [chunk for _, chunk in items]
        sparse_vectors = list(self.sparse_embed.embed(texts))
        dense_vectors = self.dense_embed.embed_documents(texts)
        payloads = [{"source": src, "text": chunk} for src, chunk in items]
        out = self.storage.upsert(ids, sparse_vectors, dense_vectors, payloads)
        return out
    
    def search_sparse(self, query: str) -> list[Document]:
        sparse_embedding = next(iter(self.sparse_embed.embed([query])))
        query_vector = SparseVector(indices=sparse_embedding.indices.tolist(),
                                    values=sparse_embedding.values.tolist())
        docs = self.storage.query_single(query_vector, self.top_k, "sparse")
        result = [Document(page_content=doc["context"],
                           metadata={"source": doc["source"]})
                           for doc in docs]
        return result
    
    def search_dense(self, query: str) -> list[Document]:
        query_vector = self.dense_embed.embed_query(query)
        docs = self.storage.query_single(query_vector, self.top_k, "dense")
        result = [Document(page_content=doc["context"],
                           metadata={"source": doc["source"]})
                           for doc in docs]
        return result
    
    def search_hybrid(self,
                      query: str,
                      prefetch_limit: int) -> list[Document]:
        sparse_embedding = next(iter(self.sparse_embed.embed([query])))
        sparse_vector = SparseVector(indices=sparse_embedding.indices.tolist(),
                                    values=sparse_embedding.values.tolist())
        
        dense_vector = self.dense_embed.embed_query(query)

        docs = self.storage.query_rrf(sparse_vector=sparse_vector,
                                      dense_vector=dense_vector,
                                      limit=self.top_k,
                                      prefetch_limit=prefetch_limit)
        result = [Document(page_content=doc["context"],
                           metadata={"source": doc["source"]})
                           for doc in docs]
        return result
    
    def search(self,
               query: str,
               using: str = "dense",
               prefetch_limit: int = 200):
        if using.lower() == "sparse":
            return self.search_sparse(query)
        if using.lower() == "hybrid":
            return self.search_hybrid(query, prefetch_limit=prefetch_limit)
        return self.search_dense(query)
    