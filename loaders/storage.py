from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct

class QdrantStorage:
    def __init__(self,
                 url: str,
                 collection: str,
                 embedding_dim: int,
                 timeout: int):
        self.client = QdrantClient(url=url, timeout=timeout)
        self.collection = collection
        self.embedding_dim = embedding_dim

        self._init_conn()

    def _init_conn(self):
        if not self.client.collection_exists(self.collection):
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=VectorParams(
                    size=self.embedding_dim,
                    distance=Distance.COSINE,
                )
            )
    
    def upsert(self,
               ids: list[str],
               vectors: list[list[float]],
               payloads: list[dict[str, Any]]):
        points = [PointStruct(id=ids[i], vector=vectors[i], payload=payloads[i])
                  for i in range(len(ids))]
        out = self.client.upsert(self.collection, points)
        return out.model_dump_json()
    
    def query_single(self, query_vector: list[float], top_k: int):
        response = self.client.query_points(
            collection_name=self.collection,
            query=query_vector,
            with_payload=True,
            limit=top_k,
        )

        try:
            results = [{
                "context": res.payload.get("text", ""),
                "source": res.payload.get("source", ""),
            } for res in response.points]
        except:
            raise LookupError
        
        return results