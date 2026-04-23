from typing import Any

from fastembed import SparseEmbedding
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance, \
    SparseVectorParams, SparseIndexParams, SparseVector, Prefetch, FusionQuery, \
    Fusion

class QdrantStorage:
    def __init__(self,
                 url: str,
                 collection: str,
                 embedding_dim: int,
                 timeout: int,
                 cleanup_on_init: bool):
        self._client = QdrantClient(url=url, timeout=timeout)
        self.collection = collection
        self.embedding_dim = embedding_dim

        self._init_conn(cleanup_on_init)

    def _init_conn(self, cleanup: bool):
        if self._client.collection_exists(self.collection) and cleanup:
            print("Cleaning up collection...")
            if self._client.delete_collection(self.collection):
                print("Done.")
                
        if not self._client.collection_exists(self.collection):
            self._client.create_collection(
                collection_name=self.collection,
                vectors_config={
                    "dense": VectorParams(
                        size=self.embedding_dim,
                        distance=Distance.COSINE,
                    )
                },
                sparse_vectors_config={
                    "sparse": SparseVectorParams(
                        index=SparseIndexParams(),
                    ),
                },
            )
    
    def count_points(self):
        return self._client.count(self.collection, exact=False).count
    
    def delete_collection(self):
        return self._client.delete_collection(self.collection)
    
    def upsert(self,
               ids: list[str],
               sparse_vectors: list[SparseEmbedding],
               dense_vectors: list[list[float]],
               payloads: list[dict[str, Any]]):
        points = [PointStruct(id=ids[i],
                              vector={
                                  "dense": dense_vectors[i],
                                  "sparse": SparseVector(
                                      indices=sparse_vectors[i].indices.tolist(),
                                      values=sparse_vectors[i].values.tolist(),
                                  ),
                              },
                              payload=payloads[i])
                  for i in range(len(ids))]
        out = self._client.upsert(self.collection, points)
        return out.model_dump_json()
    
    def query_single(self,
                     query: list[float] | SparseVector,
                     top_k: int,
                     using: str = "dense"):
        response = self._client.query_points(
            collection_name=self.collection,
            query=query,
            using=using,
            with_payload=True,
            limit=top_k,
        )

        try:
            results = [{"context": res.payload.get("text", ""),
                        "source": res.payload.get("source", "")}
                        for res in response.points]
        except:
            raise LookupError
        
        return results
    
    def query_rrf(self,
                  sparse_vector: SparseVector,
                  dense_vector: list[float],
                  limit: int,
                  prefetch_limit: int):
        response = self._client.query_points(
            collection_name=self.collection,
            prefetch=[
                Prefetch(
                    query=sparse_vector,
                    using="sparse",
                    limit=prefetch_limit,
                ),
                Prefetch(
                    query=dense_vector,
                    using="dense",
                    limit=prefetch_limit,
                ),
            ],
            query=FusionQuery(fusion=Fusion.RRF),
            limit=limit,
            with_payload=True,
        )

        try:
            results = [{"context": res.payload.get("text", ""),
                        "source": res.payload.get("source", "")}
                        for res in response.points]
        except:
            raise LookupError
        
        return results