from langchain_core.embeddings import Embeddings
from langchain_postgres import PGVector

class PGEmbeddings:
    def __init__(self, embeddings: Embeddings):
        self.connection = "postgresql+psycopg://langchain:langchain@localhost:6024/langchain"
        self.embeddings = embeddings

    def get_vectorstore(self, collection_name: str):
        return PGVector(
            embeddings=self.embeddings,
            collection_name=collection_name,
            connection=self.connection,
            use_jsonb=True,
        )
