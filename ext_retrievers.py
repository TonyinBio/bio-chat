# import QueryBundle
from llama_index.core import QueryBundle

# import NodeWithScore
from llama_index.core.schema import NodeWithScore

# Retrievers
from llama_index.core.retrievers import (
    BaseRetriever,
    VectorIndexRetriever,
    KnowledgeGraphRAGRetriever,
)

from typing import List
class GRetriever(BaseRetriever):
    """Custom retriever that performs both semantic search and hybrid search."""

    def __init__(
        self,
        vector_retriever: VectorIndexRetriever,
        kg_retriever: KnowledgeGraphRAGRetriever,
    ) -> None:
        """Init params."""
        self._vector_retriever = vector_retriever
        self._kg_retriever = kg_retriever

        super().__init__()

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """Retrieve nodes given query."""

        vector_nodes = self._vector_retriever.retrieve(query_bundle)

        entities = [n.node.node_id for n in vector_nodes]
        knowledge_sequence, rel_map = self._kg_retriever._get_knowledge_sequence(entities)

        return self._kg_retriever._build_nodes(knowledge_sequence, rel_map)
    