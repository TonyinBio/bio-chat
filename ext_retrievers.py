# logging
import logging
logger = logging.getLogger(__name__)

# import QueryBundle
from llama_index.core import QueryBundle, StorageContext

# import NodeWithScore
from llama_index.core.schema import NodeWithScore, TextNode

# Retrievers
from llama_index.core.retrievers import (
    BaseRetriever,
    VectorIndexRetriever,
    KnowledgeGraphRAGRetriever,
)

from llama_index.core.vector_stores.types import (
    VectorStore,
    VectorStoreQuery,
    VectorStoreQueryResult,
)
from llama_index.core.settings import (
    embed_model_from_settings_or_context,
    Settings,
)

from ext_graph_stores import CustomNeo4jGraphStore
from typing import List, Tuple, Optional, Dict, Any
from pcst import retrieval_via_pcst

from pprint import pprint

class GRetriever(BaseRetriever):
    """Custom retriever that performs both semantic search and hybrid search."""

    def __init__(
        self,
        storage_context: StorageContext,
        **kwargs,
    ) -> None:
        """Init params."""
        super().__init__(storage_context, **kwargs)
        self._custom_graph_store = storage_context.graph_store
        self._embed_model = embed_model_from_settings_or_context(Settings, None)
        self._similarity_top_k = 2
        
    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """Retrieve nodes given query."""

        query_bundle.embedding = self._embed_model.get_agg_embedding_from_queries(
                        query_bundle.embedding_strs
                    )
        
        query = VectorStoreQuery(
            query_embedding=query_bundle.embedding,
            similarity_top_k=self._similarity_top_k,
            query_str=query_bundle.query_str,
        )

        top_entities = self._get_similar_nodes(query)
        neighbours = self._get_all_neighbours(top_entities)
        sub_graph = self._pcst(top_entities, neighbours)
        nodes = self._build_nodes_from_sub_graph(sub_graph)

        return nodes
    
    def _get_similar_nodes(self, query: VectorStoreQuery) -> VectorStoreQueryResult:
        """Retrieve similar nodes given query."""
        ''' example:
        CALL db.index.vector.queryNodes('vector', 2, $embedding) YIELD node, score
        RETURN node.`Entity` AS text, score,
        '''

        retrieval_query = '''
            CALL db.index.vector.queryNodes($index, $k, $embedding) YIELD node, score
            RETURN score, node.id AS id
        '''

        parameters = {
            "index": self._custom_graph_store.index_name,
            "k": query.similarity_top_k,
            "embedding": query.query_embedding,
            "query": query.query_str,
        }

        results = self._custom_graph_store.query(retrieval_query, param_map=parameters)
        nodes = []
        similarities = []
        ids = []
        for record in results:
            node = NodeWithScore(
                node=TextNode(
                    text=str(record["id"]),
                ),
                score=float(record["score"])
            )
            nodes.append(node)
            similarities.append(record["score"])
            ids.append(record["id"])

        return VectorStoreQueryResult(nodes=nodes, similarities=similarities, ids=ids)
    
    def _get_all_neighbours(self, nodes: List[NodeWithScore], depth: int = 3) -> List[TextNode]:
        """Retrieve all neighbours of depth k given entities."""
        query = f'''
            MATCH p=(seedNodes:Entity)-[*0..{depth}]-(:Entity)
            WHERE (seedNodes.id) IN ['cannabinoids']
            UNWIND p as path
            UNWIND relationships(path) as r
            WITH DISTINCT r
            RETURN startNode(r).id AS subj, r.type AS pred, endNode(r).id AS obj
        '''
        params = {
            "seedNodes": [node.text for node in nodes],
        }
        
        neighbours = self._custom_graph_store.query(query, param_map=params)
        return neighbours
    
    def _pcst(self, nodes: List[NodeWithScore]) -> str:
        """Retrieve subgraph given nodes."""
        query = ""
        subgraph = self._custom_graph_store.query(query, param_map={})
        return subgraph
    
    def _build_nodes_from_sub_graph(self, knowledge_sequence, rel_map):
        pass