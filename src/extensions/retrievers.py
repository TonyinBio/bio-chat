# logging
import logging
logger = logging.getLogger(__name__)

# import QueryBundle
from llama_index.core import StorageContext

# import types
from llama_index.core.schema import NodeWithScore, TextNode, QueryBundle, QueryType
from llama_index.core.callbacks.schema import CBEventType, EventPayload

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
    callback_manager_from_settings_or_context,
    Settings,
)

from .graph_stores import CustomNeo4jGraphStore
from typing import List, Tuple, Optional, Dict, Any

# pcst stuff
from .pcst import retrieval_via_pcst
import pandas as pd
from torch_geometric.data.data import Data
import torch

# formatting
from llama_index.core.utils import print_text, truncate_text

# idk
from llama_index.core.callbacks.base import CallbackManager
from pprint import pprint

class GRetriever(BaseRetriever):
    """Custom retriever that performs both semantic search and hybrid search."""

    def __init__(
        self,
        storage_context: StorageContext,
        chainlit: bool = True,
        callback_manager: Optional[CallbackManager] = None,
        **kwargs,
    ) -> None:
        """Init params."""
        super().__init__(
            callback_manager=callback_manager
            or callback_manager_from_settings_or_context(Settings, None),
            **kwargs
        )
        self._chainlit = chainlit
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
        sgraph, desc = self._pcst(top_entities, neighbours, query_bundle)
        nodes = self._build_nodes(sgraph, desc)

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
    
    def _get_all_neighbours(self, top_entities: VectorStoreQueryResult, depth: int = 3) -> List[TextNode]:
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
            "seedNodes": [node.text for node in top_entities.nodes],
        }
        
        neighbours = self._custom_graph_store.query(query, param_map=params)
        return neighbours
    
    def _pcst(self, top_entities: List[NodeWithScore], neighbours: str, query_bundle: QueryBundle) -> str:
        """Retrieve subgraph given nodes."""

        # get distinct nodes
        edges: pd.DataFrame = pd.DataFrame(columns=["src", "edge_attr", "dst"])
        distinct_entities = {}
        avail_id = 0
        for entity in neighbours:
            # check if entity has id
            if entity["subj"] not in distinct_entities:
                distinct_entities[entity["subj"]] = avail_id
                avail_id += 1
            if entity["obj"] not in distinct_entities:
                distinct_entities[entity["obj"]] = avail_id
                avail_id += 1
            
            # get id of distinct_entities
            src_id = distinct_entities[entity["subj"]]
            dst_id = distinct_entities[entity["obj"]]
            # add to edges
            l = len(edges)
            edges.loc[l] = [src_id, entity["pred"], dst_id]
        # add to nodes
        nodes: pd.DataFrame = pd.DataFrame(
            [[node_id, node_attr] for node_attr, node_id in distinct_entities.items()], 
            columns=["node_id", "node_attr"]
        )

        # get graph
        # NOTE: inefficient, but will re-generate embeddings for now.
        # nodes
        # nodes.fillna({"node_attr": ""}, inplace=True)
        # x.shape = [num_nodes, embed_dim]
        x = self._embed_model.get_text_embedding_batch(texts=nodes.node_attr.tolist())
        x = torch.tensor(x, dtype=torch.float32)

        # edges
        # edges_attr.shape = [num_edges, embed_dim]
        edge_attr = self._embed_model.get_text_embedding_batch(texts=edges.edge_attr.tolist())
        edge_attr = torch.tensor(edge_attr, dtype=torch.float32)
        
        edge_index = torch.LongTensor([edges.src.tolist(), edges.dst.tolist()])

        graph = Data(
            x=x, 
            edge_index=edge_index,
            edge_attr=edge_attr, 
            num_nodes=len(nodes)
        )
        sgraph, desc = retrieval_via_pcst(
            graph,
            torch.tensor(query_bundle.embedding),
            nodes,
            edges,
        )

        return sgraph, desc
    
    def _build_nodes(self, sgraph: Data, desc: str) -> List[NodeWithScore]:
        """Build nodes from pcst output"""
        # if len(knowledge_sequence) == 0:
        #     logger.info("> No knowledge sequence extracted from entities.")
        #     return []
        context_string = (
f'''The following is a knowledge in the form of directed graph like:
Nodes:
node_id, node_attr

Edges:
src, edge_attr, dst

Knowledge:
{desc}''')
        if self._verbose:
            print_text(f"Graph RAG context:\n{context_string}\n", color="blue")

        node = NodeWithScore(
            node=TextNode(
                text=context_string,
                score=1.0,
            )
        )

        return [node]