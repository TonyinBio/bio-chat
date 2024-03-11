from llama_index.core import Document, SimpleDirectoryReader, Settings, StorageContext
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.extractors import TitleExtractor
from llama_index.core.ingestion import IngestionPipeline

from llama_index.vector_stores.neo4jvector import Neo4jVectorStore
from llama_index.graph_stores.neo4j import Neo4jGraphStore

from llama_index.llms.ollama import Ollama
from llama_index.core.embeddings import resolve_embed_model

from llama_index.core.indices import KnowledgeGraphIndex

from ext_transformations import TripletExtractor, GraphEmbedding

import logging

logger = logging.getLogger(__name__)

class CustomNeo4jGraphStore(Neo4jGraphStore):
    def __del__(self):
        print("here")
        # self._driver.close()
        # print("here2")