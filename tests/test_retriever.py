from llama_index.core.query_engine.retriever_query_engine import RetrieverQueryEngine
from llama_index.core.callbacks.base import CallbackManager
from llama_index.core import (
    StorageContext,
    load_index_from_storage,
)

# from llama_index.graph_stores.neo4j import Neo4jGraphStore
from llama_index.core import Settings
from llama_index.llms.ollama import Ollama
from llama_index.core.embeddings import resolve_embed_model
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader

from llama_index.core.indices import KnowledgeGraphIndex
from llama_index.core.retrievers import KnowledgeGraphRAGRetriever

from llama_index.core.schema import QueryBundle
from ext_graph_stores import CustomNeo4jGraphStore
from ext_retrievers import GRetriever

from pprint import pprint

import logging, sys
logging.basicConfig(stream=sys.stdout, level=logging.INFO)

import os
username = os.environ.get("username")
password = os.environ.get("password")
url = os.environ.get("url")

neo4j_graph = CustomNeo4jGraphStore(
    username=username,
    password=password,
    url=url,
    embedding_dimension=384
)

# init llm and embedder
Settings.llm = Ollama(model="gemma:2b", request_timeout=60.0, temperature=0)
Settings.embed_model = resolve_embed_model("local:BAAI/bge-small-en-v1.5")
Settings.chunk_size = 512

PERSIST_DIR = None
storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR, graph_store=neo4j_graph)

retriever = GRetriever(storage_context=storage_context, verbose=True)  # NOTE: had to change super to top in src code

retriever._retrieve(query_bundle=QueryBundle(query_str="cannabinoids"))