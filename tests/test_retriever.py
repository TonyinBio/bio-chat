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

from ext_graph_stores import CustomNeo4jGraphStore
import logging, sys
logging.basicConfig(stream=sys.stdout, level=logging.INFO)

import os
username = os.environ.get("username")
password = os.environ.get("password")
url = os.environ.get("url")

neo4j_graph = CustomNeo4jGraphStore(
    username=username,
    password=password,
    url=url 
)
# print(neo4j_graph.query("MATCH (n:Entity) RETURN n LIMIT 25;"))

# init llm and embedder
Settings.llm = Ollama(model="gemma:2b", request_timeout=60.0, temperature=0)
Settings.embed_model = resolve_embed_model("local:BAAI/bge-small-en-v1.5")
Settings.chunk_size = 512

PERSIST_DIR = None
# if not os.path.exists(PERSIST_DIR):
#     # load the documents and create the index
#     documents = SimpleDirectoryReader("data").load_data()
#     storage_context = StorageContext.from_defaults(graph_store=neo4j_graph)
#     index = KnowledgeGraphIndex.from_documents(
#         documents, 
#         storage_context=storage_context,
#         show_progress=True,
#     )
#     # store it for later
#     index.storage_context.persist(persist_dir=PERSIST_DIR)
# else:
#     # load the existing index
#     storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR, graph_store=neo4j_graph)
#     index = load_index_from_storage(storage_context)
storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR, graph_store=neo4j_graph)
# index = load_index_from_storage(storage_context)
# g = index.get_networkx_graph()
# from pyvis.network import Network
# net = Network(notebook=False, directed=True)
# net.from_nx(g)
# net.save_graph("non_filtered_graph.html")

retriever = KnowledgeGraphRAGRetriever(storage_context=storage_context, verbose=True)  # NOTE: had to change super to top in src code
query_engine = RetrieverQueryEngine.from_args(
    retriever
)
# query_engine = index.as_query_engine(verbose=True, include_text=True)

print(query_engine.query("What is Lisp?"))
