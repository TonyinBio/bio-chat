import logging
import sys

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

from llama_index.core import KnowledgeGraphIndex, SimpleDirectoryReader, load_index_from_storage
from llama_index.core import Settings
from llama_index.llms.ollama import Ollama
from llama_index.core.embeddings import resolve_embed_model
from llama_index.core import StorageContext

from llama_index.graph_stores.neo4j import Neo4jGraphStore
from llama_index.vector_stores.neo4jvector import Neo4jVectorStore

from llama_index.core.query_engine import RetrieverQueryEngine

from ext_retrievers import GRetriever

from llama_index.core.retrievers import VectorIndexRetriever, KnowledgeGraphRAGRetriever
from llama_index.core import VectorStoreIndex, KnowledgeGraphIndex


# init neo4j
from dotenv import load_dotenv
load_dotenv()
import os
username = os.environ.get("username")
password = os.environ.get("password")
url = os.environ.get("url")

neo4j_graph = Neo4jGraphStore(
    username=username,
    password=password,
    url=url 
)
neo4j_vector = Neo4jVectorStore(
    username=username,
    password=password,
    url=url
)

storage_context = StorageContext.from_defaults(graph_store=neo4j_graph, vector_store=neo4j_vector)

### THE plan:
# VectorIndex (VectorIndexRetriever) -> interfaces with embeddings in neo4j
# GraphIndex (KGRAGRetriever) -> interfaces with triplets and subgraph in neo4j 
# GRetriever -> combines the two
# will use both neo4jstores but have them point to the same db

# init llm and embedder
Settings.llm = Ollama(model="gemma:2b", request_timeout=60.0, temperature=0)
Settings.embed_model = resolve_embed_model("local:BAAI/bge-small-en-v1.5")
Settings.chunk_size = 512

# init retrievers
vector_index = VectorStoreIndex(storage_context=neo4j_vector)
vector_retriever = VectorIndexRetriever(
    index=vector_index,
    verbose=False,
)
kg_index = KnowledgeGraphIndex(
    storage_context=storage_context
)
graph_retriever = KnowledgeGraphRAGRetriever(
    storage_context=storage_context,
    verbose=False,
)

# the g-retriever, baby
g_retriever = GRetriever(kg_index, vector_retriever, graph_retriever)
query_engine = RetrieverQueryEngine.from_args(
    g_retriever,
    response_mode="compact"
)

# please god work
response = query_engine.query("Tell me about Interleaf.")
print(response)

