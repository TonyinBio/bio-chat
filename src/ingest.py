import logging, sys

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

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

from llama_index.core.schema import BaseNode, TextNode
from extensions.transformations import TripletExtractor, GraphEmbedding, JsonlToTriplets
from extensions.graph_stores import CustomNeo4jGraphStore


# init neo4j
from dotenv import load_dotenv
load_dotenv()
import os
username = "neo4j"
password = os.environ.get("NEO4J_PASSWORD")
url = "neo4j+s://be4c0c46.databases.neo4j.io"

neo4j_graph = CustomNeo4jGraphStore(
    username=username,
    password=password,
    url=url,
    embedding_dimension=384,
)

# init llm and embedder
Settings.llm = Ollama(model="gemma", request_timeout=60.0, temperature=0)
Settings.embed_model = resolve_embed_model("local:BAAI/bge-small-en-v1.5")
Settings.chunk_size = 512

# input: Sequence[Document], Document inherits BaseNode
pipeline = IngestionPipeline(
    transformations=[
        # SentenceSplitter(),  # Document -> BaseNode (more)
        # TitleExtractor(),  # BaseNode -> BaseNode with metadata
        # TripletExtractor(),  # BaseNode -> TripletNode
        JsonlToTriplets(),
        GraphEmbedding(),  # TripletNode -> TripletNode
    ],
    vector_store=None,  # save the subjects and objects
)
# read jsonl into triplets
fname = "../data/triplets/merged_April_11.jsonl"
with open(fname, "r") as f:
    jsonl = f.read()

nodes = [TextNode(text=jsonl)]
# Ingest directly into a vector db
print("Starting pipeline")
triplets = pipeline.run(
    show_progress=False, 
    nodes=nodes
    # documents=documents
)

print(f"Num of extracted triplets: {len(triplets)}")
# save nodes to graph store
neo4j_graph.add(triplets)
print("Done Upload")