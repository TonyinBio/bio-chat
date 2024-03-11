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
import sys

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))


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
    url=url,
    embedding_dimension=384
)

# init llm and embedder
Settings.llm = Ollama(model="gemma", request_timeout=60.0, temperature=0)
Settings.embed_model = resolve_embed_model("local:BAAI/bge-small-en-v1.5")
Settings.chunk_size = 512

# input: Sequence[Document], Document inherits BaseNode
pipeline = IngestionPipeline(
    transformations=[
        SentenceSplitter(),  # Document -> BaseNode (more)
        TitleExtractor(),  # BaseNode -> BaseNode with metadata
        TripletExtractor(),  # BaseNode -> TripletNode
        GraphEmbedding(),  # TripletNode -> TripletNode
    ],
    vector_store=None,  # save the subjects and objects
)

reader = SimpleDirectoryReader("tests/data")
documents = reader.load_data(show_progress=True)
# Ingest directly into a vector db
triplets = pipeline.run(
    show_progress=True, 
    documents=documents
)

# TODO: Write cypher query to upload both triplet and embedding at same time?
# TODO: Write cypher query to upload embedding. (so we can have the full graph and then create embeddings after)
# TODO: Write application for ChainLit
# TODO: Create NER evaluation dataset
# TODO: Decide on predicates
# chemical
# source
# disposition 
# exposure_route 
# food
# health_effect
# organoleptic_effect 
# process
# role

### THE plan:
# VectorIndex (VectorIndexRetriever) -> interfaces with embeddings in neo4j
# GraphIndex (KGRAGRetriever) -> interfaces with triplets and subgraph in neo4j 
# GRetriever -> combines the two
# will use both neo4jstores but have them point to the same db

# save nodes to vector & graph store at end
for triplet in triplets:
    neo4j_graph.upsert_triplet(triplet.subject.text, triplet.predicate.text, triplet.object.text)
subjects = [triplet.subject for triplet in triplets]
objects = [triplet.object for triplet in triplets]
neo4j_vector.add(subjects)
neo4j_vector.add(objects)


# vector_store.add([n for n in nodes if n.embedding is not None])


# # NOTE: can take a while!
# index = KnowledgeGraphIndex.from_documents(
#     documents,
#     max_triplets_per_chunk=2,
#     storage_context=storage_context,
#     include_embeddings=True
# )

# query_engine = index.as_query_engine(
#     include_text=True, 
#     response_mode="compact",
#     retriever_mode="embedding"
# )
# response = query_engine.query(
#     "Tell me more about Interleaf",
# )