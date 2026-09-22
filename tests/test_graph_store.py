from platform import node
import logging, sys
from typing import Literal
from pprint import pprint

# logging.basicConfig(stream=sys.stdout, level=logging.INFO)
# logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

from llama_index.core import Document, SimpleDirectoryReader, Settings, StorageContext, PropertyGraphIndex
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.extractors import TitleExtractor
from llama_index.core.ingestion import IngestionPipeline

from llama_index.vector_stores.neo4jvector import Neo4jVectorStore
from llama_index.graph_stores.neo4j import Neo4jGraphStore

from llama_index.llms.ollama import Ollama
from llama_index.core.embeddings import resolve_embed_model

from llama_index.core.indices import KnowledgeGraphIndex
from llama_index.core.indices.property_graph import SchemaLLMPathExtractor

from llama_index.core.schema import BaseNode, TextNode
from llama_index.core.schema import BaseNode, TransformComponent
from llama_index.core.graph_stores.types import (
    EntityNode,
    Relation,
    KG_NODES_KEY,
    KG_RELATIONS_KEY,
)

from llama_index.graph_stores.neo4j import Neo4jPGStore

from dotenv import load_dotenv
load_dotenv()
import os

from tqdm import tqdm
import json

class MyGraphExtractor(TransformComponent):
    # the init is optional
    # def __init__(self, ...):
    #     ...
    
    def __call__(
        self, llama_nodes: list[BaseNode], **kwargs
    ) -> list[BaseNode]:
        # https://developers.llamaindex.ai/python/framework/module_guides/indexing/lpg_index_guide/#sub-classing-extractors
        for llama_node in llama_nodes:
            # print(llama_node.get_content())
            # print(llama_node.metadata)
            # print(KG_NODES_KEY, KG_RELATIONS_KEY)
            # be sure to not overwrite existing entities/relations

            existing_nodes = llama_node.metadata.pop(KG_NODES_KEY, [])
            existing_relations = llama_node.metadata.pop(KG_RELATIONS_KEY, [])

            # ===== begin real code =====
            # extract triplets
            lines = llama_node.get_content().strip().split("\n")
            for line in lines:
                try:
                    datum = json.loads(line)
                    triplets: str = datum["output"]
                except:
                    continue

                # pmcid = int(datum["id"])  # make sure it is an int
                pmcid = datum["id"]
                for triplet in triplets:
                    if len(triplet) != 3:
                        continue
                    if any(len(s.encode("utf-8")) > 128 for s in triplet):
                        continue
                    if any(not s for s in triplet):
                        continue
                    if any("<" in s for s in triplet):
                        continue

                    # add nodes
                    existing_nodes.append(
                        EntityNode(
                            name=triplet[0], label="MYLABEL", properties={}
                        )
                    )
                    existing_nodes.append(
                        EntityNode(
                            name=triplet[2], label="MYLABEL", properties={}
                        )
                    )
                    existing_relations.append(
                        Relation(
                            label=triplet[1],
                            source_id=triplet[0],
                            target_id=triplet[2],
                            properties={"pmcid": pmcid},
                        )
                    )
            # ===== end real code =====

            # add back to the metadata

            llama_node.metadata[KG_NODES_KEY] = existing_nodes
            llama_node.metadata[KG_RELATIONS_KEY] = existing_relations

        return llama_nodes


USERNAME = os.environ.get("NEO4J_USERNAME")
PASSWORD = os.environ.get("NEO4J_PASSWORD")
URL = os.environ.get("NEO4J_URI")
DATABASE = os.environ.get("NEO4J_DATABASE")

# USERNAME = "neo4j"
# PASSWORD = os.environ.get("NEO4J_PASSWORD")
# URL = "neo4j+s://be4c0c46.databases.neo4j.io"
OLLAMA_URL = os.environ.get("OLLAMA_URL")

# init llm and embedder
print("Connecting to Ollama at:", OLLAMA_URL)
Settings.llm = Ollama(model="gemma:2b", base_url=OLLAMA_URL, request_timeout=120.0, temperature=0)
Settings.embed_model = resolve_embed_model("local:BAAI/bge-small-en-v1.5")
# Settings.chunk_size = 512

# print(USERNAME)
# print(PASSWORD)
print("Connecting to:", URL)
# print(DATABASE)
neo4j_graph = Neo4jPGStore(
    username=USERNAME,
    password=PASSWORD,
    url=URL,
    database=DATABASE,
)

kg_extractor = MyGraphExtractor()

index = PropertyGraphIndex.from_existing(
    property_graph_store=neo4j_graph,
    kg_extractors=[kg_extractor]
)

documents = SimpleDirectoryReader("data1").load_data()
# for doc in documents:
#     pprint(doc.metadata)

print(f"Indexing {documents[0].metadata.get('file_name', 'Unknown')}")
index.insert(documents[0])
