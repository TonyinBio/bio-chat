import logging, sys

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

from llama_index.core.schema import BaseNode, TextNode
from extensions.transformations import TripletExtractor, GraphEmbedding, JsonlToTriplets, SaveToNeo4j
from extensions.graph_stores import CustomNeo4jGraphStore

from llama_index.graph_stores.neo4j import Neo4jPGStore

from google.genai import types
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.core.embeddings import resolve_embed_model
from llama_index.core.schema import (
    TransformComponent, BaseNode, MetadataMode, TextNode
)
from dotenv import load_dotenv
load_dotenv()
import os
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple


from tqdm import tqdm
from bs4 import BeautifulSoup

USERNAME = os.environ.get("NEO4J_USERNAME")
PASSWORD = os.environ.get("NEO4J_PASSWORD")
URL = os.environ.get("NEO4J_URI")
DATABASE = os.environ.get("NEO4J_DATABASE")

# USERNAME = "neo4j"
# PASSWORD = os.environ.get("NEO4J_PASSWORD")
# URL = "neo4j+s://be4c0c46.databases.neo4j.io"
OLLAMA_URL = os.environ.get("OLLAMA_URL")

# print(USERNAME)
# print(PASSWORD)
print("Connecting to:", URL)
# print(DATABASE)

# init llm and embedder
# print("Connecting to Ollama at:", OLLAMA_URL)
# Settings.llm = Ollama(model="gemma", base_url=OLLAMA_URL, request_timeout=60.0, temperature=0)
# Settings.embed_model = resolve_embed_model("local:BAAI/bge-small-en-v1.5")
# Settings.chunk_size = 512
temp = 0.0
generate_content_config = types.GenerateContentConfig(
    temperature=temp,
    thinking_config=types.ThinkingConfig(
        thinking_level="MINIMAL",
    ),
    audio_transcription_config=types.AudioTranscriptionConfig(
    ),
    tools=[],
)
Settings.llm = GoogleGenAI(
    model="gemma-4-26b-a4b-it",
    temperature=temp,
    generation_config=generate_content_config
    # api_key="some key",  # uses GOOGLE_API_KEY env var by default
)
Settings.embed_model = resolve_embed_model("local:BAAI/bge-small-en-v1.5")

# neo4j_graph = CustomNeo4jGraphStore(
#     username=USERNAME,
#     password=PASSWORD,
#     url=URL,
#     embedding_dimension=384,
#     database=DATABASE
# )
neo4j_graph = Neo4jPGStore(
    username=USERNAME,
    password=PASSWORD,
    url=URL,
    database=DATABASE,
)


class ArticleSectionSplitter(TransformComponent):
    def __call__(
            self, 
            nodes: Sequence[BaseNode],
            show_progress = False,
            **kwargs
    ) -> Sequence[BaseNode]:
        new_nodes = []
        for node in nodes:
            soup = BeautifulSoup(node.text, "xml")

            title = soup.find("article-title")
            abstract = soup.find("abstract")

            for sec in soup.find_all("sec"):
                title_tag = sec.find("title", recursive=False)

                section_title = (
                    title_tag.get_text(" ", strip=True)
                    if title_tag
                    else ""
                )

                # Get text without the section title
                text = sec.get_text(" ", strip=True)
                
                new_nodes.append(TextNode(
                    text=text,
                    embedding=node.embedding,
                    excluded_embed_metadata_keys=node.excluded_embed_metadata_keys,
                    excluded_llm_metadata_keys=node.excluded_llm_metadata_keys,
                    metadata_seperator=node.metadata_separator,
                    metadata_template=node.metadata_template,
                    text_template=node.text_template,
                    relationships=node.relationships,
                ))

            new_nodes.append(TextNode(
                text=abstract.get_text(" ", strip=True),
                embedding=node.embedding,
                excluded_embed_metadata_keys=node.excluded_embed_metadata_keys,
                excluded_llm_metadata_keys=node.excluded_llm_metadata_keys,
                metadata_seperator=node.metadata_separator,
                metadata_template=node.metadata_template,
                text_template=node.text_template,
                relationships=node.relationships,
            ))
            new_nodes.append(TextNode(
                text=title.get_text(" ", strip=True),
                embedding=node.embedding,
                excluded_embed_metadata_keys=node.excluded_embed_metadata_keys,
                excluded_llm_metadata_keys=node.excluded_llm_metadata_keys,
                metadata_seperator=node.metadata_separator,
                metadata_template=node.metadata_template,
                text_template=node.text_template,
                relationships=node.relationships,
            ))

        return new_nodes

def create_pipeline():
    # input: Sequence[Document], Document inherits BaseNode
    pipeline = IngestionPipeline(
        transformations=[
            ArticleSectionSplitter(),
            SentenceSplitter(),  # Document -> BaseNode (more)
            # TitleExtractor(),  # BaseNode -> BaseNode with metadata
            # TripletExtractor(),  # BaseNode -> TripletNode
            # JsonlToTriplets(),
            # GraphEmbedding(),  # TripletNode -> TripletNode
            # SaveToNeo4j(neo4j_graph_store=neo4j_graph),  # TripletNode -> TripletNode
        ],
        vector_store=None,  # save the subjects and objects
    )

    return pipeline

def partition_data(fname, outfname, lines_per_file=50, ):
    # read jsonl into triplets

    with open(fname, 'r') as infile:
        file_number = 0
        outfile = None

        for i, line in enumerate(infile):
            if i % lines_per_file == 0:  # Time to start a new file
                if outfile:
                    outfile.close()
                out_filename = outfname + "_split_" + f"{file_number:03d}" + ".jsonl"
                outfile = open(out_filename, 'w')
                file_number += 1

            outfile.write(line)

        if outfile:  # Close the last file
            outfile.close()

    return file_number - 1

def main():
    try:
        documents = SimpleDirectoryReader(input_dir="data/pmc").load_data()
        # documents = documents[:1]
        # print(documents[0])
        # print(documents[0].metadata["file_name"])
        print("STARTING INGESTION PIPELINE")

        index = PropertyGraphIndex.from_existing(
            property_graph_store=neo4j_graph,
            # kg_extractors=[
                
            # ]
        )
        pipeline = create_pipeline()
        nodes = pipeline.run(
            show_progress=True, 
            # nodes=nodes
            documents=documents
        )
        # print(len(nodes))
        # for i, node in enumerate(nodes):
        #     with open(f"data/test/{i}.txt", "w") as f:
        #         f.write(node.get_content())
        # print(nodes)
        for node in tqdm(nodes):
            index.insert(node)
    finally:
        neo4j_graph.close()

if __name__ == "__main__":
    main()
