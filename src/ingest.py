import logging, sys

# logging.basicConfig(stream=sys.stdout, level=logging.INFO)
# logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

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
from extensions.transformations import TripletExtractor, GraphEmbedding, JsonlToTriplets, SaveToNeo4j
from extensions.graph_stores import CustomNeo4jGraphStore

from dotenv import load_dotenv
load_dotenv()
import os

from tqdm import tqdm

USERNAME = "neo4j"
PASSWORD = os.environ.get("NEO4J_PASSWORD")
URL = "neo4j+s://be4c0c46.databases.neo4j.io"

# init llm and embedder
Settings.llm = Ollama(model="gemma", request_timeout=60.0, temperature=0)
Settings.embed_model = resolve_embed_model("local:BAAI/bge-small-en-v1.5")
Settings.chunk_size = 512

neo4j_graph = CustomNeo4jGraphStore(
    username=USERNAME,
    password=PASSWORD,
    url=URL,
    embedding_dimension=384,
)

def create_pipeline():
    # input: Sequence[Document], Document inherits BaseNode
    pipeline = IngestionPipeline(
        transformations=[
            # SentenceSplitter(),  # Document -> BaseNode (more)
            # TitleExtractor(),  # BaseNode -> BaseNode with metadata
            # TripletExtractor(),  # BaseNode -> TripletNode
            JsonlToTriplets(),
            GraphEmbedding(),  # TripletNode -> TripletNode
            SaveToNeo4j(neo4j_graph_store=neo4j_graph),  # TripletNode -> TripletNode
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
    large_fname = "../data/triplets/merged_April_11.jsonl"
    partition_fname = "../data/triplets/partition/merged_April_11"
    end_file_number = partition_data(large_fname, partition_fname)
    
    start_file_number = 0
    # end_file_number = 0

    pipeline = create_pipeline()
    
    for fnum in tqdm(range(start_file_number, end_file_number + 1)):
        fname = partition_fname + "_split_" + f"{fnum:03d}" + ".jsonl"
        with open(fname, "r") as infile:
            nodes = [TextNode(text=infile.read())]

        triplets = pipeline.run(
            show_progress=False, 
            nodes=nodes
            # documents=documents
        )

if __name__ == "__main__":
    main()
