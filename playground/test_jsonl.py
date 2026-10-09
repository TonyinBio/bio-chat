import logging, sys
import os
logging.basicConfig(stream=sys.stdout, level=logging.INFO)

from src.extensions.transformations import JsonlToTriplets, GraphEmbedding

from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.schema import TextNode

from pprint import pprint
pipeline = IngestionPipeline(
    transformations=[
        JsonlToTriplets(),
    ],
    vector_store=None,  # save the subjects and objects
)
# read jsonl into triplets
fname = "data/triplets/pubmed_triplet_data_part_2.jsonl"
with open(fname, "r") as f:
    jsonl = f.read()

nodes = [TextNode(text=jsonl)]
# Ingest directly into a vector db
triplets = pipeline.run(
    show_progress=True, 
    nodes=nodes
)
pprint(triplets[2])