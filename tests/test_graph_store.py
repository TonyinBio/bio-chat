import logging, sys
import os
logging.basicConfig(stream=sys.stdout, level=logging.INFO)

from src.extensions.graph_stores import CustomNeo4jGraphStore
from src.extensions.schema import TripletNode
from llama_index.core.schema import TextNode
# init neo4j
from dotenv import load_dotenv
load_dotenv("../.env")
import os
username = os.environ.get("username")
password = os.environ.get("password")
url = os.environ.get("url")

neo4j_graph = CustomNeo4jGraphStore(
    username=username,
    password=password,
    url=url,
    embedding_dimension=3
)
data = [
    TripletNode(
        subject=TextNode(text=str(i), embedding=[i] * 3),
        predicate=TextNode(text=str(i) + "rel"),
        object=TextNode(text=str(-i), embedding=[-i] * 3),
    ) for i in range(1, 4)
]
neo4j_graph.add(data)