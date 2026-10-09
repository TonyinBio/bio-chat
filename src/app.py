import os
import asyncio
from pathlib import Path

import logging, sys
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

# fix llama index and chainlit bug
import llama_index
import llama_index.core
llama_index.__version__ = llama_index.core.__version__
import chainlit as cl

# llama-index core
from llama_index.core.query_engine.retriever_query_engine import RetrieverQueryEngine
from llama_index.core.callbacks.base import CallbackManager
from neo4j.exceptions import ServiceUnavailable, SessionExpired, TransientError

# Chainlit loads this file with a path-based module name, so relative imports
# are unreliable. Resolve the package from this file, independent of cwd.
project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)
from src.database import ConnectionManager

# llama-index extensions
# from extensions.graph_stores import CustomNeo4jGraphStore
from llama_index.graph_stores.neo4j import Neo4jPGStore
# from extensions.retrievers import GRetriever
from llama_index.core.indices.property_graph import VectorContextRetriever, PGRetriever
# from extensions.callbacks import CustomCallbackHandler

# from llama_index.graph_stores.neo4j import Neo4jGraphStore
from llama_index.core import Settings
# from llama_index.llms.ollama import Ollama
from google.genai import types
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.core.embeddings import resolve_embed_model

username = os.environ.get("NEO4J_USERNAME")
password = os.environ.get("NEO4J_PASSWORD")
url = os.environ.get("NEO4J_URI")
database = os.environ.get("NEO4J_DATABASE")

# rebuild storage context
# neo4j_graph = CustomNeo4jGraphStore(
#     username=username,
#     password=password,
#     url=url,
#     embedding_dimension=384,
# )
def connect_graph():
    if not all((username, password, url)):
        raise ValueError("NEO4J_USERNAME, NEO4J_PASSWORD and NEO4J_URI are required")
    return Neo4jPGStore(
        username=username,
        password=password,
        url=url,
        database=database or "neo4j",
        connection_timeout=5,
        connection_acquisition_timeout=10,
        max_transaction_retry_time=5,
    )


connections = ConnectionManager(connect_graph)
logger = logging.getLogger(__name__)
UNAVAILABLE_MESSAGE = "The database is unavailable. Please try again shortly."

# global settings
# llm_model_name = "gemma:2b"
# llm_model_name = "qwen2.5:0.5b"
# requests.post("http://ollama:11434/api/pull", json={"name": llm_model_name}, timeout=600.0)  # credit: Steven Tang :)
# Settings.llm = Ollama(
#     base_url="http://ollama:11434",
#     model=llm_model_name,
#     request_timeout=120.0, 
#     temperature=0
# )
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
# Settings.chunk_size = 512

@cl.on_chat_start
async def factory():
    await initialize_session()


async def initialize_session():
    # init llm and embedder
    Settings.callback_manager = CallbackManager([
        # CustomCallbackHandler()
        cl.LlamaIndexCallbackHandler()
    ])

    neo4j_graph = None
    try:
        neo4j_graph = await asyncio.to_thread(connections.get)
        query_engine = await asyncio.to_thread(build_query_engine, neo4j_graph)
    except Exception as exc:
        logger.exception("Unable to initialize the graph query engine")
        if neo4j_graph is not None and isinstance(
            exc, (ServiceUnavailable, SessionExpired, TransientError)
        ):
            await asyncio.to_thread(connections.invalidate, neo4j_graph)
        cl.user_session.set("query_engine", None)
        await cl.Message(content=UNAVAILABLE_MESSAGE).send()
        return None

    cl.user_session.set("neo4j_graph", neo4j_graph)
    cl.user_session.set("query_engine", query_engine)
    return query_engine


def build_query_engine(neo4j_graph):
    sub_retriever = VectorContextRetriever(
        graph_store=neo4j_graph,
    )
    retriever = PGRetriever(  # TODO: how does this use multiple retrievers?
        sub_retrievers=[sub_retriever],
    )
    query_engine = RetrieverQueryEngine.from_args(
        retriever, 
        response_mode="compact",
        include_text=True,
        streaming=True
    )

    return query_engine

@cl.on_message
async def main(message: cl.Message):
    query_engine = cl.user_session.get("query_engine")  # type: RetrieverQueryEngine
    if query_engine is None:
        query_engine = await initialize_session()
        if query_engine is None:
            return

    response_message = cl.Message(content="")
    try:
        aquery = cl.make_async(query_engine.query)
        response = await aquery(message.content)

        for token in response.response_gen:
            await response_message.stream_token(token=token)

        if response.response_txt:
            response_message.content = response.response_txt

        await response_message.send()
    except (ServiceUnavailable, SessionExpired, TransientError):
        logger.exception("Neo4j became unavailable while answering a message")
        graph = cl.user_session.get("neo4j_graph")
        await asyncio.to_thread(connections.invalidate, graph)
        cl.user_session.set("query_engine", None)
        await cl.Message(content=UNAVAILABLE_MESSAGE).send()
