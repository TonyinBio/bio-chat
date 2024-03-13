import os

import logging, sys
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

# llama-index core
from llama_index.core.query_engine.retriever_query_engine import RetrieverQueryEngine
from llama_index.core.callbacks.base import CallbackManager
from llama_index.core import (
    StorageContext,
    load_index_from_storage,
)
from llama_index.core.retrievers import KnowledgeGraphRAGRetriever

# llama-index extensions
from ext_graph_stores import CustomNeo4jGraphStore

# fix llama index and chainlit bug
import llama_index
import llama_index.core
llama_index.__version__ = llama_index.core.__version__
import chainlit as cl

# from llama_index.graph_stores.neo4j import Neo4jGraphStore
from llama_index.core import Settings
from llama_index.llms.ollama import Ollama
from llama_index.core.embeddings import resolve_embed_model
username = os.environ.get("username")
password = os.environ.get("password")
url = os.environ.get("url")

# rebuild storage context
neo4j_graph = CustomNeo4jGraphStore(
    username=username,
    password=password,
    url=url,
    embedding_dimension=3,
)
PERSIST_DIR = None
storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR, graph_store=neo4j_graph)

# global settings
Settings.llm = Ollama(model="gemma:2b", request_timeout=60.0, temperature=0)
Settings.embed_model = resolve_embed_model("local:BAAI/bge-small-en-v1.5")
Settings.chunk_size = 512

@cl.on_chat_start
async def factory():
    # init llm and embedder
    Settings.callback_manager = CallbackManager([cl.LlamaIndexCallbackHandler()])

    retriever = KnowledgeGraphRAGRetriever(storage_context=storage_context, verbose=True)  # NOTE: had to change super to top in src code
    query_engine = RetrieverQueryEngine.from_args(
        retriever, 
        response_mode="compact",
        include_text=True,
        streaming=True
    )

    cl.user_session.set("query_engine", query_engine)

@cl.on_message
async def main(message: cl.Message):
    query_engine = cl.user_session.get("query_engine")  # type: RetrieverQueryEngine
    aquery = cl.make_async(query_engine.query)
    response = await aquery(message.content)

    response_message = cl.Message(content="")

    for token in response.response_gen:
        await response_message.stream_token(token=token)

    if response.response_txt:
        response_message.content = response.response_txt

    await response_message.send()
