import os
import openai

from llama_index.core.query_engine.retriever_query_engine import RetrieverQueryEngine
from llama_index.core.callbacks.base import CallbackManager
from llama_index.core import (
    StorageContext,
    load_index_from_storage,
)

import chainlit as cl

# from llama_index.graph_stores.neo4j import Neo4jGraphStore
from llama_index.core import Settings
from llama_index.llms.ollama import Ollama
from llama_index.core.embeddings import resolve_embed_model
# username = os.environ.get("username")
# password = os.environ.get("password")
# url = os.environ.get("url")

# neo4j_graph = Neo4jGraphStore(
#     username=username,
#     password=password,
#     url=url 
# )

# # rebuild storage context
# storage_context = StorageContext.from_defaults(graph_store=neo4j_graph)

from llama_index.core import VectorStoreIndex, SimpleDirectoryReader

# fix llama index and chainlit bug
import llama_index
import llama_index.core
llama_index.__version__ = llama_index.core.__version__

PERSIST_DIR = "./storage"

@cl.on_chat_start
async def factory():
    # init llm and embedder
    Settings.llm = Ollama(model="gemma:2b", request_timeout=60.0, temperature=0)
    Settings.embed_model = resolve_embed_model("local:BAAI/bge-small-en-v1.5")
    Settings.chunk_size = 512
    Settings.callback_manager = CallbackManager([cl.LlamaIndexCallbackHandler()])
    
    storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
    index = load_index_from_storage(storage_context)
    
    query_engine = index.as_query_engine(
        streaming=True,
    )

    cl.user_session.set("query_engine", query_engine)


@cl.on_message
async def main(message: cl.Message):
    query_engine = cl.user_session.get("query_engine")  # type: RetrieverQueryEngine
    response = await cl.make_async(query_engine.query)(message.content)

    response_message = cl.Message(content="")

    for token in response.response_gen:
        await response_message.stream_token(token=token)

    if response.response_txt:
        response_message.content = response.response_txt

    await response_message.send()
