from llama_index.core.schema import MetadataMode
from llama_index.core import SimpleDirectoryReader

reader = SimpleDirectoryReader("data")
documents = reader.load_data(show_progress=True)
document = documents[0]
# Get content for LLM
print(document.get_content(metadata_mode=MetadataMode.LLM))

# # Get content for Embedding model
# print(document.get_content(metadata_mode=MetadataMode.EMBED))