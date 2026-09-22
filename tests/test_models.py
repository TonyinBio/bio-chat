from llama_index.llms.google_genai import GoogleGenAI
from google.genai import types

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

llm = GoogleGenAI(
    model="gemma-4-26b-a4b-it",
    temperature=temp,
    generation_config=generate_content_config
    # api_key="some key",  # uses GOOGLE_API_KEY env var by default
)

resp = llm.complete("Who is Paul Graham?")
print(resp)