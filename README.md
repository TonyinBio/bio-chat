# Bio Chat
An LLM using retrieval augmented generation over a biomedical knowledge graph to answer questions. Served as a website.

Set Neo4j password:
```console
export NEO4J_PASSWORD=...
```

Run the app:
```console
docker compose up -d
```

Enter docker container:
```
docker exec -it bio-chat-devcontainer-1 bash
```

Setup env without docker:
```console
conda create -n bio_chat python=3.11 -y
```
```console
conda activate bio_chat
```
```console
pip install -r requirements.txt
```
Also download [Ollama](https://ollama.com/download)

Process text and upload to neo4j:
```console
python ingest.py
```
Run a prompt:
```console
python starter.py
```

TODO: Citations
- ingest should add a new edge property
- retrieve should return edge property

TODO: Improve prompt?

TODO: Create NER evaluation dataset

TODO: Mess with CLIP/soft prompting?

TODO: Decide on predicates
- chemical
- source
- disposition 
- exposure_route 
- food
- health_effect
- organoleptic_effect 
- process
- role

Note to self:
plan:
- VectorIndex (VectorIndexRetriever) -> interfaces with embeddings in neo4j
- GraphIndex (KGRAGRetriever) -> interfaces with triplets and subgraph in neo4j 
- GRetriever -> combines the two
- will use both neo4jstores but have them point to the same db
