Setup env:
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

TODO: Display subgraph as a figure in chainlit

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

THE plan:
- VectorIndex (VectorIndexRetriever) -> interfaces with embeddings in neo4j
- GraphIndex (KGRAGRetriever) -> interfaces with triplets and subgraph in neo4j 
- GRetriever -> combines the two
- will use both neo4jstores but have them point to the same db

<sub><sub><sub><sub><sup><sup><sup><sup>pain and suffering and pain and suffering and pain and suffering and pain and suffering and pain and suffering and pain and suffering and pain and suffering and pain and suffering and pain and suffering and pain and suffering and pain and suffering and pain and suffering and pain and suffering and pain and suffering and pain and suffering and pain and suffering and pain and suffering and pain and sufferingpain and suffering and pain and suffering and pain and suffering and pain and suffering and pain and suffering and pain and suffering and pain and suffering and pain and suffering and pain and suffering and pain and suffering and pain and suffering and pain and suffering and pain and suffering and pain and suffering and pain and suffering and pain and suffering and pain and suffering and pain and sufferingpain and suffering and pain and suffering and pain and suffering and pain and suffering and pain and suffering and pain and suffering and pain and suffering and pain and suffering and pain and suffering and pain and suffering and pain and suffering and pain and suffering and pain and suffering and pain and suffering and pain and suffering and pain and suffering and pain and suffering and pain and sufferingpain and suffering and pain and suffering and pain and suffering and pain and suffering and pain and suffering and pain and suffering and pain and suffering and pain and suffering and pain and suffering and pain and suffering and pain and suffering and pain and suffering and pain and suffering and pain and suffering and pain and suffering and pain and suffering and pain and suffering and pain and suffering</sup></sup></sup></sup>