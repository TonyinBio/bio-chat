import requests
import boto3
from botocore import UNSIGNED
from botocore.config import Config
from pathlib import Path
from datasets import load_dataset
from tqdm import tqdm
from bs4 import BeautifulSoup
import json

BUCKET = "pmc-oa-opendata"
OUTPUT_DIR = Path("data/pmc")

s3 = boto3.client(
    "s3",
    config=Config(signature_version=UNSIGNED)
)

def search_pmc(disease, limit=100):
    query = f"""
    "{disease}"
    AND open access[filter]
    """

    response = requests.get(
        "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi",
        params={
            "db": "pmc",
            "term": query,
            "retmode": "json",
            "retmax": limit,
            "sort": "relevance"
        },
    )

    response.raise_for_status()

    return response.json()["esearchresult"]["idlist"]

def download_articles(pmc_ids):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for pmc_id in pmc_ids:
        pmc_id = str(pmc_id)

        response = s3.list_objects_v2(
            Bucket=BUCKET,
            Prefix=f"{pmc_id}."
        )

        objects = response.get("Contents", [])

        txt_files = [
            obj["Key"]
            for obj in objects
            if obj["Key"].endswith(".xml")
        ]

        if not txt_files:
            print(f"No xml file found for {pmc_id}")
            continue

        key = txt_files[0]
        output_path = OUTPUT_DIR / f"{pmc_id}.xml"

        print(f"Downloading {key}")

        s3.download_file(
            BUCKET,
            key,
            str(output_path)
        )

file_to_disease = {}
dataset = load_dataset("vladimire/geneturing", "gene_disease_association")
for i in tqdm(range(10)):
    disease = dataset["test"][i]["question"].lstrip("What are genes related to ").rstrip("?")
    ids = search_pmc(disease, limit=1)
    pmcids = ["PMC" + pmcid for pmcid in ids]

    print(disease)
    if pmcids:
        file_to_disease[pmcids[0]] = disease

    download_articles(pmcids)

with open(OUTPUT_DIR/"pmcid_to_disease.json", "w") as f:
    json.dump(file_to_disease, f, indent=4)