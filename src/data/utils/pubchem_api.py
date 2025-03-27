import requests
import pubchempy as pcp
from typing import List, Optional

def get_active_aids(target_gene_id: str) -> List[str]:
    """Query PubChem to get assay IDs for a given target GeneID."""
    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/assay/target/geneid/{target_gene_id}/aids/JSON"
    response = requests.get(url, timeout=10)
    response.raise_for_status()
    data = response.json()
    return data.get("InformationList", {}).get("Information", [])[0].get("AID", [])

def get_active_cids(aid: str) -> List[int]:
    """Query PubChem assay details to get active compound IDs for a given assay ID."""
    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/assay/aid/{aid}/cids/JSON"
    response = requests.get(url, timeout=10)
    response.raise_for_status()
    data = response.json()
    return data.get("InformationList", {}).get("Information", [])[0].get("CID", [])
