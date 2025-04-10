import os
from typing import List

from data.utils.logging_utils import configure_logger
from data.ingestion.bioactive_compounds_extraction import ChEMBLExtractor, PubChemExtractor

def main():
    os.makedirs('../var/log', exist_ok=True)
    logger = configure_logger('../var/log/data_ingestion.log')
    chembl_extractor = ChEMBLExtractor(bioactivity_threshold=1000, logger=logger)
    pubchem_extractor = PubChemExtractor(bioactivity_threshold=1000, logger=logger)


    pubchem_bioacties: List[str] = pubchem_extractor.get_bioactive_compounds(target_uniprot_id='P00533')

    print('Test run finished')

if __name__ == "__main__":
    main()