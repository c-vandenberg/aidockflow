from typing import List

from data.ingestion.bioactive_compounds_extraction import ChEMBLExtractor, PubChemExtractor

def main():
    chembl_extractor = ChEMBLExtractor()
    pubchem_extractor = PubChemExtractor()


    pubchem_bioacties: List[str] = pubchem_extractor.get_bioactive_compounds(target_uniprot_id='P00533')

    print('Test run finished')

if __name__ == "__main__":
    main()