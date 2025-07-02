import os
from typing import List

from ml_training_base import configure_multi_level_logger, write_strings_to_file
from biochemical_data_connectors import ChemblBioactivesConnector, PubChemBioactivesConnector


def main():
    os.makedirs('../var/log', exist_ok=True)
    logger = configure_multi_level_logger('../var/log/data_ingestion.log')

    chembl_extractor = ChemblBioactivesConnector(bioactivity_threshold=1000, logger=logger)
    pubchem_extractor = PubChemBioactivesConnector(bioactivity_threshold=1000, logger=logger)

    chembl_bioactives: List[str] = chembl_extractor.get_bioactive_compounds(target_uniprot_id='P00533')
    pubchem_bioactives: List[str] = pubchem_extractor.get_bioactive_compounds(target_uniprot_id='P00533')

    write_strings_to_file(
        file_path='../data/preprocessed/uniprot_id/P00533/chembl_bioactives',
        str_list=chembl_bioactives,
        logger=logger,
        content_name='SMILES'
    )

    write_strings_to_file(
        file_path='../data/preprocessed/uniprot_id/P00533/pubchem_bioactives',
        str_list=pubchem_bioactives,
        logger=logger,
        content_name='SMILES'
    )

    print('Test run finished')

if __name__ == "__main__":
    main()