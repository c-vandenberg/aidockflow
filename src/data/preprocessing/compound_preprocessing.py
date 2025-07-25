import logging
from typing import List, Dict

from ml_training_base import BaseDataPreprocessor
from biochemical_data_connectors import CompoundStandardizer
from biochemical_data_connectors.models import BioactiveCompound


class CompoundDataPreprocessor(BaseDataPreprocessor):
    def __init__(self, logger: logging.Logger):
        super().__init__(logger=logger)
        self._standardizer = CompoundStandardizer(logger=logger)

    def standardize_bioactive_compounds(self, raw_bioactives: List[BioactiveCompound]) -> List[BioactiveCompound]:
        standardized_actives: List[BioactiveCompound] = []
        for compound in raw_bioactives:
            standardized_data = self._standardizer.standardize_smiles(compound.smiles)
            if not standardized_data:
                self._logger.error(f'Error standardizing SMILES for {compound.smiles}')

            compound.smiles = standardized_data.get('smiles')
            compound.standardized_inchikey = standardized_data.get('inchi_key')
            standardized_actives.append(compound)

        return standardized_actives

    def standardize_centroid_compounds(self, raw_centroid_smiles: List[str]) -> List[Dict]:
        final_centroid_records = []
        for centroid_smiles in raw_centroid_smiles:
            standardized_data = self._standardizer.standardize_smiles(centroid_smiles)
            if not standardized_data:
                self._logger.error(f'Error standardizing SMILES for {centroid_smiles}')

            final_centroid_records.append(standardized_data)

        return final_centroid_records