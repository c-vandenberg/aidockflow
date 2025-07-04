import logging
from typing import Dict, List

import pandas as pd
from biochemical_data_connectors import (
    BindingDbBioactivesConnector,
    ChemblBioactivesConnector,
    IupharBioactivesConnector,
    PubChemBioactivesConnector,
    CompoundStandardizer,
)
from biochemical_data_connectors.models import BioactiveCompound

from src.data.curation.base_curator import BaseCurator


class HighFidelityActivesCurator(BaseCurator):
    def __init__(self, config: Dict, logger: logging.Logger):
        super().__init__(config=config,logger=logger)
        self._standardizer = CompoundStandardizer(logger=logger)

    def run(self):
        # 1) Validate configuration
        uniprot_id: str = self._config.get('uniprot_id')
        bioactivity_measures: Dict[str] = self._config.get('bioactivity_measures')
        if not all ([uniprot_id, bioactivity_measures]):
            raise ValueError('You must provide a `uniprot_id` and `bioactivity_measures` in the configuration file.')

        self._logger.info("Starting high-fidelity actives curation...")

        # 2) Instantiate and run biochemical data connectors
        bindingbd_actives_connector = BindingDbBioactivesConnector(
            bioactivity_measures=self._config.get('bioactivity_measures'),
            bioactivity_threshold=1000,
            cache_dir=self._config.get('cache_dir', '../data/cache/'),
            logger=self._logger
        )
        chembl_connector = ChemblBioactivesConnector(
            bioactivity_measures=self._config.get('bioactivity_measures'),
            bioactivity_threshold=1000,
            cache_dir=self._config.get('cache_dir', '../data/cache/'),
            logger=self._logger
        )
        iuphar_connector = IupharBioactivesConnector(
            bioactivity_measures=self._config.get('bioactivity_measures'),
            bioactivity_threshold=1000,
            cache_dir=self._config.get('cache_dir', '../data/cache/'),
            logger=self._logger
        )
        pubchem_connector = PubChemBioactivesConnector(
            bioactivity_measures=self._config.get('bioactivity_measures'),
            bioactivity_threshold=1000,
            cache_dir=self._config.get('cache_dir', '../data/cache/'),
            logger=self._logger
        )
        raw_actives: List[BioactiveCompound] = bindingbd_actives_connector.get_bioactive_compounds(
            self._config.get('uniprot_id')
        ) + chembl_connector.get_bioactive_compounds(
            self._config.get('uniprot_id')
        ) + iuphar_connector.get_bioactive_compounds(
            self._config.get('uniprot_id')
        ) + pubchem_connector.get_bioactive_compounds(
            self._config.get('uniprot_id')
        )

        # 3) Standardize compounds
        standardized_actives: List[BioactiveCompound] = []
        for compound in raw_actives:
            standardized_data = self._standardizer.standardize_smiles(compound.smiles)
            if not standardized_data:
                self._logger.error(f'Error standardizing SMILES for {compound.smiles}')

            compound.smiles = standardized_data.get('smiles')
            compound.standardized_inchikey = standardized_data.get('inchi_key')
            standardized_actives.append(compound)

        # 3) Deduplicate and save the final parquet file
        df = pd.DataFrame(standardized_actives)
        df.drop_duplicates(subset='standardized_inchikey')
        df['source_id'] = df['source_id'].astype(str)

        self._logger.info(f"Preparing to save {len(df)} curated compounds to Parquet file...")
        if 'raw_data' in df.columns:
            df_to_save = df.drop(columns=['raw_data'])
        else:
            df_to_save = df

        df_to_save.to_parquet(self._config.get('standardized_actives_path'))
        self._logger.info("High-fidelity actives curation complete.")
