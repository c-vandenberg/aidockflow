import logging
from typing import List, Dict, Optional

from ml_training_base import BaseDataPreprocessor
from biochemical_data_connectors import CompoundStandardizer
from biochemical_data_connectors.models import BioactiveCompound
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed, TimeoutError
from tqdm import tqdm

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

    def standardize_centroid_compounds(
        self,
        raw_centroid_smiles: List[str],
        task_timeout=60,
        batch_size=1_000_000
    ) -> List[Dict]:
        final_centroid_records = []
        with ThreadPoolExecutor(max_workers=16) as executor:
            for i in tqdm(range(0, len(raw_centroid_smiles), batch_size), desc="Processing SMILES batches"):
                smiles_batch = raw_centroid_smiles[i:i + batch_size]
                # 1. Submit all tasks to the executor.
                #    `future_to_smiles` is a dictionary that maps each running task (future)
                #    back to the original SMILES string it was given. This is crucial for logging.
                future_to_smiles = {
                    executor.submit(self._standardizer.standardize_smiles, smiles): smiles for smiles in smiles_batch
                }

                # 2. Process results as they are completed.
                self._logger.info(
                    f"Standardizing {len(smiles_batch)} compounds in batch {i//batch_size + 1} with a "
                    f"{task_timeout}s timeout per compound..."
                )

                for future in tqdm(as_completed(future_to_smiles), total=len(smiles_batch), leave=False,
                                   desc=f"Batch {i//batch_size + 1}"):
                    original_smiles = future_to_smiles[future]
                    try:
                        # 3. Try to get the result of the task with a timeout.
                        result = future.result(timeout=task_timeout)
                        if result:
                            final_centroid_records.append(result)
                    except TimeoutError:
                        # 4. If a task times out, log the problematic SMILES string.
                        self._logger.error(
                            f"TIMEOUT ERROR: Standardization of SMILES '{original_smiles}' "
                            f"took longer than {task_timeout} seconds and was skipped."
                        )
                    except Exception as e:
                        # 5. Catch any other exceptions that might occur in the worker process.
                        self._logger.error(f"ERROR processing SMILES '{original_smiles}': {e}")

        return final_centroid_records
