import os
import logging
from typing import List, Optional

def write_smiles_to_file(
    file_path: str,
    smiles: List[str],
    logger: logging.Logger,
    log_interval: Optional[int] = 1000
):
    logger.info('Starting writing SMILES to file.')
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    with open(file_path, 'w') as file:
        total: int = len(smiles)
        for idx, smile in enumerate(smiles):
            file.write(smile + '\n')

        if idx % log_interval == 0:
            logger.info(f'Written {idx} / {total} SMILES to file.')

    logger.info('Writing SMILES to file completed successfully.')