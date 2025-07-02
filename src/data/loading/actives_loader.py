import logging

import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from ml_training_base import BaseSupervisedDataLoader


class HighFidelityActivesDataLoader(BaseSupervisedDataLoader):
    def __init__(
        self,
        test_split: float,
        validation_split: float,
        logger: logging.Logger,
        actives_path: str,
        random_state: int
    ):
        super().__init__(
            test_split=test_split,
            validation_split=validation_split,
            logger=logger
        )
        self._actives_path = actives_path
        self._random_state = random_state

    @staticmethod
    def _generate_scaffold(smiles: str) -> str:
        """
        Generates a Bemis-Murcko scaffold SMILES from an input SMILES.

        Parameters
        ----------
        smiles : str
            The input SMILES string.

        Returns
        -------
        str
            The canonical SMILES string of the scaffold, or an empty string
            if the input SMILES is invalid.
        """
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return ''

        try:
            scaffold_mol = MurckoScaffold.GetScaffoldForMol(mol)
            return Chem.MolToSmiles(scaffold_mol, canonical=True)
        except (ValueError, RuntimeError):
            return ''

    def setup_datasets(self):
        """
        Loads the actives data and performs a scaffold-based split.

        This method orchestrates the entire process:
        1. Loads the actives from the specified Parquet file.
        2. Generates a Bemis-Murcko scaffold for each molecule.
        3. Groups the molecules by their scaffold.
        4. Shuffles the unique scaffolds and splits them into train, validation,
           and test sets based on the specified ratios.
        5. Assigns all molecules belonging to a scaffold to the corresponding set,
           ensuring no scaffold appears in more than one set.
                * This prevents data-leakage, ensuring the model cannot learn to
                  recognise a scaffold from the training set and then perform
                  artificially well on the validation or test sets simply because
                  it sees molecules with the same scaffold.
        """
        self._logger.info(f'Loading high-fidelity actives from {self._actives_path}...')
        try:
            df = pd.read_parquet(self._actives_path)
        except Exception as e:
            self._logger.error(f'Failed to load Parquet file: {e}')
            raise

        # 1) Generate a scaffold for each molecule in the dataset.
        self._logger.info('Generating Bemis-Murcko scaffolds for all compounds...')
        df['scaffold'] = df['smiles'].apply(self._generate_scaffold)

        # Remove any rows where the scaffold generation failed using boolean indexing/masking
        df = df[df['scaffold'] != '']

        # 2) Get a list of all unique scaffolds.
        scaffolds = df['scaffold'].unique()
        self._logger.info(f'Found {len(df)} molecules with {len(scaffolds)} unique scaffolds.')

        # 3) Shuffle the unique scaffolds to ensure random distribution.
        rng = np.random.default_rng(self._random_state)
        rng.shuffle(scaffolds)

        # 4) Calculate the split points for the list of scaffolds.
        n_scaffolds = len(scaffolds)
        test_idx = int(n_scaffolds * self._test_split)
        valid_idx = test_idx + int(n_scaffolds * self._validation_split)

        # 5) Split the list of scaffolds into three sets.
        test_scaffolds = scaffolds[:test_idx]
        valid_scaffolds = scaffolds[test_idx:valid_idx]
        train_scaffolds = scaffolds[valid_idx:]

        self._logger.info(
            f"Splitting scaffolds: "
            f"{len(train_scaffolds)} train, "
            f"{len(valid_scaffolds)} validation, "
            f"{len(test_scaffolds)} test."
        )

        # 6) Create the final DataFrames by filtering based on the scaffold sets.
        #    This ensures all molecules with the same scaffold are in the same split.
        self._train_dataset = df[df['scaffold'].isin(train_scaffolds)].copy()
        self._valid_dataset = df[df['scaffold'].isin(valid_scaffolds)].copy()
        self._test_dataset = df[df['scaffold'].isin(test_scaffolds)].copy()

        self._logger.info(
            f"Final dataset sizes (molecules): "
            f"{len(self._train_dataset)} train, "
            f"{len(self._valid_dataset)} validation, "
            f"{len(self._test_dataset)} test."
        )
