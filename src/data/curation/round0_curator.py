import os
import logging
from typing import Dict

import pandas as pd
from sklearn.model_selection import train_test_split

from src.data.curation.base_curator import BaseCurator
from src.utils.file_utils import compress_and_delete_file


class Round0DatasetCurator(BaseCurator):
    """
    Curates the initial 'round-0' training dataset, along with the validation
    and testing datasets that will be used throughout the training regime.

    This class combines high-fidelity active compounds with a diverse set of
    compounds from the ZINC20 3D druglike centroid library (which are presumed
    to be negative/non-binders for round 0). This creates the initial datasets
    required for the first round of model training.
    """
    def __init__(self, config: Dict, logger: logging.Logger):
        super().__init__(config=config,logger=logger)

    def run(self):
        """
        Executes the full round-0 dataset creation workflow.

        This method orchestrates the sampling, splitting, and merging steps
        to generate the final gzipped .smi files for training, validation,
        and testing.
        """
        self._logger.info("Starting round-0 dataset curation...")

        # 1. Sample from the centroid pool and save the remaining unsampled
        #    centroids.
        df_sampled, df_unsampled = self._sample_centroids()

        if df_sampled is None:
            return

        # 2. Split sampled centroids into train, validation, and test sets
        df_train_neg, df_val_neg, df_test_neg = self._split_negatives(df_sampled)

        # 3. Merge negatives with their corresponding positive sets and save
        #    these final sets as gzipped `.smi` files.
        self._merge_and_save_sets(df_train_neg, df_val_neg, df_test_neg)

        self._logger.info("Round-0 dataset curation complete.")

    def _sample_centroids(self) -> (pd.DataFrame, pd.DataFrame):
        """
        Randomly samples compounds from the centroid pool and saves the
        unsampled portion.

        Loads the full centroid pool parquet file, draws a random sample of
        a given size, and saves the remaining, unsampled molecules to a
        separate parquet file for use in later active learning rounds.

        Returns
        -------
        Tuple[pd.DataFrame, pd.DataFrame]
            A tuple containing the sampled DataFrame and the unsampled DataFrame.
            Returns (None, None) if the centroid pool cannot be loaded.
        """
        self._logger.info("Sampling from centroid pool...")
        centroid_pool_path = self._config.get('centroids_processed_path')
        n_sample = self._config.get('round0_n_sample', 500000)

        try:
            df_centroids = pd.read_parquet(centroid_pool_path)
            self._logger.info(f"Loaded {len(df_centroids)} centroids from {centroid_pool_path}.")
        except FileNotFoundError:
            self._logger.error(f"Centroid pool not found at: {centroid_pool_path}")

            return None, None

        # 1.1. Sample the centroids and save the remaining unsampled centroids to the
        #      unsampled pool
        df_sampled = df_centroids.sample(n=n_sample, random_state=self._config.get('random_state', 42))
        df_unsampled = df_centroids.drop(df_sampled.index)

        # 1.2. Save the unsampled pool for later active learning rounds.
        unsampled_path = self._config.get('round0_unsampled_centroid_pool_path')
        os.makedirs(os.path.dirname(unsampled_path), exist_ok=True)
        df_unsampled.to_parquet(unsampled_path, index=False)
        self._logger.info(f"Saved {len(df_unsampled)} unsampled centroids to {unsampled_path}")

        return df_sampled, df_unsampled

    def _split_negatives(self, df_sampled: pd.DataFrame) -> (pd.DataFrame, pd.DataFrame, pd.DataFrame):
        """
        Performs a random split on the sampled negative (centroids) compounds.

        Takes the DataFrame of sampled centroids and splits it into
        training, validation, and testing sets based on the configured
        ratios (e.g., 70/15/15).

        Parameters
        ----------
        df_sampled : pd.DataFrame
            The DataFrame of negative compounds sampled from the centroid pool.

        Returns
        -------
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
            A tuple containing the training, validation, and testing DataFrames
            for the negative set.
        """
        test_size = self._config.get('test_split', 0.15)
        val_size = self._config.get('validation_split', 0.15)
        random_state = self._config.get('random_state', 42)

        # 2.1. First, split into a preliminary training/validation set and a
        #      final test set.
        df_train_val_neg, df_test_neg = train_test_split(
            df_sampled, test_size=test_size, random_state=random_state
        )

        # 2.2. Next, split the preliminary set into the final training and validation sets.
        #      The validation size must be adjusted to be a fraction of the remaining data.
        #      Create the validation set from the remainder.
        df_train_neg, df_val_neg = train_test_split(
            df_train_val_neg, test_size=val_size / (1 - test_size), random_state=random_state
        )
        self._logger.info(
            f"Split negatives (centroids) into Train/Val/Test sets with sizes: "
            f"{len(df_train_neg)}/{len(df_val_neg)}/{len(df_test_neg)}"
        )

        return df_train_neg, df_val_neg, df_test_neg

    def _merge_and_save_sets(self, df_train_neg, df_val_neg, df_test_neg):
        """
        Merges positive and negative sets and saves them as gzipped .smi files.

        For each dataset (train, validation, test), it loads the corresponding
        high-fidelity actives, concatenates them with the negatives, and saves
        the final shuffled, deduplicated list to a gzipped .smi file.

        Parameters
        ----------
        df_train_neg : pd.DataFrame
            DataFrame of negative compounds for the training set.
        df_val_neg : pd.DataFrame
            DataFrame of negative compounds for the validation set.
        df_test_neg : pd.DataFrame
            DataFrame of negative compounds for the testing set.
        """
        # 3.1. Define paths for positive files and final output files
        paths = {
            'train': {
                'pos': self._config.get('actives_train_pos_path'),
                'neg': df_train_neg,
                'out': self._config.get('round0_train_smi_path')
            },
            'val': {
                'pos': self._config.get('actives_val_pos_path'),
                'neg': df_val_neg,
                'out': self._config.get('validation_smi_path')
            },
            'test': {
                'pos': self._config.get('actives_test_pos_path'),
                'neg': df_test_neg,
                'out': self._config.get('testing_smi_path')
            }
        }

        for set_name, set_paths in paths.items():
            try:
                # 3.2. Append positives to the respective negative files.
                df_pos = pd.read_parquet(set_paths['pos'])
                df_full = pd.concat([df_pos, set_paths['neg']], ignore_index=True)
                self._logger.info(
                    f"Created '{set_name}' set with {len(df_pos)} positives and {len(set_paths['neg'])} negatives."
                )

                # 3.3. Save the combined dataset to a gzipped `.smi` file.
                self._save_to_smi_gz(df_full, set_paths['out'])
            except FileNotFoundError as e:
                self._logger.error(f"Positive set file not found: {set_paths['pos']}")
            except Exception as e:
                self._logger.error(f"Failed to create or save '{set_name}' set: {e}")

    def _save_to_smi_gz(self, dataframe: pd.DataFrame, output_path: str):
        """
        Deduplicates, shuffles, and saves a DataFrame to a gzipped .smi file.

        The final dataset is deduplicated by InChIKey to ensure no compound appears
        more than once, shuffled randomly, and then written to a plain text,
        tab-separated file which is immediately gzipped.

        Parameters
        ----------
        dataframe : pd.DataFrame
            The combined DataFrame of positives and negatives to save.
        output_path : str
            The base path for the output .smi file (without extension).
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # 3.4. Deduplicate by InChIKey and shuffle the final dataset randomly.
        df_shuffled = dataframe.drop_duplicates(subset='standardized_inchikey').sample(frac=1)

        smi_path = output_path
        gz_path = f"{smi_path}.gzip"

        # 3.5. Write to a tab-separated .smi file with SMILES and InChIKey columns.
        df_shuffled.to_csv(
            smi_path,
            sep='\t',
            header=False,
            index=False,
            columns=['smiles', 'standardized_inchikey']
        )

        # 3.6. Compress the .smi file and remove the original uncompressed version.
        compress_and_delete_file(uncompressed_path=smi_path, compressed_path=gz_path, logger=self._logger)

        self._logger.info(f"Saved {len(df_shuffled)} compounds to {gz_path}")
