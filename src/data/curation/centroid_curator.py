import os
import gzip
import logging
from typing import Dict, List, Any

import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from biochemical_data_connectors import CompoundStandardizer

from src.data.curation.base_curator import BaseCurator
from src.utils.file_utils import compress_and_delete_file, stream_lines_from_gzip_file
from src.utils.clustering_utils import butina_cluster


class CentroidLibraryCurator(BaseCurator):
    def __init__(self, config: Dict, logger: logging.Logger):
        super().__init__(config=config,logger=logger)
        self._standardizer = CompoundStandardizer(logger=logger)

    def run(self):
        # --- MULTI-LEVEL HIERARCHICAL CLUSTERING ---

        # 1. ZINC20 3D druglike database has ~690 million compounds. These cannot all
        #    be loaded into memory.
        #    Multi-level hierarchical cluster is therefore needed to iteratively
        #    cluster the data until the number of centroids is manageable.
        smiles_input_path = self._config.get('zinc_concat_smiles_path') + '.gzip'
        centroids_raw_dir = self._config.get('centroids_raw_dir', '../data/raw/ZINC20-3D-druglike-centroids/')
        os.makedirs(centroids_raw_dir, exist_ok=True)

        if not os.path.exists(smiles_input_path):
            self._logger.error(f'Error: SMILES file not found at "{smiles_input_path}"')
            return

        round_num = 1
        max_in_memory_size = self._config.get('max_in_memory_centroids', 35_000)

        while True:
            centroids_output_path = f'{centroids_raw_dir}/round_{round_num}_centroids.smi'
            num_centroids = self._run_clustering_round(
                smiles_input_path=smiles_input_path,
                centroids_output_path=centroids_output_path,
                round_num=round_num
            )

            if num_centroids <= max_in_memory_size:
                self._logger.info(
                    f'Number of centroids ({num_centroids}) is small enough for final in-memory clustering.'
                )
                final_sub_centroids_path = centroids_output_path + '.gzip'
                break

            # Prepare for next clustering round
            smiles_input_path = centroids_output_path + '.gzip'
            round_num += 1

        # 2. Perform a final clustering on the aggregated sub-centroids, which now fit in memory.
        self._logger.info(f"Loading final sub-centroids from {final_sub_centroids_path} for final clustering...")
        with gzip.open(final_sub_centroids_path, 'rb') as centroids_file:
            final_sub_centroids = [line.strip() for line in centroids_file]

        final_centroid_smiles = self._process_smiles_batch(
            smiles_batch=final_sub_centroids,
            tanimoto_cutoff=self._config.get('tanimoto_cluster_cutoff', 0.6)
        )

        # 3. Standardize final centroid SMILES and calculate InChIKey
        final_centroid_records = []
        for centroid_smiles in final_centroid_smiles:
            standardized_data = self._standardizer.standardize_smiles(centroid_smiles)
            if not standardized_data:
                self._logger.error(f'Error standardizing SMILES for {centroid_smiles}')

            final_centroid_records.append(standardized_data)

        # 4. Remove duplicates, and remove any centroid that is a duplicate of a high-fidelity active
        if final_centroid_records:
            centroid_processed_path = self._config.get(
                'centroids_processed_path',
                '../data/processed/ZINC20-3D-druglike-centroids/centroid_pool.parquet'
            )
            os.makedirs(os.path.dirname(centroid_processed_path), exist_ok=True)
            self._logger.info(f"Saving {len(final_centroid_records)} final centroids to {centroid_processed_path}...")

            df_centroids = pd.DataFrame(final_centroid_records)
            df_centroids = df_centroids.drop_duplicates(subset='inchi_key')

            actives_parquet_path = self._config.get('standardized_actives_path')
            df_actives = pd.read_parquet(actives_parquet_path)

            if not df_actives.empty:
                active_keys = df_actives['standardized_inchikey']
                df_centroids = df_centroids[~df_centroids['inchi_key'].isin(active_keys)]

            df_centroids.to_parquet(centroid_processed_path, index=False)
            self._logger.info("Centroid library construction complete.")
        else:
            self._logger.error("No centroids were selected after final clustering.")

    def _run_clustering_round(self, smiles_input_path: str, centroids_output_path: str, round_num: int):
        self._logger.info(f'Starting clustering round {round_num} on file: {smiles_input_path}')
        uncompressed_centroids_path = centroids_output_path
        gzipped_centroids_path = centroids_output_path + '.gzip'
        batch_size = self._config.get('clustering_batch_size', 35_000)
        tanimoto_cutoff = self._config.get('tanimoto_cluster_cutoff', 0.6)
        smiles_stream = stream_lines_from_gzip_file(smiles_input_path)

        total_centroids_written = 0
        with open(uncompressed_centroids_path, 'w') as outfile:
            batch = []
            for i, smiles in enumerate(smiles_stream):
                batch.append(smiles)
                if len(batch) >= batch_size:
                    self._logger.info(f'Processing batch starting at molecule {i+1-batch_size}...')
                    sub_centroids = self._process_smiles_batch(smiles_batch=batch, tanimoto_cutoff=tanimoto_cutoff)
                    for sub_centroid_smiles in sub_centroids:
                        outfile.write(sub_centroid_smiles + '\n')
                    total_centroids_written += len(sub_centroids)
                    batch = []

            if batch:
                self._logger.info("Processing final batch...")
                sub_centroids = self._process_smiles_batch(smiles_batch=batch, tanimoto_cutoff=tanimoto_cutoff)
                for sub_centroid_smiles in sub_centroids:
                    outfile.write(sub_centroid_smiles + '\n')
                total_centroids_written += len(sub_centroids)

        compress_and_delete_file(
            uncompressed_path=uncompressed_centroids_path,
            compressed_path=gzipped_centroids_path,
            logger=self._logger
        )

        self._logger.info(
            f"Clustering round complete. Wrote {total_centroids_written} sub-centroids to {gzipped_centroids_path}"
        )

        return total_centroids_written

    def _process_smiles_batch(self, smiles_batch: List[str], tanimoto_cutoff: float) -> List[str]:
        fingerprints, valid_smiles = [], []
        morg_generator = Chem.rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024)
        for smiles in smiles_batch:
            mol = Chem.MolFromSmiles(smiles)
            if mol:
                valid_smiles.append(smiles)
                fingerprints.append(morg_generator.GetFingerprint(mol))

        if not fingerprints:
            self._logger.error('No fingerprints generated for SMILES batch')
            return []

        # 2. Cluster the fingerprints to give clusters of compounds whose similarities are
        #    within the Tanimoto similarity threshold.
        clusters = butina_cluster(fingerprints=fingerprints, tanimoto_cutoff=tanimoto_cutoff)

        # 3. Select the smallest molecule from each cluster as the sub-centroid of that cluster
        sub_centroids = []
        for cluster_indices in clusters:
            try:
                cluster_smiles = [valid_smiles[idx] for idx in cluster_indices]
                smallest_mol_smiles = min(
                    cluster_smiles,
                    key=lambda smiles_str: Descriptors.MolWt(Chem.MolFromSmiles(smiles_str))
                )
                sub_centroids.append(smallest_mol_smiles)
            except Exception as e:
                self._logger.error(f'Failed to calculate sub-centroid for cluster. Error: {e}')

        return sub_centroids
