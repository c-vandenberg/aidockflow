import os
import time
import logging
from typing import Dict, List, Optional

import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors
from concurrent.futures.thread import ThreadPoolExecutor

from src.data.curation.base_curator import BaseCurator
from src.utils.file_utils import compress_and_delete_file, stream_lines_from_gzip_file, count_gzip_lines
from src.utils.fingerprint_utils import smiles_to_morgan_fp, fingerprints_to_numpy
from src.utils.clustering_utils import faiss_butina_cluster
from src.data.preprocessing.compound_preprocessing import CompoundDataPreprocessor


class CentroidLibraryCurator(BaseCurator):
    def __init__(self, config: Dict, logger: logging.Logger):
        super().__init__(config=config,logger=logger)
        self._preprocessor = CompoundDataPreprocessor(logger=logger)

    def run(self):
        # --- MULTI-LEVEL HIERARCHICAL CLUSTERING ---

        # 1. ZINC20 3D druglike database has ~690 million compounds. These cannot all
        #    be loaded into memory.
        #    Multi-level hierarchical cluster is therefore needed to iteratively
        #    cluster the data until the number of centroids is manageable.
        smiles_input_path = self._config.get('zinc_concat_smiles_path') + '.gzip'
        zinc_library_reduce_dir = self._config.get(
            'zinc_library_reduction_raw_dir',
            '../data/raw/ZINC20-3D-druglike-centroids/library-reduction'
        )
        os.makedirs(zinc_library_reduce_dir, exist_ok=True)

        if not os.path.exists(smiles_input_path):
            self._logger.error(f'Error: SMILES file not found at "{smiles_input_path}"')
            return

        use_medoids_for_reduction = self._config.get('use_medoids_for_reduction', True)
        if use_medoids_for_reduction:
            self._logger.info("Using medoid (max pop-count) selection for library reduction rounds.")
            representatives = 'medoids'
        else:
            self._logger.info("Using centroid (min MolWt) selection for library reduction rounds.")
            representatives = 'centroids'

        round_num = 1
        max_in_memory_size = self._config.get('max_in_memory_centroids', 200_000_000)
        num_smiles = count_gzip_lines(smiles_input_path)

        if num_smiles <= max_in_memory_size:
            self._logger.info(
                f'Initial number of SMILES ({num_smiles}) is small enough for in-memory clustering. '
                f'No SMILES library reduction required. '
            )
            centroid_smiles = list(stream_lines_from_gzip_file(smiles_input_path))
            final_centroid_smiles = self._process_smiles_batch(
                smiles_batch=centroid_smiles,
                tanimoto_cutoff=self._config.get('tanimoto_cluster_cutoff', 0.6),
                round_num=round_num,
                use_medoids=False
            )
        else:
            self._logger.info(
                f'Initial number of SMILES ({num_smiles}) is too large for in-memory clustering. '
                f'Beginning SMILES library reduction'
            )
            while True:
                reps_output_path = f'{zinc_library_reduce_dir}/round_{round_num}_{representatives}.smi'
                num_reps = self._run_clustering_round(
                    smiles_input_path=smiles_input_path,
                    representatives_output_path=reps_output_path,
                    round_num=round_num,
                    use_medoids=use_medoids_for_reduction
                )

                if num_reps <= max_in_memory_size:
                    reps_output_path = f'{zinc_library_reduce_dir}/round_{round_num}_{representatives}.smi'
                    self._logger.info(
                        f'Number of representatives ({num_reps}) is small enough for final in-memory clustering.'
                    )
                    final_sub_centroids_path = reps_output_path + '.gzip'
                    break

                # Prepare for next clustering round
                smiles_input_path = reps_output_path + '.gzip'
                round_num += 1

            # 2. Perform a final clustering on the aggregated sub-centroids, which now fit in memory.
            self._logger.info(f"Loading final sub-centroids from {final_sub_centroids_path} for final clustering...")
            final_sub_centroids = list(stream_lines_from_gzip_file(final_sub_centroids_path))

            final_centroid_smiles = self._process_smiles_batch(
                smiles_batch=final_sub_centroids,
                tanimoto_cutoff=self._config.get('tanimoto_cluster_cutoff', 0.6),
                round_num=round_num,
                use_medoids=False
            )

        # 3. Standardize final centroid SMILES and calculate InChIKey. Use dictionary comprehension to
        #    modify InchIKey dict key to be in line with `BioactiveCompound` objects.
        preprocessed_centroid_records = self._preprocessor.standardize_centroid_compounds(
            raw_centroid_smiles=final_centroid_smiles,
            batch_size=10_000
        )
        final_centroid_records = []
        for centroid_record in preprocessed_centroid_records:
            final_centroid_record = {
                key.replace('inchi_key', 'standardized_inchikey'): value for key, value in centroid_record.items()
            }
            final_centroid_records.append(final_centroid_record)

        # 4. Remove duplicates, and remove any centroid that is a duplicate of a high-fidelity active
        if final_centroid_records:
            centroid_processed_path = self._config.get(
                'centroids_processed_path',
                '../data/processed/ZINC20-3D-druglike-centroids/centroid_pool.parquet'
            )
            os.makedirs(os.path.dirname(centroid_processed_path), exist_ok=True)
            self._logger.info(f"Saving {len(final_centroid_records)} final centroids to {centroid_processed_path}...")

            df_centroids = pd.DataFrame(final_centroid_records)
            df_centroids = df_centroids.drop_duplicates(subset='standardized_inchikey')

            actives_parquet_path = self._config.get('actives_preprocessed_path')
            df_actives = pd.read_parquet(actives_parquet_path)

            if not df_actives.empty:
                active_keys = df_actives['standardized_inchikey']
                df_centroids = df_centroids[~df_centroids['standardized_inchikey'].isin(active_keys)]

            df_centroids.to_parquet(centroid_processed_path, index=False)
            self._logger.info("Centroid library construction complete.")
        else:
            self._logger.error("No centroids were selected after final clustering.")

    def _run_clustering_round(
        self,
        smiles_input_path: str,
        representatives_output_path: str,
        round_num: int,
        use_medoids: bool
    ):
        self._logger.info(f'Starting clustering round {round_num} on file: {smiles_input_path}')
        uncompressed_reps_path = representatives_output_path
        gzipped_reps_path = uncompressed_reps_path + '.gzip'
        batch_size = self._config.get('clustering_batch_size', 100_000_000)
        tanimoto_cutoff = self._config.get('tanimoto_cluster_cutoff', 0.6)
        smiles_stream = stream_lines_from_gzip_file(smiles_input_path)

        total_reps_written = 0
        batch_num = 1
        with open(uncompressed_reps_path, 'w') as outfile:
            batch_smiles = []
            for i, smiles in enumerate(smiles_stream):
                batch_smiles.append(smiles)
                if len(batch_smiles) >= batch_size:
                    self._logger.info(f'Processing batch starting at molecule {i+1-batch_size}...')
                    sub_reps = self._process_smiles_batch(
                        smiles_batch=batch_smiles,
                        tanimoto_cutoff=tanimoto_cutoff,
                        round_num=round_num,
                        batch_num=batch_num,
                        use_medoids=use_medoids
                    )
                    for sub_rep_smiles in sub_reps:
                        outfile.write(sub_rep_smiles + '\n')
                    total_reps_written += len(sub_reps)
                    batch_num += 1
                    batch_smiles = []

            if batch_smiles:
                self._logger.info("Processing final batch...")
                sub_reps = self._process_smiles_batch(
                    smiles_batch=batch_smiles,
                    tanimoto_cutoff=tanimoto_cutoff,
                    round_num=round_num,
                    batch_num=batch_num,
                    use_medoids=use_medoids
                )
                for sub_reps_smiles in sub_reps:
                    outfile.write(sub_reps_smiles + '\n')
                total_reps_written += len(sub_reps)

        compress_and_delete_file(
            uncompressed_path=uncompressed_reps_path,
            compressed_path=gzipped_reps_path,
            logger=self._logger
        )

        self._logger.info(
            f"Clustering round complete. Wrote {total_reps_written} representatives to {gzipped_reps_path}"
        )

        return total_reps_written

    def _process_smiles_batch(
        self,
        smiles_batch: List[str],
        tanimoto_cutoff: float,
        round_num: int,
        use_medoids: bool,
        batch_num: Optional[int] = None
    ) -> List[str]:
        if batch_num is None:
            batch_num = 'Final'

        # Parallel SMILES -> (SMILES, 1024‑bit ECFP4 fingerprint)
        mfp_start = time.time()
        with ThreadPoolExecutor(max_workers=16) as executor:
            smiles_mfp_results = [
                smiles_mfp for smiles_mfp in executor.map(smiles_to_morgan_fp, smiles_batch) if smiles_mfp
            ]

        mfp_end = time.time()
        self._logger.info(
            f'Round {round_num} Batch {batch_num}: SMILES to 1024‑bit ECFP4 fingerprint calculation time '
            f'for {len(smiles_batch)}: {round(mfp_end - mfp_start)} seconds. '
            f'Found {len(smiles_mfp_results)} valid 1024‑bit ECFP4 fingerprints'
        )

        if not smiles_mfp_results:
            self._logger.error("No valid molecules in batch")
            return []

        # Unzip SMILES and 1024‑bit ECFP4 fingerprints
        valid_smiles, fingerprints = zip(*smiles_mfp_results)

        if not fingerprints:
            self._logger.error('No fingerprints generated for SMILES batch')
            return []

        mfp_to_uint8_start = time.time()
        fp_array = fingerprints_to_numpy(fingerprints)
        mfp_to_uint8_end = time.time()
        self._logger.info(
            f'Round {round_num} Batch {batch_num}: 1024‑bit ECFP4 fingerprint (RDKit `ExplicitBitVect` objects) '
            f'to (N, 128) uint8 NumPy array conversion time: {round(mfp_to_uint8_end - mfp_to_uint8_start)} seconds.'
        )

        # 2. Cluster the fingerprints to give clusters of compounds whose similarities are
        #    within the Tanimoto similarity threshold.
        cluster_start = time.time()
        clusters, popcounts = faiss_butina_cluster(
            fp_array=fp_array,
            tanimoto_cutoff=tanimoto_cutoff,
            return_popcounts=True
        )
        cluster_end = time.time()
        self._logger.info(
            f'Round {round_num} Batch {batch_num}: {len(fingerprints)} fingerprints clustering time: '
            f'{round(cluster_end - cluster_start)} seconds.'
        )

        # 3. Select the representative (centroids or medoid) from each cluster
        sub_rep_start = time.time()
        sub_reps = []
        for cluster_indices in clusters:
            if not cluster_indices:
                continue
            try:
                if use_medoids:
                    # MEDOID STRATEGY: Pick the member with the highest pop-count (densest).
                    best_idx = max(cluster_indices, key=lambda i: popcounts[i])
                    sub_reps.append(valid_smiles[best_idx])
                else:
                    # CENTROID STRATEGY: Pick the member with the smallest molecular weight.
                    cluster_smiles = [valid_smiles[idx] for idx in cluster_indices]
                    smallest_mol_smiles = min(
                        cluster_smiles,
                        key=lambda smiles_str: Descriptors.MolWt(Chem.MolFromSmiles(smiles_str))
                    )
                    sub_reps.append(smallest_mol_smiles)
            except Exception as e:
                self._logger.error(f'Round {round_num} Batch {batch_num}: Failed to select representative for cluster. '
                                   f'Error: {e}')
        sub_rep_end = time.time()
        self._logger.info(
            f'Round {round_num} Batch {batch_num}: Representative selection time: '
            f'{round(sub_rep_end - sub_rep_start)} seconds.'
        )

        self._logger.info(
            f'Round {round_num} Batch {batch_num}: {len(sub_reps)} representatives found.'
        )

        return sub_reps
