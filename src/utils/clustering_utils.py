from typing import List, Tuple, Any

import numpy as np
import faiss
from rdkit import DataStructs
from rdkit.ML.Cluster import Butina


def faiss_butina_cluster(fp_array: np.ndarray, tanimoto_cutoff: float) -> list[tuple[int, ...]]:
    """
    Performs Butina clustering using Faiss for high-speed neighbor search.

    Parameters
    ----------
    fp_array : np.ndarray
        A NumPy array of fingerprints (dtype=uint8).
    tanimoto_cutoff : float
        The Tanimoto similarity cutoff.

    Returns
    -------
    List[Tuple[int, ...]]
        A list of clusters, where each cluster is a tuple of integer indices.
    """
    n_fingerprints = fp_array.shape[0]

    # The dimension is the number of bits in the fingerprints (e.g., 1024)
    dimension = fp_array.shape[1] * 8

    # Convert the Tanimoto similarity cutoff to a Hamming distance threshold.
    # For binary vectors, range searchg in Faiss uses Hamming distance.
    hamming_threshold = int(dimension * (1.0 - tanimoto_cutoff))

    # The quantizer for an `IndexBinaryIVF` must also be a binary index.
    # The `IndexBinaryFlat` quantizer will perform exact search for the binary vectors.
    quantizer = faiss.IndexBinaryFlat(dimension)

    # A common heuristic for the number of partitions is `sqrt(n_fingerprints)`
    nlist = int(np.sqrt(n_fingerprints))

    # Ensure nlist is at least 1, as it cannot be 0.
    if nlist < 1:
        nlist = 1

    # Check for available GPUs and create the appropriate index type
    if faiss.get_num_gpus() > 0:
        print(f"GPU detected. Creating GpuIndexBinaryIVF with {nlist} partitions.")
        res = faiss.StandardGpuResources()  # Allocates GPU resources
        # Create a CPU IVF index first
        cpu_ivf_index = faiss.IndexBinaryIVF(quantizer, dimension, nlist)
        # Move it to the GPU
        index = faiss.index_cpu_to_gpu(res, 0, cpu_ivf_index)
    else:
        print(f"No GPU detected. Using CPU IndexBinaryIVF with {nlist} partitions.")
        # Create the standard CPU index
        index = faiss.IndexBinaryIVF(quantizer, dimension, nlist)

    # The index must be trained on the data to learn the partitions
    if not index.is_trained:
        print(f"Training index on {n_fingerprints} fingerprints...")
        index.train(x=fp_array)

    # Add the fingerprints to the index
    index.add(x=fp_array)

    # Set nprobe to balance speed and accuracy
    index.nprobe = 8
    print(f"Index trained and populated. Using nprobe = {index.nprobe}")

    clusters = []
    assigned = np.zeros(n_fingerprints, dtype=bool)

    for i in range(n_fingerprints):
        if assigned[i]:
            continue

        # range_search is dramatically faster on the GPU
        lims, D, I = index.range_search(x=fp_array[i:i + 1], radius=float(hamming_threshold))

        neighbor_indices = I[lims[0]:lims[1]]

        new_cluster = []
        for idx in neighbor_indices:
            # Add any neighbor that has not yet been assigned to a cluster
            if not assigned[idx]:
                new_cluster.append(idx)
                assigned[idx] = True

        if new_cluster:
            clusters.append(tuple(new_cluster))

    return clusters


def butina_cluster(fingerprints: List[Any], tanimoto_cutoff: float) -> List[Tuple[int, ...]]:
    """
    Performs a memory-efficient Butina clustering on a list of fingerprints.

    This implementation avoids creating a full N x N distance matrix, making it
    suitable for clustering large datasets.

    Parameters
    ----------
    fingerprints : List[Any]
        A list of RDKit fingerprint objects (e.g., Morgan fingerprints).
    tanimoto_cutoff : float
        The Tanimoto similarity cutoff. Molecules with a similarity greater
        than or equal to this value will be grouped in the same cluster.

    Returns
    -------
    List[Tuple[int, ...]]
        A list of clusters, where each cluster is a tuple of integer indices
        referring to the original fingerprint list. The first element of each
        tuple is the index of the cluster centroid.
    """
    n_fingerprints: int = len(fingerprints)

    # 1. Calculate the Tanimoto distance (1 - Tanimoto similarity) for all pairs.
    #    This is the most time-consuming step, but it's done efficiently by RDKit.
    dists = []
    for i in range(1, n_fingerprints):
        tanimoto_similarities = DataStructs.BulkTanimotoSimilarity(fingerprints[i], fingerprints[:i])
        dists.extend([1 - s for s in tanimoto_similarities])

    distance_threshold = 1.0 - tanimoto_cutoff
    clusters = Butina.ClusterData(
        data=dists,
        nPts=n_fingerprints,
        distThresh=distance_threshold,
        isDistData=True
    )

    return clusters
