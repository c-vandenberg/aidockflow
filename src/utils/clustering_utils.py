from typing import List, Tuple, Any, Union, Optional, Callable

import numpy as np
import faiss
from faiss.contrib.exhaustive_search import range_search_gpu
from rdkit import DataStructs
from rdkit.ML.Cluster import Butina

# FAISS_QUERY_BATCH: The number of queries to send to Faiss in a single batch.
#          This avoids a pure Python loop (1 query at a time) and reduces overhead,
#          significantly speeding up the process. A value of 4096 is small enough
#          to fit in a CPU's L3 cache, which is optimal.
FAISS_QUERY_BATCH = 4096

_BYTE_POPCOUNT = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def faiss_butina_cluster(
    fp_array: np.ndarray,
    tanimoto_cutoff: float,
    return_popcounts: bool = False
) -> Union[list[tuple[int, ...]], Tuple[List[Tuple[int, ...]], np.ndarray]]:
    """
    Performs an exact Butina clustering on a large set of binary fingerprints.
    The goal is to group molecules where the Tanimoto similarity is greater than
    or equal to a given tanimoto_cutoff.

    This approach is used when the dataset size exceeds system memory capacity, so
    RDKit `ExplicitBitVect` fingerprint objects with RDKit clustering cannot be used.

    To achieve higher speed it uses a two-stage filtering strategy:
    1. A Fast, "Generous" Search:
        * For each potential cluster center (centroid), it uses the high-speed Faiss
          library to find a broad list of candidate neighbors. This search uses a
          dynamically calculated Hamming distance radius that is deliberately
          overly generous to ensure no true neighbors are missed.
    2. An Exact Tanimoto Verification:
        * It then loops through the much smaller list of candidate neighbors and
          applies the precise mathematical formula for Tanimoto similarity. Only
          candidates that pass this exact check are included in the final cluster.

    This two-stage filtering strategy avoids the memory and speed limitations of
    traditional RDKit clustering while satisfying the strict requirement of using
    the Tanimoto metric

    Parameters
    ----------
    fp_array : np.ndarray
        A NumPy array of fingerprints (dtype=uint8).
    tanimoto_cutoff : float
        The Tanimoto similarity cutoff.
    return_popcounts : bool
        If True, returns a tuple of (clusters, popcounts).

    Returns
    -------
    List[Tuple[int, ...]] or Tuple[List[Tuple[int, ...]], np.ndarray]
        A list of clusters (where each cluster is a tuple of integer indices),
        or a tuple of (clusters, popcounts) if requested.
    """
    # --- 1. Initialization ---

    # 1.1. `n_fingerprints`: The total number of molecules in the batch.
    # 1.2. `n_bytes`: The number of bytes per fingerprint (e.g., 128 for a 1024-bit fp).
    n_fingerprints, n_bytes = fp_array.shape

    # 1.3. `dimension`: The total number of bits in each fingerprint (e.g., 1024).
    dimension = n_bytes * 8

    # 1.4. `cpu_index`: A Faiss CPU index for brute-force search on binary data.
    cpu_index = faiss.IndexBinaryFlat(dimension)

    # 1.5. If a GPU is available, create a separate GPU index for the initial fast search.
    gpu_index = None
    if faiss.get_num_gpus() > 0:
        res = faiss.StandardGpuResources()
        gpu_index = faiss.GpuIndexBinaryFlat(res, dimension)
        gpu_index.add(fp_array) # Add the fingerprints to the index
    else:
        cpu_index.add(fp_array)

    # --- 2. Pre-computation for Efficiency ---

    # 2.1. Pre-compute pop-counts (number of 1-bits) once
    # 2.1.1. `popcounts`: A NumPy array storing the number of "on" bits (1s) for every
    #              fingerprint in the batch. This is pre-computed once to avoid
    #              recalculating it thousands of times inside the main loop.
    popcounts = np.unpackbits(fp_array, axis=1).sum(1).astype(np.int16)

    # 2.1.2. `batch_max_pop`: The largest pop-count found in the entire batch. This is used
    #                         to calculate a "worst-case" search radius later.
    batch_max_pop = int(popcounts.max())

    # 2.2. `clusters`: The final list that will store the identified clusters.
    clusters: list[tuple[int, ...]] = []

    # 2.3. `assigned`: A boolean array to track which molecules have already been
    #             assigned to a cluster, which is fundamental to the Butina algorithm.
    assigned = np.zeros(n_fingerprints, dtype=bool)

    # 2.4. `coeff`: A pre-calculated constant from the Tanimoto-to-Hamming distance
    #          conversion formula to make the calculation inside the loop faster.
    #          Formula: d_max = (1-T)/(1+T) * (a+b)
    coeff = (1.0 - tanimoto_cutoff) / (1.0 + tanimoto_cutoff)

    # --- 3. Main Clustering Loop ---

    # 3.1. Iterate through every fingerprint to in mini-batches of size `BATCH_Q` to select
    # potential cluster centroids.
    for start in range(0, n_fingerprints, FAISS_QUERY_BATCH):
        # 3.1.1. Calculate end of current batch of fingerprints
        end = min(start + FAISS_QUERY_BATCH, n_fingerprints)
        mini_batch_pop = popcounts[start:end]

        # 3.1.2. `mini_batch_max_pop`: The largest pop-count found in the current mini-batch. This is used
        #                              to calculate a "worst-case" search radius later.
        mini_batch_max_pop = int(mini_batch_pop.max())


        # 3.1.3. `guess_radius`: A dynamically calculated and deliberately oversized Hamming
        #                        distance. To guarantee we don't miss any neighbors, we calculate
        #                        this "worst-case" radius assuming the neighbor has the largest
        #                        possible pop-count in the dataset (max_pop).
        guess_radius = int(coeff * (mini_batch_max_pop + batch_max_pop))

        # --- Fast, "Generous" Search with Faiss ---

        # 3.1.4. Perform a single, fast search for all queries in the mini-batch.
        if gpu_index is not None:
            # Use the GPU k-NN emulation + CPU fallback strategy for maximum speed if GPU
            # is available
            lims, dist, idx = range_search_gpu(
                fp_array[start:end],  # Queries
                guess_radius,  # Radius
                gpu_index,  # GPU Index
                cpu_index,  # CPU Index Fallback
                gpu_k=2048  # Candidates Per Query
            )
        else:
            # If no GPU available, fallback to CPU-only index
            lims, dist, idx = cpu_index.range_search(fp_array[start:end], guess_radius)

        # --- 4. Process Results and Form Clusters ---

        # 4.1. Iterate over every query fingerprint in the mini-batch
        for fp_q in range(end - start):
            # 4.1.1. `fp_idx`: Global index of the current query fingerprint (current molecule)
            fp_idx = start + fp_q

            # 4.1.2. If the molecule has already been assigned to a previous cluster, skip it.
            if assigned[fp_idx]:
                continue

            # 4.1.3. This molecule is now the "leader" of a new cluster.
            #        Immediately create its cluster and mark it as assigned.
            new_cluster = [fp_idx]
            assigned[fp_idx] = True

            # 4.1.4. `a`: The pop-count of the current "leader" molecule.
            a = int(popcounts[fp_idx])

            # `q_l, q_r`: Pointers to the slice of results for this specific query
            #             within the larger `dist` and `idx` arrays.
            q_l, q_r = lims[fp_q], lims[fp_q + 1]

            # --- Exact Tanimoto Verification ---

            # 4.1.5. Loop through only the candidate neighbors found by Faiss for this query.
            #        `j`: index of a candidate neighbor
            #        `d`: Candidate neighbour Hamming distance from molecule i.
            for j, d in zip(idx[q_l:q_r], dist[q_l:q_r]):
                # Skip any candidate that has already been assigned to a cluster.
                if assigned[j]:
                    continue

                # `b`: The pop-count of the candidate neighbor molecule.
                b = int(popcounts[j])

                # `c`: The number of shared "on" bits (the intersection). This is calculated
                #      from the two pop-counts and their Hamming distance.
                #      Hamming Formula: Hamming_Dist(a,b) = a + b - 2c
                #                       or
                #                       Hamming_Dist(a,b) = popcount(a) + popcount(b) - 2 * popcount(a&b)
                #      Rearranging for c: c = (a + b - Hamming_Dist(a,b)) / 2
                #                       or
                #                         popcount(a&b) = (popcount(a) + popcount(b) - Hamming_Dist(a,b)) / 2
                c = (a + b - d) // 2

                # Tanimoto Similarity = c / (a + b - c)
                # This is the exact, final check.
                if c / (a + b - c) >= tanimoto_cutoff:
                    # If the true Tanimoto similarity is high enough, add the neighbor
                    # to the current cluster and mark it as assigned so it won't be
                    # processed again.
                    new_cluster.append(j)
                    assigned[j] = True

            # 4.1.6. Add the newly formed, fully verified cluster to the final list.
            clusters.append(tuple(new_cluster))

    if return_popcounts:
        return clusters, popcounts
    else:
        return clusters


def consolidate_representatives(
    fp_uint8: np.ndarray,                  # (N, nbytes) uint8 packed 1024-bit fps
    smiles: List[str],                     # parallel list of SMILES (len N)
    tanimoto_cutoff: float,
    selector: str = "first",               # "first" | "max_pop" | custom via selector_fn
    selector_fn: Optional[Callable[[List[int], np.ndarray], int]] = None,
    batch_q: int = FAISS_QUERY_BATCH,
    gpu_k: int = 2048
) -> Tuple[List[str], np.ndarray]:
    """
    One global pass to merge near-duplicates across *all* batches using union–find.
    Uses the same prefilter + exact Tanimoto verification scheme as faiss_butina_cluster().
    Returns the deduplicated SMILES list and the kept indices (into fp_uint8/smiles).
    """
    assert fp_uint8.dtype == np.uint8 and fp_uint8.ndim == 2

    # --- 1. Initialization ---

    # 1.1. `n_fingerprints`: The total number of molecules in the set.
    # 1.2. `n_bytes`: The number of bytes per fingerprint (e.g., 128 for a 1024-bit fp).
    n_fingerprints, n_bytes = fp_uint8.shape

    # 1.3. `dimension`: The total number of bits in each fingerprint (e.g., 1024).
    dimension = n_bytes * 8

    # 1.4/1.5. Prepare FAISS indices (mirror the pattern from faiss_butina_cluster()).
    # GPU path: database on GPU; CPU fallback provided as NumPy table to the contrib helper.
    gpu_index = None
    if faiss.get_num_gpus() > 0:
        res = faiss.StandardGpuResources()
        gpu_index = faiss.GpuIndexBinaryFlat(res, dimension)
        gpu_index.add(fp_uint8) # Add the fingerprints to the GPU index
        cpu_index = fp_uint8    # NumPy fallback for range_search_gpu()
    else:
        cpu_index = faiss.IndexBinaryFlat(dimension)
        cpu_index.add(fp_uint8)

    # --- 2. Pre-computation for Efficiency ---

    # 2.1. Pre-compute pop-counts (number of 1-bits) once
    # 2.1.1. `popcounts`: A NumPy array storing the number of "on" bits (1s) for every fingerprint.
    popcounts = fast_popcounts_uint8(fp_uint8)

    # 2.1.2. `batch_max_pop`: The largest pop-count found in the entire set (used for worst-case radius).
    batch_max_pop = int(popcounts.max())

    # 2.2. Union–find over all elements to build connected components under the cutoff.
    uf = UnionFind(n_fingerprints)

    # 2.3. `coeff`: Pre-calculated constant from the Tanimoto→Hamming bound:
    #               d_max = (1-T)/(1+T) * (a+b)
    coeff = (1.0 - tanimoto_cutoff) / (1.0 + tanimoto_cutoff)

    # --- 3. Main Merge Loop (global pass) ---

    # 3.1. Iterate through fingerprints in mini-batches of size `batch_q`
    for start in range(0, n_fingerprints, batch_q):
        # 3.1.1. Calculate end of current mini-batch
        end = min(start + batch_q, n_fingerprints)
        mini_batch_pop = popcounts[start:end]

        # 3.1.2. `mini_batch_max_pop`: largest popcount in the current mini-batch
        mini_batch_max_pop = int(mini_batch_pop.max())

        # 3.1.3. `guess_radius`: generous Hamming radius (worst-case b = batch_max_pop)
        guess_radius = int(coeff * (mini_batch_max_pop + batch_max_pop))

        # --- Fast, "Generous" Search with Faiss ---

        # 3.1.4. Perform a single range-search for all queries in the mini-batch
        if gpu_index is not None:
            # GPU k-NN emulation + CPU fallback (NumPy table) for exact range coverage
            lims, dist, idx = range_search_gpu(
                fp_uint8[start:end], # Queries
                guess_radius,        # Radius
                gpu_index,           # GPU index (database already added)
                cpu_index,           # CPU fallback: the full NumPy table
                gpu_k=gpu_k          # candidates per query on GPU
            )
        else:
            lims, dist, idx = cpu_index.range_search(fp_uint8[start:end], guess_radius)

        # --- 4. Verify and Union ---

        # 4.1. Iterate over every query in the mini-batch
        for fp_q in range(end - start):
            # 4.1.1. `fp_idx`: global index of the current query fingerprint
            fp_idx = start + fp_q

            # 4.1.2. `a`: popcount of the current query
            a = int(popcounts[fp_idx])

            # `q_l, q_r`: slice of results for this query within `dist` and `idx`
            q_l, q_r = lims[fp_q], lims[fp_q + 1]

            # 4.1.3. Exact Tanimoto verification and union
            for j, d in zip(idx[q_l:q_r], dist[q_l:q_r]):
                if j == fp_idx:
                    continue  # Skip self
                b = int(popcounts[j])

                # exact intersection / Tanimoto via Hamming
                c = (a + b - d) // 2
                denom = a + b - c
                if denom <= 0:
                    continue
                if c / denom >= tanimoto_cutoff:
                    uf.union(fp_idx, j)

    # --- 5. Extract components and choose representatives ---

    components = uf.components()  # List[List[int]]

    if selector_fn is not None:
        keep_idx = [selector_fn(comp, popcounts) for comp in components]
    elif selector == "max_pop":
        keep_idx = [max(comp, key=lambda k: popcounts[k]) for comp in components]
    else:
        keep_idx = [comp[0] for comp in components]

    keep_idx = np.asarray(keep_idx, dtype=np.int64)
    kept_smiles = [smiles[i] for i in keep_idx.tolist()]

    return kept_smiles, keep_idx


def hamming_radius_from_tanimoto(t: float, a_max: int, b_max: int) -> int:
    """
    For binary fingerprints with popcounts a (query) and b (candidate),
    the largest Hamming distance D that can still yield Tanimoto >= T is:
        D_max = ((1 - T) / (1 + T)) * (a + b)
    """
    coeff = (1.0 - t) / (1.0 + t)
    return int(coeff * (a_max + b_max))

class UnionFind:
    def __init__(self, n: int):
        self.parent = np.arange(n, dtype=np.int64)
        self.size = np.ones(n, dtype=np.int32)

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.size[ra] < self.size[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        self.size[ra] += self.size[rb]

    def components(self) -> List[List[int]]:
        roots = {}
        for i in range(self.parent.shape[0]):
            r = self.find(i)
            roots.setdefault(r, []).append(i)
        return list(roots.values())


def fast_popcounts_uint8(fp_uint8: np.ndarray) -> np.ndarray:
    """
    fp_uint8: (N, nbytes) uint8, packed bits (e.g., 128 bytes for 1024-bit ECFP4)
    returns:  (N,) uint16, number of 1-bits per row
    """
    # map each byte to its popcount, then sum across bytes
    # keep dtype small: max per row is 1024, so uint16 is enough
    return _BYTE_POPCOUNT[fp_uint8].sum(axis=1, dtype=np.uint16)


def true_medoid_idx(
    cluster_idx: List[int],
    fp_uint8: np.ndarray,
    popcounts: np.ndarray
) -> int:
    """
    Return the index (into fp_uint8) of the true medoid:
    the member with the highest mean Tanimoto to all others.
    Only called for small clusters (<= 256) — O(|C|^2).
    """
    if len(cluster_idx) == 1:
        # Singleton cluster – The only member is its own medoid
        return cluster_idx[0]

    idx_arr = np.asarray(cluster_idx, dtype=int)  # (C,)
    sub = fp_uint8[idx_arr]  # (C, 128) uint8
    inter = np.bitwise_and(
        sub[:, None, :],  # (C, 1, 128)
        sub[None, :, :]  # (1, C, 128)
    ).sum(2, dtype=np.uint16)  # (C, C) intersection popcount

    pc = popcounts[idx_arr].astype(np.int32) # (C, 1)
    denom = pc[:, None] + pc[None, :] - inter # (C, C)
    sims = inter / denom  # (C, C) Tanimoto

    return cluster_idx[int(sims.mean(1).argmax())]


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
