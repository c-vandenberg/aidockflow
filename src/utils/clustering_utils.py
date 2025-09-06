from typing import List, Tuple, Any, Union, Optional, Callable

import numpy as np
import faiss
from faiss.contrib.exhaustive_search import range_search_gpu
from rdkit import DataStructs
from rdkit.ML.Cluster import Butina

# BATCH_Q: The number of queries to send to Faiss in a single batch.
#          This avoids a pure Python loop (1 query at a time) and reduces overhead,
#          significantly speeding up the process. A value of 4096 is small enough
#          to fit in a CPU's L3 cache, which is optimal.
BATCH_Q = 4096


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

    # --- 2. Pre-computation for Efficiency ---

    # 2.1. Pre-compute pop-counts (number of 1-bits) once
    # 2.1.1. `popcounts`: A NumPy array storing the number of "on" bits (1s) for every
    #              fingerprint in the batch. This is pre-computed once to avoid
    #              recalculating it thousands of times inside the main loop.
    popcounts = np.unpackbits(fp_array, axis=1).sum(1).astype(np.int16)

    # 2.1.2. `max_pop`: The largest pop-count found in the entire batch. This is used
    #            to calculate a "worst-case" search radius later.
    max_pop = popcounts.max()

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
    for start in range(0, n_fingerprints, BATCH_Q):
        # 3.1.1. Calculate end of current batch of fingerprints
        end = min(start + BATCH_Q, n_fingerprints)
        batch_pops = popcounts[start:end]

        # 3.1.2. `guess_radius`: A dynamically calculated and deliberately oversized Hamming
        #                 distance. To guarantee we don't miss any neighbors, we calculate
        #                 this "worst-case" radius assuming the neighbor has the largest
        #                 possible pop-count in the dataset (max_pop).
        guess_radius = int(coeff * (int(batch_pops.max()) + max_pop))

        # --- Fast, "Generous" Search with Faiss ---

        # 3.1.3. Perform a single, fast search for all queries in the mini-batch.
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
        # singleton cluster – the only member is trivially its own medoid
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


def hamming_radius_from_tanimoto(T: float, a: int, b: int) -> int:
    """
    For binary fingerprints with popcounts a (query) and b (candidate),
    the largest Hamming distance D that can still yield Tanimoto >= T is:
        D_max = ((1 - T) / (1 + T)) * (a + b)
    We return floor(D_max) as an int radius (Faiss expects int for binary).
    """
    return int(((1.0 - T) / (1.0 + T)) * (a + b))


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


def consolidate_representatives(
    fp_uint8: np.ndarray,                 # (N, nbytes) uint8 packed 1024-bit fps
    smiles: List[str],                    # parallel list of SMILES (len N)
    tanimoto_cutoff: float,
    selector: str = "first",              # "first" | "max_pop" | custom via selector_fn
    selector_fn: Optional[Callable[[List[int], np.ndarray], int]] = None,
    batch_q: int = BATCH_Q,
    gpu_k: int = 2048
) -> Tuple[List[str], np.ndarray]:
    """
    One global pass to merge near-duplicates across batches using union-find.
    - Uses FAISS for a generous Hamming prefilter (GPU if available).
    - Verifies with exact Tanimoto computed from popcounts + Hamming.
    - Returns the deduplicated SMILES list and indices kept.
    """
    assert fp_uint8.dtype == np.uint8 and fp_uint8.ndim == 2
    N, nbytes = fp_uint8.shape
    dim = nbytes * 8

    # Precompute popcounts once
    pop = np.unpackbits(fp_uint8, axis=1).sum(1).astype(np.int16)  # (N,)
    b_max = int(pop.max())

    uf = UnionFind(N)

    # Prepare FAISS indices.
    # GPU path: keep all data on GPU index; give CPU fallback to contrib helper as numpy
    gpu_index = None
    if faiss.get_num_gpus() > 0:
        res = faiss.StandardGpuResources()
        gpu_index = faiss.GpuIndexBinaryFlat(res, dim)
        gpu_index.add(fp_uint8)  # database on GPU

    # CPU index only needed if no GPU available (or as fallback inside contrib helper via numpy)
    cpu_index = None
    if gpu_index is None:
        cpu_index = faiss.IndexBinaryFlat(dim)
        cpu_index.add(fp_uint8)

    # Batch over queries to avoid Python overhead
    for start in range(0, N, batch_q):
        end = min(start + batch_q, N)
        batch = fp_uint8[start:end]

        # tight-but-safe batch radius: use per-batch max(a) vs global max(b)
        a_max = int(pop[start:end].max())
        r = hamming_radius_from_tanimoto(tanimoto_cutoff, a_max, b_max)

        if gpu_index is not None:
            # Pass the whole DB as numpy for CPU fallback; helper will build a flat CPU index on demand
            lims, D, I = range_search_gpu(batch, r, gpu_index, fp_uint8, gpu_k=gpu_k)
        else:
            lims, D, I = cpu_index.range_search(batch, r)

        # Verify and union
        for q in range(end - start):
            i = start + q
            ql, qr = lims[q], lims[q + 1]
            a = int(pop[i])
            # leader itself might appear; skip self-pairs
            for j, d in zip(I[ql:qr], D[ql:qr]):
                if j == i:
                    continue
                b = int(pop[j])
                # exact intersection / Tanimoto from Hamming
                c = (a + b - d) // 2
                denom = a + b - c
                if denom <= 0:
                    continue
                if c / denom >= tanimoto_cutoff:
                    uf.union(i, j)

    comps = uf.components()

    # Choose representative per component
    if selector_fn is not None:
        keep_idx = [selector_fn(comp, pop) for comp in comps]
    elif selector == "max_pop":
        keep_idx = [max(comp, key=lambda k: pop[k]) for comp in comps]
    elif selector == "first":
        keep_idx = [comp[0] for comp in comps]
    else:
        # default to first
        keep_idx = [comp[0] for comp in comps]

    keep_idx = np.array(keep_idx, dtype=np.int64)
    kept_smiles = [smiles[i] for i in keep_idx.tolist()]
    return kept_smiles, keep_idx
