# AiDockFlow

# Abstract
Structure-based virtual screening of ultra-large chemical libraries is a powerful strategy in early-stage drug discovery. However, the computational expense of physics-based docking necessitates more efficient methods for navigating chemical space. Building upon the active learning framework proposed by *Zhou et al.* in their work "An artificial intelligence accelerated virtual screening platform for drug discovery", **<sup>1</sup>** we present AiDockFlow, a fully-automated pipeline designed to improve the efficiency and robustness of hit discovery campaigns.

Our methodology introduces several key enhancements to the data curation and model training stages. The process begins by curating a high-fidelity set of known actives for a given target, which are partitioned using a scaffold-based split to ensure robust training and validation sets. This is coupled with the generation of a diverse "druglike-centroid" library from ZINC15 to facilitate broad exploration of chemical space, as per the approach proposed by *Zhou et al.*. These high-fidelity actives are labelled as 'binders' and are used for initial training of the surrogate Graph Neural Network (GNN), along with a large sample of the diverse centroids (labelled as 'non-binders').

Another core contribution of our work is to refine the active learning loop where a Graph Neural Network (GNN) surrogate is iteratively retrained. We introduce a three-tiered labelling strategy based on RosetteVS docking scores where the top 10% of docked compounds are labelled as "binders", the bottom 10% are labelled as "non-binders", and the ambiguous middle 50% are ignored to ensure the model is trained only on high-confidence data. Additionally, for reduced computational expense, we replace the cumulative training dataset approach with a "replay buffer" (a smaller, random sample of previously seen compounds) to prevent catastrophic forgetting of model learning. Our final contribution is the addition of PAINS and Brenk filters in addition to the suite of medicinal chemistry filters proposed by *Zhou et al.*

# Contents


# Protocol
## 1. Data Ingestion & Curation
### 1.1. Retrieve High-Fidelity Actives
1. Target to be defined by Uniprot Accession Number in training config file.
2. Gather SMILES and bioactivity measure values for compounds that are known
binders to the target from public sources (ChEMBL, PubChem, etc.) via Python APIs or bulk downloads (SDF/CSV).
3. Likely parameters to use to determine known binders include inhibition constant (Ki)
and dissociation constant (Kd). Set threshold value (in nM) for binding parameter. These will be defined in training config file.
4. When extracting compounds, keep InChIKey, binding parameter (Ki or Kd), and SMILES string.
5. Merge compounds from each public source and deduplicate based on InChIKey.
6. Clean and normalize/standardize molecules:
    * Remove salts.
    * Protonation/tautomer standardization at pH 7.4 with `dimorphite-dl`.
    * Normalize binding parameter to nM.
    * Calculate InChIKey if not present in public sources.
    * Canonicalize SMILES with RDKit.
    * If a compound appears multiple times keep the lowest value and record median/σ for reference.
7. Save compounds as `validated_actives.parquet` file:
    * Apache Parquet is an open source, column-oriented data file format designed for efficient data storage and retrieval.
    * Parquet table (columnar, compressed) will have the following schema:
<br>
  <div align="center">
    <table>
        <thead>
          <tr>
            <th>column</th>
            <th>dtype</th>
            <th>comment</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td><code>inchi_key</code></td>
            <td>string</td>
            <td>27‑char standard InChIKey (UU‑case)</td>
          </tr>
          <tr>
            <td><code>smiles</code></td>
            <td>string</td>
            <td>RDKit canonical isomeric SMILES</td>
          </tr>
          <tr>
            <td><code>measure</code></td>
            <td>category</td>
            <td>“Ki” or “Kd”</td>
          </tr>
          <tr>
            <td><code>value_nM`</code></td>
            <td>float32</td>
            <td>Best (lowest) experimental value in nM</td>
          </tr>
          <tr>
            <td><code>n_measurements</code></td>
            <td>int16</td>
            <td>Count of independent measurements</td>
          </tr>
          <tr>
            <td><code>source_db</code></td>
            <td>category</td>
            <td>“ChEMBL” / “PubChem”</td>
          </tr>
          <tr>
            <td><code>target_uniprot</code></td>
            <td>string</td>
            <td>Target protein accession</td>
          </tr>
        </tbody>
      </table>
  </div>
<br>

### 1.2. Split High-Fidelity Actives Data
1. Do random stratified split 70/15/15% (train, validation, test). Exact split ratio to be defined in training config file:
    * Group actives by Bemis‑Murcko scaffold (`rdMolDescriptors.GetScaffoldForMol`).
    * Shuffle scaffolds with fixed RNG seed.
    * Allocate 70 % of scaffolds to train, 15 % to validation, 15 % to test.
2. Write three Parquet files with same schema as before - `train_pos.parquet`, `val_pos.parquet`, `test_pos.parquet`.

### 1.3. Build “Druglike-Centroid Library”
Based on the paper by Zhou et al., we will build a specialized subset of ~13 million molecules, referred to as the druglike-centroid library. These will be extracted from the ZINC15 3D druglike database (~493.6 million molecules) by:
1. Clustering similar molecules from the ZINC 3D druglike database, using a cutoff of 0.6 Tanimoto similarity (defined in training config file).
2. From each cluster, the smallest molecule will be selected and added to the library, serving as the centroid of the cluster.
3. This process will result in the formation of the druglike-centroid library, which will contain ~13 million molecules.

The purpose of creating the druglike-centroid library is to ensure that the model is exposed to a wide range of chemical space during the initial iteration.
The steps for this include:
1. Download ZINC15 drug-like 3D SMILES dataset.
2. Generate 1024‑bit ECFP4 fingerprints in streamed fashion (RDKit).
3. Cluster with Tanimoto 0.6 (defined in training config file) using Butina‑like algorithm (Faiss for speed).
4. Remove any centroid whose Tanimoto similarity to any of the high-fidelity actives is 1.0 (i.e. they are the same molecule).
5. Calculate InChIKey for each molecule and save druglike-centroid library to
`centroid_pool.parquet` the following schema:
   1. `inchi_key` (str)
   2. `smiles` (str)
  
### 1.4. Create “round-0” Candidate Training Dataset, & Validation/Testing Datasets
In the training workflow from Zhou et al., for the first iteration, 0.5 million and 1 million molecules were randomly selected from the centroid library as the training and testing datasets respectively.

We will slightly alter this to sample 0.5 million molecules, followed by a random stratified 70/15/15% split (with both the sample and split to be tunable in the config file). The thinking is:
1. Training Dataset:
   1. We do not want to dilute the high-fidelity Ki/Kd positives in the training dataset too much, but at the same time dilute them enough with negatives from the `centroid_pool.parquet` pool so that the model learns from broad chemical space.
   2. Therefore, ~350,000 negatives from the `centroid_pool.parquet` pool should be a good starting point.
2. Validation/Testing Dataset:
   1. We will start with ~75,000 negatives in both the validation and testing datasets.
   2. It is hoped that ~75,000 negatives, plus the positives in each will be enough to get a stable AUROC estimate.
  

The steps to create the round 0 training dataset, and the validation/testing datasets are:
1. Randomly sample 500,000 compounds (defined in training config file) from `centroid_pool.parquet` (**exploration**) and save to `round0_sampled_centroid_pool.parquet`.
2. Save unsampled pool to `round0_unsampled_centroid_pool.parquet`. This will be used in Step 4.1. to sample 250,000 undocked ligands for active learning loop at the end of round 0.
3. Do a random stratified 70/15/15% training/validation/testing split of `round0_sampled_centroid_pool.parquet`, saving to `train_neg.parquet`, `val_neg.parquet`, and `test_neg.parquet`.
4. Append all positives from `train_pos.parquet`, `val_pos.parquet`, and `test_pos.parquet` to the respective negative files from step 3 (**exploitation**).
5. We should now have `round0_full_train.parquet`, `full_val.parquet`, and `full_test.parquet`.
6. Deduplicate all three datasets based on InChiKeys, shuffle and write to `.smi` files, followed by compression with gzip (e.g. `round0_full_train.smi`, then `round0_full_train.smi.gzip`).
7. **N.B.** To prevent any data leakage validation and test datasets:
   * Are to be frozen throughout the training regime. This means they should not be relabeled, nor should they be replaced with a new set of molecules.
   * Should not be added to `graphs_master.pt` (see later) and should never be included in the active learning loop.

A `.smi` file is plain text, tab-separated list of molecules:
  * First column – SMILES string
  * Second column – InchiKey
