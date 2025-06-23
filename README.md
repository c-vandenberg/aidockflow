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

### 1.5. Prepare Target Structure
1. Target to be defined by Uniprot Accession Number in training config file.
2. Download PDB or AlphaFold model (target source database to be defined in training config file).
3. Use PDBFixer to add missing residues/atoms, assign bonds, and remove water molecules.
4. Use `propka` to add pH 7.4 protonation.
5. Use OpenBabel to add hydrogens.
6. Energy-minimize side chains with Rosetta FastRelax (2,000 cycles, coordinates constraints).
7. Convert to PDBQT with AutoDockTools script (`prepare_receptor4.py`).
8. Store copy of relaxed target in both PDB and PDBQT files (`target_prepped.pdb` and `target_prepped.pdbqt`).

## 2. Featurization (Round 0 Initially)
**Data Type Conversions:**
1. SMILES → Graphs (for GNN machine learning).
2. SMILES → ETKDG conformer in SDF files (for docking).
3. SMILES → 1024-bit ECFP4 (for clustering, similarity searches, FNN pre-screen etc.)

### 2.1. Canonicalize Ligands and Add Explicit Hydrogens
1. Load SMILES from training (and validation and testing if round 0) `.smi` files, convert to `rdkit.Mol` objects (`Chem.MolFromSmiles(smi, sanitize=True)`).
2. Check if each SMILES are correct by checking that `rdkit.Mol` object is not `None`.
3. Add stereochemistry tags to each SMILES string via `rdkit.Chem.rdmolops.AssignStereochemistry(mol, cleanIt=True, force=True)`
4. Add explicit hydrogens via `rdkit.Chem.AddHs(mol)`. This is needed for 3-D embedding and for graphs that encode hydrogen counts.
5. Save cleaned SMILES as `cleaned_round0_full_train.smi.gz`, `cleaned_full_val.smi.gz`, and `cleaned_full_test.smi.gz`.

### 2.2. Generate Graph Objects & Attach Initial Labels
Because the GNN needs node/edge features, we need to build graph objects for each SMILES string.
1. Load SMILES from `cleaned_round0_full_train.smi.gz`, `cleaned_full_val.smi.gz`, and
`cleaned_full_test.smi.gz`.
2. For every molecule, build an atom-bond graph with:
   1. Element
   2. Degree
   3. Formal charge
   4. Aromatic flag
   5. Hybridization
   6. Ring flag
   7. Bond type
   8. Conjugated ring flag
3. Recommended libraries include PyTorch‑Geometric, DGL or Chemprop (they all expose a “Mol ⇒ Data” helper).
4. Each molecule graph needs to have its InChiKey assigned to it for identification, along with a label. (See below).
5. Serialize the list with `torch.save` or `joblib` (~3 GB per 1 million molecules, compressible) to `graphs_round0_train.pt`, `graphs_val.pt`, and `graphs_test.pt` (binary; tens of GB).

The `graphs_*.pt` files are single PyTorch files that holds the following triplet for every ligand processed in round 0:
1. **InChiKey** (attribute `inchi_key`): Immutable ID used throughout the pipeline.
2. **Graph Object**: Atoms/bonds and their features for the GNN.
3. **Label** (attribute `y`): 1 = binder, 0 = non-binder, –1 = masked/unknown.

The graph object will be a PyG `torch_geometric.data.Data` instance. The InChiKey will be the `Data.inchi_key` attribute that is a Python `str` data type. The Label will be the `Data.y` attribute that is a Python `int` data type that has been cast to a `torch.long` data type.

For the initial labels, we want to set all the seed positives as binders (1), and all others as non-binders (0) initially:
1. `seed_positives = set(train_pos.parquet.inchi_key)`
2. For every `mol_graph`, if `inchi_key` in `seed_positives`:
   1. `mol_graph.y = tensor([1])`
3. Else:
   1. `mol_graph.y = tensor([0])`

We will also do the same for `graphs_val.pt`, and `graphs_test.pt`, but their labels will remain frozen for the entire training regime.

### 2.3. Create Cumulative Training Dataset (`graphs_master.pt`)
The `graphs_master.pt` file is a single PyTorch file that holds the same triplet for every ligand processed so far.

For the initial `graphs_master.pt` file we will simply make a copy of `graphs_round0_train.pt` (N.B. we do not include any ligands from the validation or test datasets. However, this will be the cumulative training dataset and at the end of each round we will append all the new ligand graphs used in that round (i.e. `graphs_round{r}_train.pt`), as well as update any newly calculated labels from that training round.

### 2.4. Create 3-D Conformers
Because fast VSX docking needs initial 3-D coordinates, we must convert each ligand to an ETKDG conformer.
1. Load SMILES from `cleaned_round0_full_train.smi.gz`.
2. Convert each SMILES string to an `rdkit.Mol` object via `rdkit.Chem.AllChem.EmbedMolecule`.
3. Energy-minimize with 200‑step MMFF or UFF minimization. This is a good enough minimization with minimal computational overhead.
4. Batch-write ligands to 50 SDF files, with 10,000 ligands per SDF file, and `gzip` each file. This format is preferred by Rosetta’s “multi-ligand” mode.
5. E.g. `batch_0001.sdf.gz`, `batch_0001.sdf.gz`, etc. (~20 MB each).

### 2.5. Cache Fingerprints (1024-Bit ECFP4)
By caching each molecule as a 1024-bit ECFP4, we can do fast similarity searches, clustering, and an optional FNN pre-screen.
1. Load SMILES from `cleaned_round0_full_train.smi.gz`.
2. Compute 1024-bit ECFP4 for each ligand and pack into NumPy `uint8` arrays via `fps = [np.packbits(AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(smiles) ,2,1024)) for smiles in round0_candidates]`.
3. Save as `round0_train.fp.npy`.
4. N.B. If we later want to compare chemical space coverage between train and test, we can also load `cleaned_full_val.smi.gz` and `cleaned_full_test.smi.gz` and save to `val.fp.npy` and `test.fp.npy` respectively.

### 2.6. Create “Seen” Hash-Set
We need to keep track of what training ligands the GNN will have seen after round 0. This includes all training molecules that have already been converted to graphs and ETKDG conformers.
**N.B.** To avoid data leakage we do not include validation or test ligands as they are not to be used in the active learning loop.
1. **Read the Training Ligand InChiKeys Just Processed:**
   1. Load InChiKeys from `cleaned_round0_full_train.smi.gz`.
2. **Save InChiKeys to Disk:**
   1. Insert every key into an in-memory `set()` and then serialize it to disk by saving to `seen_inchikeys.pkl.gz`.
  
## 3. Surrogate GNN Model Training & Tuning
At a high-level, our active learning technique can be broken down into:
1. **Surrogate GNN Model Training & Tuning**
   1. The surrogate GNN model is continuously retrained with ~500,000 molecules/ligands at the start of each full training iteration.
2. **Active Learning Loop (VSX-Derived Labels)**
   1. The GNN model that just finished training is then used in the active learning loop to pick the next ~500,000 molecules → Dock them with VSX → Calculate each ligand’s ΔG → Turn the ΔG scores into new labels (binders=1, non-binders=0, masked=-1).
  
For the Surrogate GNN Model Training & Tuning, round 0 will differ from subsequent rounds:
1. **Round 0:**
   1. We first train the GNN (GNN0) using the graphs from `graphs_round0_train.pt`.
   2. As per step 2.2:
      1. The seed positives (i.e. the “high-fidelity actives” from step 1.1) are given the label 1 as they are known binders.
      2. The centroid molecules are assumed to be non-binders and are given the label 0.
2. **Round k (≥ 1):**
   1. We retrain the GNN and fine-tune using everything that has been labelled from the previous round’s active learning loop.
  
### 3.1. Model Choice (Round 0 Only)
1. Can use Direct-MPNN from Chemprop, use 3-4 message-passing steps, and a hidden size of 300.
2. The Pytorch-Geometric equivalent is `GINConv` or `GATv2Conv`.

### 3.2. Bayesian/Monte-Carlo Hyperparameter Search (Round 0 Only)
1. Prepare a stratified 80/20 split of the labelled set `graphs_round0_train.pt`.
2. Define initial hyperparameters as:
   1. `hidden_size` = 300
   2. `message_depth` = 3
   3. `dropout` = 0.2
   4. `learning_rate` = 3e-4
   5. `weight_decay` = 1e-6
3. Use a small Optuna or Ray Tune search to find the Area Under the Receiver Operating Characteristic curve (AUROC) on the 20% validation slice (binary 1 v 0), with a TPE or BOHB sampler.
4. Do ~25 trails, with each trial training for 10 epochs with early stopping.
5. Pick best AUROC trial and dump its parameters (hidden size, message depth, dropout, learning rate, weight decay) to `best_hyperparams.json`.
6. Feed these `best_hyperparams.json` values into the training regime for round 0 and disable further hyperparameter searching. Only the weights are to evolve in later rounds, not the hyperparameters or model architecture.

### 3.3. Training/Fine-Tuning
In a molecule-focused GNN, the encoder is a stack of message-passing layers. Each layer:
1. Collect information from every atom’s neighbours.
2. Merges (e.g. sum/mean/attention) those messages.
3. Updates the atom’s hidden vector.
   
After `k` rounds every atom’s vector encodes information from atoms that are up to `k` bonds away.

In its final layer, the encoder pools all atom vectors into one fixed-length molecule vector, usually by a simple sum or mean. The output head is the final classification or regression function that sits on top of this final encoder layer. It takes the single fixed-length molecule vector as input.

In most GNN libraries, this output head is just a two-layer MLP that turns this final vector into either:
  * A single sigmoid score (in our case binder/non-binder).
  * A single linear value (e.g. Predicted ΔG, Kd etc.).

### 3.3.1. Round 0
The training dataset is `graphs_round0_train.pt` and the general steps are:
1. Initial Stage:
   1. Create a brand-new model with the hyperparameters chosen in Step 3.3.
2. Epoch Loop Stage:
   1. Run for 20-30 epochs, with a minibatch shuffle.
   2. All weights (message-passing and output head) to be updated with AdamW.
   3. Early-stop when validation AUROC hasn’t improved for 5 epochs.
3. External Validation:
   1. Run the freshly trained model on `graphs_val.pt`.
   2. Compute AUROC, PR-AUC, accuracy, etc.
   3. Log to `val_metrics_round{r}.json` and TensorBoard.
4. Save Stage:
   1. Save round 0 model full checkpoint (encoder + head) to `gnn_round0.pt`.
  
### 3.3.2. Rounds ≥ 1 — “Light Fine-Tune”
As the encoder has already learned useful chemistry, we mainly want to let the output head adjust to the new labels produced by the latest VSX batch without overfitting.

The training dataset is `graphs_round{r}_train.pt` and the general steps are:
1. **Load Previous Weights**
2. **Freeze Encoder:**
   1. If we have plenty of GPU and want a bit more flexibility, we could “thaw” the
encoder at a lower LR (e.g. 0.1x) instead of a full freeze.
3. **Optimizer:**
   1. Once we freeze the encoder, we will have two kinds of weights in the model:
      1. Frozen Encoder Weights: These have `param.requires_grad = False`, and so PyTorch will not calculate gradients for them.
      2. Still Trainable Output Head Weights: The few dense layers (often a 2-layer MLP) that sit on top of the graph’s encoder/message layers. These have `param.requires_grad = True`.
   2. We must build the optimizer so that it only updates the weights that are still trainable. A common way to do this is `optim = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr = 2 * BASE_LR)`.
   3. Note that because the list is small (maybe 20 k weights instead of 1 M+) we can raise the learning rate a little—often double the rate you used when the whole network was trainable. The higher LR helps the output head adapt quickly to the new labels without risking instability in the frozen encoder.
4. **Epochs:**
   1. 2-3 epochs are usually enough. We will need to monitor validation AUROC—if it jumps quickly then plateaus, stop.
5. **External Validation:**
   1. Run the freshly trained model on `graphs_val.pt`.
   2. Compute AUROC, PR-AUC, accuracy, etc.
   3. Log to `val_metrics_round{r}.json` and TensorBoard.
6. **External Test (If End of Training):**
   1. Run the freshly trained model on `graphs_test.pt`.
   2. Compute AUROC, PR-AUC, accuracy, etc.
   3. Log to `test_metrics.json` and TensorBoard.
7. **Save:**
   1. Save round `r` model full checkpoint (encoder + head) to `gnn_round{r}.pt`.
  
### 3.4. Inference
For each round, once the training stage finishes, we run production-quality scoring/inference to generate fresh scores that the active learning loop will use to pick the next 0.5M candidates.

This is the same routine that will be used once the active learning loop has converged and model is fully trained and is ready to be used to score a commercial-sized library (e.g. hundreds of millions of compounds).

The general steps are:
1. Mini-Batch the Library:
   1. This will feed the huge ligand list to the GPU in bite-size chunks.
   2. Only use centroid ligands that have not been docked yet (or the whole set if is the first round). I.E. Use `round{r}_unsampled_centroid_pool.smi`.
   3. Mini-batch sizes should be 4096 molecules. This is small enough to fit in GPU memory but large enough for speed.
2. Forward Pass Without Gradients:
   1. Ask the trained model to predict binding scores; no training, so we disable gradient bookkeeping.
3. Write Results to Disk
   1. Combine each SMILES with their predicted scores and save to `scores_round{r}.csv` for round `r`.
  
It should be noted that inference will be running while VSX docking from the previous round is still running on CPU. Therefore, GPU and CPU work will overlap, and GPU inference is not the bottleneck; VSX docking is.

## 4. Active Learning Loop (VSX-Derived Labels)
