# AiDockFlow

# Abstract
Structure-based virtual screening of ultra-large chemical libraries is a powerful strategy in early-stage drug discovery. However, the computational expense of physics-based docking necessitates more efficient methods for navigating chemical space. Building upon the active learning framework proposed by *Zhou et al.* in their work "An artificial intelligence accelerated virtual screening platform for drug discovery", **<sup>1</sup>** we present AiDockFlow, a fully-automated pipeline designed to improve the efficiency and robustness of hit discovery campaigns.

Our methodology introduces several key enhancements to the data curation and model training stages. The process begins by curating a high-fidelity set of known actives for a given target, which are partitioned using a scaffold-based split to ensure robust training and validation sets. This is coupled with the generation of a diverse "druglike-centroid" library from ZINC15 to facilitate broad exploration of chemical space, as per the approach proposed by *Zhou et al.*. These high-fidelity actives are labelled as 'binders' and are used for initial training of the surrogate Graph Neural Network (GNN), along with a large sample of the diverse centroids (labelled as 'non-binders').

Another core contribution of our work is to refine the active learning loop where a Graph Neural Network (GNN) surrogate is iteratively retrained. We introduce a three-tiered labelling strategy based on RosetteVS docking scores where the top 10% of docked compounds are labelled as "binders", the bottom 10% are labelled as "non-binders", and the ambiguous middle 50% are ignored to ensure the model is trained only on high-confidence data. Additionally, for reduced computational expense, we replace the cumulative training dataset approach with a "replay buffer" (a smaller, random sample of previously seen compounds) to prevent catastrophic forgetting of model learning. Our final contribution is the addition of PAINS and Brenk filters in addition to the suite of medicinal chemistry filters proposed by *Zhou et al.*

# Contents
<details>
   <summary><b>1. Data Ingestion & Curation</b></summary>
   
   &nbsp; &nbsp; &nbsp; &nbsp; 1.1 [Retrieve High-Fidelity Actives](https://github.com/c-vandenberg/aidockflow?tab=readme-ov-file#11-retrieve-high-fidelity-actives)<br>
   &nbsp; &nbsp; &nbsp; &nbsp; 1.2 [Split High-Fidelity Actives Data](https://github.com/c-vandenberg/aidockflow?tab=readme-ov-file#12-split-high-fidelity-actives-data)<br>
   &nbsp; &nbsp; &nbsp; &nbsp; 1.3 [Build “Druglike-Centroid Library”](https://github.com/c-vandenberg/aidockflow?tab=readme-ov-file#13-build-druglike-centroid-library)<br>
   &nbsp; &nbsp; &nbsp; &nbsp; 1.4 [Create “round-0” Candidate Training Dataset, & Validation/Testing Datasets](https://github.com/c-vandenberg/aidockflow?tab=readme-ov-file#14-create-round-0-candidate-training-dataset--validationtesting-datasets)<br>
   &nbsp; &nbsp; &nbsp; &nbsp; 1.5 [Prepare Target Structure](https://github.com/c-vandenberg/aidockflow?tab=readme-ov-file#15-prepare-target-structure)<br>
</details>
   
<details>
   <summary><b>2. Featurization (Round 0 Initially)</b></summary>
   
   &nbsp; &nbsp; &nbsp; &nbsp; 2.1 [Canonicalize Ligands and Add Explicit Hydrogens](https://github.com/c-vandenberg/aidockflow?tab=readme-ov-file#21-canonicalize-ligands-and-add-explicit-hydrogens)<br>
   &nbsp; &nbsp; &nbsp; &nbsp; 2.2 [Generate Graph Objects & Attach Initial Labels](https://github.com/c-vandenberg/aidockflow?tab=readme-ov-file#22-generate-graph-objects--attach-initial-labels)<br>
   &nbsp; &nbsp; &nbsp; &nbsp; 2.3 [Create Cumulative Training Dataset (`graphs_master.pt`)](https://github.com/c-vandenberg/aidockflow?tab=readme-ov-file#23-create-cumulative-training-dataset-graphs_masterpt)<br>
   &nbsp; &nbsp; &nbsp; &nbsp; 2.4 [Create 3-D Conformers](https://github.com/c-vandenberg/aidockflow?tab=readme-ov-file#24-create-3-d-conformers)<br>
   &nbsp; &nbsp; &nbsp; &nbsp; 2.5 [Cache Fingerprints (1024-Bit ECFP4)](https://github.com/c-vandenberg/aidockflow?tab=readme-ov-file#25-cache-fingerprints-1024-bit-ecfp4)<br>
   &nbsp; &nbsp; &nbsp; &nbsp; 2.4 [Create “Seen” Hash-Set](https://github.com/c-vandenberg/aidockflow?tab=readme-ov-file#26-create-seen-hash-set)<br>
</details>

<details>
   <summary><b>3. Surrogate GNN Model Training & Tuning</b></summary>
   
   &nbsp; &nbsp; &nbsp; &nbsp; 3.1 [Model Choice (Round 0 Only)](https://github.com/c-vandenberg/aidockflow?tab=readme-ov-file#31-model-choice-round-0-only)<br>
   &nbsp; &nbsp; &nbsp; &nbsp; 3.2 [Bayesian/Monte-Carlo Hyperparameter Search (Round 0 Only)](https://github.com/c-vandenberg/aidockflow?tab=readme-ov-file#32-bayesianmonte-carlo-hyperparameter-search-round-0-only)<br>
   &nbsp; &nbsp; &nbsp; &nbsp; 3.3 [Training/Fine-Tuning](https://github.com/c-vandenberg/aidockflow?tab=readme-ov-file#33-trainingfine-tuning)<br>
   &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; 3.3.1 [Round 0](https://github.com/c-vandenberg/aidockflow?tab=readme-ov-file#331-round-0)<br>
   &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; 3.3.2 [Rounds ≥ 1 — “Light Fine-Tune”](https://github.com/c-vandenberg/aidockflow?tab=readme-ov-file#332-rounds--1--light-fine-tune)<br>
   &nbsp; &nbsp; &nbsp; &nbsp; 3.4 [Inference](https://github.com/c-vandenberg/aidockflow?tab=readme-ov-file#34-inference)<br>
</details>

<details>
   <summary><b>4. Active Learning Loop (VSX-Derived Labels)</b></summary>
   
   &nbsp; &nbsp; &nbsp; &nbsp; 4.1 [Candidate Selection](https://github.com/c-vandenberg/aidockflow?tab=readme-ov-file#41-candidate-selection)<br>
   &nbsp; &nbsp; &nbsp; &nbsp; 4.2 [Featurise Previously Unseen Ligands (Canonicalize, Generate Graph Object, Create 3-D Conformers, Cache Fingerprints & Update “Seen” Hash-Set)](https://github.com/c-vandenberg/aidockflow?tab=readme-ov-file#42-featurise-previously-unseen-ligands-canonicalize-generate-graph-object-create-3-d-conformers-cache-fingerprints--update-seen-hash-set)<br>
   &nbsp; &nbsp; &nbsp; &nbsp; 4.3 [VSX Docking (Fast Rosetta Run)](https://github.com/c-vandenberg/aidockflow?tab=readme-ov-file#43-vsx-docking-fast-rosetta-run)<br>
   &nbsp; &nbsp; &nbsp; &nbsp; 4.4 [Calculate ΔG & Add Labels](https://github.com/c-vandenberg/aidockflow?tab=readme-ov-file#44-calculate-%CE%B4g--add-labels)<br>
   &nbsp; &nbsp; &nbsp; &nbsp; 4.5 [Add Labels to Cumulative Training Pool](https://github.com/c-vandenberg/aidockflow?tab=readme-ov-file#45-add-labels-to-cumulative-training-pool)<br>
   &nbsp; &nbsp; &nbsp; &nbsp; 4.6 [Convergence Check](https://github.com/c-vandenberg/aidockflow?tab=readme-ov-file#46-convergence-check)<br>
   &nbsp; &nbsp; &nbsp; &nbsp; 4.7 [Build Next Round Training Dataset (If Not Converged)](https://github.com/c-vandenberg/aidockflow?tab=readme-ov-file#47-build-next-round-training-dataset-if-not-converged)<br>
</details>

<details>
   <summary><b>5. High-Precision Docking & Post-Processing</b></summary>
   
   &nbsp; &nbsp; &nbsp; &nbsp; 5.1 [Candidate List](https://github.com/c-vandenberg/aidockflow?tab=readme-ov-file#51-candidate-list)<br>
   &nbsp; &nbsp; &nbsp; &nbsp; 5.2 [VSH Docking](https://github.com/c-vandenberg/aidockflow?tab=readme-ov-file#52-vsh-docking)<br>
   &nbsp; &nbsp; &nbsp; &nbsp; 5.3 [Medicinal Chemistry Filters](https://github.com/c-vandenberg/aidockflow?tab=readme-ov-file#53-medicinal-chemistry-filters)<br>
   &nbsp; &nbsp; &nbsp; &nbsp; 5.4 [Clustering & Diversity](https://github.com/c-vandenberg/aidockflow?tab=readme-ov-file#54-clustering--diversity)<br>
   &nbsp; &nbsp; &nbsp; &nbsp; 5.5 [Exports](https://github.com/c-vandenberg/aidockflow?tab=readme-ov-file#55-exports)<br>
</details>

<details>
   <summary><b>6. Visualization & Reporting</b></summary>
   
</details>


# Protocol
## 1. Data Ingestion & Curation
### 1.1. Retrieve High-Fidelity Actives
1. **Target** to be defined by **Uniprot Accession Number** in training config file.
2. Gather **SMILES** and **bioactivity measure values** for compounds that are **known binders** to the target from public sources (**ChEMBL**, **PubChem**, etc.) via Python APIs or bulk downloads (SDF/CSV).
3. Likely parameters to use to determine known binders include **inhibition constant** (**Ki**) and **dissociation constant** (**Kd**). Set **threshold value** (in **nM**) for binding parameter. These will be defined in training config file.
4. When extracting compounds, keep **InChIKey**, **binding parameter** (**Ki** or **Kd**), and **SMILES string**.
5. If a compound appears **multiple times** keep the **lowest value** and record **count**, **mean**, **median**, and **σ**.
6. **Merge** compounds from each public source and **deduplicate based on InChIKey**.
7. **Clean** and **normalize/standardize** molecules:
    * Remove **salts**.
    * **Protonation/tautomer** standardization at **pH 7.4** with `dimorphite-dl`.
    * Normalize binding parameter to **nM**.
    * **Calculate InChIKey** if not present in public sources.
    * **Canonicalize SMILES** with RDKit.
8. Save compounds as `validated_actives.parquet` file:
    * **Apache Parquet** is an open source, column-oriented data file format designed for efficient data storage and retrieval.
    * Parquet table (**columnar, compressed**) will have the following schema:
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
1. Do **random stratified split** 70/15/15% (train, validation, test). Exact split ratio to be defined in training config file:
    * Group actives by **Bemis‑Murcko scaffold** (`rdMolDescriptors.GetScaffoldForMol`).
    * **Shuffle scaffolds** with **fixed RNG seed**.
    * Allocate 70 % of scaffolds to train, 15 % to validation, 15 % to test.
2. Write three Parquet files with same schema as before - `train_pos.parquet`, `val_pos.parquet`, `test_pos.parquet`.

### 1.3. Build “Druglike-Centroid Library”
Based on the paper by Zhou et al., we will build a **specialized subset of ~13 million molecules**, referred to as the **druglike-centroid library**. These will be extracted from the **ZINC15 3D druglike database** (~493.6 million molecules) by:
1. **Clustering similar molecules** from the ZINC 3D druglike database, using a **cutoff of 0.6 Tanimoto similarity** (defined in training config file).
2. From each cluster, the **smallest molecule**will be selected and added to the library, serving as the **centroid of the cluster**.
3. This process will result in the formation of the **druglike-centroid library**, which will contain **~13 million molecules**.

The purpose of creating the druglike-centroid library is to ensure that the model is exposed to a **wide range of chemical space** during the initial iteration.
The steps for this include:
1. **Download** ZINC15 drug-like 3D SMILES dataset.
2. **Generate 1024‑bit ECFP4 fingerprints** in streamed fashion (RDKit).
3. **Cluster** with Tanimoto 0.6 (defined in training config file) using **Butina‑like algorithm** (Faiss for speed).
4. **Remove** any centroid whose Tanimoto similarity to any of the high-fidelity actives is **1.0** (i.e. they are the same molecule).
5. **Calculate InChIKey** for each molecule and save druglike-centroid library to `centroid_pool.parquet` the following schema:
   1. `inchi_key` (str)
   2. `smiles` (str)
  
### 1.4. Create “round-0” Candidate Training Dataset, & Validation/Testing Datasets
In the training workflow from Zhou et al., for the first iteration, **0.5 million** and **1 million** molecules were randomly selected from the centroid library as the **training** and **testing datasets** respectively.

We will slightly alter this to sample 0.5 million molecules, followed by a **random stratified 70/15/15% split** (with both the sample and split to be tunable in the config file). The thinking is:
1. **Training Dataset:**
   1. We do not want to dilute the high-fidelity Ki/Kd positives in the training dataset too much, but at the same time dilute them enough with negatives from the `centroid_pool.parquet` pool so that the model learns from broad chemical space.
   2. Therefore, ~350,000 negatives from the `centroid_pool.parquet` pool should be a good starting point.
2. **Validation/Testing Dataset:**
   1. We will start with ~75,000 negatives in both the validation and testing datasets.
   2. It is hoped that ~75,000 negatives, plus the positives in each will be enough to get a stable AUROC estimate.
  

The steps to create the round 0 training dataset, and the validation/testing datasets are:
1. Randomly sample 500,000 compounds (defined in training config file) from `centroid_pool.parquet` (**exploration**) and save to `round0_sampled_centroid_pool.parquet`.
2. Save **unsampled pool** to `round0_unsampled_centroid_pool.parquet`. This will be used in **Step 4.1.** to sample **250,000 undocked ligands** for **active learning loop** at the end of round 0.
3. Do a **random stratified 70/15/15% training/validation/testing split** of `round0_sampled_centroid_pool.parquet`, saving to `train_neg.parquet`, `val_neg.parquet`, and `test_neg.parquet`.
4. **Append** all positives from `train_pos.parquet`, `val_pos.parquet`, and `test_pos.parquet` to the respective negative files from step 3 (**exploitation**).
5. We should now have `round0_full_train.parquet`, `full_val.parquet`, and `full_test.parquet`.
6. **Deduplicate all three datasets based on InChiKeys**, shuffle and write to `.smi` files, followed by compression with gzip (e.g. `round0_full_train.smi`, then `round0_full_train.smi.gzip`).
7. **N.B.** To **prevent any data leakage** validation and test datasets:
   * Are to be frozen throughout the training regime. This means they should not be relabeled, nor should they be replaced with a new set of molecules.
   * Should not be added to `graphs_master.pt` (see later) and should **never be included in the active learning loop**.

A `.smi` file is **plain text, tab-separated** list of molecules:
  * **First column** – SMILES string
  * **Second column** – InchiKey

### 1.5. Prepare Target Structure
1. **Target** to be defined by **Uniprot Accession Number** in training config file.
2. Download **PDB** or **AlphaFold** model (target source database to be defined in training config file).
3. Use **PDBFixer** to add missing residues/atoms, assign bonds, and remove water molecules.
4. Use `propka` to add pH 7.4 protonation.
5. Use **OpenBabel** to add hydrogens.
6. **Energy-minimize side chains** with **Rosetta FastRelax** (2,000 cycles, coordinates constraints).
7. Convert to **PDBQT** with AutoDockTools script (`prepare_receptor4.py`).
8. Store copy of relaxed target in both **PDB** and **PDBQT** files (`target_prepped.pdb` and `target_prepped.pdbqt`).

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
### 4.1. Candidate Selection
1. For current round `r`, open `scores_round{r}.csv` from current round inference.
2. Sort by GNN score (high → low) and extract top 250,000 ligands.
3. Randomly sample 250,000 SMILES from `round{r}_unsampled_centroid_pool.smi.gz` and save unsampled SMILES to `round{r+1}_unsampled_centroid_pool.smi.gz`. This will be used to sample the next 250,000 SMILES in the next active learning iteration.
4. Combine top GNN-scored ligands from 2. and sample ligands from 3. to give 500,000 ligands for the upcoming VSX run.
5. Save to `round{r}_active_learning.smi.gz` for round `r`.

### 4.2. Featurise Previously Unseen Ligands (Canonicalize, Generate Graph Object, Create 3-D Conformers, Cache Fingerprints & Update “Seen” Hash-Set)
1. Read `round{r}_active_learning.smi.gz` and compare each InChiKey to the “seen” hash- set `seen_inchikeys.pkl.gz`.
2. Create `graphs_round{r}_train.pt` file.
3. For every new ligand:
   1. Clean/canonicalize SMILES as per Step 2.1.
   2. Build graph object as per Step 2.2., append to both `graphs_round{r+1}_train.pt` and `graphs_master.pt`.
   3. Generate ETKDG conformer as per Step 2.3. and collect into the SDF batches.
   4. Compute and store ECFP4 fingerprint as per Step 2.4.
   5. Append InChiKey to `seen_inchikeys.pkl.gz`.
  
### 4.3. VSX Docking (Fast Rosetta Run)
1. Convert each SMILES in `round{r}_candidates.smi.gz` to a 3D SDF as per Step 2.3.
2. Bundle the ligands into 50 SDF files of 10,000 ligands each. This format is preferred by Rosetta’s “multi-ligand” mode.
3. On the cluster: submit 50 array jobs, each job uses for example 20 CPU cores. Rosetta’s VSX protocol (vsx.xml) docks one ligand in about 2 ½ minutes on one core → ~25 ligands / core / hour.
4. Rosetta writes a `.sc` score table for every SDF batch (one row per ligand, ΔG in kcal mol⁻¹). This will give many `score_XXXXX.sc` tables.

### 4.4. Calculate ΔG & Add Labels
1. Merge the 50 score tables into one spreadsheet (ligand ID + ΔG).
2. Sort by ΔG (more negative = better):
   1. **Top 10% (50,000 ligands)**** → Label 1 (“binder”).
   2. **Bottom 40% (200,000 ligands)** → Label 0 (“non-binder”).
   3. **Middle 50% (250,000 ligands)** → Label -1 (“ignore”).
   4. **N.B.** We could make the top 10% cutoff tunable or dynamic, like the Zhou et al. paper where the cutoff becomes stricter over time. For example, we could start at 10% and gradually decrease it to 5% or 1% in later rounds to focus the model on identifying only the most potent binders as the search progresses.
3. **BUT**, if any ligand is in the original high fidelity Kd dataset (`train_pos.parquet`), ensure it stays label 1 even if its ΔG isn’t in the top 10%.
4. Save the table as CSV file `round{r}_active_learning_labels.csv` with four columns:
   1. **InChiKey**
   2. **SMILES**
   3. **ΔG**
   4. **Label**
  
### 4.5. Add Labels to Cumulative Training Pool
1. For every ligand in `round{r}_active_learning_labels.csv` locate its graph object in `graphs_master.pt`.
2. Attach the integer label (1/0/-1) to that graph.

### 4.6. Convergence Check
To check for convergence, we will be checking three measures:
1. **Surrogate GNN Best Percentile Score:** This reflects how much the surrogate GNN still believes there are unexplored “good” regions after this round.
2. **VSX ΔG Best Percentile:** This tells us whether the highly scored GNN compounds for this round survive the first physics filter.
3. **Round Limit:** Predefined maximum round limit.

<br>
  <div align="center">
    <table>
    <thead>
      <tr>
        <th>Metric / Check</th>
        <th>How to Measure</th>
        <th>Why It Matters (Rationale)</th>
        <th>Action / Output (After Each Round)</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td><strong>Surrogate GNN Best Percentile Score</strong> (Early Enrichment)</td>
        <td>
          <ol>
            <li>Take <code>scores_round{r}.csv</code> (all remaining centroids).</li>
            <li>Sort descending by the GNN probability (sigmoid output).</li>
            <li>Grab the top 0.1% (e.g., 10,000 ligands if 10M are left).</li>
            <li>Record their mean and median score.</li>
          </ol>
        </td>
        <td>If the model keeps discovering higher-scoring ligands, the search space still has fertile regions.</td>
        <td>Calculated from <code>scores_round{r}.csv</code> and used in the stopping rule below.</td>
      </tr>
      <tr>
        <td><strong>Artefact Stopping Rule</strong> (Tunable in Config File)</td>
        <td>
            Compute Δ(mean-score) relative to the previous round.
            <br><br>
            <strong>Stop criterion:</strong> <code>abs(Δ) &lt; 0.01</code> (scores are 0-1) for two consecutive rounds.
        </td>
        <td>Provides an automatic stopping rule to terminate the search when enrichment plateaus.</td>
        <td>Append a results line to <code>progress.tsv</code>.</td>
      </tr>
      <tr>
        <td><strong>Fast-dock (VSX) Best Percentile</strong></td>
        <td>
          <ol>
            <li>Concatenate all <code>score_*.sc</code> files from the latest VSX job.</li>
            <li>Sort ascending by Rosetta ΔG (kcal mol<sup>-1</sup>, more negative is better).</li>
            <li>Take the top 0.1%.</li>
            <li>Record their mean and median ΔG.</li>
          </ol>
           <strong>Stop criterion:</strong> <code>abs(Δ) &lt; 0.1</code> kcal mol<sup>-1</sup> for two rounds.
        </td>
        <td>This is the first physics-based sanity check. If these numbers plateau, your "cheap" GNN search is no longer finding better physical binders.</td>
        <td>Append to <code>progress.tsv</code> and write a human-readable line to <code>train.log</code>.</td>
      </tr>
      <tr>
        <td><strong>Round Limit (Safety Brake)</strong></td>
        <td>
            Count the number of completed rounds.
            <br><br>
            <strong>Hard cap:</strong> E.g., 10 rounds.
        </td>
        <td>Prevents endless runs if metrics fluctuate around the stopping threshold without converging.</td>
        <td>Checked before starting a new round; logged in <code>progress.tsv</code>.</td>
      </tr>
    </tbody>
  </table>
  </div>
<br>

An example of how the `progress.tsv` file may look:
<br>
  <div align="center">
    <table>
    <thead>
      <tr>
        <th class="text-center">Round</th>
        <th><code>best_gnn_mean</code></th>
        <th><code>best_gnn_delta</code></th>
        <th><code>best_dG_mean</code></th>
        <th><code>best_dG_delta</code></th>
        <th><code>n_ligands_scored</code></th>
        <th>Comment</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td class="text-center">0</td>
        <td class="text-right">0.934</td>
        <td class="null-value">--</td>
        <td class="text-right">-10.7</td>
        <td class="null-value">--</td>
        <td class="text-right">5,000,000</td>
        <td>Initial</td>
      </tr>
      <tr>
        <td class="text-center">1</td>
        <td class="text-right">0.941</td>
        <td class="text-right">0.007</td>
        <td class="text-right">-11.1</td>
        <td class="text-right">-0.4</td>
        <td class="text-right">5,000,000</td>
        <td>Ok</td>
      </tr>
      <tr>
        <td class="text-center">2</td>
        <td class="text-right">0.942</td>
        <td class="text-right">0.001</td>
        <td class="text-right">-11.18</td>
        <td class="text-right">-0.08</td>
        <td class="text-right">5,000,000</td>
        <td>Plateau?</td>
      </tr>
      <tr>
        <td class="text-center">3</td>
        <td class="text-right">0.943</td>
        <td class="text-right">0.001</td>
        <td class="text-right">-11.22</td>
        <td class="text-right">-0.04</td>
        <td class="text-right">5,000,000</td>
        <td>Stopping</td>
      </tr>
    </tbody>
  </table>
  </div>
<br>

### 4.7. Build Next Round Training Dataset (If Not Converged)
There are two things to consider when deciding how to curate the next rounds training dataset from the `graph_master.pt` cumulative dataset:
1. **Why not train using the whole `graph_master.pt` dataset?**
   1. After a few rounds `graph_master.pt` will contain millions of graphs, therefore using this full cumulative training dataset would explode GPU hours with diminishing returns.
   2. A lot of these graphs will be labelled –1 (masked) and therefore will only add noise, not signal.
   3. Active-learning theory assumes we focus on the newest informative labels while keeping a stable set of high-confidence positives/negatives.
2. **Why not train on a brand new 500k ligands?**
   1. This would result in the GNN model “forgetting” what it has learned about seed positives and strong negatives from previous rounds.b. As a result, validation metrics would bounce wildly because the label distribution would change on each new round.

Therefore, to build the next rounds training dataset (`graphs_round{r+1}_train.pt`), we will include:
1. **All Seed Positives:** All ligand graphs in `graph_master.pt` that are in the high-fidelity dataset `train_pos.parquet`.
2. **Latest VSX Labelled Batch:** All ligand graphs in that are in `round{r}_active_learning.smi`
3. **Replay Buffer:** A random 50 – 100k graphs from earlier rounds whose labels are not –1. This prevents forgetting and stabilizes AUROC. In the example below, this is a fixed 2% random sample, but we could make this tunable in the config file.

The psuedocode for this would look like:
```
master = torch.load('graphs_master.pt')
seed_pos_inchi = pd.read_parquet(‘train_pos.parquet`, columns=['inchkey’])
latest_active_learning_inchikeys = []

with gzip.open(‘round{r}_candidates.smi.gz’, ‘rt’) as file:
   for line in file:
      if not line.strip() # Skip blank lines
         continue
      inchikey = line.split(‘\t’, 1)[1].rstrip() # Second column
      latest_active_learning_inchikeys.append(inchikey)
      train_set = []

      for graph in master:
         if graph.inchi_key in seed_pos_inchi or graph.inchi_key in latest_active_learning_inchikeys:
            train_set.append(graph)
         elif random.random() < 0.02: # 2% Replay buffer:
            train_set.append(graph)torch.save(train_set, f' graphs_round{r+1}_train.pt )
```

## 5. High-Precision Docking & Post-Processing
### 5.1. Candidate List
1. Gather union of top‑0.5 % GNN‑ranked molecules across all rounds (~50 k‑100 k) and save to `final_candidates.smi`.

### 5.2. VSH Docking
1. Redock these top `final_candidates.smi` on a slower, higher‑quality docking mode.
2. For example, RosettaVS‑VSH with flexible side chains and a finer search.
3. Save these scores to `vsh_scores.csv`

### 5.3. Medicinal Chemistry Filters
We can then run various cheap cheminformatics quality filters and to remove any ligands that have poor druglikeness and those with pharmacologically undesirable substructures or potential toxicity.
For example:
1. **cLogP:**
   1. Those ligands with a calculated LogP (cLogP) that is too low (too hydrophillic) means that it will not be orally bioavailable and will not be able to pass through the lipid bilayers of target cells.
   2. Conversely, those ligands with a cLogP that is too high (too hydrophobic), means that it will accumulate in fatty tissues and lipid bilayers.
   3. For efficient transport, the drug must be hydrophobic enough to partition into the lipid bilayer, but not so hydrophobic, that once it is in the bilayer, it will not partition out again.
   4. Additionally, hydrophobicity plays a major role in determining where drugs are distributed within the body after absorption and, consequently, in how rapidly they are metabolized and excreted.
   5. We can therefore calculate the cLogP of each of these top candidates using the `rdkit.Chem.Crippen` module on the parent (desalted) structure, and remove those under a certain value (e.g. ≤ 3.5)
2. **Buried unsatisfied hydrogen-bond donors / acceptors:**
   1. A buried polar atom that cannot donate/accept H-bonds in the pocket is an enthalpic penalty and often predicts low potency.
   2. After docking we can therefore run PLIP (`plip -f complex.pdb -o …`) or Rosetta’s InterfaceAnalyzer mover to list buried unsats Parse per-ligand
   3. counts.
   4. If any ligand is below a certain threshold (e.g. ≤ 1 buried unsat per ligand), we can therefore remove it.
3. **Torsion outliers (CSD Torsion Library):**
   1. Unusual torsion/dihedral angles indicate strained conformations with high internal strain. This indicates instability and a synthetic difficulty.
   2. We can therefore use the `csd.analysis.TorsionAnalyser.analyse_molecule` method from the Mogul library from the Cambridge Structural Database (CSD) software suite, accessible by the CSD Python API
      1. Export the ligand from the docked pose as SDF and load the ligand using `ligand = sd.io.MoleculeReader('path/to/your/ligand.sdf')[0]`.
      2. The core of this analysis is the `csd.analysis.TorsionAnalyser`. This tool uses data from the Mogul library, which is derived from the CSD, to assess the likelihood of observed torsion angles.
      3. Running `analysed_torsions = csd.analysis.TorsionAnalyser.analyse_molecule` will give the `analysed_torsions` object containing a list of all the rotatable bond torsions analyzed as `Torsion` objects, each having a `classification` attribute.
      4. This `classification` attribute can have values such as “Common”, “Infrequent”, “Unusual” (or “Allowed”, “Outlier”, “Unknown”).
      5. We can then count the number of “Unusual” or “Outlier” for a given ligand, and if it is above a certain threshold, we can drop it.
4. **PAINS Filter:**
   1. **Pan-assay interference compounds (PAINS)** are chemical compounds that often give false positive results in high-throughput screens.
   2. PAINS tend to react nonspecifically with numerous biological targets rather than specifically affecting one desired target, and several disruptive functional groups are shared by many PAINS.
   3. We can use RDKit’s built-in SMARTS catalogues to filter out any ligands with PAINS substructures:
      1. For each ligand SMILES, convert to `Mol` object via `rdkit.Chem.MolFromSmiles(ligand)`
      2. Initialise RDKit PAINS filter catalogue via `params = rdkit.Chem.FilterCatalog.FilterCatalogParams()` and `params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS)`
      3. Then, create the `FilterCatalog` object via `catalog = FilterCatalog(params)`.
      4. We then iterate through each of our ligand `Mol` objects, passing them to `matches = catalog.GetMatches(mol)`. If `matches` is not `None`, a PAINS alert is fired.
      5. We can then drop that ligand, and log the details:
         ```
         rejected_alerts = []
         for match in matches:
            Alert_name = match.GetDescription()
            Smiles = Chem.MolToSmiles(mol)
            rejected_alerts.append({‘SMILES’: smiles, ‘Alert’: alert_name})
         print(f"Alert fired for molecule {SMILES}: {alert_name}")
         ```
5. **Brenk Filter:**
   1. The Brenk filter comes from the paper "Lessons Learnt from Assembling Screening Libraries for Drug Discovery for Neglected Diseases" R. Brenk et al.” and removes molecules containing substructures with undesirable pharmacokinetics or toxicity.
   2. To implement this, we can simply add the Brenk filter to the RDKit filter catalogue in step ii when implementing the PAINS filter: `params.AddCatalog(FilterCatalogParams.FilterCatalogs.BRENK`.
  
### 5.4. Clustering & Diversity
**Final-Stage Clustering: From Exploration to Exploitation**
Throughout this pipeline, the strategic goal has evolved:
1. **Initial Stage (Exploration):**
   1. When we created the Druglike centroid library, the goal was maximum exploration. We started with ~494 million molecules and clustered them to get ~13 million diverse representatives.
   2. The purpose was explicitly to ensure that the model was exposed to a wide range of chemical space.
2. **Final Stage (Exploitation & Efficiency):**
   1. After the active learning loop converges and we have run high-precision VSH docking, we have a list of potentially 50,000-100,000 top-ranked compounds. It is financially and logistically impossible to synthesize and test all of them.
   2. Therefore, the goal shifts to exploitation. We now must select a manageable number (e.g., 500 – 2,000) that has the highest probability of success.

However, when reducing the top-ranked candidates we must balance potency with diversity. A method to do this is to apply Butina clustering at Tanimoto 0.6 and keep the ligand with best VSH ΔG at each cluster
This approach intelligently balances the two competing priorities:
1. **Diversity (The Clustering):**
    * If we simply took the top 500 compounds by VSH score, we might end up with 500 molecules that are slight variations of the same 5 or 6 chemical scaffolds.
    * This is an inefficient use of resources because they will likely have similar biological activity and properties (a concept known as an "activity cliff").
    * By clustering the compounds based on structural similarity (Tanimoto similarity), we group these variations together.
2. **Potency (The Selection):**
    * Within each structural cluster, we then select the single compound with the best (most negative) VSH ΔG score.
    * This member is considered the best representative of its chemical family.

By taking one representative from each cluster, we ensure that the final selection of 500 - 2,000 compounds is composed of structurally distinct molecules, each with the highest predicted potency in its class.

### 5.5. Exports

## 6. Visualization & Reporting
