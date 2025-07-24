import os
from typing import Dict
import logging

import pandas as pd
import numpy as np
import torch
import gzip
import pickle
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import AllChem, rdmolops
from torch_geometric.data import Data as PtGeometricData
from tqdm.auto import tqdm

from src.utils.file_utils import validate_file_extension


class CompoundDataProcessor:
    def __init__(
        self,
        data_config: Dict,
        logger: logging.Logger
    ):
        self._data_config = data_config
        self._logger = logger

    def add_stereochemistry_and_hydrogens(
        self,
        smiles_input_path: str,
        smiles_output_path: str,
        smiles_key: str,
        inchikey_key: str
    ):
        validate_file_extension(file_path=smiles_input_path, valid_file_ext='.parquet', logger=self._logger)

        os.makedirs(smiles_output_path, exist_ok=True)
        self._logger.info(
            f'Canonicalizing, assigning stereochemistry, and adding hydrogens to SMILES from {smiles_input_path}'
        )
        smiles_df = pd.read_parquet(smiles_input_path)

        processed_data = []
        for _, row in tqdm(smiles_df.iterrows(), total=smiles_df.shape[0], desc='Canonicalizing'):
            mol = Chem.MolFromSmiles(row[smiles_key])

            if mol is None:
                self._logger.warning(
                    f'Invalid SMILES. Skipped molecule - SMILES: {row[smiles_key]}, InChIKey: {row[inchikey_key]}'
                )

            rdmolops.AssignStereochemistry(mol=mol, cleanIt=True, force=True)
            mol_with_hs = Chem.AddHs(mol)
            canonical_smiles = Chem.MolToSmiles(mol_with_hs)
            canonical_inchikey = Chem.MolToInchiKey(mol_with_hs)

            processed_data.append({'smiles': canonical_smiles, 'inchikey': canonical_inchikey})

        pd.DataFrame(processed_data).to_parquet(smiles_output_path, index=False)
        self._logger.info(
            f'Saved canonicalized SMILES with stereochemistry and hydrogens to {smiles_output_path}'
        )

    def generate_graph_objects(
        self,
        smiles_path: str,
        seed_positives_path: str,
        graphs_output_path: str,
        smiles_key: str,
        inchikey_key: str
    ):
        validate_file_extension(file_path=smiles_path, valid_file_ext='.parquet', logger=self._logger)
        validate_file_extension(file_path=seed_positives_path, valid_file_ext='.parquet', logger=self._logger)

        os.makedirs(graphs_output_path, exist_ok=True)
        self._logger.info(
            f'Generating graph objects for {smiles_path}'
        )

        smiles_df = pd.read_parquet(smiles_path)
        seed_pos_df = pd.read_parquet(seed_positives_path)
        seed_pos_inchikeys = set(seed_pos_df[inchikey_key])

        mol_graph_list = []
        for _, row in tqdm(smiles_df.iterrows(), total=smiles_df.shape[0], desc='Generating Graphs'):
            mol = Chem.MolFromSmiles(row[smiles_key])

            if not mol:
                self._logger.warning(
                    f'Invalid SMILES. Skipped molecule - SMILES: {row[smiles_key]}, InChIKey: {row[inchikey_key]}'
                )

            # Define node (atom) features
            node_features = [
                [
                    atom.GetAtomicNum(), atom.GetDegree(), atom.GetFormalCharge, int(atom.GetIsAromatic()),
                    int(atom.GetHydridization()), int(atom.IsInRing())
                ]
                for atom in mol.GetAtoms()
            ]

            # Define edge (bond) features
            edge_indices, edge_attrs = [], []
            for bond in mol.GetBonds():
                bond_atom_a_idx, bond_atom_b_idx = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                edge_indices.extend(
                    [
                        [bond_atom_a_idx, bond_atom_b_idx],
                        [bond_atom_b_idx, bond_atom_a_idx]
                    ]
                )

            # Create the graph object
            mol_graph = PtGeometricData(
                x=torch.tensor(node_features, dtype=torch.float),
                edge_index=torch.tensor(edge_indices, dtype=torch.long).t().contiguous(),
                edge_attr=torch.tensor(edge_attrs, dtype=torch.float)
            )

            mol_graph.inchi_key = row[inchikey_key]
            mol_graph.y = torch.tensor([1] if row[inchikey_key] in seed_pos_inchikeys else [0], dtype=torch.long)

            mol_graph_list.append(mol_graph)

        torch.save(mol_graph_list, graphs_output_path)
        self._logger.info(
            f'Saved {len(mol_graph_list)} graph objects to {graphs_output_path}'
        )

    def generate_etkdg_3d_conformers(
        self,
        smiles_path: str,
        conformers_output_dir: str,
        smiles_key: str,
        inchikey_key: str,
        batch_size = 10000
    ):
        validate_file_extension(file_path=smiles_path, valid_file_ext='.parquet', logger=self._logger)

        os.makedirs(conformers_output_dir, exist_ok=True)
        self._logger.info(
            f'Generating ETKDG 3D conformers for {smiles_path}'
        )

        smiles_df = pd.read_parquet(smiles_path)
        writer = None
        file_count = 0
        num_batches = (len(smiles_df) - 1) // batch_size + 1

        with tqdm(total=len(smiles_df), desc='Generating ETKDG 3D Conformers') as progress_bar:
            for idx, row in smiles_df.iterrows():
                if idx % batch_size == 0:
                    if writer:
                        writer.close()
                    file_count += 1
                    sdf_path = f'{conformers_output_dir}/batch_{file_count:04d}.sdf.gzip'
                    writer = Chem.SDWriter(gzip.open(str(sdf_path), 'wt'))

                mol = Chem.MolFromSmiles(row[smiles_key])
                if not mol:
                    self._logger.warning(
                        f'Invalid SMILES. Skipped molecule - SMILES: {row[smiles_key]}, InChIKey: {row[inchikey_key]}'
                    )
                    progress_bar.update(1)

                # ETKDG conformer generation and minimization
                AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())
                AllChem.MMFFOptimizeMolecule(mol)

                # Add InChIKey as property for tracking
                mol.SetProp('InChIKey', row[inchikey_key])
                writer.write(mol)
                progress_bar.update(1)

        if writer:
            writer.close()

        self._logger.info(
            f'Saved {len(smiles_df)} conformers into {num_batches} SDF files in {conformers_output_dir}'
        )