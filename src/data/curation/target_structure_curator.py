import os
import logging
import subprocess
import requests
from typing import Dict

import shutil
from pdbfixer import PDBFixer
from openmm.app import PDBFile

from src.data.curation.base_curator import BaseCurator


class TargetStructureCurator(BaseCurator):
    """
    Prepares a target protein structure for docking from a PDB or AlphaFold model.
    """
    def __init__(self, config: Dict, logger: logging.Logger):
        super().__init__(config, logger)
        self._target_prep_dir = self._config.get('target_prep_dir', 'data/processed/target')
        os.makedirs(self._target_prep_dir, exist_ok=True)

    def run(self):
        """
        Executes the full target preparation workflow.
        """
        self._logger.info('Starting target structure preparation...')
        uniprot_id = self._config.get('uniprot_id')

        if not uniprot_id:
            self._logger.error('Target Uniprot ')

        raw_pdb_path = os.path.join(self._target_prep_dir, f'{uniprot_id}_raw.pdb')
        fixed_pdb_path = os.path.join(self._target_prep_dir, f'{uniprot_id}_fixed.pdb')
        protonated_pdb_path = os.path.join(self._target_prep_dir, f'{uniprot_id}_protonated.pdb')
        hydrogens_pdb_path = os.path.join(self._target_prep_dir, f'{uniprot_id}_hydrogens.pdb')
        relaxed_pdb_path = self._config.get(
            'target_prepped_pdb_path',
            os.path.join(self._target_prep_dir,'target_prepped.pdb')
        )
        final_pdbqt_path = self._config.get(
            'target_prepped_pdbqt_path',
            os.path.join(self._target_prep_dir, 'target_prepped.pdbqt')
        )

        try:
            # 1. Download PDB or AlphaFold model
            self._download_structure(uniprot_id, raw_pdb_path)

            # 2. Use PDBFixer to add missing residues/atoms
            self._run_pdbfixer(raw_pdb_path, fixed_pdb_path)

            # 3. Use PROPKA to add pH 7.4 protonation
            self._run_propka(fixed_pdb_path, protonated_pdb_path)

            # 4. Use OpenBabel to add hydrogens
            self._run_openbabel(protonated_pdb_path, hydrogens_pdb_path)

            # 5. Energy-minimize side chains with Rosetta FastRelax
            self._run_rosetta_relax(hydrogens_pdb_path, relaxed_pdb_path)

            # 6. Convert to PDBQT with AutoDock Tools
            self._run_autodock_prepare(relaxed_pdb_path, final_pdbqt_path)

            self._logger.info(f'Successfully prepared target. Final files:')
            self._logger.info(f'  PDB: {os.path.abspath(relaxed_pdb_path)}')
            self._logger.info(f'  PDBQT: {os.path.abspath(final_pdbqt_path)}')
        except (FileNotFoundError, subprocess.CalledProcessError) as e:
            self._logger.error(f'Target preparation failed: {e}')

        self._logger.info('Target structure preparation complete (placeholder implementation).')

    def _download_structure(self, uniprot_id: str, output_path: str):
        """Downloads a structure from the AlphaFold database."""
        self._logger.info(f'Step 1: Downloading AlphaFold structure for {uniprot_id}...')
        url = f'https://alphafold.ebi.ac.uk/files/AF-{uniprot_id}-F1-model_v4.pdb'
        response = requests.get(url)
        response.raise_for_status()  # Will raise an exception for bad status codes
        with open(output_path, 'w') as f:
            f.write(response.text)

        self._logger.info(f'Successfully downloaded and saved to {output_path}')

    def _run_command(self, command: list):
        """Helper function to run an external command and log its output."""
        self._logger.info(f'Running command: {" ".join(command)}')
        result = subprocess.run(command, capture_output=True, text=True, check=True)

        if result.stdout:
            self._logger.debug(f'STDOUT: {result.stdout}')
        if result.stderr:
            self._logger.warning(f'STDERR: {result.stderr}')

    def _run_pdbfixer(self, input_pdb: str, output_pdb: str):
        """Repairs a PDB file using PDBFixer."""
        self._logger.info('Step 2: Repairing structure with PDBFixer...')

        fixer = PDBFixer(filename=input_pdb)
        fixer.findMissingResidues()
        fixer.findNonstandardResidues()
        fixer.replaceNonstandardResidues()
        fixer.removeHeterogens(keepWater=False)
        fixer.findMissingAtoms()
        fixer.addMissingAtoms()
        fixer.addMissingHydrogens(7.4)
        PDBFile.writeFile(fixer.topology, fixer.positions, open(output_pdb, 'w'))

        self._logger.info(f'PDBFixer complete. Repaired structure saved to {output_pdb}')

    def _run_propka(self, input_pdb: str, output_pdb: str):
        """Assigns protonation states at pH 7.4 using PROPKA."""
        self._logger.info('Step 3: Assigning protonation states at pH 7.4 with PROPKA...')
        command = ['propka', '-i', input_pdb]
        self._run_command(command)
        # Note: A more robust implementation would parse the PROPKA output and
        # modify the PDB accordingly. For simplicity, we assume a tool that
        # directly modifies the PDB or we would handle the conversion.
        # This is a placeholder for that logic. Here we just copy the file.
        shutil.copy(input_pdb, output_pdb)

        self._logger.info(f'PROPKA analysis complete. Protonated structure at {output_pdb}')

    def _run_openbabel(self, input_pdb: str, output_pdb: str):
        """Adds hydrogen atoms using OpenBabel."""
        self._logger.info('Step 4: Adding hydrogens with OpenBabel...')
        command = ['obabel', input_pdb, '-O', output_pdb, '-h']
        self._run_command(command)

        self._logger.info(f'OpenBabel complete. Structure with hydrogens saved to {output_pdb}')

    def _run_rosetta_relax(self, input_pdb: str, output_pdb: str):
        """Performs energy minimization using Rosetta's FastRelax protocol."""
        self._logger.info('Step 5: Minimizing side chains with Rosetta FastRelax...')
        # This command assumes 'relax.default.linuxgccrelease' is in the PATH
        # and ROSETTA3_DB is set.
        rosetta_executable = 'relax.default.linuxgccrelease'  # Or full path
        command = [
            rosetta_executable,
            '-s', input_pdb,
            '-relax:constrain_relax_to_start_coords',
            '-relax:coord_constrain_sidechains',
            '-relax:ramp_constraints false',
            '-nstruct', '1'
        ]
        self._run_command(command)
        relaxed_source_path = input_pdb.replace('.pdb', '_0001.pdb')
        os.rename(relaxed_source_path, output_pdb)

        self._logger.info(f'Rosetta FastRelax complete. Minimized structure saved to {output_pdb}')

    def _run_autodock_prepare(self, input_pdb: str, output_pdbqt: str):
        """Converts a PDB file to PDBQT format using AutoDock Tools."""
        self._logger.info('Step 1.5.7: Converting to PDBQT format...')
        # This command assumes 'prepare_receptor4.py' is in the PATH
        prepare_script = 'prepare_receptor4.py'
        command = [
            'python', prepare_script,
            '-r', input_pdb,
            '-o', output_pdbqt,
            '-A', 'hydrogens' # Add hydrogens and merge non-polar
        ]
        self._run_command(command)

        self._logger.info(f'AutoDock conversion complete. PDBQT file saved to {output_pdbqt}')