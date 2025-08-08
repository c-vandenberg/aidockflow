import os
import logging
import subprocess

import requests
from typing import Dict, Tuple, Optional

from pdbfixer import PDBFixer
from openmm.app import PDBFile
from propka.run import single
from propka.molecular_container import MolecularContainer
from openbabel import openbabel

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
        self._logger.info('Starting target 3D structure preparation...')
        target_source = self._config.get('target_structure_database', 'AlphaFold').lower()

        if target_source == 'alphafold':
            target_id = self._config.get('uniprot_id')
            if not target_id:
                raise ValueError('Config error: "uniprot_id" is required when target_source is "AlphaFold".')
        elif target_source == 'pdb':
            target_id = self._config.get('pdb_id')
            if not target_id:
                raise ValueError('Config error: "pdb_id" is required when target_source is "PDB".')
        else:
            raise ValueError(f'Invalid target_source: "{target_source}". Must be "AlphaFold" or "PDB".')

        raw_pdb_path = os.path.join(self._target_prep_dir, f'{target_id}_raw.pdb')
        fixed_pdb_path = os.path.join(self._target_prep_dir, f'{target_id}_fixed.pdb')
        protonated_pdb_path = os.path.join(self._target_prep_dir, f'{target_id}_protonated.pdb')
        hydrogens_pdb_path = os.path.join(self._target_prep_dir, f'{target_id}_hydrogens.pdb')
        relaxed_pdb_path = os.path.join(self._target_prep_dir,f'{target_id}_relaxed.pdb')
        final_pdbqt_path = os.path.join(self._target_prep_dir, f'{target_id}_target_final.pdbqt')

        try:
            # 1. Download target 3D structure from AlphaFold DB
            #self._download_structure(target_id=target_id, structure_source=target_source, output_path=raw_pdb_path)

            # 2. Use PDBFixer to add missing residues/atoms
            #self._run_pdbfixer(raw_pdb_path, fixed_pdb_path)

            # 3. Use PROPKA to add pH 7.4 protonation
            #self._run_propka(fixed_pdb_path, protonated_pdb_path)

            # 4. Use OpenBabel to add hydrogens
            #self._run_openbabel(protonated_pdb_path, hydrogens_pdb_path)

            # 5. Energy-minimize side chains with Rosetta FastRelax
            self._run_rosetta_relax(hydrogens_pdb_path, relaxed_pdb_path)

            # 6. Convert to PDBQT with AutoDock Tools
            self._run_autodock_prepare(relaxed_pdb_path, final_pdbqt_path)

            self._logger.info(f'Successfully prepared target. Final files:')
            self._logger.info(f'  PDB: {os.path.abspath(relaxed_pdb_path)}')
            self._logger.info(f'  PDBQT: {os.path.abspath(final_pdbqt_path)}')
        except (FileNotFoundError, subprocess.CalledProcessError) as e:
            self._logger.error(f'Target preparation failed: {e}')
            return

        self._logger.info('Target structure preparation complete.')

    def _download_structure(self, target_id: str, structure_source: str, output_path: str):
        """
        Downloads the 3D structure of a target from either AlphaFold DB or the Protein Databank (PDB).

        Parameters
        ----------
        target_id : str
            The identifier for the structure (UniProt ID for AlphaFold, PDB ID for PDB).
        structure_source : str
            The source database, either 'alphafold' or 'pdb'.
        output_path : str
            The file path to save the downloaded PDB file.
        """
        self._logger.info(f'Step 1: Downloading target 3D structure for {target_id} from {structure_source.upper()}...')
        if structure_source.lower() == 'alphafold':
            url = f'https://alphafold.ebi.ac.uk/files/AF-{target_id}-F1-model_v4.pdb'
        else:
            url = f"https://files.rcsb.org/download/{target_id}.pdb"

        self._logger.info(f"Requesting URL: {url}")
        response = requests.get(url)
        response.raise_for_status()
        with open(output_path, 'w') as f:
            f.write(response.text)

        self._logger.info(f'Successfully downloaded and saved to {output_path}')

    def _run_command(self, command: list, stdout_path: Optional[str] = None):
        """Helper function to run an external command and log its output."""
        self._logger.info(f'Running command: `{" ".join(command)}`')

        if stdout_path:
            with open(stdout_path, 'w') as outfile:
                result = subprocess.run(command, stdout=outfile, stderr=subprocess.PIPE, text=True, check=True)
        else:
            result = subprocess.run(command, capture_output=True, text=True, check=True)
            if result.stdout:
                self._logger.debug(f'STDOUT: {result.stdout}')

        if result.stderr:
            self._logger.warning(f'STDERR: {result.stderr}')

    def _run_pdbfixer(self, input_pdb: str, output_pdb: str):
        """Repairs a PDB file using PDBFixer."""
        self._logger.info('Step 2: Repairing structure with PDBFixer...')

        pdb_fixer = PDBFixer(filename=input_pdb)
        pdb_fixer.findMissingResidues()
        pdb_fixer.findNonstandardResidues()
        pdb_fixer.replaceNonstandardResidues()
        pdb_fixer.removeHeterogens(keepWater=False)
        pdb_fixer.findMissingAtoms()
        pdb_fixer.addMissingAtoms()
        pdb_fixer.addMissingHydrogens(7.4)
        PDBFile.writeFile(pdb_fixer.topology, pdb_fixer.positions, open(output_pdb, 'w'))

        self._logger.info(f'PDBFixer complete. Repaired structure saved to {output_pdb}')

    def _run_propka(self, input_pdb: str, output_pdb: str):
        """Assigns protonation states at pH 7.4 using PROPKA."""
        self._logger.info('Step 3: Assigning protonation states at pH 7.4 with PROPKA...')

        # --- Step 1: Run PROPKA to generate the .pka file ---
        pka_file = os.path.splitext(input_pdb)[0] + '.pka'
        propka_mol = single(input_pdb, write_pka=False)
        propka_mol.calculate_pka()
        propka_mol.write_pka(pka_file)

        # --- Step 2: Parse the .pka file ---
        protonation_map = self._get_protonation_map_from_propka(propka_mol=propka_mol)

        # --- Step 3: Modify the `.pka` file with the new residue names ---
        self._logger.info(f"Applying new protonation states to create {output_pdb}...")
        with open(input_pdb, 'r') as infile, open(output_pdb, 'w') as outfile:
            for line in infile:
                if line.startswith("ATOM") or line.startswith("HETATM"):
                    chain_id = line[21]
                    res_num = int(line[22:26])

                    # Check if this residue needs to be renamed
                    if (chain_id, res_num) in protonation_map:
                        new_res_name = protonation_map[(chain_id, res_num)]
                        line = line[:17] + new_res_name.ljust(3) + line[20:]

                outfile.write(line)

        self._logger.info(f"PROPKA analysis and PDB modification complete. Protonated structure at {output_pdb}")

    def _run_openbabel(self, input_pdb: str, output_pdb: str):
        """Adds hydrogen atoms using OpenBabel."""
        self._logger.info('Step 4: Adding hydrogens with OpenBabel...')
        obconversion = openbabel.OBConversion()
        obconversion.SetInAndOutFormats('pdb', 'pdb')
        mol = openbabel.OBMol()
        obconversion.ReadFile(mol, input_pdb)
        mol.AddHydrogens()
        obconversion.WriteFile(mol, output_pdb)

        self._logger.info(f'OpenBabel complete. Structure with hydrogens saved to {output_pdb}')

    def _run_rosetta_relax(self, input_pdb: str, output_pdb: str):
        """Performs energy minimization using Rosetta's FastRelax protocol."""
        self._logger.info('Step 5: Minimizing side chains with Rosetta FastRelax...')
        # This command assumes 'relax.default.linuxgccrelease' is in the PATH
        # and ROSETTA3_DB is set.
        rosetta_executable = "/opt/rosetta.source.release-371/main/source/bin/relax.default.linuxgccrelease"

        # Check if the executable exists before trying to run it
        if not os.path.exists(rosetta_executable):
            raise FileNotFoundError(f"Rosetta executable not found at: {rosetta_executable}")

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

    def _get_protonation_map_from_propka(
        self,
        propka_mol: MolecularContainer,
        ph_val: float = 7.4
    ) -> Dict[Tuple[str, int], str]:
        """
        Parses a PROPKA `MolecularContainer` object to map residues to their
        correct protonation state.

        This function iterates through the groups identified by PROPKA and uses the
        calculated, environment-corrected pKa (`pka_value`) to determine the
        appropriate three-letter residue name at a given pH.

        Parameters
        ----------
        propka_mol : propka.molecular_container.MolecularContainer
            PROPKA `MolecularContainer` object for storing all contents of PDB files,
            returned by a single `propka.run.single()` call.
        ph_val : float
            The pH at which to determine the protonation state, by default 7.4.

        Returns
        -------
        Dict[Tuple[str, int], str]
            A dictionary mapping (chain_id, residue_number) to the new
            three-letter residue name (e.g., HID, HIE, ASP, GLU).
        """
        protonation_map = {}
        for conformer in propka_mol.conformations.values():
            for group in conformer.groups:
                if group.pka_value != 0.0:
                    res_name = group.atom.res_name
                    res_num = group.atom.res_num
                    chain_id = group.atom.chain_id
                    pka_val = group.pka_value

                    new_res_name = res_name
                    if res_name == "ASP" and pka_val > ph_val:
                        new_res_name = "ASH"  # Protonated Aspartic Acid
                    elif res_name == "GLU" and pka_val > ph_val:
                        new_res_name = "GLH"  # Protonated Glutamic Acid
                    elif res_name == "HIS":
                        if pka_val > ph_val:
                            new_res_name = "HIP"  # Positively charged Histidine
                        else:
                            new_res_name = "HIE"  # Neutral, proton on Epsilon

                    if new_res_name != res_name:
                        protonation_map[(chain_id, res_num)] = new_res_name

        self._logger.info(f"Parsed {len(protonation_map)} residue states from PROPKA results.")

        return protonation_map