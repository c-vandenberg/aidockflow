import requests
from abc import ABC, abstractmethod
from typing import List, Optional

from chembl_webresource_client.new_client import new_client
import pubchempy as pcp
from pubchempy import request

from src.data.utils.mappings import uniprot_to_gene_id_mapping


class BioactivityExtractorInterface(ABC):
    """
    Abstract base class for extracting bioactive compounds from a data source.

    Methods
    -------
    get_bioactive_compounds(target: str) -> List[str]
        Abstract method to return a list of canonical SMILES for bioactive compounds given a target UniProt ID
        identifier.
    """
    @abstractmethod
    def get_bioactive_compounds(self, target_uniprot_id: str) -> List[str]:
        """
        Retrieve a list of canonical SMILES for bioactive compounds for a given target.

        Parameters
        ----------
        target_uniprot_id : str
            The target identifier (UniProt accession, e.g. "P00533").

        Returns
        -------
        List[str]
            A list of canonical SMILES strings representing bioactive compounds.
        """
        pass

class ChEMBLExtractor(BioactivityExtractorInterface):
    """
    Extracts bioactive compounds for a given target from ChEMBL using a UniProt accession.

    Parameters
    ----------
    client : object, optional
        A ChEMBL client instance for dependency injection. If None, the default client from
        `chembl_webresource_client.new_client` will be used.
    bioactivity_threshold : float, optional
        The maximum standard_value (in nM) to consider a compound bioactive.
        Default is 1000 (i.e. compounds with IC50 <= 1 µM).
    bioactivity_measure : str, optional
        The bioactivity measurement type to filter on (e.g. "IC50"). Default is "IC50".

    Methods
    -------
    get_bioactive_compounds(target: str) -> List[str]
        Retrieves canonical SMILES for bioactive compounds for the given UniProt target.
    """
    def __init__(
        self,
        client = None,
        bioactivity_threshold: float = 1000, # for compounds with IC50 <= 1 µM
        bioactivity_measure: str = 'IC50'
    ):
        if client is None:
            self._client = new_client
        else:
            self._client = client
        self._bioactivity_threshold = bioactivity_threshold
        self._bioactivity_measure = bioactivity_measure


    def get_bioactive_compounds(self, target_uniprot_id: str) -> List[str]:
        """
        Retrieve canonical SMILES for bioactive compounds for a given target from ChEMBL.
        The target is provided as a UniProt accession (e.g., "P00533").

        Parameters
        ----------
        target_uniprot_id : str
            The UniProt accession for the target.

        Returns
        -------
        List[str]
            A list of canonical SMILES strings for compounds with activity measurements meeting
            the specified threshold.

        Raises
        ------
        ValueError
            If no matching target is found for the provided UniProt accession.
        """
        # Search for the target by name and retrieve the first matching result
        target_results = self._client.target.filter(target_components__accession=target_uniprot_id)
        if not target_results:
            raise ValueError(f"No matching target found for UniProt ID {target_uniprot_id}")
        target_data = target_results[0]
        target_id = target_data['target_chembl_id']

        # Filter activities based on the specified standard type (e.g. IC50)
        bioactive_compounds = self._client.activity.filter(
            target_chembl_id=target_id,
            standard_type=self._bioactivity_measure
        )
        bioactive_smiles: List = []
        for compound in bioactive_compounds:
            compound_smiles: str = compound.get('canonical_smiles')

            try:
                bioactivity_value = float(compound.get('standard_value', 0))
            except (ValueError, TypeError):
                bioactivity_value = None

            if compound_smiles and bioactivity_value is not None and bioactivity_value <= self._bioactivity_threshold:
                bioactive_smiles.append(compound_smiles)

        return bioactive_smiles


class PubChemExtractor(BioactivityExtractorInterface):
    """
    Extracts bioactive compounds for a given target from PubChem using a UniProt accession.

    For PubChem, the provided UniProt accession must first be mapped to an NCBI GeneID using a
    modified lookup that searches by protein accession.

    Parameters
    ----------
    bioactivity_threshold : float, optional
        The maximum potency (IC50 in nM) a compound must have to be considered bioactive.
        Default is 1000.

    Methods
    -------
    get_bioactive_compounds(target: str) -> List[str]
        Retrieves canonical SMILES for compounds from PubChem for the given UniProt target.
    """
    def __init__(
        self,
        bioactivity_threshold: float = 1000,  # for compounds with IC50 <= 1 µM
    ):
        self._bioactivity_threshold = bioactivity_threshold

    def get_bioactive_compounds(self, target_uniprot_id: str) -> List[str]:
        """
        Retrieve canonical SMILES for compounds for a given target from PubChem.
        The target is provided as a UniProt accession (e.g. "P00533").

        Parameters
        ----------
        target_uniprot_id : str
            The UniProt accession for the target.

        Returns
        -------
        List[str]
            A list of canonical SMILES strings for compounds matching the target UniProt accession.
        """
        target_gene_id = self._lookup_target_gene_id(target_uniprot_id)
        if not target_gene_id:
            print(f"Could not determine GeneID for target '{target_uniprot_id}'.")
            return []

        bioactive_compounds = pcp.get_compounds(target_gene_id, 'name')
        bioactive_smiles: List = []

        for compound in bioactive_compounds:
            compound_smiles = compound.canonical_smiles
            if not compound_smiles:
                continue

            compound_potency = self._get_compound_potency(compound=compound, target_gene_id=target_gene_id)
            if compound_potency is None or compound_potency > self._bioactivity_threshold:
                continue

            bioactive_smiles.append(compound_smiles)

        return bioactive_smiles

    @staticmethod
    def _lookup_target_gene_id(target: str) -> Optional[str]:
        """
        Look up the target gene identifier (GeneID) for the given UniProt accession by
        using the UniProt ID mapping API.

        Parameters
        ----------
        target : str
            The UniProt accession (e.g., "P00533").

        Returns
        -------
        Optional[str]
            The corresponding NCBI GeneID if found, otherwise None.
        """
        return uniprot_to_gene_id_mapping(target)


    @staticmethod
    def _get_compound_potency(compound: pcp.Compound, target_gene_id: str) -> Optional[float]:
        """
        Retrieve a potency value (e.g., IC50 in nM) for a compound by querying the
        PubChem bioassay endpoint.

        Parameters
        ----------
        compound : pcp.Compound
            A compound object from PubChem.

        Returns
        -------
        Optional[float]
            The potency value if available, otherwise None.
        """
        cid = compound.cid
        pubchem_bioassay_url = f'https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/assaysummary/JSON'
        try:
            response = requests.get(pubchem_bioassay_url, timeout=10)
            response.raise_for_status()
            response_json = response.json()

            response_table = response_json.get('Table')
            if not response_table:
                return

            response_columns = response_table.get('Columns')
            response_rows = response_table.get('Row')
            if not response_columns or not response_rows:
                return None

            try:
                columns_list = response_columns.get('Column', [])
                target_gene_idx = columns_list.index('Target GeneID')
                activity_name_idx = columns_list.index('Activity Name')
                activity_value_idx = columns_list.index('Activity Value [uM]')
            except ValueError as e:
                print(f'Column not found in bioassay data: {e}')
                return None

            ic50_values = []
            for row in response_rows:
                row_cell = row.get('Cell', [])
                if not row_cell:
                    continue

                row_target_gene = row_cell[target_gene_idx]
                row_activity_name = row_cell[activity_name_idx]
                if str(row_target_gene).strip() != str(target_gene_id).strip():
                    continue
                if row_activity_name.strip().upper() != "IC50":
                    continue

                # Extract the activity value (in µM) and convert it to nM
                try:
                    value_um = float(row_cell[activity_value_idx])
                    value_nm = value_um * 1000.0
                    ic50_values.append(value_nm)
                except (ValueError, TypeError):
                    continue

            if ic50_values:
                return min(ic50_values)

        except Exception as e:
            print(f'Error retrieving potency for CID {cid}: {e}')
            return None

        return None
