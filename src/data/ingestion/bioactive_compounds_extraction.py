import time
import requests
import logging
import concurrent.futures
from abc import ABC, abstractmethod
from functools import partial
from typing import List, Optional

from chembl_webresource_client.new_client import new_client
import pubchempy as pcp
from click.core import batch
from pubchempy import request

from constants import RestApiEndpoints
from data.utils.logging_utils import configure_logger
from data.utils.mappings import uniprot_to_gene_id_mapping
from data.utils.pubchem_api import (get_active_aids, get_active_cids,
                                    get_active_cids_wrapper, get_compounds_in_batches,
                                    batch_iterable, get_compound_potency)


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
        bioactivity_measure: str = 'Kd',
        bioactivity_threshold: Optional[float] = None, # In nM (e.g. 1000 nM threshold to filter for compounds with Kd <= 1 µM)
        logger: Optional[logging.Logger] = None
    ):
        if client is None:
            self._client = new_client
        else:
            self._client = client
        self._bioactivity_measure = bioactivity_measure
        self._bioactivity_threshold = bioactivity_threshold
        self._logger = logger if logger else logging.getLogger(__name__)

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
        # 1) Search for the target by UniProt ID and retrieve the first matching result
        target_results = self._client.target.filter(target_components__accession=target_uniprot_id)
        if not target_results:
            self._logger.error(f"No matching target found for UniProt ID {target_uniprot_id}")
            return []

        target_data = target_results[0]
        target_id = target_data['target_chembl_id']

        # 2) Build the base parameters for ChEMBL REST API.
        #    Filter activities based on the specified standard type (e.g. IC50)
        params = {
            "target_chembl_id": target_id,
            "standard_type": self._bioactivity_measure,
            "standard_units": "nM"
        }

        # 2.1) Add threshold value to params if set
        if self._bioactivity_threshold is not None:
            # standard_value__lte is the ChEMBL REST parameter for “≤”
            params["standard_value__lte"] = self._bioactivity_threshold

        # 2.2) Paginate REST API requests
        limit: int = 1000
        offset: int = 0
        chembl_activity_url: str = RestApiEndpoints.CHEMBL_ACTIVITY.url()
        bioactive_smiles: List = []
        chembl_start = time.time()

        while True:
            page_params = {
                **params,
                "limit": limit,
                "offset": offset,
                "fields": "canonical_smiles"
            }

            chembl_activity_request = requests.get(
                chembl_activity_url,
                params=page_params,
                timeout=15
            )

            chembl_activity_request.raise_for_status()
            activity_data = chembl_activity_request.json()

            records = activity_data.get('activities', [])
            if not records:
                break

            for record in records:
                smiles = record.get('canonical_smiles')
                if smiles:
                    bioactive_smiles.append(smiles)

            offset += limit

        chembl_end = time.time()
        self._logger.info(f'ChEMBL Total Query Time: {round(chembl_end - chembl_start)} seconds')

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
        bioactivity_measure: str = 'Kd',
        bioactivity_threshold: Optional[float] = None, # In nM (e.g. 1000 nM threshold to filter for compounds with Kd <= 1 µM)
        logger: Optional[logging.Logger] = None
    ):
        self._bioactivity_measure = bioactivity_measure
        self._bioactivity_threshold = bioactivity_threshold
        self._logger = logger if logger else logging.getLogger(__name__)

    def get_bioactive_compounds(self, target_uniprot_id: str) -> List[str]:
        """
        Retrieve canonical SMILES for compounds for a given target from PubChem.
        The target is provided as a UniProt accession (e.g. "P00533").

        This method performs the following steps:

        1. Maps the UniProt accession to an NCBI GeneID.
        2. Uses the GeneID to query PubChem’s BioAssay API for assay IDs (AIDs).
        3. For each assay, extracts the active compound IDs (CIDs).
        4. Retrieves compound details (including canonical SMILES) for the aggregated CIDs.
        5. Optionally filters compounds based on potency.

        Parameters
        ----------
        target_uniprot_id : str
            The UniProt accession for the target.

        Returns
        -------
        List[str]
            A list of canonical SMILES strings for compounds matching the target criteria.
        """
        # 1) Map the UniProt accession to an NCBI GeneID.
        target_gene_id = self._lookup_target_gene_id(target_uniprot_id)
        if not target_gene_id:
            self._logger.error(f"Could not determine GeneID for target '{target_uniprot_id}'.")
            return []

        # 2) Query the BioAssay API to get assay IDs (AIDs) for the target GeneID.
        try:
            aid_list = get_active_aids(target_gene_id)
        except Exception as e:
            self._logger.error(f"Error retrieving assay IDs for GeneID {target_gene_id}: {e}")
            return []

        # 3) For each assay, retrieve active compound IDs (CIDs) and aggregate them.
        #    Create thread pool using Python’s `ThreadPoolExecutor` to issue multiple API calls concurrently in batches
        cids_api_start: float = time.time()
        active_cids = set()

        # Create a new partial function with `logger` argument fixed. This allows us to pass a fixed `logger` argument
        # to the `get_active_cids_wrapper()` function when it is mapped to each AID element in `aid_list` via
        # `concurrent.futures.ThreadPoolExecutor.map()`
        get_active_cids_wrapper_partial = partial(get_active_cids_wrapper, logger=self._logger)

        with concurrent.futures.ThreadPoolExecutor(max_workers=9) as executor:
            # Map and apply partial function of `cids_for_aid_wrapper()` to every element in `aid_list` concurrently
            results = list(executor.map(get_active_cids_wrapper_partial, aid_list))

            for cids in results:
                active_cids.update(cids)

        cids_api_end: float = time.time()
        self._logger.info(f'PubChem CID Total API Query Time: {round(cids_api_end - cids_api_start)} seconds')

        if not active_cids:
            self._logger.error(f"No active compounds found for GeneID {target_gene_id}.")
            return []

        # 4) Retrieve compound details using PubChemPy with the aggregated CIDs.
        bioactive_compound_api_start: float = time.time()
        bioactive_compounds = get_compounds_in_batches(cids=list(active_cids), logger=self._logger)
        bioactive_compound_api_end: float = time.time()
        self._logger.info(f'PubChem Bioactive Compounds From CIDs Total API Query Time: '
                          f'{round(bioactive_compound_api_end - bioactive_compound_api_start)} seconds')

        bioactive_smiles: List[str] = []
        if not self._bioactivity_threshold:
            for compound in bioactive_compounds:
                compound_smiles = compound.canonical_smiles
                if not compound_smiles:
                    continue

                bioactive_smiles.append(compound_smiles)

            return bioactive_smiles
        else:
            compound_potency_api_start: float = time.time()

            # Create a new partial function with `target_gene_id` and `logger` argument fixed. As before, this allows
            # us to pass these fixed arguments to `self._get_bioactive_compound_potency()` when it is mapped to each
            # compound element in the batched `bioactive_compounds` iterable via
            # `concurrent.futures.ThreadPoolExecutor.map()`
            get_bioactive_compound_potency_partial = partial(
                self._get_bioactive_compound_potency,
                target_gene_id=target_gene_id,
                logger=self._logger
            )
            for compound_batch in batch_iterable(iterable=bioactive_compounds):
                # Process the current `bioactive_compounds` batch concurrently using a thread pool
                with concurrent.futures.ThreadPoolExecutor(max_workers=9) as executor:
                    # Map and apply partial function of `self._get_bioactive_compound_potency()` to every element in
                    # current `bioactive_compounds` batch concurrently
                    potencies = list(
                        executor.map(
                            get_bioactive_compound_potency_partial,
                            compound_batch
                        )
                    )

                # Iterate over compounds in the batch alongside their potencies
                for compound, potency in zip(compound_batch, potencies):
                    compound_smiles = compound.canonical_smiles
                    if not compound_smiles:
                        continue

                    # Filter by potency if a potency is provided.
                    if potency is None or potency > self._bioactivity_threshold:
                        continue

                    bioactive_smiles.append(compound_smiles)

            compound_potency_api_end: float = time.time()
            self._logger.info(f'PubChem Bioactive Compound Potencies Total API Query Time: '
                              f'{round(compound_potency_api_end - compound_potency_api_start)} seconds')

            return bioactive_smiles

    def _get_compound_potency(self, compound: pcp.Compound, target_gene_id: str) -> Optional[float]:
        """
        Retrieve a potency value (e.g., Kd in nM) for a compound by querying the
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
                if row_activity_name.strip().upper() != self._bioactivity_measure:
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

    def _get_bioactive_compound_potency(
        self,
        compound: pcp.Compound,
        target_gene_id: str,
        logger: logging.Logger = None
    ) -> Optional[float]:
        return get_compound_potency(
            compound=compound,
            target_gene_id=target_gene_id,
            bioactivity_measure=self._bioactivity_measure,
            logger=logger
        )
