import logging
import sqlite3
from typing import List, Tuple, Optional

class SmilesDataPreprocessor:
    def __init__(
        self,
        logger: Optional[logging.Logger] = None
    ):
        self._logger = logger if logger else logging.getLogger(__name__)

    def concatenate_datasets(
        self,
        dataset_a: List[str],
        dataset_b: List[str]
    ) -> List[str]:
        if dataset_a is not None and dataset_b is not None:
            self._logger.info("Concatenating product datasets.")
            self._logger.info(f"Dataset A size before concatenation: {len(dataset_a)}")
            dataset_a.extend(dataset_b)
            self._logger.info(f"Dataset size after concatenation: {len(dataset_b)}")

        return dataset_a

    def deduplicate_dataset_in_memory(self, dataset: List[str]) -> List[str]:
        """
        Deduplicate reaction pairs using in-memory sets.

        Returns:
        -------

        """
        self._logger.info("Starting in-memory deduplication.")
        seen = set()
        unique_datapoints = []

        for datapoint in dataset:
            if datapoint not in seen:
                seen.add(datapoint)
                unique_datapoints.append(datapoint)
            else:
                self._logger.debug(f"Duplicate within batch skipped: Datapoint={datapoint}")

        self._logger.info(f"Deduplication completed. Unique datapoints: {len(unique_datapoints)}")

        return unique_datapoints

    def deduplicate_dataset_on_disk(
        self,
        dataset: List[str],
        db_path: str = 'seen_pairs.db',
        batch_size: int = 1000,
        log_interval: int = 1000
    ):
        self._logger.info("Starting SQLite-based deduplication.")

        conn = sqlite3.connect(db_path)
        try:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS unique_datapoints (
                        datapoint TEXT,
                        PRIMARY KEY (datapoint)
                    )
            """)
            cursor = conn.cursor()

            total = len(dataset)
            current_idx = 0
            batch_number = 1

            cursor.execute("BEGIN TRANSACTION;")

            while current_idx < total:
                batch_datapoints = dataset[current_idx:current_idx + batch_size]

                try:
                    cursor.executemany(
                        "INSERT OR IGNORE INTO unique_datapoints (datapoint) VALUE (?)",
                        batch_datapoints
                    )
                    self._logger.debug(f"Batch {batch_number} inserted with {len(batch_datapoints)} datapoints.")
                except sqlite3.Error as e:
                    self._logger.error(f"SQLite error during batch {batch_number} insertion: {e}")
                    continue

                current_idx += batch_size
                batch_number += 1

                if current_idx % log_interval == 0:
                    self._logger.info(f"Processed {current_idx}/{total} datapoints.")

            conn.commit()
            self._logger.info("SQLite-based deduplication completed successfully.")
        finally:
            conn.close()

        return self._extract_unique_datapoints_from_db(db_path=db_path)

    def _extract_unique_datapoints_from_db(self, db_path: str = 'seen_pairs.db') -> List[str]:
        """
        Extract all unique datapoints from the SQLite database and assign them to in-memory datasets.

        Parameters
        ----------
        db_path : str, optional
            Path to the SQLite database used for deduplication. Defaults to 'seen_pairs.db'.

        Returns
        -------
        None
        """
        self._logger.info("Extracting unique reactions from the SQLite database.")
        unique_dataset: List = []
        conn = sqlite3.connect(db_path)
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT datapoint FROM unique_datapoints")
            rows = cursor.fetchall()
            self._logger.debug(f"Fetched {len(rows)} unique reactions from the database.")

            unique_dataset = [reactant for reactant in rows]

            self._logger.info(f"Assigned {len(unique_dataset)} unique reactions to in-memory datasets.")
        except sqlite3.Error as e:
            self._logger.error(f"SQLite error during extraction: {e}")
        finally:
            conn.close()

        return unique_dataset