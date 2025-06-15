import logging
import sqlite3
from typing import List, Tuple, Optional

from ml_training_base import BaseDataPreprocessor

class SmilesDataPreprocessor(BaseDataPreprocessor):
    def __init__(self, logger: Optional[logging.Logger] = None):
        super().__init__(logger)
