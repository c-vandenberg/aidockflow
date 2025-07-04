import logging
from abc import abstractmethod, ABC
from typing import Dict


class BaseCurator(ABC):
    def __init__(self, config: Dict, logger: logging.Logger):
        self._config = config
        self._logger = logger

    @abstractmethod
    def run(self):
        pass
