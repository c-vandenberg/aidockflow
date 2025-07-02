import os
import logging
from os.path import exists
from typing import Dict, Any, List, Union

from ml_training_base import BasePyTorchSupervisedTrainer
from ml_training_base import PyTorchTrainingEnvironment
from ml_training_base import configure_multi_level_logger
from ml_training_base import load_config

class GNNTrainer(BasePyTorchSupervisedTrainer):
    def __init__(self, config_path: str, training_env: PyTorchTrainingEnvironment):
        super().__init__(config_path, training_env)
        self._config: Dict[str, Any] = load_config(config_path)

        self._logger_path = self._config.get('data', {}).get('logger_path', '../../../var/log/default_logs.log')
        os.makedirs(os.path.dirname(self._logger_path), exist_ok=True)
        self._logger = configure_multi_level_logger(self._logger_path)

    def _setup_data(self):
        pass

    def _setup_model(self):
        pass

    def _build_model(self):
        pass

    def _setup_callbacks(self):
        pass

    def _train(self):
        pass

    def _save_model(self):
        pass

    def _evaluate(self):
        pass
