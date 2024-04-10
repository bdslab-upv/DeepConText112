"""
DESCRIPTION: classes and operations for meta-learning for continual learning.
AUTHOR: Pablo Ferri-Borredà
DATE: 04/07/22
"""

# MODULES IMPORT
import datetime
from os.path import join

from optuna import create_study
from optuna.logging import disable_default_handler

from learning.objective import Objective


# METALEARNER
class Metalearner:

    # INITIALIZATION
    def __init__(self, objective: Objective, number_trials: int = 1):
        # INITIALIZATION
        self._objective = objective
        self._number_trials = number_trials
        self._display_progress = False
        self._study = create_study(direction='maximize')
        self._trials_log = None
        self._best_hyperparams = None

    # EXTERNAL ATTRIBUTE ACCESS AND EDITION CONTROL
    @property
    def trials_log(self):
        return self._trials_log

    @property
    def best_hyperparams(self):
        return self._best_hyperparams

    # EXECUTION
    def run(self):
        # Progress display
        if not self._display_progress:
            disable_default_handler()

        # Trials execution
        self._study.optimize(self._objective, n_trials=self._number_trials, show_progress_bar=True)
        self._trials_log = self._study.trials_dataframe()
        self._best_hyperparams = self._study.best_params

        # Results exporting
        # Settings
        results_directory = './Results/'
        filename = 'trialslog_' + self._objective.strategy_identifier + '_' + datetime.datetime.now().strftime(
            "%Y%m%d-%H%M%S") + '.csv'
        absolute_filepath = join(results_directory, filename)

        # Exporting
        self._trials_log.to_csv(absolute_filepath, sep=';', encoding='latin-1')
