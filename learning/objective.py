"""
DESCRIPTION: classes and operations for meta-learning.
AUTHOR: Pablo Ferri-Borredà
DATE: 04/07/22
"""

# MODULES IMPORT
import warnings

from abc import abstractmethod
from math import isnan

from avalanche.benchmarks.scenarios.generic_cl_scenario import GenericCLScenario

import learning.contlearn as cl
import learning.datahand as dh
import learning.modeling as mod
from learning.hyperpars import HyperparamContainer
from learning.tracking import track_settings_results


# OPTUNA OBJECTIVE DEFINITION
class OptunaObjective(object):

    # INITIALIZATION
    @abstractmethod
    def __init__(self, hyperparam_container):
        # Inputs checking
        if type(hyperparam_container) is not HyperparamContainer:
            raise TypeError('Hyperparameters are not properly encapsulated in an Hyperparameter Container.')

        # Attributes assignation
        self._hyperparam_container = hyperparam_container

    # EXTERNAL ATTRIBUTE ACCESS AND EDITION CONTROL
    @property
    def hyperparam_container(self):
        return self._hyperparam_container

    # CONVERSION TO CALLABLE
    @abstractmethod
    def __call__(self, trial):
        # Iteration over hyperparameters
        for identifier, hyperparameter in self._hyperparam_container._content.items():
            # It the hyperparameter is not prefixed and has to be tuned
            if hyperparameter.tune:
                search_space = hyperparameter.search_space
                self._hyperparam_container.set_value(hyperparam_identifier=identifier,
                                                     value=trial.suggest_categorical(identifier, search_space))
            # It the hyperparameter is prefixed
            else:
                default_value = hyperparameter.default
                self._hyperparam_container.set_value(hyperparam_identifier=identifier, value=default_value)


# LEARNING OBJECTIVE DEFINITION
class Objective(OptunaObjective):

    # INITIALIZATION
    def __init__(self, hyperparam_container: HyperparamContainer, model_identifier: str, strategy_identifier: str,
                 scenario: GenericCLScenario) -> None:
        super().__init__(hyperparam_container)

        # Attributes assignation
        self._model_identifier = model_identifier
        self._strategy_identifier = strategy_identifier
        self._scenario = scenario
        self._trial = 0  # trial counter

    # EXTERNAL ATTRIBUTE ACCESS AND EDITION CONTROL
    # Model identifier
    @property
    def model_identifier(self):
        return self._model_identifier

    # Strategy identifier
    @property
    def strategy_identifier(self):
        return self._strategy_identifier

    # Scenario
    @property
    def scenario(self):
        return self._scenario

    # CALLABLE METHOD OVERWRITING
    def __call__(self, trial):
        # PARENT METHOD CALL
        super().__call__(trial)

        # MODEL INITIALIZATION
        model = mod.define_model(self.model_identifier)

        # OPTIMIZER DEFINITION
        optimizer = mod.define_optimizer(model=model, hyperparam_container=self.hyperparam_container)

        # LOSS FUNCTION DEFINITION
        loss_function = mod.define_loss_function(hyperparam_container=self.hyperparam_container)

        # EVALUATOR DEFINITION
        evaluator = cl.define_evaluator(strategy_identifier=self.strategy_identifier)

        # STRATEGY DEFINITION
        strategy = cl.define_strategy(strategy_identifier=self.strategy_identifier, model=model, optimizer=optimizer,
                                      loss_function=loss_function, evaluation_plugin=evaluator,
                                      hyperparam_container=self.hyperparam_container)

        # OBJECTIVE METRIC INITIALIZATION
        objective_metric = 0

        # LAUNCHING OF CONTINUAL LEARNING TRAINING AND EVALUATION
        try:
            # CONTINUAL MODEL TRAINING AND EVALUATION
            results = cl.train_eval_model_continually(self.strategy_identifier, self.scenario, strategy)

            # SETTINGS AND RESULTS TRACKING
            settings_frame, results_frame = track_settings_results(results, model_identifier=self.model_identifier,
                                                                   strategy_identifier=self.strategy_identifier,
                                                                   hyperparam_container=self.hyperparam_container)

            # OBJECTIVE METRIC EXTRACTION
            if self.strategy_identifier != 'JointTraining':
                objective_metric = self._calculate_objective_metric_standard(results)
            else:
                objective_metric = self._calculate_objective_metric_joint(results)

            if isnan(objective_metric):
                objective_metric = -1

            # SETTINGS AND RESULTS EXPORTING
            dh.export_settings_results(strategy_identifier=self.strategy_identifier, settings_frame=settings_frame,
                                       results_frame=results_frame)

        except Exception:
            warnings.warn('An error ocurred during execution, probably due to memory size problems.')

        # TRIAL UPGRADING
        self._trial += 1

        # OUTPUT
        return objective_metric

    # OBJECTIVE METRIC CALCULATION
    # Joint training
    @staticmethod
    def _calculate_objective_metric_joint(joint_results: list) -> float:
        # Initialization
        metric_identifier = 'Top1_AUC_Stream/eval_phase/test_stream/Task000'

        # Metric extraction
        metric = joint_results[0][metric_identifier]

        # Output
        return metric

    # Standard continual learning strategies
    @staticmethod
    def _calculate_objective_metric_standard(contlearn_results: list) -> float:
        # Initialization
        # metric header
        metric_header = 'Top1_AUC_Exp/eval_phase/test_stream/Task000/Exp'
        # memory allocation
        metrics = []

        # Iterative extraction
        for i in range(len(contlearn_results) - 1):
            contlearn_results_i = contlearn_results[i]

            for metric_idf, metric_val in contlearn_results_i.items():
                if metric_header in metric_idf:
                    experience_int = int(metric_idf.split('Task000/Exp')[1])

                    if experience_int == i + 1:
                        metrics.append(metric_val)

        # Average calculation
        metric = sum(metrics) / len(metrics)

        # Output
        return metric
