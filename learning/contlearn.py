"""
DESCRIPTION: continual learning strategies and continual learning loop.
AUTHOR: Pablo Ferri-Borredà
DATE: 01/07/22
"""

# MODULES IMPORT
import datetime
from os.path import join

from avalanche.benchmarks.scenarios.generic_cl_scenario import GenericCLScenario
from avalanche.evaluation.metrics import forgetting_metrics, accuracy_metrics, loss_metrics, confusion_matrix_metrics
from avalanche.logging import InteractiveLogger, TensorboardLogger
from avalanche.training import BaseStrategy, JointTraining, Cumulative, Naive, Replay, SynapticIntelligence
from avalanche.training.plugins import EvaluationPlugin
from torch.cuda import is_available
from torch.nn import Module
from torch.optim import Optimizer

from evaluation.auc import area_under_curve_metrics
from evaluation.f1score import f1_score_metrics
from evaluation.negpredval import negative_predictive_value_metrics
from evaluation.precision import precision_metrics
from evaluation.recall import recall_metrics
from evaluation.specificity import specificity_metrics
from learning.hyperpars import HyperparamContainer

# SETTINGS
BATCH_SIZE_EVAL = 256
EVAL_STEP_EPOCS = 0


# EVALUATOR DEFINITION
def define_evaluator(strategy_identifier: str):
    # Tensorboard path setting
    # absolute log filepath extraction
    log_directory = './Results/'
    log_filepath = join(log_directory, 'TextClassifierDistilBERT', strategy_identifier,
                        datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

    # Loggers definition
    # interactive logger
    interactive_logger = InteractiveLogger()
    # tensorboard logger
    tensorboard_logger_flag = True
    if tensorboard_logger_flag:
        tensorboard_logger = TensorboardLogger(tb_log_dir=log_filepath)
        loggers_list = [interactive_logger, tensorboard_logger]
    else:
        loggers_list = [interactive_logger]

    # Evaluation plugin definition
    eval_plugin = EvaluationPlugin(
        area_under_curve_metrics(minibatch=True, epoch=True, experience=True, stream=True),
        accuracy_metrics(minibatch=True, epoch=True, experience=True, stream=True),
        recall_metrics(minibatch=True, epoch=True, experience=True, stream=True),
        specificity_metrics(minibatch=True, epoch=True, experience=True, stream=True),
        precision_metrics(minibatch=True, epoch=True, experience=True, stream=True),
        negative_predictive_value_metrics(minibatch=True, epoch=True, experience=True, stream=True),
        f1_score_metrics(minibatch=True, epoch=True, experience=True, stream=True),
        loss_metrics(minibatch=True, epoch=True, experience=True, stream=True),
        forgetting_metrics(experience=True, stream=True),
        confusion_matrix_metrics(num_classes=2, save_image=False, stream=True),
        loggers=loggers_list)

    # Output
    return eval_plugin


# STRATEGY DEFINITION
# High-level strategy definition
def define_strategy(strategy_identifier: str, *, model: Module, optimizer: Optimizer, loss_function: Module,
                    evaluation_plugin: EvaluationPlugin, hyperparam_container: HyperparamContainer) -> BaseStrategy:
    # Strategy definition
    strategy = _define_strategy(strategy_identifier, model=model, optimizer=optimizer, loss_function=loss_function,
                                batch_size_eval=BATCH_SIZE_EVAL, evaluation_step_epochs=EVAL_STEP_EPOCS,
                                evaluation_plugin=evaluation_plugin, hyperparam_container=hyperparam_container)

    # Output
    return strategy


# Low-level strategy definition
def _define_strategy(strategy_identifier: str, *, model: Module, optimizer: Optimizer, loss_function: Module,
                     batch_size_eval: int, evaluation_step_epochs: int, evaluation_plugin: EvaluationPlugin,
                     hyperparam_container: HyperparamContainer) -> BaseStrategy:
    # Device definition
    device = 'cuda' if is_available() else 'cpu'

    # Hyperparameters extraction
    batch_size_train = hyperparam_container.get_value('batch_size_train')
    number_epochs = hyperparam_container.get_value('number_epochs')

    # Strategy definition
    # joint training
    if strategy_identifier == 'JointTraining':
        strategy = JointTraining(
            model=model, optimizer=optimizer, criterion=loss_function, evaluator=evaluation_plugin,
            train_mb_size=batch_size_train, eval_mb_size=batch_size_eval, train_epochs=number_epochs,
            eval_every=evaluation_step_epochs, device=device)
    # cumulative
    elif strategy_identifier == 'Cumulative':
        strategy = Cumulative(model=model, optimizer=optimizer, criterion=loss_function, evaluator=evaluation_plugin,
                              train_mb_size=batch_size_train, eval_mb_size=batch_size_eval, train_epochs=number_epochs,
                              eval_every=evaluation_step_epochs, device=device)
    # single-experience
    elif strategy_identifier == 'SingleFineTuning':
        strategy = BaseStrategy(model=model, optimizer=optimizer, criterion=loss_function, evaluator=evaluation_plugin,
                                train_mb_size=batch_size_train, eval_mb_size=batch_size_eval,
                                train_epochs=number_epochs,
                                eval_every=evaluation_step_epochs, device=device)
    # naive
    elif strategy_identifier == 'ContinualFineTuning':
        strategy = Naive(model=model, optimizer=optimizer, criterion=loss_function, evaluator=evaluation_plugin,
                         train_mb_size=batch_size_train, eval_mb_size=batch_size_eval, train_epochs=number_epochs,
                         eval_every=evaluation_step_epochs, device=device)
    # replay
    elif strategy_identifier == 'Replay':
        # hyperparameters extraction
        memory_size = hyperparam_container.get_value('memory_size')
        # strategy definition
        strategy = Replay(model=model, optimizer=optimizer, criterion=loss_function, evaluator=evaluation_plugin,
                          train_mb_size=batch_size_train, eval_mb_size=batch_size_eval, train_epochs=number_epochs,
                          eval_every=evaluation_step_epochs, device=device, mem_size=memory_size)
    # synaptic intelligence
    elif strategy_identifier == 'SynapticIntelligence':
        # hyperparameters extraction
        lambda_ = hyperparam_container.get_value('lambda')
        # strategy definition
        strategy = SynapticIntelligence(
            model=model, optimizer=optimizer, criterion=loss_function, evaluator=evaluation_plugin,
            train_mb_size=batch_size_train, eval_mb_size=batch_size_eval, train_epochs=number_epochs,
            eval_every=evaluation_step_epochs, device=device,
            si_lambda=lambda_)
    # unrecognized strategy identifier
    else:
        raise ValueError('Unrecognized strategy identifier.')

    # Output
    return strategy


# CONTINUAL LEARNING AND EVALUATION
# High-level
def train_eval_model_continually(strategy_identifier: str, scenario: GenericCLScenario, strategy: BaseStrategy) -> list:
    # Training and evaluation following a specific continual learning strategy
    results = _train_eval_model_continually(strategy_identifier, scenario, strategy)

    # Output
    return results


# Low-level
def _train_eval_model_continually(strategy_identifier: str, scenario: GenericCLScenario,
                                  strategy: BaseStrategy) -> list:
    # Initialization
    # memory allocation
    results = []
    # strategy object backup (if required)
    if strategy_identifier == 'SingleFineTuning':
        strategy_ = strategy

    # Offline training
    if isinstance(strategy, JointTraining):
        # Progress display
        print('Joint training')

        # Training and evaluation
        results.append(strategy.train(scenario.train_stream, eval_streams=[scenario.test_stream]))

    # Continual training
    else:
        for batch_train in scenario.train_stream:
            # Progress display
            print('Training on experience ', batch_train.current_experience)

            # Training and evaluation
            # recovery of initial state (if required)
            if strategy_identifier == 'SingleFineTuning':
                strategy = strategy_
            # training and evaluation
            results.append(strategy.train(batch_train, eval_streams=[scenario.test_stream]))

    # Output
    return results
