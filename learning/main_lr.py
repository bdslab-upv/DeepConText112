"""
DESCRIPTION: main script to implement a continual learning pipeline for text data from Cordex and Coordcom system 112.
AUTHOR: Pablo Ferri-Borredà
DATE: 01/07/22
"""

# MODULES IMPORT
from os.path import join

import learning.datahand as dh
import learning.modeling as mod
from filehandling import load_data
from learning.hyperpars import define_hyperparameters
from learning.metalearn import Metalearner
from learning.objective import Objective

# SETTINGS
ROUTINE = 'train_test'  # 'train_val', 'train_test'
MODEL_IDENTIFIER = 'TextClassifierDistilBERT'
STRATEGIES = ('JointTraining', 'Cumulative', 'SingleFineTuning', 'ContinualFineTuning', 'Replay',
              'SynapticIntelligence')

data_directory = './Data/'
data_filename = 'working_matrices.pkl'

# EXECUTION
if __name__ == '__main__':
    # DATA LOADING
    # Filepath definition
    absolute_filepath = join(data_directory, data_filename)

    # Loading
    data = load_data(absolute_filepath)

    # SCENARIO GENERATION
    scenario = dh.generate_scenario(data, routine=ROUTINE)

    # ITERATIVE CONTINUAL LEARNING
    # Settings
    # number of trials map
    if ROUTINE == 'train_val':
        number_trials_map = {'JointTraining': 15, 'Cumulative': 15, 'SingleFineTuning': 15, 'ContinualFineTuning': 15,
                             'Replay': 20, 'SynapticIntelligence': 20}
    elif ROUTINE == 'train_test':
        number_trials_map = {'JointTraining': 1, 'Cumulative': 1, 'SingleFineTuning': 1, 'ContinualFineTuning': 1,
                             'Replay': 1, 'SynapticIntelligence': 1}
    else:
        raise ValueError('Unrecognized routine identifier.')

    # Execution
    for strategy_idf in STRATEGIES:
        # NUMBER OF TRIALS EXTRACTION
        number_trials = number_trials_map[strategy_idf]

        # HYPERPARAMETERS DEFINITION
        hyperpam_container = define_hyperparameters(strategy_idf)

        # LEARNING OBJECTIVE DEFINITION
        model = mod.define_model(MODEL_IDENTIFIER)

        # LEARNING, CONTINUAL LEARNING AND META-LEARNING
        # Objective definition
        objective = Objective(hyperparam_container=hyperpam_container, model_identifier=MODEL_IDENTIFIER,
                              strategy_identifier=strategy_idf, scenario=scenario)

        # Meta-learner definition
        meta_learner = Metalearner(objective=objective, number_trials=number_trials)

        # Execution
        meta_learner.run()
