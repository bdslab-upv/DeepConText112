"""
DESCRIPTION: functions and classes to track and export results.
AUTHOR: Pablo Ferri-Borredà
DATE: 01/07/22
"""

# MODULES IMPORT
from typing import Tuple

from pandas import concat, DataFrame

from arrangement.arrangement import TEST_RATIO, VAL_RATIO, TIME_WINDOWS
from preparation.dataprep import TOKENIZER_TYPE
from learning.contlearn import BATCH_SIZE_EVAL
from learning.hyperpars import HyperparamContainer


# SETTINGS AND RESULTS TRACKING
def track_settings_results(results: list, *, model_identifier: str, strategy_identifier: str,
                           hyperparam_container: HyperparamContainer) -> Tuple[DataFrame, DataFrame]:
    # Settings frame generation
    settings_frame = _generate_settings_frame(model_identifier, strategy_identifier, hyperparam_container)

    # Results frame generation
    if strategy_identifier != 'JointTraining':
        # standard continual learning technique
        results_frame = _generate_results_frame_standard(results)
    else:
        # joint training
        results_frame = _generate_results_frame_joint(results)

    # Output
    return settings_frame, results_frame


# SETTINGS FRAME GENERATION
def _generate_settings_frame(model_identifier: str, strategy_identifier: str,
                             hyperparam_container: HyperparamContainer) -> DataFrame:
    # Settings dictionary generation
    # initialization
    settings_map = dict()
    # filling
    settings_map['tokenizer_type'] = TOKENIZER_TYPE
    settings_map['test_ratio'] = TEST_RATIO
    settings_map['val_ratio'] = VAL_RATIO
    settings_map['time_windows'] = TIME_WINDOWS
    settings_map['model_identifier'] = model_identifier
    settings_map['learning_rate'] = hyperparam_container.get_value('learning_rate')
    settings_map['class_weighting'] = hyperparam_container.get_value('class_weighting')
    settings_map['class_weights'] = hyperparam_container.get_value('class_weights')
    settings_map['strategy_identifier'] = strategy_identifier
    settings_map['batch_size_train'] = hyperparam_container.get_value('batch_size_train')
    settings_map['batch_size_eval'] = BATCH_SIZE_EVAL
    settings_map['number_epochs'] = hyperparam_container.get_value('number_epochs')
    if strategy_identifier == 'Replay':
        settings_map['memory_size'] = hyperparam_container.get_value('memory_size')
    if strategy_identifier == 'SynapticIntelligence':
        settings_map['lambda'] = hyperparam_container.get_value('lambda')

    # Settings frame generation
    # settings map formatting
    settings_map_formatted = {set_idf: [set_val] for set_idf, set_val in settings_map.items()}
    # casting to data frame
    settings_frame = DataFrame.from_dict(settings_map_formatted, orient='index', columns=['VALUE'])

    # Output
    return settings_frame


# RESULTS FRAME GENERATION
# Common settings
experience2year_map = {'Exp000': 2009, 'Exp001': 2010, 'Exp002': 2011, 'Exp003': 2012, 'Exp004': 2014,
                       'Exp005': 2015, 'Exp006': 2016, 'Exp007': 2017, 'Exp008': 2018, 'Exp009': 2019}


# experience2year_map = {'Exp000': 2009, 'Exp001': 2014, 'Exp002': 2019}


# Joint training
def _generate_results_frame_joint(results: list) -> DataFrame:
    # Initialization
    results_map = dict()

    # Joint training results extraction
    joint_results = results[0]

    # Data frame generation
    for result_exp_idf, result_exp_val in joint_results.items():
        if 'eval_phase' in result_exp_idf and 'Top1_' in result_exp_idf:
            result_exp_idf_clean = result_exp_idf.replace('Top1_', '')
            result_exp_idf_clean = result_exp_idf_clean.replace('Task000/', '')
            result_exp_idf_clean = result_exp_idf_clean.replace('eval_phase/', '')
            for exp_idf, year_val in experience2year_map.items():
                result_exp_idf_clean = result_exp_idf_clean.replace(exp_idf, str(year_val))
            results_map[result_exp_idf_clean] = result_exp_val

    # Results frame generation
    # formatting
    results_map_exp_formatted = {res_idf: [res_val] for res_idf, res_val in results_map.items()}
    # casting to data frame
    results_frame = DataFrame.from_dict(results_map_exp_formatted, orient='index', columns=['VALUE'])

    # Output
    return results_frame


# Standard
def _generate_results_frame_standard(results: list) -> DataFrame:
    # Results map generation
    # initialization
    results_map = dict()
    counter = 0
    # filling
    for results_experience in results:
        # Time window identifier generation
        time_window_idf = TIME_WINDOWS[counter]

        # Dictionary initialization
        results_map[time_window_idf] = dict()

        # Filling
        for result_exp_idf, result_exp_val in results_experience.items():
            if 'eval_phase' in result_exp_idf and 'Top1_' in result_exp_idf:
                result_exp_idf_clean = result_exp_idf.replace('Top1_', '')
                result_exp_idf_clean = result_exp_idf_clean.replace('Task000/', '')
                result_exp_idf_clean = result_exp_idf_clean.replace('eval_phase/', '')
                for exp_idf, year_val in experience2year_map.items():
                    result_exp_idf_clean = result_exp_idf_clean.replace(exp_idf, str(year_val))
                results_map[time_window_idf][result_exp_idf_clean] = result_exp_val

        # Results frame generation
        # formatting
        results_map_exp_formatted = {res_idf: [res_val] for res_idf, res_val in results_map[time_window_idf].items()}
        # casting to data frame
        results_exp_frame = DataFrame.from_dict(results_map_exp_formatted, orient='index', columns=[time_window_idf])

        # Arrangement
        if counter == 0:
            results_frame = results_exp_frame
        else:
            results_frame = concat([results_frame, results_exp_frame], axis=1)

        # Counter updating
        counter += 1

    # Output
    return results_frame
