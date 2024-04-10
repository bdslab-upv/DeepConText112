"""
DESCRIPTION: functions and classes to handle data.
AUTHOR: Pablo Ferri-Borredà
DATE: 01/07/22
"""

# MODULES IMPORT
import datetime
from os.path import join

from avalanche.benchmarks.generators import dataset_benchmark
from avalanche.benchmarks.scenarios.generic_cl_scenario import GenericCLScenario
from avalanche.benchmarks.utils.avalanche_dataset import AvalancheTensorDataset
from pandas import DataFrame, ExcelWriter
from torch import stack, from_numpy


# SCENARIO GENERATION
def generate_scenario(data: dict, routine: str) -> GenericCLScenario:
    # Settings
    # time window identifiers
    time_window_identifiers = (2009, 2010, 2011, 2012, 2014, 2015, 2016, 2017, 2018, 2019)
    # data set identifiers
    if routine == 'train_val':
        data_set_identifiers = ('train', 'val')
    elif routine == 'train_test':
        data_set_identifiers = ('train_val', 'test')
    else:
        raise ValueError('Unrecognized routine.')

    # Scenario generation
    scenario = _generate_scenario_text_classifier(data, time_window_identifiers, data_set_identifiers)

    # Output
    return scenario


# AVALANCHE SCENARIO GENERATION
# Scenario for text classification with attention mask
def _generate_scenario_text_classifier(data: dict, time_window_identifiers: tuple,
                                       data_set_identifiers: tuple) -> GenericCLScenario:
    """
    References: https://avalanche.continualai.org/how-tos/avalanchedataset/creating-avalanchedatasets
    """

    # Initialization
    # memory allocation
    datasets = {data_set: [] for data_set in data_set_identifiers}
    # task label
    task_label = 0

    # Iterative loop for feeders generation
    for time_idf in time_window_identifiers:
        for data_set in data_set_identifiers:
            # Data slicing
            text_matrix = data[time_idf][data_set]['indexes']
            attention_matrix = data[time_idf][data_set]['attention_mask']
            label_matrix = data[time_idf][data_set]['labels']

            # Casting to tensor
            text_tensor = _array2tensor(text_matrix)
            attention_tensor = _array2tensor(attention_matrix)
            label_tensor = _array2tensor(label_matrix)

            # Additional casting operations
            text_tensor = text_tensor.long()
            attention_tensor = attention_tensor.long()

            # Tensor stacking
            text_attention_tensor = stack(tensors=(text_tensor, attention_tensor), dim=2)

            # Avalanche datasets
            avalanche_dataset = AvalancheTensorDataset(
                text_attention_tensor, label_tensor, task_labels=task_label, targets=label_tensor)

            # Addition
            datasets[data_set].append(avalanche_dataset)

    # Avalanche scenario generation
    if 'val' in data_set_identifiers:  # Training and validation routine
        scenario = dataset_benchmark(train_datasets=datasets['train'], test_datasets=datasets['val'])
    else:  # Training and testing routine
        scenario = dataset_benchmark(train_datasets=datasets['train_val'], test_datasets=datasets['test'])

    # Output
    return scenario


# Array to tensor casting
def _array2tensor(array):
    return from_numpy(array).float()


# SETTINGS AND RESULTS EXPORTING
def export_settings_results(*, strategy_identifier: str, settings_frame: DataFrame, results_frame: DataFrame) -> None:
    # Settings
    results_directory = './Results/'
    filename = 'setres_' + strategy_identifier + '_' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + '.xlsx'
    absolute_filepath = join(results_directory, filename)

    # Arrangement
    data_map = {'Settings': settings_frame, 'Results': results_frame}

    # Exporting
    with ExcelWriter(absolute_filepath) as writer:
        for frame_identifier, data_frame in data_map.items():
            data_frame.to_excel(writer, sheet_name=frame_identifier, encoding='latin-1')
