"""
DESCRIPTION: functions and classes to arrange data for covariate shifts detection.
AUTHOR: Pablo Ferri-Borredà
DATE: 19/07/22
"""

# MODULES IMPORT
from numpy import array
from pandas import DataFrame, concat
from torch import Tensor
from torch import from_numpy
from torch.utils.data import Dataset, DataLoader

from arrangement.arrangement import HoldoutSplitter


# DATA SEGREGATION
# Global segregation function
def segregate_data(data: DataFrame) -> dict:
    # Years
    years = (2009, 2010, 2011, 2012, 2014, 2015, 2016, 2017, 2018, 2019)

    # Memory allocation
    data_seg = dict()

    # Data segregation by years
    for year_i in years:
        for year_j in years:
            if year_i != year_j:
                if (year_j, year_i) not in data_seg.keys():
                    # Experiment key generation
                    experiment_key = (year_i, year_j)

                    # Data slicing and shuffling
                    data_seg[experiment_key] = slice_shuffle_data(data, year_i, year_j)

    # Output
    return data_seg


# Slicing and shuffling
random_seed = 112
columns_common = ['TEXT_IDX', 'TEXT_ATM']


def slice_shuffle_data(data: DataFrame, year_i: int, year_j: int) -> DataFrame:
    # Year columns generation
    year_i_col = 'YEAR_' + str(year_i)
    year_j_col = 'YEAR_' + str(year_j)

    # Data slicing
    # masks generation
    mask_year_i = data[year_i_col] == 1
    mask_year_j = data[year_j_col] == 1
    # per year row slicing
    data_year_i = data[mask_year_i]
    data_year_j = data[mask_year_j]
    # concatenation
    data_sliced = concat([data_year_i, data_year_j])

    # Data shuffling
    data_sliced_shuffled = data_sliced.sample(frac=1, replace=False, random_state=random_seed)

    # Columns selection
    data_sel = data_sliced_shuffled[columns_common + [year_i_col, year_j_col]]

    # Output
    return data_sel


# DATA SPLITTING
def split_data(data_seg: dict) -> dict:
    # Memory allocation
    data_split = dict()

    # Splitter initialization
    splitter = HoldoutSplitter(eval_ratio=0.2)

    # Iterative data splitting (just train and test because hyperparameters are assumed to be optimal)
    for experiment_key, data_exp in data_seg.items():
        # Dictionary creation
        data_split[experiment_key] = dict()

        # Splitting
        data_train, data_test = splitter.split(data_exp)

        # Arrangement
        data_split[experiment_key]['train'] = data_train
        data_split[experiment_key]['test'] = data_test

    # Output
    return data_split


# WORKING MATRICES GENERATION
def generate_working_matrices(data_split: dict) -> dict:
    # Memory allocation
    working_matrices = dict()

    # Generation
    for experiment_key, data_train_test in data_split.items():
        working_matrices[experiment_key] = dict()

        year_i = experiment_key[0]
        year_j = experiment_key[1]

        year_i_col = 'YEAR_' + str(year_i)
        year_j_col = 'YEAR_' + str(year_j)

        year_i_j_cols = [year_i_col, year_j_col]

        for data_set, data in data_train_test.items():
            working_matrices[experiment_key][data_set] = dict()

            working_matrices[experiment_key][data_set]['indexes'] = array(data['TEXT_IDX'].to_list())
            working_matrices[experiment_key][data_set]['attention_mask'] = array(data['TEXT_ATM'].to_list())
            working_matrices[experiment_key][data_set]['labels'] = data[year_i_j_cols].values

    # Output
    return working_matrices


# DATA LOADERS GENERATION
# Settings
batch_size_train = 32
batch_size_test = 256


# Data loader generation and arrangement
def generate_data_loaders(working_matrices: dict) -> dict:
    # Memory allocation
    data_loaders = dict()

    # Generation
    for experiment_key, data_train_test in working_matrices.items():
        data_loaders[experiment_key] = dict()

        for data_set, data in data_train_test.items():
            # Number of data extraction
            number_data = data['indexes'].shape[0]

            # Casting
            indexes_tensor = _array2tensor(data['indexes']).long()
            attention_mask_tensor = _array2tensor(data['attention_mask'])
            labels_tensor = _array2tensor(data['labels'])

            # Dataset generation
            dataset = TextClassificationDataset(indexes_tensor=indexes_tensor,
                                                attention_mask_tensor=attention_mask_tensor,
                                                labels_tensor=labels_tensor, number_data=number_data)

            # Data loader generation
            if data_set == 'train':
                data_loader = DataLoader(dataset=dataset, batch_size=batch_size_train, shuffle=True)
            elif data_set == 'test':
                data_loader = DataLoader(dataset=dataset, batch_size=batch_size_test, shuffle=False)
            else:
                raise ValueError('Unrecognized data set identifier.')

            # Arrangement
            data_loaders[experiment_key][data_set] = data_loader

    # Output
    return data_loaders


# Data casting
def _array2tensor(array):
    return from_numpy(array).float()


# Text classification dataset class
class TextClassificationDataset(Dataset):
    # INITIALIZATION
    def __init__(self, indexes_tensor: Tensor, attention_mask_tensor, labels_tensor, number_data: int) -> None:
        # Inputs checking
        if type(indexes_tensor) is not Tensor:
            raise TypeError('Indexes must be specified as a PyTorch tensor.')
        if type(attention_mask_tensor) is not Tensor:
            raise TypeError('Attention mask must be specified as a PyTorch tensor.')
        if type(labels_tensor) is not Tensor:
            raise TypeError('Labels must be specified as a PyTorch tensor.')
        if type(number_data) is not int:
            raise TypeError('Number data must be an integer.')

        # Attributes assignation
        self._indexes = indexes_tensor
        self._attention_mask = attention_mask_tensor
        self._labels = labels_tensor
        self._number_data = number_data

    # EXTERNAL ATTRIBUTE ACCESS AND EDITION CONTROL
    @property
    def indexes(self):
        return self._indexes

    @property
    def attention_mask(self):
        return self._attention_mask

    @property
    def labels(self):
        return self._labels

    @property
    def number_data(self):
        return self._number_data

    # ITEM EXTRACTION
    def __getitem__(self, indexes4slicing) -> dict:
        # Indexes, attention mask and labels extraction
        indexes_sliced = self.indexes[indexes4slicing, :]
        attention_mask_sliced = self.attention_mask[indexes4slicing, :]
        labels_sliced = self.labels[indexes4slicing, :]

        # Arrangement
        data_batch = {'indexes': indexes_sliced, 'attention_mask': attention_mask_sliced, 'labels': labels_sliced}

        # Output
        return data_batch

    # NUMBER OF DATA EXTRACTION
    def __len__(self):
        return self.number_data
