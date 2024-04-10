"""
DESCRIPTION: functions to arrange data.
AUTHOR: Pablo Ferri-Borredà
DATE: 27/06/22
"""

# MODULES IMPORT
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
from numpy import array
from pandas import DataFrame

# SETTINGS
TEXT_COLUMN = 'TEXT'
EXPERIENCE_COLUMN = 'EXPERIENCE_IDF'
TEST_RATIO = 0.2
VAL_RATIO = 0.3
TIME_WINDOWS = (2009, 2010, 2011, 2012, 2014, 2015, 2016, 2017, 2018, 2019)


# EXPERIENCE IDENTIFIERS GENERATION
def generate_experience_identifiers(data: DataFrame) -> DataFrame:
    # Year identifier
    # Grouping variable generation
    data[EXPERIENCE_COLUMN] = data['YEAR']

    # Discarding of the date column
    data = data.drop(columns=['YEAR'])

    # Output
    return data


# DATA SPLITTING
def split_data(data: DataFrame) -> dict:
    # Train and test split
    # splitter initialization
    train_test_splitter = HoldoutSplitter(eval_ratio=TEST_RATIO)
    # split
    data_train, data_test = train_test_splitter.split(data)

    # Pure train and validation split
    # splitter initialization
    train_val_splitter = HoldoutSplitter(eval_ratio=VAL_RATIO)
    # split
    data_trainpure, data_val = train_val_splitter.split(data_train)

    # Arrangement
    data_split = {'train': data_trainpure, 'val': data_val, 'train_val': data_train, 'test': data_test}

    # Output
    return data_split


# SPLITTER
class Splitter(ABC):

    # INITIALIZATION
    def __init__(self, random_seed=8374):
        # Inputs checking
        if type(random_seed) is not int:
            raise TypeError

        # Splitting attributes
        self._random_seed = random_seed

    # EXTERNAL ATTRIBUTE ACCES AND EDITION CONTROL
    @property
    def random_seed(self):
        return self._random_seed

    # SPLITTING
    @abstractmethod
    def split(self, data_frame: DataFrame):
        raise NotImplementedError

    # NUMBER OF DATA EXTRACTION
    @staticmethod
    def _get_number_data(data_frame: DataFrame) -> int:
        return data_frame.shape[0]


# HOLDOUT SPLITTER
class HoldoutSplitter(Splitter):
    # INITIALIZATION
    def __init__(self, eval_ratio=0.2):
        super().__init__()
        self._eval_ratio = eval_ratio

    # SPLITTING
    def split(self, data_frame: DataFrame) -> tuple:
        # Masks getting
        mask_train, mask_eval = self._get_split_masks(data_frame)

        # Data splitting
        data_train, data_eval = data_frame[mask_train], data_frame[mask_eval]

        # Output
        return data_train, data_eval

    def _get_split_masks(self, data_frame):
        number_data = self._get_number_data(data_frame)
        indexes = np.linspace(start=0, stop=(number_data - 1), num=number_data, dtype=int)
        indexes_full = pd.DataFrame(indexes, columns=['SAMPLE_IDXS'])
        indexes_evaluation = indexes_full.sample(frac=self._eval_ratio, replace=False, random_state=self._random_seed)
        indexes_training = indexes_full.iloc[indexes_full.index.difference(indexes_evaluation.index)]
        mask_train = self._get_mask(indexes_training, number_data)
        mask_eval = self._get_mask(indexes_evaluation, number_data)

        return mask_train, mask_eval

    @staticmethod
    def _get_mask(indexes_series, number_data):
        indexes_array = indexes_series['SAMPLE_IDXS'].values
        mask_list = [True if i in indexes_array else False for i in range(0, number_data)]

        return pd.Series(mask_list).values


# WORKING MATRICES GENERATION
def generate_working_matrices(data_split: dict) -> dict:
    # Initialization
    # memory allocation
    working_matrices = {year: {'train': {'indexes': None, 'attention_mask': None, 'labels': None},
                               'val': {'indexes': None, 'attention_mask': None, 'labels': None},
                               'train_val': {'indexes': None, 'attention_mask': None, 'labels': None},
                               'test': {'indexes': None, 'attention_mask': None, 'labels': None}} for year in
                        TIME_WINDOWS}

    # Iterative loop for feeders generation
    for time_idf in TIME_WINDOWS:
        # Time masks generation
        # training
        time_mask_train = data_split['train'][EXPERIENCE_COLUMN] == time_idf
        # validation
        time_mask_val = data_split['val'][EXPERIENCE_COLUMN] == time_idf
        # training and validation
        time_mask_train_val = data_split['train_val'][EXPERIENCE_COLUMN] == time_idf
        # test
        time_mask_test = data_split['test'][EXPERIENCE_COLUMN] == time_idf

        # Data slicing
        # training
        data_train_time_window = data_split['train'][time_mask_train.values]
        # validation
        data_val_time_window = data_split['val'][time_mask_val.values]
        # training and validation
        data_train_val_time_window = data_split['train_val'][time_mask_train_val.values]
        # test
        data_test_time_window = data_split['test'][time_mask_test.values]

        # Index matrices extraction and arrangement
        # training
        working_matrices[time_idf]['train']['indexes'] = array(data_train_time_window[TEXT_COLUMN + '_IDX'].to_list())
        # validation
        working_matrices[time_idf]['val']['indexes'] = array(data_val_time_window[TEXT_COLUMN + '_IDX'].to_list())
        # training and validation
        working_matrices[time_idf]['train_val']['indexes'] = array(
            data_train_val_time_window[TEXT_COLUMN + '_IDX'].to_list())
        # test
        working_matrices[time_idf]['test']['indexes'] = array(data_test_time_window[TEXT_COLUMN + '_IDX'].to_list())

        # Attention matrices extraction and arrangement
        # training
        working_matrices[time_idf]['train']['attention_mask'] = array(
            data_train_time_window[TEXT_COLUMN + '_ATM'].to_list())
        # validation
        working_matrices[time_idf]['val']['attention_mask'] = array(
            data_val_time_window[TEXT_COLUMN + '_ATM'].to_list())
        # training and validation
        working_matrices[time_idf]['train_val']['attention_mask'] = array(
            data_train_val_time_window[TEXT_COLUMN + '_ATM'].to_list())
        # test
        working_matrices[time_idf]['test']['attention_mask'] = array(
            data_test_time_window[TEXT_COLUMN + '_ATM'].to_list())

        # Label matrices extraction and arrangement
        # training
        working_matrices[time_idf]['train']['labels'] = data_train_time_window[
            ['LIFE_THREATENING_NO', 'LIFE_THREATENING_YES']].values
        # validation
        working_matrices[time_idf]['val']['labels'] = data_val_time_window[
            ['LIFE_THREATENING_NO', 'LIFE_THREATENING_YES']].values
        # training and validation
        working_matrices[time_idf]['train_val']['labels'] = data_train_val_time_window[
            ['LIFE_THREATENING_NO', 'LIFE_THREATENING_YES']].values
        # test
        working_matrices[time_idf]['test']['labels'] = data_test_time_window[
            ['LIFE_THREATENING_NO', 'LIFE_THREATENING_YES']].values

    # Output
    return working_matrices
