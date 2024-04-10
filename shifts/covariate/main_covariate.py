"""
DESCRIPTION: main script to explore covariate shifts.
AUTHOR: Pablo Ferri-Borredà
DATE: 01/07/22
"""

# MODULES IMPORT
from os.path import join

from filehandling import load_data, export_data
from shifts.covariate import arrange_covariate as ar
from shifts.covariate import prep_covariate as pr
from shifts.covariate import traintest_covariate as tt

# SETTINGS
# Data directory
data_directory = './Data/'

# Data filename
data_filename = 'prepared_data.pkl'

# Results directory
results_directory = './Results/'

# Results filename
results_filename = 'covariate_shift_analysis.pkl'

# EXECUTION
if __name__ == '__main__':
    # DATA LOADING
    # Filepath definition
    absolute_filepath = join(data_directory, data_filename)

    # Loading
    data = load_data(absolute_filepath)

    # ONE-HOT ENCODING YEAR VARIABLE
    data = pr.one_hot_encode_years(data)

    # DATA SEGREGATION
    # segregation
    data_seg = ar.segregate_data(data)
    # workspace cleaning
    del data

    # DATA SPLITTING
    # splitting
    data_split = ar.split_data(data_seg)
    # workspace cleaning
    del data_seg

    # WORKING MATRICES GENERATION
    # generation
    working_matrices = ar.generate_working_matrices(data_split)
    # workspace cleaning
    del data_split

    # DATA LOADERS GENERATION
    # generation
    data_loaders = ar.generate_data_loaders(working_matrices)
    # workspace cleaning
    del working_matrices

    # ITERATIVE MODELS TRAINING AND EVALUATION
    results = tt.train_test(data_loaders)

    # RESULTS EXPORTING
    export_data(data=results, filepath=results_filename)
