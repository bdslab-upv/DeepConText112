"""
DESCRIPTION: main script to prepare text data to learn continually.
AUTHOR: Pablo Ferri-Borredà
DATE: 27/06/22
"""

# MODULES IMPORT
from os.path import join

from arrangement import arrangement as ar
from filehandling import load_data, export_data

# SETTINGS
# Data directory
data_directory = './Data/'

# Data files names
data2load_filename = 'prepared_data.pkl'
data2export_filename = 'working_matrices.pkl'

# EXECUTION
if __name__ == '__main__':
    # DATA LOADING
    # Filepath definition
    absolute_filepath_loading = join(data_directory, data2load_filename)

    # Loading
    data = load_data(absolute_filepath_loading)

    # IDENTIFIERS GENERATION
    data = ar.generate_experience_identifiers(data)

    # DATA SPLITTING
    # splitting
    data_split = ar.split_data(data)
    # workspace cleaning
    del data

    # WORKING MATRICES GENERATION
    working_matrices = ar.generate_working_matrices(data_split)
    del data_split

    # WORKING MATRICES EXPORTING
    # Filepath definition
    absolute_filepath_exporting = join(data_directory, data2export_filename)

    # Exporting
    export_data(data=working_matrices, filepath=absolute_filepath_exporting)
