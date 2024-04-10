"""
DESCRIPTION: main script to prepare text data from Cordex and Coordcom system 112 to learn continually.
AUTHOR: Pablo Ferri-Borredà
DATE: 01/07/22
"""

# MODULES IMPORT
from os.path import join

from preparation.dataprep import prepare_data
from preparation.filtering import filter_data
from filehandling import load_data, export_data

# SETTINGS
# Data directory
data_directory = './Data/'

# Data filenames
data_filename_load = 'data_original.pkl'
data_filename_export = 'prepared_data.pkl'

# EXECUTION
if __name__ == '__main__':
    # DATA LOADING
    # Filepath definition
    absolute_filepath_load = join(data_directory, data_filename_load)

    # Loading
    data = load_data(absolute_filepath_load)

    # DATA FILTERING
    data = filter_data(data)

    # DATA PREPARATION
    data = prepare_data(data)

    # PREPARED DATA EXPORTING
    # Filepath definition
    absolute_filepath_export = join(data_directory, data_filename_export)

    # Exporting
    export_data(data=data, filepath=absolute_filepath_export)
