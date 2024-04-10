"""
DESCRIPTION: main script to explore data.
AUTHOR: Pablo Ferri-Borredà
DATE: 12/07/22
"""

# MODULES IMPORT
from os.path import join

from filehandling import load_data
from shifts.concept import arrange_concept as ar
from shifts.concept import plot_concept as pt

# SETTINGS
# Metric to explore
METRIC = 'F1s'

# Results directory
results_directory = './Results/'

# Results filename
results_filename = 'results_SingleFineTuning.pkl'

# EXECUTION
if __name__ == '__main__':
    # RESULTS LOADING
    # Filepath definition
    absolute_filepath = join(results_directory, results_filename)

    # Loading
    results = load_data(absolute_filepath)

    # PERFORMANCE MATRIX CONSTRUCTION
    perf_matrix = ar.get_performance_matrix(results, METRIC)

    # MATRIX PLOT
    pt.plot_performance_matrix(perf_matrix, METRIC)
