"""
DESCRIPTION: main script to explore prior probability shifts.
AUTHOR: Pablo Ferri-Borredà
DATE: 01/07/22
"""

# MODULES IMPORT
from os.path import join

from statsmodels.tsa.stattools import kpss

from filehandling import load_data
from shifts.prior import plot_prior as pt

# SETTINGS
# Data directory
data_directory = './Data/'

# Data filename
data_filename = 'prepared_data.pkl'

# Temporal variable
temporal_variable = 'YEAR_MONTH'

# Label variable
label_variable = 'LIFE_THREATENING_YES'

# Number of months defining the time window with
months_window_width = 12

# Number of months defining the time window step
time_window_step = 4

# EXECUTION
if __name__ == '__main__':
    # DATA LOADING
    # Filepath definition
    absolute_filepath = join(data_directory, data_filename)

    # Loading
    data = load_data(absolute_filepath)

    # DATA SELECTION
    data = data[[temporal_variable, label_variable]]

    # METADATA EXTRACTION
    # sorted unique months
    sorted_unique_months = sorted(data[temporal_variable].unique())
    # number months
    number_months = len(sorted_unique_months)

    # EXPLORATION
    # Initialization
    # pointers
    pointer_low = 0
    pointer_high = months_window_width
    # month integer to abbreviation map
    month2abv = {'01': 'JAN', '02': 'FEB', '03': 'MAR', '04': 'APR', '05': 'MAY', '06': 'JUN', '07': 'JUL', '08': 'AUG',
                 '09': 'SEP', '10': 'OCT', '11': 'NOV', '12': 'DEC'}

    # memory allocation
    time_var = []
    no_proba = []
    yes_proba = []

    # Values calculation
    while pointer_high <= number_months:
        # Months extraction
        months = sorted_unique_months[pointer_low: pointer_high]

        # Time variable generation
        year_month_low_i = str(months[0])
        year_month_high_i = str(months[-1])
        month_low_i = year_month_low_i[-2:]
        month_high_i = year_month_high_i[-2:]
        year_low_i = year_month_low_i[2:4]
        year_high_i = year_month_high_i[2:4]
        time_var_i = month2abv[month_low_i] + year_low_i + '->' + month2abv[month_high_i] + year_high_i

        # Data slicing
        # mask generation
        mask = data[temporal_variable].isin(months)
        # slicing
        data_i = data[mask]

        # Calculation
        number_cases_i = data_i.shape[0]
        yes_proba_i = data_i[label_variable].sum() / number_cases_i

        # Arrangement
        time_var.append(time_var_i)
        yes_proba.append(yes_proba_i)

        # Pointers upgrading
        pointer_low += time_window_step
        pointer_high += time_window_step

    # Exploration
    # life-threatening
    pt.plot_label_one_class_dots(time_var, yes_proba)

    # STATIONARY STATISTICAL TESTS
    # Memory allocation
    pvalue_map = {'yes_probability': {'kpss': None}}

    # KPSS test
    pvalue_map['yes_probability']['kpss'] = round(kpss(yes_proba)[1], 3)
