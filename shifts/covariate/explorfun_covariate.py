"""
DESCRIPTION: auxiliar functions for figures generation.
AUTHOR: Pablo Ferri-Borredà
DATE: 25/07/22
"""

# MODULES IMPORT
from math import isnan
from os.path import join

import matplotlib.pyplot as mpl
from pandas import DataFrame
from seaborn import heatmap

from filehandling import load_data

# COMMON SETTINGS
# Two column text
TWO_COLS_TEXT = True

# Configuration for one column text
if not TWO_COLS_TEXT:
    fontsize = 14

# Configuration of two column text
else:
    fontsize = 27


# RESULTS LOADING
def load_results() -> dict:
    # Settings
    results_directory = './Results/'
    filename = 'covariate_shift_analysis.pkl'
    absolute_filepath = join(results_directory, filename)

    # Results loading
    results_raw = load_data(absolute_filepath)

    # Output
    return results_raw


# RESULTS ARRANGEMENT
def arrange_results(results_raw: dict, metric_identifier: str) -> DataFrame:
    # Inputs checking
    if metric_identifier not in ('AUC', 'F1s'):
        raise ValueError('Unrecognized metric identifier.')

    # Data frame initialization
    years = (2009, 2010, 2011, 2012, 2014, 2015, 2016, 2017, 2018, 2019)
    results_frame = DataFrame(index=years[:-1], columns=years[1:], dtype='float')

    # Filling
    for experiment_key, pre_post_metrics in results_raw.items():
        year_i, year_j = experiment_key[0], experiment_key[1]

        if metric_identifier == 'AUC':
            metric_value = pre_post_metrics['presaturation']['AUC_MACRO']
        elif metric_identifier == 'F1s':
            metric_value = pre_post_metrics['postsaturation']['F1-SCORE_MACRO']
        else:
            raise ValueError('Unrecognized metric identifier.')

        results_frame[year_j][results_frame.index == year_i] = metric_value

    # Output
    return results_frame


# RESULTS PLOTTING
def plot_results(results_arranged: DataFrame, metric_identifier: str):
    # Captions generation
    if metric_identifier == 'AUC':
        title_ = 'Area under curve matrix'
    elif metric_identifier == 'F1s':
        title_ = 'F1-score macro matrix'
    else:
        raise ValueError('Unrecognized metric identifier.')

    # Nan mask generation
    nan_mask = results_arranged.applymap(lambda x: 1 if isnan(x) else 0)

    # Plotting
    mpl.rc('font', family='serif')
    mpl.figure()
    axes = mpl.axes()
    graph = heatmap(data=results_arranged, cmap='RdYlGn_r', ax=axes, xticklabels=True, yticklabels=True, mask=nan_mask,
                    annot=True, fmt='.3g', annot_kws={'size': fontsize - 2})  # cmap='RdYlGn', 'seismic'
    # set(font_scale=5)

    cax = graph.figure.axes[-1]
    cax.tick_params(labelsize=fontsize - 2)

    mpl.xticks(fontsize=fontsize - 2, fontname='serif')
    mpl.xlabel('Year', fontsize=fontsize, fontname='serif')

    mpl.yticks(rotation=0, fontsize=fontsize - 2, fontname='serif')
    mpl.ylabel('Year', fontsize=fontsize, fontname='serif')

    axes.set_title(title_, fontsize=fontsize + 2, fontname='serif')
    mpl.show()
