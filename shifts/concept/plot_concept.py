"""
DESCRIPTION: classes and functions to draw plots.
AUTHOR: Pablo Ferri-Borredà
DATE: 12/07/22
"""

# MODULES IMPORT
import matplotlib.pyplot as mpl
from pandas import DataFrame
from seaborn import heatmap

# COMMON SETTINGS
# Two column text
TWO_COLS_TEXT = True

# Configuration for one column text
if not TWO_COLS_TEXT:
    fontsize = 14

# Configuration of two column text
else:
    fontsize = 30


# PLOTTING FUNCTIONS
def plot_performance_matrix(perf_matrix: DataFrame, metric: str) -> None:
    # Header generation
    if metric == 'AUC':
        header = 'Area under curve'
    elif metric == 'F1s':
        header = 'F1-score'
    else:
        raise ValueError

    # Data selection
    years = [2009, 2010, 2011, 2012, 2014, 2015, 2016, 2017, 2018, 2019]
    auc_matrix_sel = perf_matrix[years]

    # Transposition
    data_frame_transposed = auc_matrix_sel.transpose()

    # Plotting
    mpl.rc('font', family='serif')
    mpl.figure()
    axes = mpl.axes()
    graph = heatmap(data=data_frame_transposed, cmap='RdYlGn', ax=axes, xticklabels=years, annot=True, fmt='.3g',
            annot_kws={'size': fontsize - 2})  # cmap='RdYlGn', 'seismic'
    # set(font_scale=5)

    cax = graph.figure.axes[-1]
    cax.tick_params(labelsize=fontsize - 2)

    mpl.xticks(fontsize=fontsize - 2, fontname='serif')
    mpl.xlabel('Evaluation year', fontsize=fontsize, fontname='serif')

    mpl.yticks(rotation=0, fontsize=fontsize - 2, fontname='serif')
    mpl.ylabel('Training year', fontsize=fontsize, fontname='serif')

    axes.set_title(f'{header} matrix', fontsize=fontsize + 2, fontname='serif')
    mpl.show()
