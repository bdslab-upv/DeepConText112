"""
DESCRIPTION: classes and functions to draw plots.
AUTHOR: Pablo Ferri-Borredà
DATE: 01/07/22
"""

# MODULES IMPORT
import matplotlib.pyplot as mpl

# COMMON SETTINGS
# Two column text
TWO_COLS_TEXT = True

alpha = 0.5
markeredgewidth = 1

# Configuration for one column text
if not TWO_COLS_TEXT:
    linewidth = 3
    markersize = 15
    fontsize = 14

# Configuration of two column text
else:
    linewidth = 5
    markersize = 25
    fontsize = 30


# PLOTTING FUNCTION
def plot_label_one_class_dots(time_var: list, yes_probs: list) -> None:
    mpl.rc('font', family='serif')

    mpl.figure()
    mpl.plot(time_var, yes_probs, color='red', marker='o', linewidth=linewidth, markersize=markersize,
             markeredgecolor='black', markeredgewidth=markeredgewidth, label='LIFE-THREATENING=YES')

    mpl.ylim([0.32, 0.425])
    mpl.yticks(fontsize=fontsize)
    mpl.ylabel('Empirical probability', fontsize=fontsize)

    mpl.title('Empirical life-threatening probability', fontsize=fontsize + 2)

    if not TWO_COLS_TEXT:
        mpl.xticks(fontsize=fontsize-4, rotation=45)
    else:
        mpl.xticks(fontsize=fontsize - 8, rotation=60)

    mpl.legend(ncol=2, fontsize=fontsize)
    mpl.grid(color='grey', linestyle='dashed', linewidth=0.5)
    mpl.show()

