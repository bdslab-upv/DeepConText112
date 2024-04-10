"""
DESCRIPTION: script for generating exploration figures for dataset shifts.
AUTHOR: Pablo Ferri-Borredà
DATE: 25/07/22
"""

# MODULES IMPORT
from shifts.covariate import explorfun_covariate as ef

# SETTINGS
METRIC = 'AUC'

# RESULTS LOADING
results_raw = ef.load_results()

# RESULTS ARRANGEMENT
metric_idf = 'AUC'  # 'AUC', 'F1s'
results_arranged = ef.arrange_results(results_raw, metric_identifier=metric_idf)

# RESULTS PLOTTING
ef.plot_results(results_arranged, metric_identifier=metric_idf)













