"""
DESCRIPTION: functions and classes to arrange results from Cordex and Coordcom continual learning experiments.
AUTHOR: Pablo Ferri-Borredà
DATE: 12/07/22
"""

# MODULES IMPORT
from pandas import DataFrame


# PERFORMANCE OBTENTION
def get_performance_matrix(results: DataFrame, metric: str) -> DataFrame:
    # Rows filtering
    auc_matrix = results[results['METRIC'].str.contains(f'{metric}_Exp')]

    # Output
    return auc_matrix
