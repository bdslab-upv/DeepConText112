"""
DESCRIPTION: functions and classes to filter data.
AUTHOR: Pablo Ferri-Borredà
DATE: 27/06/22
"""

# MODULES IMPORT
from pandas import DataFrame


# DATA FILTERING
def filter_data(data: DataFrame) -> DataFrame:
    # Missing values discarding
    data = data.dropna(subset=['TEXT'])

    # Output
    return data
