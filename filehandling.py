"""
DESCRIPTION: functions to read and export data files.
AUTHOR: Pablo Ferri-Borredà
DATE: 01/07/22
"""

# MODULES IMPORT
import pickle as pk


# DATA LOADING
def load_data(filepath: str):
    with open(filepath, 'rb') as handle:
        return pk.load(handle)


# DATA EXPORTING
def export_data(*, data, filepath: str):
    with open(filepath, 'wb') as handle:
        pk.dump(data, handle, protocol=pk.HIGHEST_PROTOCOL)
