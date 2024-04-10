"""
DESCRIPTION: script for launching tensorboard.
AUTHOR: Pablo Ferri-Borredà
DATE: 01/07/22
"""

# MODULES IMPORT
from os import system

# LOGGING PATH DEFINITION
log_directory = 'Results/tensorboard/'

# TENSORBOARD LAUNCHING
system('tensorboard --logdir ' + log_directory)
