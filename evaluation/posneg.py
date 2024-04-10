"""
DESCRIPTION: auxiliar functions to calculate true and false positives and true and false negatives.
AUTHOR: Pablo Ferri-Borredà
DATE: 13/06/22
"""

# MODULES IMPORT
from torch import Tensor

# POSITIVES AND NEGATIVES CALCULATOR
class PositivesNegativesCalculator:
    # INITIALIZATION
    def __init__(self, true_y: Tensor, predicted_y: Tensor, positive_class_index: int) -> None:
        # Boolean comparisons
        # true values
        ytrue_eq = true_y == positive_class_index  # equal
        ytrue_neq = true_y != positive_class_index  # not equal
        # predicted values
        yhat_eq = predicted_y == positive_class_index  # equal
        yhat_neq = predicted_y != positive_class_index  # not equal

        # Values calculation
        # true positives
        true_pos = (yhat_eq & ytrue_eq).sum().item()
        # true negatives
        true_neg = (yhat_neq & ytrue_neq).sum().item()
        # false positives
        false_pos = (yhat_eq & ytrue_neq).sum().item()
        # false negatives
        false_neg = (yhat_neq & ytrue_eq).sum().item()

        # Attributes assignation
        self._true_positives = true_pos
        self._true_negatives = true_neg
        self._false_positives = false_pos
        self._false_negatives = false_neg

    # EXTERNAL ATTRIBUTE ACCESS AND EDITION CONTROL
    # True positives
    @property
    def true_positives(self):
        return self._true_positives

    # True negatives
    @property
    def true_negatives(self):
        return self._true_negatives

    # False positives
    @property
    def false_positives(self):
        return self._false_positives

    # False negatives
    @property
    def false_negatives(self):
        return self._false_negatives





