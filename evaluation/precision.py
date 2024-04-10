"""
DESCRIPTION: precision metric avalanche.
AUTHOR: Pablo Ferri-Borredà
DATE: 13/06/22
"""

# MODULES IMPORT
from collections import defaultdict
from typing import List, Union, Dict

import torch
from avalanche.evaluation import Metric, PluginMetric, GenericPluginMetric
from avalanche.evaluation.metric_utils import phase_and_task
from avalanche.evaluation.metrics.mean import Mean
from torch import Tensor

from evaluation.posneg import PositivesNegativesCalculator as PosNegCalc


class Precision(Metric[float]):
    """
    The Precision metric. This is a standalone metric.

    The metric keeps a dictionary of <task_label, Precision value> pairs.
    and update the values through a running average over multiple
    <prediction, target> pairs of Tensors, provided incrementally.
    The "prediction" and "target" tensors may contain plain labels or
    one-hot/logit vectors.

    Each time `result` is called, this metric emits the average Precision
    across all predictions made since the last `reset`.

    The reset method will bring the metric to its initial state. By default
    this metric in its initial state will return a precision value of 0.
    """

    _positive_class_index = 1  # FIXME This is a provisional implementation, assuming a binary label.

    def __init__(self):
        """
        Creates an instance of the standalone precision metric.

        By default this metric in its initial state will return an precision
        value of 0. The metric can be updated by using the `update` method
        while the running precision can be retrieved using the `result` method.
        """
        self._mean_precision = defaultdict(Mean)
        """
        The mean utility that will be used to store the running precision
        for each task label.
        """

    @torch.no_grad()
    def update(self, predicted_y: Tensor, true_y: Tensor,
               task_labels: Union[float, Tensor]) -> None:
        """
        Update the running precision given the true and predicted labels.
        Parameter `task_labels` is used to decide how to update the inner
        dictionary: if Float, only the dictionary value related to that task
        is updated. If Tensor, all the dictionary elements belonging to the
        task labels will be updated.

        :param predicted_y: The model prediction. Both labels and logit vectors
            are supported.
        :param true_y: The ground truth. Both labels and one-hot vectors
            are supported.
        :param task_labels: the int task label associated to the current
            experience or the task labels vector showing the task label
            for each pattern.

        :return: None.
        """
        if len(true_y) != len(predicted_y):
            raise ValueError('Size mismatch for true_y and predicted_y tensors')

        if isinstance(task_labels, Tensor) and len(task_labels) != len(true_y):
            raise ValueError('Size mismatch for true_y and task_labels tensors')

        true_y = torch.as_tensor(true_y)
        predicted_y = torch.as_tensor(predicted_y)

        # Check if logits or labels
        if len(predicted_y.shape) > 1:
            # Logits -> transform to labels
            predicted_y = torch.max(predicted_y, 1)[1]

        if len(true_y.shape) > 1:
            # Logits -> transform to labels
            true_y = torch.max(true_y, 1)[1]

        if isinstance(task_labels, int):
            # Positives and negatives calculation
            posneg = PosNegCalc(true_y=true_y, predicted_y=predicted_y, positive_class_index=self._positive_class_index)

            # precision calculation
            precision = posneg.true_positives / (posneg.true_positives + posneg.false_positives) \
                if (posneg.true_positives + posneg.false_positives) != 0 else float('nan')

            # Arrangement
            self._mean_precision[task_labels].update(precision)

        elif isinstance(task_labels, Tensor):
            raise NotImplementedError
        else:
            raise ValueError(f"Task label type: {type(task_labels)}, "
                             f"expected int/float or Tensor")

    def result(self, task_label=None) -> Dict[int, float]:
        """
        Retrieves the running precision.

        Calling this method will not change the internal state of the metric.

        :param task_label: if None, return the entire dictionary of precisions
            for each task. Otherwise return the dictionary
            `{task_label: precision}`.
        :return: A dict of running precisions for each task label,
            where each value is a float value between 0 and 1.
        """
        assert (task_label is None or isinstance(task_label, int))
        if task_label is None:
            return {k: v.result() for k, v in self._mean_precision.items()}
        else:
            return {task_label: self._mean_precision[task_label].result()}

    def reset(self, task_label=None) -> None:
        """
        Resets the metric.
        :param task_label: if None, reset the entire dictionary.
            Otherwise, reset the value associated to `task_label`.

        :return: None.
        """
        assert (task_label is None or isinstance(task_label, int))
        if task_label is None:
            self._mean_precision = defaultdict(Mean)
        else:
            self._mean_precision[task_label].reset()


class PrecisionPluginMetric(GenericPluginMetric[float]):
    """
    Base class for all precisions plugin metrics
    """

    def __init__(self, reset_at, emit_at, mode):
        self._precision = Precision()
        super(PrecisionPluginMetric, self).__init__(
            self._precision, reset_at=reset_at, emit_at=emit_at,
            mode=mode)

    def reset(self, strategy=None) -> None:
        if self._reset_at == 'stream' or strategy is None:
            self._metric.reset()
        else:
            self._metric.reset(phase_and_task(strategy)[1])

    def result(self, strategy=None) -> float:
        if self._emit_at == 'stream' or strategy is None:
            return self._metric.result()
        else:
            return self._metric.result(phase_and_task(strategy)[1])

    def update(self, strategy):
        # task labels defined for each experience
        task_labels = strategy.experience.task_labels
        if len(task_labels) > 1:
            # task labels defined for each pattern
            task_labels = strategy.mb_task_id
        else:
            task_labels = task_labels[0]
        self._precision.update(strategy.mb_output, strategy.mb_y, task_labels)


class MinibatchPrecision(PrecisionPluginMetric):
    """
    The minibatch plugin precision metric.
    This metric only works at training time.

    This metric computes the average precision over patterns
    from a single minibatch.
    It reports the result after each iteration.

    If a more coarse-grained logging is needed, consider using
    :class:`EpochPrecision` instead.
    """

    def __init__(self):
        """
        Creates an instance of the MinibatchPrecision metric.
        """
        super(MinibatchPrecision, self).__init__(
            reset_at='iteration', emit_at='iteration', mode='train')

    def __str__(self):
        return "Top1_Pre_MB"


class EpochPrecision(PrecisionPluginMetric):
    """
    The average precision over a single training epoch.
    This plugin metric only works at training time.

    The precision will be logged after each training epoch by computing
    the number of correctly predicted patterns during the epoch divided by
    the overall number of patterns encountered in that epoch.
    """

    def __init__(self):
        """
        Creates an instance of the EpochPrecision metric.
        """

        super(EpochPrecision, self).__init__(
            reset_at='epoch', emit_at='epoch', mode='train')

    def __str__(self):
        return "Top1_Pre_Epoch"


class RunningEpochPrecision(PrecisionPluginMetric):
    """
    The average precision across all minibatches up to the current
    epoch iteration.
    This plugin metric only works at training time.

    At each iteration, this metric logs the precision averaged over all patterns
    seen so far in the current epoch.
    The metric resets its state after each training epoch.
    """

    def __init__(self):
        """
        Creates an instance of the RunningEpochPrecision metric.
        """

        super(RunningEpochPrecision, self).__init__(
            reset_at='epoch', emit_at='iteration', mode='train')

    def __str__(self):
        return "Top1_RunningPre_Epoch"


class ExperiencePrecision(PrecisionPluginMetric):
    """
    At the end of each experience, this plugin metric reports
    the average precision over all patterns seen in that experience.
    This metric only works at eval time.
    """

    def __init__(self):
        """
        Creates an instance of ExperiencePrecision metric
        """
        super(ExperiencePrecision, self).__init__(
            reset_at='experience', emit_at='experience', mode='eval')

    def __str__(self):
        return "Top1_Pre_Exp"


class StreamPrecision(PrecisionPluginMetric):
    """
    At the end of the entire stream of experiences, this plugin metric
    reports the average precision over all patterns seen in all experiences.
    This metric only works at eval time.
    """

    def __init__(self):
        """
        Creates an instance of StreamPrecision metric
        """
        super(StreamPrecision, self).__init__(
            reset_at='stream', emit_at='stream', mode='eval')

    def __str__(self):
        return "Top1_Pre_Stream"


class TrainedExperiencePrecision(PrecisionPluginMetric):
    """
    At the end of each experience, this plugin metric reports the average
    precision for only the experiences that the model has been trained on so far.

    This metric only works at eval time.
    """

    def __init__(self):
        """
        Creates an instance of TrainedExperiencePrecision metric by first
        constructing PrecisionPluginMetric
        """
        super(TrainedExperiencePrecision, self).__init__(
            reset_at='stream', emit_at='stream', mode='eval')
        self._current_experience = 0

    def after_training_exp(self, strategy) -> None:
        self._current_experience = strategy.experience.current_experience
        # Reset average after learning from a new experience
        PrecisionPluginMetric.reset(self, strategy)
        return PrecisionPluginMetric.after_training_exp(self, strategy)

    def update(self, strategy):
        """
        Only update the precision with results from experiences that have been
        trained on
        """
        if strategy.experience.current_experience <= self._current_experience:
            PrecisionPluginMetric.update(self, strategy)

    def __str__(self):
        return "Precision_On_Trained_Experiences"


def precision_metrics(*,
                      minibatch=False,
                      epoch=False,
                      epoch_running=False,
                      experience=False,
                      stream=False,
                      trained_experience=False) -> List[PluginMetric]:
    """
    Helper method that can be used to obtain the desired set of
    plugin metrics.

    :param minibatch: If True, will return a metric able to log
        the minibatch precision at training time.
    :param epoch: If True, will return a metric able to log
        the epoch precision at training time.
    :param epoch_running: If True, will return a metric able to log
        the running epoch precision at training time.
    :param experience: If True, will return a metric able to log
        the precision on each evaluation experience.
    :param stream: If True, will return a metric able to log
        the precision averaged over the entire evaluation stream of experiences.
    :param trained_experience: If True, will return a metric able to log
        the average evaluation precision only for experiences that the
        model has been trained on

    :return: A list of plugin metrics.
    """

    metrics = []
    if minibatch:
        metrics.append(MinibatchPrecision())

    if epoch:
        metrics.append(EpochPrecision())

    if epoch_running:
        metrics.append(RunningEpochPrecision())

    if experience:
        metrics.append(ExperiencePrecision())

    if stream:
        metrics.append(StreamPrecision())

    if trained_experience:
        metrics.append(TrainedExperiencePrecision())

    return metrics


__all__ = [
    'Precision',
    'MinibatchPrecision',
    'EpochPrecision',
    'RunningEpochPrecision',
    'ExperiencePrecision',
    'StreamPrecision',
    'TrainedExperiencePrecision',
    'precision_metrics'
]
