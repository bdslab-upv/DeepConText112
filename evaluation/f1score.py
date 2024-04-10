"""
DESCRIPTION: f1_score metric avalanche.
AUTHOR: Pablo Ferri-Borredà
DATE: 13/06/22
"""

# MODULES IMPORT
from typing import List, Union, Dict

import torch
from torch import Tensor
from avalanche.evaluation import Metric, PluginMetric, GenericPluginMetric
from avalanche.evaluation.metrics.mean import Mean
from avalanche.evaluation.metric_utils import phase_and_task
from collections import defaultdict

from evaluation.posneg import PositivesNegativesCalculator as PosNegCalc


class F1_score(Metric[float]):
    """
    The F1_score metric. This is a standalone metric.

    The metric keeps a dictionary of <task_label, f1_score value> pairs.
    and update the values through a running average over multiple
    <prediction, target> pairs of Tensors, provided incrementally.
    The "prediction" and "target" tensors may contain plain labels or
    one-hot/logit vectors.

    Each time `result` is called, this metric emits the average f1_score
    across all predictions made since the last `reset`.

    The reset method will bring the metric to its initial state. By default
    this metric in its initial state will return an f1_score value of 0.
    """

    _positive_class_index = 1  # FIXME This is a provisional implementation, assuming a binary label.

    def __init__(self):
        """
        Creates an instance of the standalone F1_score metric.

        By default this metric in its initial state will return an f1_score
        value of 0. The metric can be updated by using the `update` method
        while the running f1_score can be retrieved using the `result` method.
        """
        self._mean_f1_score = defaultdict(Mean)
        """
        The mean utility that will be used to store the running f1_score
        for each task label.
        """

    @torch.no_grad()
    def update(self, predicted_y: Tensor, true_y: Tensor,
               task_labels: Union[float, Tensor]) -> None:
        """
        Update the running f1_score given the true and predicted labels.
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

            # Recall calculation
            recall = posneg.true_positives / (posneg.true_positives + posneg.false_negatives) \
                if (posneg.true_positives + posneg.false_negatives) != 0 else float('nan')

            # Precision calculation
            precision = posneg.true_positives / (posneg.true_positives + posneg.false_positives) \
                if (posneg.true_positives + posneg.false_positives) != 0 else float('nan')

            # F1_score calculation
            f1_score = (2 * recall * precision) / (recall + precision) if (recall + precision) != 0 else float('nan')

            # Arrangement
            self._mean_f1_score[task_labels].update(f1_score)

        elif isinstance(task_labels, Tensor):
            raise NotImplementedError
        else:
            raise ValueError(f"Task label type: {type(task_labels)}, "
                             f"expected int/float or Tensor")

    def result(self, task_label=None) -> Dict[int, float]:
        """
        Retrieves the running f1_score.

        Calling this method will not change the internal state of the metric.

        :param task_label: if None, return the entire dictionary of f1_scores
            for each task. Otherwise return the dictionary
            `{task_label: f1_score}`.
        :return: A dict of running f1_scores for each task label,
            where each value is a float value between 0 and 1.
        """
        assert (task_label is None or isinstance(task_label, int))
        if task_label is None:
            return {k: v.result() for k, v in self._mean_f1_score.items()}
        else:
            return {task_label: self._mean_f1_score[task_label].result()}

    def reset(self, task_label=None) -> None:
        """
        Resets the metric.
        :param task_label: if None, reset the entire dictionary.
            Otherwise, reset the value associated to `task_label`.

        :return: None.
        """
        assert (task_label is None or isinstance(task_label, int))
        if task_label is None:
            self._mean_f1_score = defaultdict(Mean)
        else:
            self._mean_f1_score[task_label].reset()


class F1_scorePluginMetric(GenericPluginMetric[float]):
    """
    Base class for all f1_scores plugin metrics
    """

    def __init__(self, reset_at, emit_at, mode):
        self._f1_score = F1_score()
        super(F1_scorePluginMetric, self).__init__(
            self._f1_score, reset_at=reset_at, emit_at=emit_at,
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
        self._f1_score.update(strategy.mb_output, strategy.mb_y, task_labels)


class MinibatchF1_score(F1_scorePluginMetric):
    """
    The minibatch plugin f1_score metric.
    This metric only works at training time.

    This metric computes the average f1_score over patterns
    from a single minibatch.
    It reports the result after each iteration.

    If a more coarse-grained logging is needed, consider using
    :class:`EpochF1_score` instead.
    """

    def __init__(self):
        """
        Creates an instance of the MinibatchF1_score metric.
        """
        super(MinibatchF1_score, self).__init__(
            reset_at='iteration', emit_at='iteration', mode='train')

    def __str__(self):
        return "Top1_F1s_MB"


class EpochF1_score(F1_scorePluginMetric):
    """
    The average f1_score over a single training epoch.
    This plugin metric only works at training time.

    The f1_score will be logged after each training epoch by computing
    the number of correctly predicted patterns during the epoch divided by
    the overall number of patterns encountered in that epoch.
    """

    def __init__(self):
        """
        Creates an instance of the EpochF1_score metric.
        """

        super(EpochF1_score, self).__init__(
            reset_at='epoch', emit_at='epoch', mode='train')

    def __str__(self):
        return "Top1_F1s_Epoch"


class RunningEpochF1_score(F1_scorePluginMetric):
    """
    The average f1_score across all minibatches up to the current
    epoch iteration.
    This plugin metric only works at training time.

    At each iteration, this metric logs the f1_score averaged over all patterns
    seen so far in the current epoch.
    The metric resets its state after each training epoch.
    """

    def __init__(self):
        """
        Creates an instance of the RunningEpochF1_score metric.
        """

        super(RunningEpochF1_score, self).__init__(
            reset_at='epoch', emit_at='iteration', mode='train')

    def __str__(self):
        return "Top1_RunningF1s_Epoch"


class ExperienceF1_score(F1_scorePluginMetric):
    """
    At the end of each experience, this plugin metric reports
    the average f1_score over all patterns seen in that experience.
    This metric only works at eval time.
    """

    def __init__(self):
        """
        Creates an instance of ExperienceF1_score metric
        """
        super(ExperienceF1_score, self).__init__(
            reset_at='experience', emit_at='experience', mode='eval')

    def __str__(self):
        return "Top1_F1s_Exp"


class StreamF1_score(F1_scorePluginMetric):
    """
    At the end of the entire stream of experiences, this plugin metric
    reports the average f1_score over all patterns seen in all experiences.
    This metric only works at eval time.
    """

    def __init__(self):
        """
        Creates an instance of StreamF1_score metric
        """
        super(StreamF1_score, self).__init__(
            reset_at='stream', emit_at='stream', mode='eval')

    def __str__(self):
        return "Top1_F1s_Stream"


class TrainedExperienceF1_score(F1_scorePluginMetric):
    """
    At the end of each experience, this plugin metric reports the average
    f1_score for only the experiences that the model has been trained on so far.

    This metric only works at eval time.
    """

    def __init__(self):
        """
        Creates an instance of TrainedExperienceF1_score metric by first
        constructing F1_scorePluginMetric
        """
        super(TrainedExperienceF1_score, self).__init__(
            reset_at='stream', emit_at='stream', mode='eval')
        self._current_experience = 0

    def after_training_exp(self, strategy) -> None:
        self._current_experience = strategy.experience.current_experience
        # Reset average after learning from a new experience
        F1_scorePluginMetric.reset(self, strategy)
        return F1_scorePluginMetric.after_training_exp(self, strategy)

    def update(self, strategy):
        """
        Only update the f1_score with results from experiences that have been
        trained on
        """
        if strategy.experience.current_experience <= self._current_experience:
            F1_scorePluginMetric.update(self, strategy)

    def __str__(self):
        return "F1_score_On_Trained_Experiences"


def f1_score_metrics(*,
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
        the minibatch f1_score at training time.
    :param epoch: If True, will return a metric able to log
        the epoch f1_score at training time.
    :param epoch_running: If True, will return a metric able to log
        the running epoch f1_score at training time.
    :param experience: If True, will return a metric able to log
        the f1_score on each evaluation experience.
    :param stream: If True, will return a metric able to log
        the f1_score averaged over the entire evaluation stream of experiences.
    :param trained_experience: If True, will return a metric able to log
        the average evaluation f1_score only for experiences that the
        model has been trained on

    :return: A list of plugin metrics.
    """

    metrics = []
    if minibatch:
        metrics.append(MinibatchF1_score())

    if epoch:
        metrics.append(EpochF1_score())

    if epoch_running:
        metrics.append(RunningEpochF1_score())

    if experience:
        metrics.append(ExperienceF1_score())

    if stream:
        metrics.append(StreamF1_score())

    if trained_experience:
        metrics.append(TrainedExperienceF1_score())

    return metrics


__all__ = [
    'F1_score',
    'MinibatchF1_score',
    'EpochF1_score',
    'RunningEpochF1_score',
    'ExperienceF1_score',
    'StreamF1_score',
    'TrainedExperienceF1_score',
    'f1_score_metrics'
]
