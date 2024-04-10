"""
DESCRIPTION: area under curve metric avalanche.
AUTHOR: Pablo Ferri-Borredà
DATE: 16/06/22
"""

# MODULES IMPORT
from typing import List, Union, Dict

import torch
from torch import Tensor
from avalanche.evaluation import Metric, PluginMetric, GenericPluginMetric
from avalanche.evaluation.metrics.mean import Mean
from avalanche.evaluation.metric_utils import phase_and_task
from collections import defaultdict
from sklearn.metrics import roc_auc_score


class AreaUnderCurve(Metric[float]):
    """
    The AreaUnderCurve metric. This is a standalone metric.

    The metric keeps a dictionary of <task_label, area_under_curve value> pairs.
    and update the values through a running average over multiple
    <prediction, target> pairs of Tensors, provided incrementally.
    The "prediction" and "target" tensors may contain plain labels or
    one-hot/logit vectors.

    Each time `result` is called, this metric emits the average area_under_curve
    across all predictions made since the last `reset`.

    The reset method will bring the metric to its initial state. By default
    this metric in its initial state will return an area_under_curve value of 0.
    """

    _positive_class_index = 1  # FIXME This is a provisional implementation, assuming a binary label.

    def __init__(self):
        """
        Creates an instance of the standalone AreaUnderCurve metric.

        By default this metric in its initial state will return an area_under_curve
        value of 0. The metric can be updated by using the `update` method
        while the running area_under_curve can be retrieved using the `result` method.
        """
        self._mean_area_under_curve = defaultdict(Mean)
        """
        The mean utility that will be used to store the running area_under_curve
        for each task label.
        """

    @torch.no_grad()
    def update(self, predicted_y: Tensor, true_y: Tensor,
               task_labels: Union[float, Tensor]) -> None:
        """
        Update the running area_under_curve given the true and predicted labels.
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
        if len(true_y.shape) > 1:
            # Logits -> transform to labels
            true_y = torch.max(true_y, 1)[1]

        if isinstance(task_labels, int):
            # Casting
            predicted_y_numpy = _tensor2array(predicted_y)
            predicted_y_numpy_class_positive = predicted_y_numpy[:, self._positive_class_index]
            true_y_numpy = _tensor2array(true_y)

            # Calculation
            try:
                area_under_curve = roc_auc_score(y_true=true_y_numpy, y_score=predicted_y_numpy_class_positive)
            except ValueError:
                area_under_curve = float('NaN')

            # Arrangement
            self._mean_area_under_curve[task_labels].update(area_under_curve)

        elif isinstance(task_labels, Tensor):
            raise NotImplementedError
        else:
            raise ValueError(f"Task label type: {type(task_labels)}, "
                             f"expected int/float or Tensor")

    def result(self, task_label=None) -> Dict[int, float]:
        """
        Retrieves the running area_under_curve.

        Calling this method will not change the internal state of the metric.

        :param task_label: if None, return the entire dictionary of area_under_curves
            for each task. Otherwise return the dictionary
            `{task_label: area_under_curve}`.
        :return: A dict of running area_under_curves for each task label,
            where each value is a float value between 0 and 1.
        """
        assert (task_label is None or isinstance(task_label, int))
        if task_label is None:
            return {k: v.result() for k, v in self._mean_area_under_curve.items()}
        else:
            return {task_label: self._mean_area_under_curve[task_label].result()}

    def reset(self, task_label=None) -> None:
        """
        Resets the metric.
        :param task_label: if None, reset the entire dictionary.
            Otherwise, reset the value associated to `task_label`.

        :return: None.
        """
        assert (task_label is None or isinstance(task_label, int))
        if task_label is None:
            self._mean_area_under_curve = defaultdict(Mean)
        else:
            self._mean_area_under_curve[task_label].reset()


class AreaUnderCurvePluginMetric(GenericPluginMetric[float]):
    """
    Base class for all area_under_curves plugin metrics
    """

    def __init__(self, reset_at, emit_at, mode):
        self._area_under_curve = AreaUnderCurve()
        super(AreaUnderCurvePluginMetric, self).__init__(
            self._area_under_curve, reset_at=reset_at, emit_at=emit_at,
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
        self._area_under_curve.update(strategy.mb_output, strategy.mb_y, task_labels)


class MinibatchAreaUnderCurve(AreaUnderCurvePluginMetric):
    """
    The minibatch plugin area_under_curve metric.
    This metric only works at training time.

    This metric computes the average area_under_curve over patterns
    from a single minibatch.
    It reports the result after each iteration.

    If a more coarse-grained logging is needed, consider using
    :class:`EpochAreaUnderCurve` instead.
    """

    def __init__(self):
        """
        Creates an instance of the MinibatchAreaUnderCurve metric.
        """
        super(MinibatchAreaUnderCurve, self).__init__(
            reset_at='iteration', emit_at='iteration', mode='train')

    def __str__(self):
        return "Top1_AUC_MB"


class EpochAreaUnderCurve(AreaUnderCurvePluginMetric):
    """
    The average area_under_curve over a single training epoch.
    This plugin metric only works at training time.

    The area_under_curve will be logged after each training epoch by computing
    the number of correctly predicted patterns during the epoch divided by
    the overall number of patterns encountered in that epoch.
    """

    def __init__(self):
        """
        Creates an instance of the EpochAreaUnderCurve metric.
        """

        super(EpochAreaUnderCurve, self).__init__(
            reset_at='epoch', emit_at='epoch', mode='train')

    def __str__(self):
        return "Top1_AUC_Epoch"


class RunningEpochAreaUnderCurve(AreaUnderCurvePluginMetric):
    """
    The average area_under_curve across all minibatches up to the current
    epoch iteration.
    This plugin metric only works at training time.

    At each iteration, this metric logs the area_under_curve averaged over all patterns
    seen so far in the current epoch.
    The metric resets its state after each training epoch.
    """

    def __init__(self):
        """
        Creates an instance of the RunningEpochAreaUnderCurve metric.
        """

        super(RunningEpochAreaUnderCurve, self).__init__(
            reset_at='epoch', emit_at='iteration', mode='train')

    def __str__(self):
        return "Top1_RunningAUC_Epoch"


class ExperienceAreaUnderCurve(AreaUnderCurvePluginMetric):
    """
    At the end of each experience, this plugin metric reports
    the average area_under_curve over all patterns seen in that experience.
    This metric only works at eval time.
    """

    def __init__(self):
        """
        Creates an instance of ExperienceAreaUnderCurve metric
        """
        super(ExperienceAreaUnderCurve, self).__init__(
            reset_at='experience', emit_at='experience', mode='eval')

    def __str__(self):
        return "Top1_AUC_Exp"


class StreamAreaUnderCurve(AreaUnderCurvePluginMetric):
    """
    At the end of the entire stream of experiences, this plugin metric
    reports the average area_under_curve over all patterns seen in all experiences.
    This metric only works at eval time.
    """

    def __init__(self):
        """
        Creates an instance of StreamAreaUnderCurve metric
        """
        super(StreamAreaUnderCurve, self).__init__(
            reset_at='stream', emit_at='stream', mode='eval')

    def __str__(self):
        return "Top1_AUC_Stream"


class TrainedExperienceAreaUnderCurve(AreaUnderCurvePluginMetric):
    """
    At the end of each experience, this plugin metric reports the average
    area_under_curve for only the experiences that the model has been trained on so far.

    This metric only works at eval time.
    """

    def __init__(self):
        """
        Creates an instance of TrainedExperienceAreaUnderCurve metric by first
        constructing AreaUnderCurvePluginMetric
        """
        super(TrainedExperienceAreaUnderCurve, self).__init__(
            reset_at='stream', emit_at='stream', mode='eval')
        self._current_experience = 0

    def after_training_exp(self, strategy) -> None:
        self._current_experience = strategy.experience.current_experience
        # Reset average after learning from a new experience
        AreaUnderCurvePluginMetric.reset(self, strategy)
        return AreaUnderCurvePluginMetric.after_training_exp(self, strategy)

    def update(self, strategy):
        """
        Only update the area_under_curve with results from experiences that have been
        trained on
        """
        if strategy.experience.current_experience <= self._current_experience:
            AreaUnderCurvePluginMetric.update(self, strategy)

    def __str__(self):
        return "AreaUnderCurve_On_Trained_Experiences"


def area_under_curve_metrics(*,
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
        the minibatch area_under_curve at training time.
    :param epoch: If True, will return a metric able to log
        the epoch area_under_curve at training time.
    :param epoch_running: If True, will return a metric able to log
        the running epoch area_under_curve at training time.
    :param experience: If True, will return a metric able to log
        the area_under_curve on each evaluation experience.
    :param stream: If True, will return a metric able to log
        the area_under_curve averaged over the entire evaluation stream of experiences.
    :param trained_experience: If True, will return a metric able to log
        the average evaluation area_under_curve only for experiences that the
        model has been trained on

    :return: A list of plugin metrics.
    """

    metrics = []
    if minibatch:
        metrics.append(MinibatchAreaUnderCurve())

    if epoch:
        metrics.append(EpochAreaUnderCurve())

    if epoch_running:
        metrics.append(RunningEpochAreaUnderCurve())

    if experience:
        metrics.append(ExperienceAreaUnderCurve())

    if stream:
        metrics.append(StreamAreaUnderCurve())

    if trained_experience:
        metrics.append(TrainedExperienceAreaUnderCurve())

    return metrics

# TENSOR TO ARRAY
def _tensor2array(tensor: Tensor):
    if tensor.is_cuda:
        try:
            return _tensor2array_gpu(tensor)
        except:
            return _tensor2array_detach_gpu(tensor)
    else:
        try:
            return _tensor2array_cpu(tensor)
        except:
            return _tensor2array_detach_cpu(tensor)

def _tensor2array_cpu(tensor: Tensor):
    return tensor.numpy()

def _tensor2array_gpu(tensor: Tensor):
    return tensor.cpu().numpy()

def _tensor2array_detach_cpu(tensor: Tensor):
    return tensor.detach().numpy()

def _tensor2array_detach_gpu(tensor: Tensor):
    return tensor.cpu().detach().numpy()

__all__ = [
    'AreaUnderCurve',
    'MinibatchAreaUnderCurve',
    'EpochAreaUnderCurve',
    'RunningEpochAreaUnderCurve',
    'ExperienceAreaUnderCurve',
    'StreamAreaUnderCurve',
    'TrainedExperienceAreaUnderCurve',
    'area_under_curve_metrics'
]
