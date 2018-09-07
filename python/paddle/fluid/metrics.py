#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fluid Metrics

The metrics are accomplished via Python natively. 
"""
import numpy as np
import copy
import warnings

__all__ = [
    'MetricBase',
    'CompositeMetric',
    'Accuracy',
    'ChunkEvaluator',
    'EditDistance',
    'DetectionMAP',
    'Auc',
]


def _is_numpy_(var):
    return isinstance(var, (np.ndarray, np.generic))


def _is_number_(var):
    return isinstance(var, int) or isinstance(var, float) or (isinstance(
        var, np.ndarray) and var.shape == (1, ))


def _is_number_or_matrix_(var):
    return _is_number_(var) or isinstance(var, np.ndarray)


class MetricBase(object):
    """
    Base Class for all evaluators

    Args:
        name(str): The name of evaluator. such as, "accuracy". Used for generate
            temporary variable name.
    Interface:
        Note(*) : the states is the attributes who not has _ prefix.

        get_config(): print current states and configuration
        reset(): clear the states. If the Metrics states type is not (int, float, np.ndarray),
                Please override this method.
        update(): update states at every minibatch
        eval(): get metric evaluation in numpy type.
    """

    def __init__(self, name, **kwargs):
        self._name = str(name) if name != None else self.__class__.__name__
        self._kwargs = kwargs if kwargs != None else dict()
        self.reset()

    def __str__(self):
        return self._name

    def reset(self):
        """
        states is the attributes who not has _ prefix.
        reset the states of metrics.
        """
        states = {
            attr: value
            for attr, value in self.__dict__.iteritems()
            if not attr.startswith("_")
        }
        for attr, value in states.iteritems():
            if isinstance(value, int):
                setattr(self, attr, 0)
            elif isinstance(value, float):
                setattr(self, attr, .0)
            elif isinstance(value, (np.ndarray, np.generic)):
                setattr(self, attr, np.zeros_like(value))
            else:
                setattr(self, attr, None)

    def get_config(self):
        states = {
            attr: value
            for attr, value in self.__dict__.iteritems()
            if not attr.startswith("_")
        }
        config = copy.deepcopy(self._kwargs)
        config.update({"name": self._name, "states": copy.deepcopy(states)})
        return config

    def update(self):
        raise NotImplementedError()

    def eval(self):
        raise NotImplementedError()


class CompositeMetric(MetricBase):
    """
    Compute multiple metrics in each minibatch.
    for example, merge F1, accuracy, recall into one Metric.
    """

    def __init__(self, name=None, **kwargs):
        super(CompositeMetric, self).__init__(name, kwargs)
        self._metrics = []

    def add_metric(self, metric):
        if not isinstance(metric, MetricBase):
            raise ValueError("SubMetric should be inherit from MetricBase.")
        self._metrics.append(metric)

    def eval(self):
        ans = []
        for m in self._metrics:
            ans.append(m.eval())
        return ans


class Accuracy(MetricBase):
    """
    Accumulate the accuracy from minibatches and compute the average accuracy
    for every pass.

    Args:
       name: the metrics name

    Example:
        minibatch_accuracy = fluid.layers.accuracy(pred, label)
        accuracy_evaluator = fluid.metrics.Accuracy()
        for epoch in PASS_NUM:
            accuracy_evaluator.reset()
            for data in batches:
                loss = exe.run(fetch_list=[cost, minibatch_accuracy])
            accuracy_evaluator.update(value=minibatch_accuracy, weight=batches)
            accuracy = accuracy_evaluator.eval()
    """

    def __init__(self, name=None):
        super(Accuracy, self).__init__(name)
        self.value = .0
        self.weight = .0

    def update(self, value, weight):
        if not _is_number_or_matrix_(value):
            raise ValueError(
                "The 'value' must be a number(int, float) or a numpy ndarray.")
        if not _is_number_(weight):
            raise ValueError("The 'weight' must be a number(int, float).")
        self.value += value * weight
        self.weight += weight

    def eval(self):
        if self.weight == 0:
            raise ValueError(
                "There is no data in Accuracy Metrics. Please check layers.accuracy output has added to Accuracy."
            )
        return self.value / self.weight


class ChunkEvaluator(MetricBase):
    """
    Accumulate counter numbers output by chunk_eval from mini-batches and
    compute the precision recall and F1-score using the accumulated counter
    numbers.
    """

    def __init__(self, name=None):
        super(ChunkEvaluator, self).__init__(name)
        self.num_infer_chunks = 0
        self.num_label_chunks = 0
        self.num_correct_chunks = 0

    def update(self, num_infer_chunks, num_label_chunks, num_correct_chunks):
        if not _is_number_or_matrix_(num_infer_chunks):
            raise ValueError(
                "The 'num_infer_chunks' must be a number(int, float) or a numpy ndarray."
            )
        if not _is_number_or_matrix_(num_label_chunks):
            raise ValueError(
                "The 'num_label_chunks' must be a number(int, float) or a numpy ndarray."
            )
        if not _is_number_or_matrix_(num_correct_chunks):
            raise ValueError(
                "The 'num_correct_chunks' must be a number(int, float) or a numpy ndarray."
            )
        self.num_infer_chunks += num_infer_chunks
        self.num_label_chunks += num_label_chunks
        self.num_correct_chunks += num_correct_chunks

    def eval(self):
        precision = float(
            self.num_correct_chunks
        ) / self.num_infer_chunks if self.num_infer_chunks else 0
        recall = float(self.num_correct_chunks
                       ) / self.num_label_chunks if self.num_label_chunks else 0
        f1_score = float(2 * precision * recall) / (
            precision + recall) if self.num_correct_chunks else 0
        return precision, recall, f1_score


class EditDistance(MetricBase):
    """
    Accumulate edit distance sum and sequence number from mini-batches and
    compute the average edit_distance and instance error of all batches.

    Args:
        name: the metrics name

    Example:
        edit_distance_metrics = fluid.layers.edit_distance(input, label)
        distance_evaluator = fluid.metrics.EditDistance()
        for epoch in PASS_NUM:
            distance_evaluator.reset()
            for data in batches:
                loss = exe.run(fetch_list=[cost] + list(edit_distance_metrics))
            distance_evaluator.update(*edit_distance_metrics)
            distance, instance_error = distance_evaluator.eval()

        In the above example:
        'distance' is the average of the edit distance in a pass.
        'instance_error' is the instance error rate in a pass.

    """

    def __init__(self, name):
        super(EditDistance, self).__init__(name)
        self.total_distance = .0
        self.seq_num = 0
        self.instance_error = 0

    def update(self, distances, seq_num):
        if not _is_numpy_(distances):
            raise ValueError("The 'distances' must be a numpy ndarray.")
        if not _is_number_(seq_num):
            raise ValueError("The 'seq_num' must be a number(int, float).")
        seq_right_count = np.sum(distances == 0)
        total_distance = np.sum(distances)
        self.seq_num += seq_num
        self.instance_error += seq_num - seq_right_count
        self.total_distance += total_distance

    def eval(self):
        if self.seq_num == 0:
            raise ValueError(
                "There is no data in EditDistance Metric. Please check layers.edit_distance output has been added to EditDistance."
            )
        avg_distance = self.total_distance / self.seq_num
        avg_instance_error = self.instance_error / self.seq_num
        return avg_distance, avg_instance_error


class DetectionMAP(MetricBase):
    """
    Calculate the detection mean average precision (mAP).

    TODO (Dang Qingqing): update the following doc.
    The general steps are as follows:
    1. calculate the true positive and false positive according to the input
        of detection and labels.
    2. calculate mAP value, support two versions: '11 point' and 'integral'.

    Please get more information from the following articles:
      https://sanchom.wordpress.com/tag/average-precision/
      https://arxiv.org/abs/1512.02325
    """

    def __init__(self, name=None):
        super(DetectionMAP, self).__init__(name)
        # the current map value
        self.value = .0
        self.weight = .0

    def update(self, value, weight):
        if not _is_number_or_matrix_(value):
            raise ValueError(
                "The 'value' must be a number(int, float) or a numpy ndarray.")
        if not _is_number_(weight):
            raise ValueError("The 'weight' must be a number(int, float).")
        self.value += value
        self.weight += weight

    def eval(self):
        if self.weight == 0:
            raise ValueError(
                "There is no data in DetectionMAP Metrics. "
                "Please check layers.detection_map output has added to DetectionMAP."
            )
        return self.value / self.weight


class Auc(MetricBase):
    """
    Auc Metrics which adapts to binary classification.
    Need to note that auc metrics compute the value via Python natively.
    If you concern the speed, please use the fluid.layers.auc instead.

    The `auc` function creates four local variables, `true_positives`,
      `true_negatives`, `false_positives` and `false_negatives` that are used to
      compute the AUC. To discretize the AUC curve, a linearly spaced set of
      thresholds is used to compute pairs of recall and precision values. The area
      under the ROC-curve is therefore computed using the height of the recall
      values by the false positive rate, while the area under the PR-curve is the
      computed using the height of the precision values by the recall.

    Args:
        name: metric name
        curve: Specifies the name of the curve to be computed, 'ROC' [default] or
          'PR' for the Precision-Recall-curve.
        num_thresholds: The number of thresholds to use when discretizing the roc
            curve.

    "NOTE: only implement the ROC curve type via Python now."
    """

    def __init__(self, name, curve='ROC', num_thresholds=200):
        super(MetricBase, self).__init__(name, curve, num_thresholds)
        self._curve = curve
        self._num_thresholds = num_thresholds
        self._epsilon = 1e-6
        self.tp_list = np.ndarray((num_thresholds, ))
        self.fn_list = np.ndarray((num_thresholds, ))
        self.tn_list = np.ndarray((num_thresholds, ))
        self.fp_list = np.ndarray((num_thresholds, ))

    def update(self, labels, predictions, axis=1):
        if not _is_numpy_(labels):
            raise ValueError("The 'labels' must be a numpy ndarray.")
        if not _is_numpy_(predictions):
            raise ValueError("The 'predictions' must be a numpy ndarray.")

        kepsilon = 1e-7  # to account for floating point imprecisions
        thresholds = [(i + 1) * 1.0 / (self._num_thresholds - 1)
                      for i in range(self._num_thresholds - 2)]
        thresholds = [0.0 - kepsilon] + thresholds + [1.0 + kepsilon]

        # caculate TP, FN, TN, FP count
        for idx_thresh, thresh in enumerate(thresholds):
            tp, fn, tn, fp = 0, 0, 0, 0
            for i, lbl in enumerate(labels):
                if lbl:
                    if predictions[i, 0] >= thresh:
                        tp += 1
                    else:
                        fn += 1
                else:
                    if predictions[i, 0] >= thresh:
                        fp += 1
                    else:
                        tn += 1
            self.tp_list[idx_thresh] += tp
            self.fn_list[idx_thresh] += fn
            self.tn_list[idx_thresh] += tn
            self.fp_list[idx_thresh] += fp

    def eval(self):
        epsilon = self._epsilon
        num_thresholds = self._num_thresholds
        tpr = (self.tp_list.astype("float32") + epsilon) / (
            self.tp_list + self.fn_list + epsilon)
        fpr = self.fp_list.astype("float32") / (
            self.fp_list + self.tn_list + epsilon)
        rec = (self.tp_list.astype("float32") + epsilon) / (
            self.tp_list + self.fp_list + epsilon)

        x = fpr[:num_thresholds - 1] - fpr[1:]
        y = (tpr[:num_thresholds - 1] + tpr[1:]) / 2.0
        auc_value = np.sum(x * y)
        return auc_value
