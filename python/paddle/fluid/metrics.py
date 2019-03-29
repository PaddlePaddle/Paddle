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
"""

from __future__ import print_function

import numpy as np
import copy
import warnings
import six

from .layer_helper import LayerHelper
from .initializer import Constant
from . import unique_name
from .framework import Program, Variable, program_guard
from . import layers

__all__ = [
    'MetricBase',
    'CompositeMetric',
    'Precision',
    'Recall',
    'Accuracy',
    'ChunkEvaluator',
    'EditDistance',
    'DetectionMAP',
    'Auc',
]


def _is_numpy_(var):
    return isinstance(var, (np.ndarray, np.generic))


def _is_number_(var):
    return isinstance(var, int) or isinstance(var, np.int64) or isinstance(
        var, float) or (isinstance(var, np.ndarray) and var.shape == (1, ))


def _is_number_or_matrix_(var):
    return _is_number_(var) or isinstance(var, np.ndarray)


class MetricBase(object):
    """
    Base Class for all Metrics.
    MetricBase define a group of interfaces for the
    model evaluation methods. Metrics accumulate metric states between
    consecutive minibatches, at every minibatch, use update
    interface to add current minibatch value to global states.
    Use eval to compute accumative metric value from last reset()
    or from scratch on.
    If you need to custom a new metric, please inherit from MetricBase and
    custom implementation.

    Args:
        name(str): The name of metric instance. such as, "accuracy".
                  It needed if you want to distinct different metrics in a model.

    """

    def __init__(self, name):
        self._name = str(name) if name != None else self.__class__.__name__

    def __str__(self):
        return self._name

    def reset(self):
        """
        reset clear the states of metrics. By default, the states
        are the members who do not has _ prefix, reset set them to inital states.
        If you violate the implicit name rule, please also custom the reset
        interface.
        """
        states = {
            attr: value
            for attr, value in six.iteritems(self.__dict__)
            if not attr.startswith("_")
        }
        for attr, value in six.iteritems(states):
            if isinstance(value, int):
                setattr(self, attr, 0)
            elif isinstance(value, float):
                setattr(self, attr, .0)
            elif isinstance(value, (np.ndarray, np.generic)):
                setattr(self, attr, np.zeros_like(value))
            else:
                setattr(self, attr, None)

    def get_config(self):
        """
        Get the metric and current states.
        The states are the members who do not has "_" prefix.

        Args:
            None

        Returns:
            dict: a dict of metric and states
        """
        states = {
            attr: value
            for attr, value in six.iteritems(self.__dict__)
            if not attr.startswith("_")
        }
        config = {}
        config.update({"name": self._name, "states": copy.deepcopy(states)})
        return config

    def update(self, preds, labels):
        """
        Updates the metric states at every minibatch.
        One user can compute the minibatch metric via pure Python, or
        via a c++ operator.

        Args:
            preds(numpy.array): the predictions of current minibatch
            labels(numpy.array): the labels of current minibatch, if the label is one-hot
                               or soft-label, should custom the corresponding update rule.
        """
        raise NotImplementedError(
            "Should not use it directly, please extend it.")

    def eval(self):
        """
        Evalute the current metrics based the accumulated states.

        Returns:
            float|list(float)|numpy.array: the metrics via Python.
        """
        raise NotImplementedError(
            "Should not use it directly, please extend it.")


class CompositeMetric(MetricBase):
    """
    Composite multiple metrics in one instance.
    for example, merge F1, accuracy, recall into one Metric.

    Examples:
        .. code-block:: python

          labels = fluid.layers.data(name="data", shape=[1], dtype="int32")
          data = fluid.layers.data(name="data", shape=[32, 32], dtype="int32")
          pred = fluid.layers.fc(input=data, size=1000, act="tanh")
          comp = fluid.metrics.CompositeMetric()
          acc = fluid.metrics.Precision()
          recall = fluid.metrics.Recall()
          comp.add_metric(acc)
          comp.add_metric(recall)
          for pass in range(PASSES):
            comp.reset()
            for data in train_reader():
                loss, preds, labels = exe.run(fetch_list=[cost, preds, labels])
            comp.update(preds=preds, labels=labels)
            numpy_acc, numpy_recall = comp.eval()
    """

    def __init__(self, name=None):
        super(CompositeMetric, self).__init__(name)
        self._metrics = []

    def add_metric(self, metric):
        """
        add one metric instance to CompositeMetric.

        Args:
            metric: a instance of MetricBase.
        """
        if not isinstance(metric, MetricBase):
            raise ValueError("SubMetric should be inherit from MetricBase.")
        self._metrics.append(metric)

    def update(self, preds, labels):
        """
        Update every metrics in sequence.

        Args:
            preds(numpy.array): the predictions of current minibatch
            labels(numpy.array): the labels of current minibatch, if the label is one-hot
                               or soft-label, should custom the corresponding update rule.
        """
        for m in self._metrics:
            m.update(preds, labels)

    def eval(self):
        """
        Evaluate every metrics in sequence.

        Returns:
            list(float|numpy.array): a list of metrics value in Python.
        """
        ans = []
        for m in self._metrics:
            ans.append(m.eval())
        return ans


class Precision(MetricBase):
    """
    Precision (also called positive predictive value) is the fraction of
    relevant instances among the retrieved instances.
    https://en.wikipedia.org/wiki/Evaluation_of_binary_classifiers

    Note Precision is different with Accuracy in binary classifiers.
    accuracy = true positive / total instances
    precision = true positive / all positive instance

    Examples:
        .. code-block:: python

            metric = fluid.metrics.Precision()
            for pass in range(PASSES):
                metric.reset()
                for data in train_reader():
                    loss, preds, labels = exe.run(fetch_list=[cost, preds, labels])
                    metric.update(preds=preds, labels=labels)
                numpy_precision = metric.eval()
    """

    def __init__(self, name=None):
        super(Precision, self).__init__(name)
        self.tp = 0  # true positive
        self.fp = 0  # false positive

    def update(self, preds, labels):
        if not _is_numpy_(preds):
            raise ValueError("The 'preds' must be a numpy ndarray.")
        if not _is_numpy_(labels):
            raise ValueError("The 'labels' must be a numpy ndarray.")
        sample_num = labels.shape[0]
        preds = np.rint(preds).astype("int32")
        for i in range(sample_num):
            pred = preds[i]
            label = labels[i]
            if label == 1:
                if pred == label:
                    self.tp += 1
                else:
                    self.fp += 1

    def eval(self):
        ap = self.tp + self.fp
        return float(self.tp) / ap if ap != 0 else .0


class Recall(MetricBase):
    """
    Recall (also known as sensitivity) is the fraction of
    relevant instances that have been retrieved over the
    total amount of relevant instances

    https://en.wikipedia.org/wiki/Precision_and_recall

    Examples:
        .. code-block:: python

            metric = fluid.metrics.Recall()
            for pass in range(PASSES):
                metric.reset()
                for data in train_reader():
                    loss, preds, labels = exe.run(fetch_list=[cost, preds, labels])
                metric.update(preds=preds, labels=labels)
                numpy_recall = metric.eval()
    """

    def __init__(self, name=None):
        super(Recall, self).__init__(name)
        self.tp = 0  # true positive
        self.fn = 0  # false negtive

    def update(self, preds, labels):
        if not _is_numpy_(preds):
            raise ValueError("The 'preds' must be a numpy ndarray.")
        if not _is_numpy_(labels):
            raise ValueError("The 'labels' must be a numpy ndarray.")
        sample_num = labels[0]
        for i in range(sample_num):
            pred = preds[i].astype("int32")
            label = labels[i]
            if label == 1:
                if pred == label:
                    self.tp += 1
            else:
                if pred != label:
                    self.fn += 1

    def eval(self):
        recall = self.tp + self.fn
        return float(self.tp) / recall if recall != 0 else .0


class Accuracy(MetricBase):
    """
    Accumulate the accuracy from minibatches and compute the average accuracy
    for every pass.
    https://en.wikipedia.org/wiki/Accuracy_and_precision

    Args:
       name: the metrics name

    Examples:
        .. code-block:: python

            labels = fluid.layers.data(name="data", shape=[1], dtype="int32")
            data = fluid.layers.data(name="data", shape=[32, 32], dtype="int32")
            pred = fluid.layers.fc(input=data, size=1000, act="tanh")
            minibatch_accuracy = fluid.layers.accuracy(pred, label)
            accuracy_evaluator = fluid.metrics.Accuracy()
            for pass in range(PASSES):
                accuracy_evaluator.reset()
                for data in train_reader():
                    batch_size = data[0]
                    loss = exe.run(fetch_list=[cost, minibatch_accuracy])
                accuracy_evaluator.update(value=minibatch_accuracy, weight=batch_size)
                numpy_acc = accuracy_evaluator.eval()
    """

    def __init__(self, name=None):
        super(Accuracy, self).__init__(name)
        self.value = .0
        self.weight = .0

    def update(self, value, weight):
        """
        Update minibatch states.

        Args:
            value(float|numpy.array): accuracy of one minibatch.
            weight(int|float): batch size.
        """
        if not _is_number_or_matrix_(value):
            raise ValueError(
                "The 'value' must be a number(int, float) or a numpy ndarray.")
        if not _is_number_(weight):
            raise ValueError("The 'weight' must be a number(int, float).")
        self.value += value * weight
        self.weight += weight

    def eval(self):
        if self.weight == 0:
            raise ValueError("There is no data in Accuracy Metrics. \
                Please check layers.accuracy output has added to Accuracy.")
        return self.value / self.weight


class ChunkEvaluator(MetricBase):
    """
    Accumulate counter numbers output by chunk_eval from mini-batches and
    compute the precision recall and F1-score using the accumulated counter
    numbers.
    For some basics of chunking, please refer to 
    `Chunking with Support Vector Machines <https://aclanthology.info/pdf/N/N01/N01-1025.pdf>`_ .
    ChunkEvalEvaluator computes the precision, recall, and F1-score of chunk detection,
    and supports IOB, IOE, IOBES and IO (also known as plain) tagging schemes.

    Examples:
        .. code-block:: python

            labels = fluid.layers.data(name="data", shape=[1], dtype="int32")
            data = fluid.layers.data(name="data", shape=[32, 32], dtype="int32")
            pred = fluid.layers.fc(input=data, size=1000, act="tanh")
            precision, recall, f1_score, num_infer_chunks, num_label_chunks, num_correct_chunks = layers.chunk_eval(
                input=pred,
                label=label)
            metric = fluid.metrics.ChunkEvaluator()
            for data in train_reader():
                loss, preds, labels = exe.run(fetch_list=[cost, preds, labels])
                metric.update(num_infer_chunks, num_label_chunks, num_correct_chunks)
                numpy_precision, numpy_recall, numpy_f1 = metric.eval()
    """

    def __init__(self, name=None):
        super(ChunkEvaluator, self).__init__(name)
        self.num_infer_chunks = 0
        self.num_label_chunks = 0
        self.num_correct_chunks = 0

    def update(self, num_infer_chunks, num_label_chunks, num_correct_chunks):
        """
        Update the states based on the layers.chunk_eval() ouputs.

        Args:
            num_infer_chunks(int|numpy.array): The number of chunks in Inference on the given minibatch.
            num_label_chunks(int|numpy.array): The number of chunks in Label on the given mini-batch.
            num_correct_chunks(int|float|numpy.array): The number of chunks both in Inference and Label on the
                                                  given mini-batch.
        """
        if not _is_number_or_matrix_(num_infer_chunks):
            raise ValueError(
                "The 'num_infer_chunks' must be a number(int) or a numpy ndarray."
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
    Edit distance is a way of quantifying how dissimilar two strings
    (e.g., words) are to one another by counting the minimum number
    of operations required to transform one string into the other.
    Refer to https://en.wikipedia.org/wiki/Edit_distance

    Accumulate edit distance sum and sequence number from mini-batches and
    compute the average edit_distance and instance error of all batches.

    Args:
        name: the metrics name

    Examples:
        .. code-block:: python

            distances, seq_num = fluid.layers.edit_distance(input, label)
            distance_evaluator = fluid.metrics.EditDistance()
            for epoch in PASS_NUM:
                distance_evaluator.reset()
                for data in batches:
                    loss = exe.run(fetch_list=[cost] + list(edit_distance_metrics))
                distance_evaluator.update(distances, seq_num)
                distance, instance_error = distance_evaluator.eval()

    In the above example:

        - 'distance' is the average of the edit distance in a pass.
        - 'instance_error' is the instance error rate in a pass.

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
        avg_instance_error = self.instance_error / float(self.seq_num)
        return avg_distance, avg_instance_error


class Auc(MetricBase):
    """
    Auc metric adapts to the binary classification.
    Refer to https://en.wikipedia.org/wiki/Receiver_operating_characteristic#Area_under_the_curve
    Need to note that auc metric compute the value via Python natively.
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

    "NOTE: only implement the ROC curve type via Python now."

    Examples:
        .. code-block:: python

            pred = fluid.layers.fc(input=data, size=1000, act="tanh")
            metric = fluid.metrics.Auc()
            for data in train_reader():
                loss, preds, labels = exe.run(fetch_list=[cost, preds, labels])
                metric.update(preds, labels)
                numpy_auc = metric.eval()
    """

    def __init__(self, name, curve='ROC', num_thresholds=4095):
        super(Auc, self).__init__(name=name)
        self._curve = curve
        self._num_thresholds = num_thresholds

        _num_pred_buckets = num_thresholds + 1
        self._stat_pos = [0] * _num_pred_buckets
        self._stat_neg = [0] * _num_pred_buckets

    def update(self, preds, labels):
        if not _is_numpy_(labels):
            raise ValueError("The 'labels' must be a numpy ndarray.")
        if not _is_numpy_(preds):
            raise ValueError("The 'predictions' must be a numpy ndarray.")

        for i, lbl in enumerate(labels):
            value = preds[i, 1]
            bin_idx = int(value * self._num_thresholds)
            assert bin_idx <= self._num_thresholds
            if lbl:
                self._stat_pos[bin_idx] += 1.0
            else:
                self._stat_neg[bin_idx] += 1.0

    @staticmethod
    def trapezoid_area(x1, x2, y1, y2):
        return abs(x1 - x2) * (y1 + y2) / 2.0

    def eval(self):
        tot_pos = 0.0
        tot_neg = 0.0
        auc = 0.0

        idx = self._num_thresholds
        while idx >= 0:
            tot_pos_prev = tot_pos
            tot_neg_prev = tot_neg
            tot_pos += self._stat_pos[idx]
            tot_neg += self._stat_neg[idx]
            auc += self.trapezoid_area(tot_neg, tot_neg_prev, tot_pos,
                                       tot_pos_prev)
            idx -= 1

        return auc / tot_pos / tot_neg if tot_pos > 0.0 and tot_neg > 0.0 else 0.0


class DetectionMAP(object):
    """
    Calculate the detection mean average precision (mAP).

    The general steps are as follows:

    1. calculate the true positive and false positive according to the input
       of detection and labels.
    2. calculate mAP value, support two versions: '11 point' and 'integral'.

    Please get more information from the following articles:

      https://sanchom.wordpress.com/tag/average-precision/

      https://arxiv.org/abs/1512.02325

    Args:
        input (Variable): The detection results, which is a LoDTensor with shape
            [M, 6]. The layout is [label, confidence, xmin, ymin, xmax, ymax].
        gt_label (Variable): The ground truth label index, which is a LoDTensor
            with shape [N, 1].
        gt_box (Variable): The ground truth bounding box (bbox), which is a
            LoDTensor with shape [N, 4]. The layout is [xmin, ymin, xmax, ymax].
        gt_difficult (Variable|None): Whether this ground truth is a difficult
            bounding bbox, which can be a LoDTensor [N, 1] or not set. If None,
            it means all the ground truth labels are not difficult bbox.
        class_num (int): The class number.
        background_label (int): The index of background label, the background
            label will be ignored. If set to -1, then all categories will be
            considered, 0 by defalut.
        overlap_threshold (float): The threshold for deciding true/false
            positive, 0.5 by defalut.
        evaluate_difficult (bool): Whether to consider difficult ground truth
            for evaluation, True by defalut. This argument does not work when
            gt_difficult is None.
        ap_version (string): The average precision calculation ways, it must be
            'integral' or '11point'. Please check
            https://sanchom.wordpress.com/tag/average-precision/ for details.
            - 11point: the 11-point interpolated average precision.
            - integral: the natural integral of the precision-recall curve.

    Examples:
        .. code-block:: python

            exe = fluid.Executor(place)
            map_evaluator = fluid.Evaluator.DetectionMAP(input,
                gt_label, gt_box, gt_difficult)
            cur_map, accum_map = map_evaluator.get_map_var()
            fetch = [cost, cur_map, accum_map]
            for epoch in PASS_NUM:
                map_evaluator.reset(exe)
                for data in batches:
                    loss, cur_map_v, accum_map_v = exe.run(fetch_list=fetch)

    In the above example:

            - 'cur_map_v' is the mAP of current mini-batch.
            - 'accum_map_v' is the accumulative mAP of one pass.

 
    """

    def __init__(self,
                 input,
                 gt_label,
                 gt_box,
                 gt_difficult=None,
                 class_num=None,
                 background_label=0,
                 overlap_threshold=0.5,
                 evaluate_difficult=True,
                 ap_version='integral'):

        self.helper = LayerHelper('map_eval')
        gt_label = layers.cast(x=gt_label, dtype=gt_box.dtype)
        if gt_difficult:
            gt_difficult = layers.cast(x=gt_difficult, dtype=gt_box.dtype)
            label = layers.concat([gt_label, gt_difficult, gt_box], axis=1)
        else:
            label = layers.concat([gt_label, gt_box], axis=1)

        # calculate mean average precision (mAP) of current mini-batch
        map = layers.detection_map(
            input,
            label,
            class_num,
            background_label,
            overlap_threshold=overlap_threshold,
            evaluate_difficult=evaluate_difficult,
            ap_version=ap_version)

        states = []
        states.append(
            self._create_state(
                dtype='int32', shape=None, suffix='accum_pos_count'))
        states.append(
            self._create_state(
                dtype='float32', shape=None, suffix='accum_true_pos'))
        states.append(
            self._create_state(
                dtype='float32', shape=None, suffix='accum_false_pos'))
        var = self._create_state(dtype='int32', shape=[1], suffix='has_state')
        self.helper.set_variable_initializer(
            var, initializer=Constant(value=int(0)))
        self.has_state = var

        # calculate accumulative mAP
        accum_map = layers.detection_map(
            input,
            label,
            class_num,
            background_label,
            overlap_threshold=overlap_threshold,
            evaluate_difficult=evaluate_difficult,
            has_state=self.has_state,
            input_states=states,
            out_states=states,
            ap_version=ap_version)

        layers.fill_constant(
            shape=self.has_state.shape,
            value=1,
            dtype=self.has_state.dtype,
            out=self.has_state)

        self.cur_map = map
        self.accum_map = accum_map

    def _create_state(self, suffix, dtype, shape):
        """
        Create state variable.
        Args:
            suffix(str): the state suffix.
            dtype(str|core.VarDesc.VarType): the state data type
            shape(tuple|list): the shape of state
        Returns: State variable
        """
        state = self.helper.create_variable(
            name="_".join([unique_name.generate(self.helper.name), suffix]),
            persistable=True,
            dtype=dtype,
            shape=shape)
        return state

    def get_map_var(self):
        """
        Returns: mAP variable of current mini-batch and
            accumulative mAP variable cross mini-batches.
        """
        return self.cur_map, self.accum_map

    def reset(self, executor, reset_program=None):
        """
        Reset metric states at the begin of each pass/user specified batch.

        Args:
            executor(Executor): a executor for executing
                the reset_program.
            reset_program(Program|None): a single Program for reset process.
                If None, will create a Program.
        """

        def _clone_var_(block, var):
            assert isinstance(var, Variable)
            return block.create_var(
                name=var.name,
                shape=var.shape,
                dtype=var.dtype,
                type=var.type,
                lod_level=var.lod_level,
                persistable=var.persistable)

        if reset_program is None:
            reset_program = Program()
        with program_guard(main_program=reset_program):
            var = _clone_var_(reset_program.current_block(), self.has_state)
            layers.fill_constant(
                shape=var.shape, value=0, dtype=var.dtype, out=var)
        executor.run(reset_program)
