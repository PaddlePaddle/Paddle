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
from .layers import detection

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

            import numpy as np
            preds = [[0.1], [0.7], [0.8], [0.9], [0.2],
                     [0.2], [0.3], [0.5], [0.8], [0.6]]
            labels = [[0], [1], [1], [1], [1],
                      [0], [0], [0], [0], [0]]
            preds = np.array(preds)
            labels = np.array(labels)

            comp = fluid.metrics.CompositeMetric()
            precision = fluid.metrics.Precision()
            recall = fluid.metrics.Recall()
            comp.add_metric(precision)
            comp.add_metric(recall)

            comp.update(preds=preds, labels=labels)
            numpy_precision, numpy_recall = comp.eval()

            print("expect precision: %.2f, got %.2f" % ( 3. / 5, numpy_precision ) )
            print("expect recall: %.2f, got %.2f" % (3. / 4, numpy_recall ) )
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

    This class mangages the precision score for binary classification task.

    Examples:
        .. code-block:: python

            import numpy as np

            metric = fluid.metrics.Precision()

            # generate the preds and labels

            preds = [[0.1], [0.7], [0.8], [0.9], [0.2],
                     [0.2], [0.3], [0.5], [0.8], [0.6]]

            labels = [[0], [1], [1], [1], [1],
                      [0], [0], [0], [0], [0]]

            preds = np.array(preds)
            labels = np.array(labels)

            metric.update(preds=preds, labels=labels)
            numpy_precision = metric.eval()

            print("expct precision: %.2f and got %.2f" % ( 3.0 / 5.0, numpy_precision))
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
            if pred == 1:
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

    This class mangages the recall score for binary classification task.

    Examples:
        .. code-block:: python

            import numpy as np

            metric = fluid.metrics.Recall()

            # generate the preds and labels

            preds = [[0.1], [0.7], [0.8], [0.9], [0.2],
                     [0.2], [0.3], [0.5], [0.8], [0.6]]

            labels = [[0], [1], [1], [1], [1],
                      [0], [0], [0], [0], [0]]

            preds = np.array(preds)
            labels = np.array(labels)

            metric.update(preds=preds, labels=labels)
            numpy_precision = metric.eval()

            print("expct precision: %.2f and got %.2f" % ( 3.0 / 4.0, numpy_precision))
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
        sample_num = labels.shape[0]
        preds = np.rint(preds).astype("int32")

        for i in range(sample_num):
            pred = preds[i]
            label = labels[i]
            if label == 1:
                if pred == label:
                    self.tp += 1
                else:
                    self.fn += 1

    def eval(self):
        recall = self.tp + self.fn
        return float(self.tp) / recall if recall != 0 else .0


class Accuracy(MetricBase):
    """
    Calculate the mean accuracy over multiple batches.
    https://en.wikipedia.org/wiki/Accuracy_and_precision

    Args:
       name: the metrics name

    Examples:
        .. code-block:: python

            #suppose we have batch_size = 128
            batch_size=128
            accuracy_manager = fluid.metrics.Accuracy()

            #suppose the accuracy is 0.9 for the 1st batch
            batch1_acc = 0.9
            accuracy_manager.update(value = batch1_acc, weight = batch_size)
            print("expect accuracy: %.2f, get accuracy: %.2f" % (batch1_acc, accuracy_manager.eval()))

            #suppose the accuracy is 0.8 for the 2nd batch
            batch2_acc = 0.8

            accuracy_manager.update(value = batch2_acc, weight = batch_size)
            #the joint acc for batch1 and batch2 is (batch1_acc * batch_size + batch2_acc * batch_size) / batch_size / 2
            print("expect accuracy: %.2f, get accuracy: %.2f" % ((batch1_acc * batch_size + batch2_acc * batch_size) / batch_size / 2, accuracy_manager.eval()))

            #reset the accuracy_manager
            accuracy_manager.reset()
            #suppose the accuracy is 0.8 for the 3rd batch
            batch3_acc = 0.8
            accuracy_manager.update(value = batch3_acc, weight = batch_size)
            print("expect accuracy: %.2f, get accuracy: %.2f" % (batch3_acc, accuracy_manager.eval()))
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
        if _is_number_(weight) and weight < 0:
            raise ValueError("The 'weight' can not be negative")
        self.value += value * weight
        self.weight += weight

    def eval(self):
        """
        Return the mean accuracy (float or numpy.array) for all accumulated batches.
        """
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

            # init the chunck-level evaluation manager
            metric = fluid.metrics.ChunkEvaluator()

            # suppose the model predict 10 chuncks, while 8 ones are correct and the ground truth has 9 chuncks.
            num_infer_chunks = 10
            num_label_chunks = 9 
            num_correct_chunks = 8

            metric.update(num_infer_chunks, num_label_chunks, num_correct_chunks)
            numpy_precision, numpy_recall, numpy_f1 = metric.eval()

            print("precision: %.2f, recall: %.2f, f1: %.2f" % (numpy_precision, numpy_recall, numpy_f1))

            # the next batch, predicting 3 prefectly correct chuncks.
            num_infer_chunks = 3
            num_label_chunks = 3
            num_correct_chunks = 3

            metric.update(num_infer_chunks, num_label_chunks, num_correct_chunks)
            numpy_precision, numpy_recall, numpy_f1 = metric.eval()

            print("precision: %.2f, recall: %.2f, f1: %.2f" % (numpy_precision, numpy_recall, numpy_f1))

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
    (e.g., words) are to each another by counting the minimum number
    of edit operations (add, remove or replace) required to transform
    one string into the other.
    Refer to https://en.wikipedia.org/wiki/Edit_distance

    This EditDistance class takes two inputs by using update function:
    1. distances: a (batch_size, 1) numpy.array, each element represents the
    edit distance between two sequences.
    2. seq_num: a int|float value, standing for the number of sequence pairs.

    and returns the overall edit distance of multiple sequence-pairs.

    Args:
        name: the metrics name

    Examples:
        .. code-block:: python

            import numpy as np

            # suppose that batch_size is 128
            batch_size = 128

            # init the edit distance manager
            distance_evaluator = fluid.metrics.EditDistance("EditDistance")

            # generate the edit distance across 128 sequence pairs, the max distance is 10 here
            edit_distances_batch0 = np.random.randint(low = 0, high = 10, size = (batch_size, 1))
            seq_num_batch0 = batch_size

            distance_evaluator.update(edit_distances_batch0, seq_num_batch0)
            avg_distance, wrong_instance_ratio = distance_evaluator.eval()
            print("the average edit distance for batch0 is %.2f and the wrong instance ratio is %.2f " % (avg_distance, wrong_instance_ratio))

            edit_distances_batch1 = np.random.randint(low = 0, high = 10, size = (batch_size, 1))
            seq_num_batch1 = batch_size

            distance_evaluator.update(edit_distances_batch1, seq_num_batch1)
            avg_distance, wrong_instance_ratio = distance_evaluator.eval()
            print("the average edit distance for batch0 and batch1 is %.2f and the wrong instance ratio is %.2f " % (avg_distance, wrong_instance_ratio))

            distance_evaluator.reset()

            edit_distances_batch2 = np.random.randint(low = 0, high = 10, size = (batch_size, 1))
            seq_num_batch2 = batch_size

            distance_evaluator.update(edit_distances_batch2, seq_num_batch2)
            avg_distance, wrong_instance_ratio = distance_evaluator.eval()
            print("the average edit distance for batch2 is %.2f and the wrong instance ratio is %.2f " % (avg_distance, wrong_instance_ratio))

    """

    def __init__(self, name):
        super(EditDistance, self).__init__(name)
        self.total_distance = .0
        self.seq_num = 0
        self.instance_error = 0

    def update(self, distances, seq_num):
        """
        Update the overall edit distance

        Args:
            distances: a (batch_size, 1) numpy.array, each element represents the 
            edit distance between two sequences.
            seq_num: a int|float value, standing for the number of sequence pairs.

        """
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
        """
        Return two floats:
        avg_distance: the average distance for all sequence pairs updated using the update function.
        avg_instance_error: the ratio of sequence pairs whose edit distance is not zero.
        """
        if self.seq_num == 0:
            raise ValueError(
                "There is no data in EditDistance Metric. Please check layers.edit_distance output has been added to EditDistance."
            )
        avg_distance = self.total_distance / self.seq_num
        avg_instance_error = self.instance_error / float(self.seq_num)
        return avg_distance, avg_instance_error


class Auc(MetricBase):
    """
    The auc metric is for binary classification.
    Refer to https://en.wikipedia.org/wiki/Receiver_operating_characteristic#Area_under_the_curve
    Please notice that the auc metric is implemented with python, which may be a little bit slow.
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

            import numpy as np
            # init the auc metric
            auc_metric = fluid.metrics.Auc("ROC")

            # suppose that batch_size is 128
            batch_num = 100
            batch_size = 128

            for batch_id in range(batch_num):

                class0_preds = np.random.random(size = (batch_size, 1))
                class1_preds = 1 - class0_preds

                preds = np.concatenate((class0_preds, class1_preds), axis=1)

                labels = np.random.randint(2, size = (batch_size, 1))
                auc_metric.update(preds = preds, labels = labels)

                # shall be some score closing to 0.5 as the preds are randomly assigned
                print("auc for iteration %d is %.2f" % (batch_id, auc_metric.eval()))
    """

    def __init__(self, name, curve='ROC', num_thresholds=4095):
        super(Auc, self).__init__(name=name)
        self._curve = curve
        self._num_thresholds = num_thresholds

        _num_pred_buckets = num_thresholds + 1
        self._stat_pos = [0] * _num_pred_buckets
        self._stat_neg = [0] * _num_pred_buckets

    def update(self, preds, labels):
        """
        Update the auc curve with the given predictions and labels

        Args:
             preds: an numpy array in the shape of (batch_size, 2), preds[i][j] denotes the probability
             of classifying the instance i into the class j.
             labels: an numpy array in the shape of (batch_size, 1), labels[i] is either o or 1, representing
             the label of the instance i.
        """
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
        """
        Return the area (a float score) under auc curve
        """
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
            considered, 0 by default.
        overlap_threshold (float): The threshold for deciding true/false
            positive, 0.5 by default.
        evaluate_difficult (bool): Whether to consider difficult ground truth
            for evaluation, True by default. This argument does not work when
            gt_difficult is None.
        ap_version (string): The average precision calculation ways, it must be
            'integral' or '11point'. Please check
            https://sanchom.wordpress.com/tag/average-precision/ for details.
            - 11point: the 11-point interpolated average precision.
            - integral: the natural integral of the precision-recall curve.

    Examples:
        .. code-block:: python

            import paddle.fluid.layers as layers

            batch_size = -1 # can be any size
            image_boxs_num = 10
            bounding_bboxes_num = 21

            pb = layers.data(name='prior_box', shape=[image_boxs_num, 4],
                append_batch_size=False, dtype='float32')

            pbv = layers.data(name='prior_box_var', shape=[image_boxs_num, 4],
                append_batch_size=False, dtype='float32')

            loc = layers.data(name='target_box', shape=[batch_size, bounding_bboxes_num, 4],
                append_batch_size=False, dtype='float32')

            scores = layers.data(name='scores', shape=[batch_size, bounding_bboxes_num, image_boxs_num],
                append_batch_size=False, dtype='float32')

            nmsed_outs = fluid.layers.detection_output(scores=scores,
                loc=loc, prior_box=pb, prior_box_var=pbv)

            gt_box = fluid.layers.data(name="gt_box", shape=[batch_size, 4], dtype="float32")
            gt_label = fluid.layers.data(name="gt_label", shape=[batch_size, 1], dtype="float32")
            difficult = fluid.layers.data(name="difficult", shape=[batch_size, 1], dtype="float32")

            exe = fluid.Executor(fluid.CUDAPlace(0))
            map_evaluator = fluid.metrics.DetectionMAP(nmsed_outs, gt_label, gt_box, difficult, class_num = 3)

            cur_map, accum_map = map_evaluator.get_map_var()

 
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
        map = detection.detection_map(
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
        accum_map = detection.detection_map(
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
