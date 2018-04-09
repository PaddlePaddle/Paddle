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
import numpy as np
import copy


def _is_numpy_(var):
    return isinstance(var, (np.ndarray, np.generic))


def _is_number_(var):
    return isinstance(var, int) or isinstance(var, float) or (isinstance(
        var, np.ndarray) and var.shape == (1, ))


def _is_number_or_matrix_(var):
    return _is_number_(var) or isinstance(var, np.ndarray)


class MetricBase(object):
    """
    """

    def __init__(self, name=None, **kwargs):
        self._name = name if name != None else self.__class__.__name__
        self._kwargs = kwargs
        self.reset()

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
        for attr, value in states:
            if isinstance(value, int):
                states[attr] = 0
            elif isinstance(value, float):
                states[attr] = .0
            elif isinstance(value, (np.ndarray, np.generic)):
                states[attr] = np.zeros_like(value)
            else:
                states[attr] = None

    def get_config(self):
        states = {
            attr: value
            for attr, value in self.__dict__.iteritems()
            if not attr.startswith("_")
        }
        config = copy.deepcopy(self._kwargs)
        return config.update({
            "name": self._name,
            "states": copy.deepcopy(states)
        })

    def update(self):
        raise NotImplementedError()

    def eval(self):
        raise NotImplementedError()


class CompositeMetric(MetricBase):
    """
    Compute multiple metrics in each minibatch.
    for example, merge F1, accuracy, recall into one Metric.
    """

    def __init__(self):
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
    def __init__(self):
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


class ChunkEvalutor(MetricBase):
    def __init__(self):
        self.num_infer_chunks = 0
        self.num_label_chunks = 0
        self.num_correct_chunks = 0

    def update(self, precision, recall, f1_score, num_infer_chunks,
               num_label_chunks, num_correct_chunks):
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
    def __init__(self):
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

    def eval():
        if self.seq_num == 0:
            raise ValueError(
                "There is no data in EditDistance Metric. Please check layers.edit_distance output has been added to EditDistance."
            )
        avg_distance = self.total_distance / self.seq_num
        avg_instance_error = self.instance_error / self.seq_num
        return avg_distance, avg_instance_error
