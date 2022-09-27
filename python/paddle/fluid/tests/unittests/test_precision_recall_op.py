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

import unittest
import numpy as np
from op_test import OpTest


def calc_precision(tp_count, fp_count):
    if tp_count > 0.0 or fp_count > 0.0:
        return tp_count / (tp_count + fp_count)
    return 1.0


def calc_recall(tp_count, fn_count):
    if tp_count > 0.0 or fn_count > 0.0:
        return tp_count / (tp_count + fn_count)
    return 1.0


def calc_f1_score(precision, recall):
    if precision > 0.0 or recall > 0.0:
        return 2 * precision * recall / (precision + recall)
    return 0.0


def get_states(idxs, labels, cls_num, weights=None):
    ins_num = idxs.shape[0]
    # TP FP TN FN
    states = np.zeros((cls_num, 4)).astype('float32')
    for i in range(ins_num):
        w = weights[i] if weights is not None else 1.0
        idx = idxs[i][0]
        label = labels[i][0]
        if idx == label:
            states[idx][0] += w
            for j in range(cls_num):
                states[j][2] += w
            states[idx][2] -= w
        else:
            states[label][3] += w
            states[idx][1] += w
            for j in range(cls_num):
                states[j][2] += w
            states[label][2] -= w
            states[idx][2] -= w
    return states


def compute_metrics(states, cls_num):
    total_tp_count = 0.0
    total_fp_count = 0.0
    total_fn_count = 0.0
    macro_avg_precision = 0.0
    macro_avg_recall = 0.0
    for i in range(cls_num):
        total_tp_count += states[i][0]
        total_fp_count += states[i][1]
        total_fn_count += states[i][3]
        macro_avg_precision += calc_precision(states[i][0], states[i][1])
        macro_avg_recall += calc_recall(states[i][0], states[i][3])
    metrics = []
    macro_avg_precision /= cls_num
    macro_avg_recall /= cls_num
    metrics.append(macro_avg_precision)
    metrics.append(macro_avg_recall)
    metrics.append(calc_f1_score(macro_avg_precision, macro_avg_recall))
    micro_avg_precision = calc_precision(total_tp_count, total_fp_count)
    metrics.append(micro_avg_precision)
    micro_avg_recall = calc_recall(total_tp_count, total_fn_count)
    metrics.append(micro_avg_recall)
    metrics.append(calc_f1_score(micro_avg_precision, micro_avg_recall))
    return np.array(metrics).astype('float32')


class TestPrecisionRecallOp_0(OpTest):

    def setUp(self):
        self.op_type = "precision_recall"
        ins_num = 64
        cls_num = 10
        max_probs = np.random.uniform(0, 1.0, (ins_num, 1)).astype('float32')
        idxs = np.random.choice(range(cls_num), ins_num).reshape(
            (ins_num, 1)).astype('int32')
        labels = np.random.choice(range(cls_num), ins_num).reshape(
            (ins_num, 1)).astype('int32')
        states = get_states(idxs, labels, cls_num)
        metrics = compute_metrics(states, cls_num)

        self.attrs = {'class_number': cls_num}

        self.inputs = {'MaxProbs': max_probs, 'Indices': idxs, 'Labels': labels}

        self.outputs = {
            'BatchMetrics': metrics,
            'AccumMetrics': metrics,
            'AccumStatesInfo': states
        }

    def test_check_output(self):
        self.check_output()


class TestPrecisionRecallOp_1(OpTest):

    def setUp(self):
        self.op_type = "precision_recall"
        ins_num = 64
        cls_num = 10
        max_probs = np.random.uniform(0, 1.0, (ins_num, 1)).astype('float32')
        idxs = np.random.choice(range(cls_num), ins_num).reshape(
            (ins_num, 1)).astype('int32')
        weights = np.random.uniform(0, 1.0, (ins_num, 1)).astype('float32')
        labels = np.random.choice(range(cls_num), ins_num).reshape(
            (ins_num, 1)).astype('int32')

        states = get_states(idxs, labels, cls_num, weights)
        metrics = compute_metrics(states, cls_num)

        self.attrs = {'class_number': cls_num}

        self.inputs = {
            'MaxProbs': max_probs,
            'Indices': idxs,
            'Labels': labels,
            'Weights': weights
        }

        self.outputs = {
            'BatchMetrics': metrics,
            'AccumMetrics': metrics,
            'AccumStatesInfo': states
        }

    def test_check_output(self):
        self.check_output()


class TestPrecisionRecallOp_2(OpTest):

    def setUp(self):
        self.op_type = "precision_recall"
        ins_num = 64
        cls_num = 10
        max_probs = np.random.uniform(0, 1.0, (ins_num, 1)).astype('float32')
        idxs = np.random.choice(range(cls_num), ins_num).reshape(
            (ins_num, 1)).astype('int32')
        weights = np.random.uniform(0, 1.0, (ins_num, 1)).astype('float32')
        labels = np.random.choice(range(cls_num), ins_num).reshape(
            (ins_num, 1)).astype('int32')
        states = np.random.randint(0, 30, (cls_num, 4)).astype('float32')

        accum_states = get_states(idxs, labels, cls_num, weights)
        batch_metrics = compute_metrics(accum_states, cls_num)
        accum_states += states
        accum_metrics = compute_metrics(accum_states, cls_num)

        self.attrs = {'class_number': cls_num}

        self.inputs = {
            'MaxProbs': max_probs,
            'Indices': idxs,
            'Labels': labels,
            'Weights': weights,
            'StatesInfo': states
        }

        self.outputs = {
            'BatchMetrics': batch_metrics,
            'AccumMetrics': accum_metrics,
            'AccumStatesInfo': accum_states
        }

    def test_check_output(self):
        self.check_output()


if __name__ == '__main__':
    unittest.main()
