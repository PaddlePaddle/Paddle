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

import paddle


def _compute_auc_ref(pred, labels, num_thresholds=200, slide_steps=1):
    """Compute AUC reference values matching the C++ auc op behavior."""
    pred_pos = pred[:, 1]
    label = labels.reshape(-1)

    thresholds = np.array(
        [i / num_thresholds for i in range(num_thresholds + 1)],
        dtype='float64',
    )

    stat_len = (1 + slide_steps) * (num_thresholds + 1) + 1
    stat_pos = np.zeros(stat_len, dtype='int64')
    stat_neg = np.zeros(stat_len, dtype='int64')

    for i in range(len(label)):
        for j in range(num_thresholds + 1):
            if pred_pos[i] >= thresholds[j]:
                if label[i] == 1:
                    stat_pos[j] += 1
                else:
                    stat_neg[j] += 1
            else:
                break

    auc = 0.0
    total_pos = stat_pos.sum()
    total_neg = stat_neg.sum()
    if total_pos > 0 and total_neg > 0:
        cum_neg = 0.0
        for i in range(num_thresholds):
            cum_neg_prev = cum_neg
            cum_neg += stat_neg[i]
            auc += (cum_neg - cum_neg_prev) * (stat_pos[i] + stat_pos[i + 1]) / 2.0
        auc /= (total_pos * total_neg)

    return auc, stat_pos, stat_neg


class TestAucSinglePredOp(OpTest):
    def setUp(self):
        self.op_type = "auc"
        pred = np.random.random((128, 2)).astype("float32")
        pred0 = pred[:, 0].reshape(128, 1)
        labels = np.random.randint(0, 2, (128, 1)).astype("int64")
        num_thresholds = 200
        slide_steps = 1

        stat_pos = np.zeros(
            (1 + slide_steps) * (num_thresholds + 1) + 1,
        ).astype("int64")
        stat_neg = np.zeros(
            (1 + slide_steps) * (num_thresholds + 1) + 1,
        ).astype("int64")

        self.inputs = {
            'Predict': pred0,
            'Label': labels,
            "StatPos": stat_pos,
            "StatNeg": stat_neg,
        }
        self.attrs = {
            'curve': 'ROC',
            'num_thresholds': num_thresholds,
            "slide_steps": slide_steps,
        }

        for i in range(128):
            pred[i][1] = pred[i][0]
        auc_val, pos, neg = _compute_auc_ref(pred, labels, num_thresholds, slide_steps)

        pos_out = pos.tolist() * 2
        pos_out.append(1)
        neg_out = neg.tolist() * 2
        neg_out.append(1)
        self.outputs = {
            'AUC': np.array(auc_val).astype("float64"),
            'StatPosOut': np.array(pos_out),
            'StatNegOut': np.array(neg_out),
        }

    def test_check_output(self):
        self.check_output(check_dygraph=False)


class TestAucGlobalSinglePredOp(OpTest):
    def setUp(self):
        self.op_type = "auc"
        pred = np.random.random((128, 2)).astype("float32")
        pred0 = pred[:, 0].reshape(128, 1)
        labels = np.random.randint(0, 2, (128, 1)).astype("int64")
        num_thresholds = 200
        slide_steps = 0

        stat_pos = np.zeros((1, (num_thresholds + 1))).astype("int64")
        stat_neg = np.zeros((1, (num_thresholds + 1))).astype("int64")

        self.inputs = {
            'Predict': pred0,
            'Label': labels,
            "StatPos": stat_pos,
            "StatNeg": stat_neg,
        }
        self.attrs = {
            'curve': 'ROC',
            'num_thresholds': num_thresholds,
            "slide_steps": slide_steps,
        }

        for i in range(128):
            pred[i][1] = pred[i][0]
        auc_val, pos, neg = _compute_auc_ref(pred, labels, num_thresholds, slide_steps)

        self.outputs = {
            'AUC': np.array(auc_val).astype("float64"),
            'StatPosOut': np.array([pos]),
            'StatNegOut': np.array([neg]),
        }

    def test_check_output(self):
        self.check_output(check_dygraph=False)


if __name__ == "__main__":
    unittest.main()
