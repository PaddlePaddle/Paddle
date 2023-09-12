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

        python_auc = paddle.metric.Auc(
            name="auc", curve='ROC', num_thresholds=num_thresholds
        )
        for i in range(128):
            pred[i][1] = pred[i][0]
        python_auc.update(pred, labels)

        pos = python_auc._stat_pos.tolist() * 2
        pos.append(1)
        neg = python_auc._stat_neg.tolist() * 2
        neg.append(1)
        self.outputs = {
            'AUC': np.array(python_auc.accumulate()),
            'StatPosOut': np.array(pos),
            'StatNegOut': np.array(neg),
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

        python_auc = paddle.metric.Auc(
            name="auc", curve='ROC', num_thresholds=num_thresholds
        )
        for i in range(128):
            pred[i][1] = pred[i][0]
        python_auc.update(pred, labels)

        pos = python_auc._stat_pos
        neg = python_auc._stat_neg
        self.outputs = {
            'AUC': np.array(python_auc.accumulate()),
            'StatPosOut': np.array([pos]),
            'StatNegOut': np.array([neg]),
        }

    def test_check_output(self):
        self.check_output(check_dygraph=False)


if __name__ == "__main__":
    unittest.main()
