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

from __future__ import print_function

import unittest
import numpy as np
from op_test import OpTest
from paddle.fluid import metrics


class TestAucOp(OpTest):
    def setUp(self):
        self.op_type = "auc"
        pred = np.random.random((128, 2)).astype("float32")
        labels = np.random.randint(0, 2, (128, 1)).astype("int64")
        num_thresholds = 200

        stat_pos = np.zeros((num_thresholds + 1, )).astype("int64")
        stat_neg = np.zeros((num_thresholds + 1, )).astype("int64")

        self.inputs = {
            'Predict': pred,
            'Label': labels,
            "StatPos": stat_pos,
            "StatNeg": stat_neg
        }
        self.attrs = {
            'curve': 'ROC',
            'num_thresholds': num_thresholds,
            "slide_steps": 1
        }

        python_auc = metrics.Auc(name="auc",
                                 curve='ROC',
                                 num_thresholds=num_thresholds)
        python_auc.update(pred, labels)

        self.outputs = {
            'AUC': np.array(python_auc.eval()),
            'StatPosOut': np.array(python_auc._stat_pos),
            'StatNegOut': np.array(python_auc._stat_neg)
        }

    def test_check_output(self):
        self.check_output()


if __name__ == "__main__":
    unittest.main()
