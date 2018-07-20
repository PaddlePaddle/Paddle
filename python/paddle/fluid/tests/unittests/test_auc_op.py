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
from paddle.fluid import metrics


class TestAucOp(OpTest):
    def setUp(self):
        self.op_type = "auc"
        pred = np.random.random((128, 2)).astype("float32")
        labels = np.random.randint(0, 2, (128, 1))
        num_thresholds = 200
        tp = np.zeros((num_thresholds, )).astype("int64")
        tn = np.zeros((num_thresholds, )).astype("int64")
        fp = np.zeros((num_thresholds, )).astype("int64")
        fn = np.zeros((num_thresholds, )).astype("int64")

        self.inputs = {
            'Predict': pred,
            'Label': labels,
            'TP': tp,
            'TN': tn,
            'FP': fp,
            'FN': fn
        }
        self.attrs = {'curve': 'ROC', 'num_thresholds': num_thresholds}

        python_auc = metrics.Auc(name="auc",
                                 curve='ROC',
                                 num_thresholds=num_thresholds)
        python_auc.update(pred, labels)

        self.outputs = {
            'AUC': python_auc.eval(),
            'TPOut': python_auc.tp_list,
            'FNOut': python_auc.fn_list,
            'TNOut': python_auc.tn_list,
            'FPOut': python_auc.fp_list
        }

    def test_check_output(self):
        self.check_output()


if __name__ == "__main__":
    unittest.main()
