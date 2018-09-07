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


class TestAucOp(OpTest):
    def setUp(self):
        self.op_type = "auc"
        pred = np.random.random((128, 2)).astype("float32")
        indices = np.random.randint(0, 2, (128, 2))
        labels = np.random.randint(0, 2, (128, 1))
        num_thresholds = 200
        self.inputs = {'Out': pred, 'Indices': indices, 'Label': labels}
        self.attrs = {'curve': 'ROC', 'num_thresholds': num_thresholds}
        # NOTE: sklearn use a different way to generate thresholds
        #       which will cause the result differs slightly:
        # from sklearn.metrics import roc_curve, auc
        # fpr, tpr, thresholds = roc_curve(labels, pred)
        # auc_value = auc(fpr, tpr)
        # we caculate AUC again using numpy for testing
        kepsilon = 1e-7  # to account for floating point imprecisions
        thresholds = [(i + 1) * 1.0 / (num_thresholds - 1)
                      for i in range(num_thresholds - 2)]
        thresholds = [0.0 - kepsilon] + thresholds + [1.0 + kepsilon]

        # caculate TP, FN, TN, FP count
        tp_list = np.ndarray((num_thresholds, ))
        fn_list = np.ndarray((num_thresholds, ))
        tn_list = np.ndarray((num_thresholds, ))
        fp_list = np.ndarray((num_thresholds, ))
        for idx_thresh, thresh in enumerate(thresholds):
            tp, fn, tn, fp = 0, 0, 0, 0
            for i, lbl in enumerate(labels):
                if lbl:
                    if pred[i, 0] >= thresh:
                        tp += 1
                    else:
                        fn += 1
                else:
                    if pred[i, 0] >= thresh:
                        fp += 1
                    else:
                        tn += 1
            tp_list[idx_thresh] = tp
            fn_list[idx_thresh] = fn
            tn_list[idx_thresh] = tn
            fp_list[idx_thresh] = fp

        epsilon = 1e-6
        tpr = (tp_list.astype("float32") + epsilon) / (
            tp_list + fn_list + epsilon)
        fpr = fp_list.astype("float32") / (fp_list + tn_list + epsilon)
        rec = (tp_list.astype("float32") + epsilon) / (
            tp_list + fp_list + epsilon)

        x = fpr[:num_thresholds - 1] - fpr[1:]
        y = (tpr[:num_thresholds - 1] + tpr[1:]) / 2.0
        auc_value = np.sum(x * y)

        self.outputs = {'AUC': auc_value}

    def test_check_output(self):
        self.check_output()


if __name__ == "__main__":
    unittest.main()
