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

from __future__ import division
import unittest
import numpy as np
from op_test import OpTest


def compute_mean_iou(predictions, labels, num_classes, in_cm):
    assert predictions.shape == labels.shape
    predictions = predictions.flatten()
    labels = labels.flatten()
    out_cm = np.zeros(in_cm.shape).astype(in_cm.dtype)
    for pred, label in zip(predictions, labels):
        out_cm[pred][label] = in_cm[pred][label] + 1

    row_sum = out_cm.sum(axis=0)
    col_sum = out_cm.sum(axis=1)
    diag = np.diag(out_cm)
    denominator = row_sum + col_sum - diag
    valid_count = (denominator != 0).sum()
    denominator = np.where(denominator > 0, denominator,
                           np.ones(denominator.shape))
    mean_iou = (diag / denominator).sum()

    return mean_iou, out_cm


class TestMeanIOUOp(OpTest):
    def setUp(self):
        self.config()
        self.op_type = "mean_iou"
        predictions = np.random.randint(0, self.num_classes,
                                        self.image_size).astype("int32")
        labels = np.random.randint(0, self.num_classes,
                                   self.image_size).astype("int32")

        in_cm = np.ones([self.num_classes, self.num_classes]).astype("float32")
        self.inputs = {
            'predictions': predictions,
            'labels': predictions,
            'in_confusion_matrix': in_cm
        }
        self.attrs = {'num_classes': long(self.num_classes)}

        mean_iou, out_cm = compute_mean_iou(predictions, labels,
                                            self.num_classes, in_cm)
        print "out_cm from python: %s" % out_cm
        print "mean_iou from python: %s" % mean_iou

        self.outputs = {'mean_iou': mean_iou, 'out_confusion_matrix': out_cm}

    def config(self):
        self.num_classes = 10
        self.image_size = [128, 128]

    def test_check_output(self):
        self.check_output()


if __name__ == '__main__':
    unittest.main()
