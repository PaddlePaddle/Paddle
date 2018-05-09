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


def compute_mean_iou(predictions, labels, num_classes, in_cms, im_mean_ious):
    assert predictions.shape == labels.shape
    predictions = predictions.flatten()
    labels = labels.flatten()

    out_cm = np.zeros([num_classes, num_classes]).astype("float32")
    for _, cm in in_cms:
        out_cm += cm

    for pred, label in zip(predictions, labels):
        out_cm[pred][label] += 1

    row_sum = out_cm.sum(axis=0)
    col_sum = out_cm.sum(axis=1)
    diag = np.diag(out_cm)
    denominator = row_sum + col_sum - diag
    valid_count = (denominator != 0).sum()
    denominator = np.where(denominator > 0, denominator,
                           np.ones(denominator.shape))
    mean_iou = (diag / denominator).sum() / valid_count

    for _, im_mean_iou in im_mean_ious:
        mean_iou += im_mean_iou
    return mean_iou, out_cm


class TestMeanIOUOp(OpTest):
    def setUp(self):
        self.config()
        self.op_type = "mean_iou"
        predictions = np.random.randint(0, self.num_classes,
                                        self.image_size).astype("int32")
        labels = np.random.randint(0, self.num_classes,
                                   self.image_size).astype("int32")

        in_cms = []
        for i in range(self.in_cm_num):
            in_cms.append(("in_cms_%d" % i, np.zeros(
                [self.num_classes, self.num_classes]).astype("float32")))

        in_mean_ious = []
        for i in range(self.in_mean_iou_num):
            in_mean_ious.append(("in_mean_iou_%d" % i, np.random.uniform(
                0, 1, [1]).astype("float32")))
        self.inputs = {
            'predictions': predictions,
            'labels': labels,
            'in_confusion_matrix': in_cms,
            'in_mean_iou': in_mean_ious
        }
        self.attrs = {'num_classes': long(self.num_classes)}

        mean_iou, out_cm = compute_mean_iou(
            predictions, labels, self.num_classes, in_cms, in_mean_ious)

        self.outputs = {'mean_iou': mean_iou, 'out_confusion_matrix': out_cm}

    def config(self):
        self.num_classes = 10
        self.image_size = [128, 128]
        self.in_cm_num = 0
        self.in_mean_iou_num = 0

    def test_check_output(self):
        self.check_output()


class TestCase1(TestMeanIOUOp):
    def config(self):
        self.num_classes = 5
        self.image_size = [100, 128]
        self.in_cm_num = 2
        self.in_mean_iou_num = 2


if __name__ == '__main__':
    unittest.main()
