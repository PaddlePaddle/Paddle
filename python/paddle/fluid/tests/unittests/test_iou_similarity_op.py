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
import numpy.random as random
import sys
import math
from op_test import OpTest


class TestIOUSimilarityOp(OpTest):

    def test_check_output(self):
        self.check_output(check_dygraph=False)

    def setUp(self):
        self.op_type = "iou_similarity"
        self.boxes1 = random.rand(2, 4).astype('float32')
        self.boxes2 = random.rand(3, 4).astype('float32')
        self.output = random.rand(2, 3).astype('float32')
        self.box_normalized = False
        # run python iou computation
        self._compute_iou()
        self.inputs = {'X': self.boxes1, 'Y': self.boxes2}
        self.attrs = {"box_normalized": self.box_normalized}
        self.outputs = {'Out': self.output}

    def _compute_iou(self, ):
        for row in range(self.boxes1.shape[0]):
            for col in range(self.boxes2.shape[0]):
                xmin1, ymin1, xmax1, ymax1 = self.boxes1[row]
                xmin2, ymin2, xmax2, ymax2 = self.boxes2[col]
                if not self.box_normalized:
                    area1 = (ymax1 - ymin1 + 1) * (xmax1 - xmin1 + 1)
                    area2 = (ymax2 - ymin2 + 1) * (xmax2 - xmin2 + 1)
                else:
                    area1 = (ymax1 - ymin1) * (xmax1 - xmin1)
                    area2 = (ymax2 - ymin2) * (xmax2 - xmin2)

                inter_xmax = min(xmax1, xmax2)
                inter_ymax = min(ymax1, ymax2)
                inter_xmin = max(xmin1, xmin2)
                inter_ymin = max(ymin1, ymin2)
                inter_height = inter_ymax - inter_ymin
                inter_width = inter_xmax - inter_xmin
                if not self.box_normalized:
                    inter_height += 1
                    inter_width += 1
                inter_height = max(inter_height, 0)
                inter_width = max(inter_width, 0)
                inter_area = inter_width * inter_height
                union_area = area1 + area2 - inter_area
                sim_score = inter_area / union_area
                self.output[row, col] = sim_score


class TestIOUSimilarityOpWithLoD(TestIOUSimilarityOp):

    def test_check_output(self):
        self.check_output(check_dygraph=False)

    def setUp(self):
        super(TestIOUSimilarityOpWithLoD, self).setUp()
        self.boxes1_lod = [[1, 1]]
        self.output_lod = [[1, 1]]
        self.box_normalized = False
        # run python iou computation
        self._compute_iou()
        self.inputs = {'X': (self.boxes1, self.boxes1_lod), 'Y': self.boxes2}
        self.attrs = {"box_normalized": self.box_normalized}
        self.outputs = {'Out': (self.output, self.output_lod)}


class TestIOUSimilarityOpWithBoxNormalized(TestIOUSimilarityOp):

    def test_check_output(self):
        self.check_output(check_dygraph=False)

    def setUp(self):
        super(TestIOUSimilarityOpWithBoxNormalized, self).setUp()
        self.boxes1_lod = [[1, 1]]
        self.output_lod = [[1, 1]]
        self.box_normalized = True
        # run python iou computation
        self._compute_iou()
        self.inputs = {'X': (self.boxes1, self.boxes1_lod), 'Y': self.boxes2}
        self.attrs = {"box_normalized": self.box_normalized}
        self.outputs = {'Out': (self.output, self.output_lod)}


if __name__ == '__main__':
    unittest.main()
