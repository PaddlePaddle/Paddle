#   Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

sys.path.append("..")
import math
import paddle
from op_test import OpTest

paddle.enable_static()

np.random.seed(2022)


class TestMluIouSimilarityOp(OpTest):

    def setUp(self):
        self.op_type = "iou_similarity"
        self.set_mlu()
        self.init_dtype()
        self.set_init_config()
        self.set_attrs()
        self.set_inputs()
        self.set_outputs()

    def set_mlu(self):
        self.__class__.use_mlu = True
        self.place = paddle.MLUPlace(0)

    def init_dtype(self):
        self.dtype = np.float32

    def set_init_config(self):
        self.N = 2
        self.M = 3
        self.box_normalized = False
        self.use_lod = False

    def set_inputs(self):
        self.boxes1 = random.rand(self.N, 4).astype(self.dtype)
        self.boxes2 = random.rand(self.M, 4).astype(self.dtype)
        if self.use_lod:
            self.boxes1_lod = [[1 for _ in range(self.N)]]
            self.inputs = {
                'X': (self.boxes1, self.boxes1_lod),
                'Y': self.boxes2
            }
        else:
            self.inputs = {'X': self.boxes1, 'Y': self.boxes2}

    def set_attrs(self):
        self.attrs = {"box_normalized": self.box_normalized}

    def set_outputs(self):
        self.output = random.rand(self.N, self.M).astype(self.dtype)
        self._compute_iou()
        self.outputs = {'Out': self.output}

    def test_check_output(self):
        self.check_output_with_place(self.place)

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


class TestMluIouSimilarityOpWithLoD(TestMluIouSimilarityOp):

    def set_init_config(self):
        super(TestMluIouSimilarityOpWithLoD, self).set_init_config()
        self.box_normalized = True
        self.use_lod = True


class TestMluIouSimilarityOpWithBoxNormalized(TestMluIouSimilarityOp):

    def set_init_config(self):
        super(TestMluIouSimilarityOpWithBoxNormalized, self).set_init_config()
        self.box_normalized = True
        self.use_lod = True


def TestMluIouSimilarityOpFp16(TestMluIouSimilarityOp):

    def init_dtype(self):
        self.dtype = np.float16


if __name__ == '__main__':
    unittest.main()
