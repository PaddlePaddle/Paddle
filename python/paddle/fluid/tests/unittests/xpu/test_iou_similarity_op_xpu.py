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

sys.path.append("..")
from op_test_xpu import XPUOpTest
from xpu.get_test_cover_info import create_test_class, get_xpu_op_support_types, XPUOpTestWrapper
import paddle

paddle.enable_static()


class XPUTestIOUSimilarityOp(XPUOpTestWrapper):

    def __init__(self):
        self.op_name = 'iou_similarity'
        self.use_dynamic_create_class = False

    class TestXPUIOUSimilarityOp(XPUOpTest):

        def init(self):
            self.dtype = self.in_type
            self.place = paddle.XPUPlace(0)
            self.op_type = 'iou_similarity'

        def test_check_output(self):
            self.check_output_with_place(self.place)

        def setUp(self):
            self.init()
            self.boxes1 = random.rand(2, 4).astype(self.dtype)
            self.boxes2 = random.rand(3, 4).astype(self.dtype)
            self.output = random.rand(2, 3).astype(self.dtype)
            self.box_normalized = False
            # run python iou computation
            self._compute_iou()
            self.inputs = {'X': self.boxes1, 'Y': self.boxes2}
            self.attrs = {
                "box_normalized": self.box_normalized,
                'use_xpu': True
            }
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

    class TestXPUIOUSimilarityOpWithLoD(TestXPUIOUSimilarityOp):

        def test_check_output(self):
            self.check_output_with_place(self.place, check_dygraph=False)

        def setUp(self):
            super().setUp()
            self.boxes1_lod = [[1, 1]]
            self.output_lod = [[1, 1]]
            self.box_normalized = False
            # run python iou computation
            self._compute_iou()
            self.inputs = {
                'X': (self.boxes1, self.boxes1_lod),
                'Y': self.boxes2
            }
            self.attrs = {"box_normalized": self.box_normalized}
            self.outputs = {'Out': (self.output, self.output_lod)}

    class TestXPUIOUSimilarityOpWithBoxNormalized(TestXPUIOUSimilarityOp):

        def test_check_output(self):
            self.check_output_with_place(self.place, check_dygraph=False)

        def setUp(self):
            super().setUp()
            self.boxes1_lod = [[1, 1]]
            self.output_lod = [[1, 1]]
            self.box_normalized = True
            # run python iou computation
            self._compute_iou()
            self.inputs = {
                'X': (self.boxes1, self.boxes1_lod),
                'Y': self.boxes2
            }
            self.attrs = {"box_normalized": self.box_normalized}
            self.outputs = {'Out': (self.output, self.output_lod)}


support_types = get_xpu_op_support_types('iou_similarity')
for stype in support_types:
    create_test_class(globals(), XPUTestIOUSimilarityOp, stype)

if __name__ == '__main__':
    unittest.main()
