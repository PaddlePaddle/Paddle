# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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


def iou(box_a, box_b):
    """Apply intersection-over-union overlap between box_a and box_b"""
    xmin_a = min(box_a[0], box_a[2])
    ymin_a = min(box_a[1], box_a[3])
    xmax_a = max(box_a[0], box_a[2])
    ymax_a = max(box_a[1], box_a[3])

    xmin_b = min(box_b[0], box_b[2])
    ymin_b = min(box_b[1], box_b[3])
    xmax_b = max(box_b[0], box_b[2])
    ymax_b = max(box_b[1], box_b[3])

    area_a = (ymax_a - ymin_a) * (xmax_a - xmin_a)
    area_b = (ymax_b - ymin_b) * (xmax_b - xmin_b)
    if area_a <= 0 and area_b <= 0:
        return 0.0

    xa = max(xmin_a, xmin_b)
    ya = max(ymin_a, ymin_b)
    xb = min(xmax_a, xmax_b)
    yb = min(ymax_a, ymax_b)

    inter_area = max(xb - xa, 0.0) * max(yb - ya, 0.0)

    iou_ratio = inter_area / (area_a + area_b - inter_area)
    return iou_ratio


def nms(boxes, nms_threshold):
    selected_indices = np.zeros(boxes.shape[0], dtype=np.int64)
    keep = np.ones(boxes.shape[0], dtype=int)
    io_ratio = np.ones((boxes.shape[0], boxes.shape[0]), dtype=np.float64)
    cnt = 0
    for i in range(boxes.shape[0]):
        if keep[i] == 0:
            continue
        selected_indices[cnt] = i
        cnt += 1
        for j in range(i + 1, boxes.shape[0]):
            io_ratio[i][j] = iou(boxes[i], boxes[j])
            if keep[j]:
                overlap = iou(boxes[i], boxes[j])
                keep[j] = 1 if overlap <= nms_threshold else 0
            else:
                continue

    return selected_indices[:cnt]


class TestNMSOp(OpTest):
    def setUp(self):
        self.op_type = 'nms'
        self.python_api = paddle.vision.ops.nms
        self.dtype = np.float64
        self.init_dtype_type()
        boxes = np.random.rand(32, 4).astype(self.dtype)
        boxes[:, 2] = boxes[:, 0] + boxes[:, 2]
        boxes[:, 3] = boxes[:, 1] + boxes[:, 3]

        paddle.disable_static()
        self.inputs = {'Boxes': boxes}
        self.attrs = {'iou_threshold': 0.5}
        out_py = nms(boxes, self.attrs['iou_threshold'])
        self.outputs = {'KeepBoxesIdxs': out_py}
        paddle.enable_static()

    def init_dtype_type(self):
        pass

    def test_check_output(self):
        self.check_output(check_pir=True)


if __name__ == "__main__":
    unittest.main()
