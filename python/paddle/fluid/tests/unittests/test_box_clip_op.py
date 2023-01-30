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

<<<<<<< HEAD
import unittest

import numpy as np
from op_test import OpTest
=======
from __future__ import print_function

import unittest
import numpy as np
import sys
import math
from op_test import OpTest
import copy
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81


def box_clip(input_box, im_info, output_box):
    im_w = round(im_info[1] / im_info[2])
    im_h = round(im_info[0] / im_info[2])
<<<<<<< HEAD
    output_box[:, :, 0] = np.maximum(
        np.minimum(input_box[:, :, 0], im_w - 1), 0
    )
    output_box[:, :, 1] = np.maximum(
        np.minimum(input_box[:, :, 1], im_h - 1), 0
    )
    output_box[:, :, 2] = np.maximum(
        np.minimum(input_box[:, :, 2], im_w - 1), 0
    )
    output_box[:, :, 3] = np.maximum(
        np.minimum(input_box[:, :, 3], im_h - 1), 0
    )
=======
    output_box[:, :, 0] = np.maximum(np.minimum(input_box[:, :, 0], im_w - 1),
                                     0)
    output_box[:, :, 1] = np.maximum(np.minimum(input_box[:, :, 1], im_h - 1),
                                     0)
    output_box[:, :, 2] = np.maximum(np.minimum(input_box[:, :, 2], im_w - 1),
                                     0)
    output_box[:, :, 3] = np.maximum(np.minimum(input_box[:, :, 3], im_h - 1),
                                     0)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81


def batch_box_clip(input_boxes, im_info, lod):
    n = input_boxes.shape[0]
    m = input_boxes.shape[1]
    output_boxes = np.zeros((n, m, 4), dtype=np.float32)
    cur_offset = 0
    for i in range(len(lod)):
<<<<<<< HEAD
        box_clip(
            input_boxes[cur_offset : (cur_offset + lod[i]), :, :],
            im_info[i, :],
            output_boxes[cur_offset : (cur_offset + lod[i]), :, :],
        )
=======
        box_clip(input_boxes[cur_offset:(cur_offset + lod[i]), :, :],
                 im_info[i, :],
                 output_boxes[cur_offset:(cur_offset + lod[i]), :, :])
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        cur_offset += lod[i]
    return output_boxes


class TestBoxClipOp(OpTest):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def test_check_output(self):
        self.check_output()

    def setUp(self):
        self.op_type = "box_clip"
        lod = [[1, 2, 3]]
        input_boxes = np.random.random((6, 10, 4)) * 5
<<<<<<< HEAD
        im_info = np.array([[5, 8, 1.0], [6, 6, 1.0], [7, 5, 1.0]])
=======
        im_info = np.array([[5, 8, 1.], [6, 6, 1.], [7, 5, 1.]])
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        output_boxes = batch_box_clip(input_boxes, im_info, lod[0])

        self.inputs = {
            'Input': (input_boxes.astype('float32'), lod),
            'ImInfo': im_info.astype('float32'),
        }
        self.outputs = {'Output': output_boxes}


if __name__ == '__main__':
    unittest.main()
