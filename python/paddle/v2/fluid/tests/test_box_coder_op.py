#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserve.
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
import sys
import math
from op_test import OpTest


def box_coder(target_box, prior_box, prior_box_var, output_box, code_type):
    prior_box_x = (
        (prior_box[:, 2] + prior_box[:, 0]) / 2).reshape(1, prior_box.shape[0])
    prior_box_y = (
        (prior_box[:, 3] + prior_box[:, 1]) / 2).reshape(1, prior_box.shape[0])
    prior_box_width = (
        (prior_box[:, 2] - prior_box[:, 0])).reshape(1, prior_box.shape[0])
    prior_box_height = (
        (prior_box[:, 3] - prior_box[:, 1])).reshape(1, prior_box.shape[0])
    prior_box_var = prior_box_var.reshape(1, prior_box_var.shape[0],
                                          prior_box_var.shape[1])

    if (code_type == "EncodeCenterSize"):
        target_box_x = ((target_box[:, 2] + target_box[:, 0]) / 2).reshape(
            target_box.shape[0], 1)
        target_box_y = ((target_box[:, 3] + target_box[:, 1]) / 2).reshape(
            target_box.shape[0], 1)
        target_box_width = ((target_box[:, 2] - target_box[:, 0])).reshape(
            target_box.shape[0], 1)
        target_box_height = ((target_box[:, 3] - target_box[:, 1])).reshape(
            target_box.shape[0], 1)

        output_box[:,:,0] = (target_box_x - prior_box_x) / prior_box_width / \
                prior_box_var[:,:,0]
        output_box[:,:,1] = (target_box_y - prior_box_y) / prior_box_height / \
                prior_box_var[:,:,1]
        output_box[:,:,2] = np.log(np.fabs(target_box_width / prior_box_width)) / \
                prior_box_var[:,:,2]
        output_box[:,:,3] = np.log(np.fabs(target_box_height / prior_box_height)) / \
                prior_box_var[:,:,3]

    elif (code_type == "DecodeCenterSize"):
        target_box = target_box.reshape(target_box.shape[0], 1,
                                        target_box.shape[1])
        target_box_x = prior_box_var[:,:,0] * target_box[:,:,0] * \
                       prior_box_width + prior_box_x
        target_box_y = prior_box_var[:,:,1] * target_box[:,:,1] * \
                       prior_box_height + prior_box_y
        target_box_width = np.exp(prior_box_var[:,:,2] * target_box[:,:,2]) * \
                           prior_box_width
        target_box_height = np.exp(prior_box_var[:,:,3] * target_box[:,:,3]) * \
                            prior_box_height
        output_box[:, :, 0] = target_box_x - target_box_width / 2
        output_box[:, :, 1] = target_box_y - target_box_height / 2
        output_box[:, :, 2] = target_box_x + target_box_width / 2
        output_box[:, :, 3] = target_box_y + target_box_height / 2


def batch_box_coder(prior_box, prior_box_var, target_box, lod, code_type):
    n = target_box.shape[0]
    m = prior_box.shape[0]
    output_box = np.zeros((n, m, 4), dtype=np.float32)
    for i in range(len(lod) - 1):
        box_coder(target_box[lod[i]:lod[i + 1], :], prior_box, prior_box_var,
                  output_box[lod[i]:lod[i + 1], :, :], code_type)
    return output_box


class TestBoxCoderOp(OpTest):
    def test_check_output(self):
        self.check_output()

    def setUp(self):
        self.op_type = "box_coder"
        lod = [[0, 20]]
        prior_box = np.random.random((10, 4)).astype('float32')
        prior_box_var = np.random.random((10, 4)).astype('float32')
        target_box = np.random.random((20, 4)).astype('float32')
        code_type = "DecodeCenterSize"
        output_box = batch_box_coder(prior_box, prior_box_var, target_box,
                                     lod[0], code_type)

        self.inputs = {
            'PriorBox': prior_box,
            'PriorBoxVar': prior_box_var,
            'TargetBox': target_box,
        }
        self.attrs = {'code_type': 'decode_center_size'}
        self.outputs = {'OutputBox': output_box}


class TestBoxCoderOpWithLoD(OpTest):
    def test_check_output(self):
        self.check_output()

    def setUp(self):
        self.op_type = "box_coder"
        lod = [[0, 4, 12, 20]]
        prior_box = np.random.random((10, 4)).astype('float32')
        prior_box_var = np.random.random((10, 4)).astype('float32')
        target_box = np.random.random((20, 4)).astype('float32')
        code_type = "EncodeCenterSize"
        output_box = batch_box_coder(prior_box, prior_box_var, target_box,
                                     lod[0], code_type)

        self.inputs = {
            'PriorBox': prior_box,
            'PriorBoxVar': prior_box_var,
            'TargetBox': (target_box, lod),
        }
        self.attrs = {'code_type': 'encode_center_size'}
        self.outputs = {'OutputBox': output_box}


if __name__ == '__main__':
    unittest.main()
