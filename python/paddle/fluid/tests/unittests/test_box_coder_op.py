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
import sys
import math
from op_test import OpTest
import paddle
import paddle.fluid.core as core


def box_decoder(t_box, p_box, pb_v, output_box, norm, axis=0):
    pb_w = p_box[:, 2] - p_box[:, 0] + (norm == False)
    pb_h = p_box[:, 3] - p_box[:, 1] + (norm == False)
    pb_x = pb_w * 0.5 + p_box[:, 0]
    pb_y = pb_h * 0.5 + p_box[:, 1]
    shape = (1, p_box.shape[0]) if axis == 0 else (p_box.shape[0], 1)

    pb_w = pb_w.reshape(shape)
    pb_h = pb_h.reshape(shape)
    pb_x = pb_x.reshape(shape)
    pb_y = pb_y.reshape(shape)

    if pb_v.ndim == 2:
        var_shape = (1, pb_v.shape[0],
                     pb_v.shape[1]) if axis == 0 else (pb_v.shape[0], 1,
                                                       pb_v.shape[1])
        pb_v = pb_v.reshape(var_shape)
    if pb_v.ndim == 1:
        tb_x = pb_v[0] * t_box[:, :, 0] * pb_w + pb_x
        tb_y = pb_v[1] * t_box[:, :, 1] * pb_h + pb_y
        tb_w = np.exp(pb_v[2] * t_box[:, :, 2]) * pb_w
        tb_h = np.exp(pb_v[3] * t_box[:, :, 3]) * pb_h
    else:
        tb_x = pb_v[:, :, 0] * t_box[:, :, 0] * pb_w + pb_x
        tb_y = pb_v[:, :, 1] * t_box[:, :, 1] * pb_h + pb_y
        tb_w = np.exp(pb_v[:, :, 2] * t_box[:, :, 2]) * pb_w
        tb_h = np.exp(pb_v[:, :, 3] * t_box[:, :, 3]) * pb_h
    output_box[:, :, 0] = tb_x - tb_w / 2
    output_box[:, :, 1] = tb_y - tb_h / 2
    output_box[:, :, 2] = tb_x + tb_w / 2 - (not norm)
    output_box[:, :, 3] = tb_y + tb_h / 2 - (not norm)


def box_encoder(t_box, p_box, pb_v, output_box, norm):
    pb_w = p_box[:, 2] - p_box[:, 0] + (norm == False)
    pb_h = p_box[:, 3] - p_box[:, 1] + (norm == False)
    pb_x = pb_w * 0.5 + p_box[:, 0]
    pb_y = pb_h * 0.5 + p_box[:, 1]
    shape = (1, p_box.shape[0])

    pb_w = pb_w.reshape(shape)
    pb_h = pb_h.reshape(shape)
    pb_x = pb_x.reshape(shape)
    pb_y = pb_y.reshape(shape)

    if pb_v.ndim == 2:
        pb_v = pb_v.reshape(1, pb_v.shape[0], pb_v.shape[1])
    tb_x = ((t_box[:, 2] + t_box[:, 0]) / 2).reshape(t_box.shape[0], 1)
    tb_y = ((t_box[:, 3] + t_box[:, 1]) / 2).reshape(t_box.shape[0], 1)
    tb_w = (t_box[:, 2] - t_box[:, 0]).reshape(t_box.shape[0], 1) + (not norm)
    tb_h = (t_box[:, 3] - t_box[:, 1]).reshape(t_box.shape[0], 1) + (not norm)
    if pb_v.ndim == 1:
        output_box[:, :, 0] = (tb_x - pb_x) / pb_w / pb_v[0]
        output_box[:, :, 1] = (tb_y - pb_y) / pb_h / pb_v[1]
        output_box[:, :, 2] = np.log(np.fabs(tb_w / pb_w)) / pb_v[2]
        output_box[:, :, 3] = np.log(np.fabs(tb_h / pb_h)) / pb_v[3]
    else:
        output_box[:, :, 0] = (tb_x - pb_x) / pb_w / pb_v[:, :, 0]
        output_box[:, :, 1] = (tb_y - pb_y) / pb_h / pb_v[:, :, 1]
        output_box[:, :, 2] = np.log(np.fabs(tb_w / pb_w)) / pb_v[:, :, 2]
        output_box[:, :, 3] = np.log(np.fabs(tb_h / pb_h)) / pb_v[:, :, 3]


def batch_box_coder(p_box, pb_v, t_box, lod, code_type, norm, axis=0):
    n = t_box.shape[0]
    m = p_box.shape[0]
    if code_type == "DecodeCenterSize":
        m = t_box.shape[1]
    output_box = np.zeros((n, m, 4), dtype=np.float32)
    cur_offset = 0
    for i in range(len(lod)):
        if (code_type == "EncodeCenterSize"):
            box_encoder(t_box[cur_offset:(cur_offset + lod[i]), :], p_box, pb_v,
                        output_box[cur_offset:(cur_offset + lod[i]), :, :],
                        norm)
        elif (code_type == "DecodeCenterSize"):
            box_decoder(t_box, p_box, pb_v, output_box, norm, axis)
        cur_offset += lod[i]
    return output_box


class TestBoxCoderOp(OpTest):

    def test_check_output(self):
        self.check_output(check_eager=True)

    def setUp(self):
        self.op_type = "box_coder"
        self.python_api = paddle.fluid.layers.box_coder
        lod = [[1, 1, 1, 1, 1]]
        prior_box = np.random.random((81, 4)).astype('float32')
        prior_box_var = np.random.random((81, 4)).astype('float32')
        target_box = np.random.random((20, 81, 4)).astype('float32')
        code_type = "DecodeCenterSize"
        box_normalized = False
        output_box = batch_box_coder(prior_box, prior_box_var, target_box,
                                     lod[0], code_type, box_normalized)
        self.inputs = {
            'PriorBox': prior_box,
            'PriorBoxVar': prior_box_var,
            'TargetBox': target_box,
        }
        self.attrs = {
            'code_type': 'decode_center_size',
            'box_normalized': False
        }
        self.outputs = {'OutputBox': output_box}


class TestBoxCoderOpWithoutBoxVar(OpTest):

    def test_check_output(self):
        self.check_output(check_eager=True)

    def setUp(self):
        self.python_api = paddle.fluid.layers.box_coder
        self.op_type = "box_coder"
        lod = [[0, 1, 2, 3, 4, 5]]
        prior_box = np.random.random((81, 4)).astype('float32')
        prior_box_var = np.ones((81, 4)).astype('float32')
        target_box = np.random.random((20, 81, 4)).astype('float32')
        code_type = "DecodeCenterSize"
        box_normalized = False
        output_box = batch_box_coder(prior_box, prior_box_var, target_box,
                                     lod[0], code_type, box_normalized)

        self.inputs = {
            'PriorBox': prior_box,
            'PriorBoxVar': prior_box_var,
            'TargetBox': target_box,
        }
        self.attrs = {
            'code_type': 'decode_center_size',
            'box_normalized': False
        }
        self.outputs = {'OutputBox': output_box}


class TestBoxCoderOpWithLoD(OpTest):

    def test_check_output(self):
        self.check_output(check_eager=True)

    def setUp(self):
        self.python_api = paddle.fluid.layers.box_coder
        self.op_type = "box_coder"
        lod = [[10, 20, 20]]
        prior_box = np.random.random((20, 4)).astype('float32')
        prior_box_var = np.random.random((20, 4)).astype('float32')
        target_box = np.random.random((50, 4)).astype('float32')
        code_type = "EncodeCenterSize"
        box_normalized = True
        output_box = batch_box_coder(prior_box, prior_box_var, target_box,
                                     lod[0], code_type, box_normalized)

        self.inputs = {
            'PriorBox': prior_box,
            'PriorBoxVar': prior_box_var,
            'TargetBox': (target_box, lod),
        }
        self.attrs = {'code_type': 'encode_center_size', 'box_normalized': True}
        self.outputs = {'OutputBox': output_box}


class TestBoxCoderOpWithAxis(OpTest):

    def test_check_output(self):
        self.check_output(check_eager=True)

    def setUp(self):
        self.python_api = paddle.fluid.layers.box_coder
        self.op_type = "box_coder"
        lod = [[1, 1, 1, 1, 1]]
        prior_box = np.random.random((30, 4)).astype('float32')
        prior_box_var = np.random.random((30, 4)).astype('float32')
        target_box = np.random.random((30, 81, 4)).astype('float32')
        code_type = "DecodeCenterSize"
        box_normalized = False
        axis = 1
        output_box = batch_box_coder(prior_box, prior_box_var, target_box,
                                     lod[0], code_type, box_normalized, axis)

        self.inputs = {
            'PriorBox': prior_box,
            'PriorBoxVar': prior_box_var,
            'TargetBox': target_box,
        }
        self.attrs = {
            'code_type': 'decode_center_size',
            'box_normalized': False,
            'axis': axis
        }
        self.outputs = {'OutputBox': output_box}


class TestBoxCoderOpWithVariance(OpTest):

    def test_check_output(self):
        self.check_output()

    def setUp(self):
        self.op_type = "box_coder"
        lod = [[1, 1, 1, 1, 1]]
        prior_box = np.random.random((30, 4)).astype('float32')
        prior_box_var = np.random.random((4)).astype('float32')
        target_box = np.random.random((30, 81, 4)).astype('float32')
        code_type = "DecodeCenterSize"
        box_normalized = False
        axis = 1
        output_box = batch_box_coder(prior_box, prior_box_var, target_box,
                                     lod[0], code_type, box_normalized, axis)

        self.inputs = {
            'PriorBox': prior_box,
            'TargetBox': target_box,
        }
        self.attrs = {
            'code_type': 'decode_center_size',
            'box_normalized': False,
            'variance': prior_box_var.astype(np.float64).flatten(),
            'axis': axis
        }
        self.outputs = {'OutputBox': output_box}


class TestBoxCoderOpWithVarianceDygraphAPI(unittest.TestCase):

    def setUp(self):
        self.lod = [[1, 1, 1, 1, 1]]
        self.prior_box = np.random.random((30, 4)).astype('float32')
        self.prior_box_var = np.random.random((4)).astype('float32')
        self.target_box = np.random.random((30, 81, 4)).astype('float32')
        self.code_type = "DecodeCenterSize"
        self.box_normalized = False
        self.axis = 1
        self.output_ref = batch_box_coder(self.prior_box, self.prior_box_var,
                                          self.target_box, self.lod[0],
                                          self.code_type, self.box_normalized,
                                          self.axis)
        self.place = [paddle.CPUPlace()]
        if core.is_compiled_with_cuda():
            self.place.append(paddle.CUDAPlace(0))

    def test_dygraph_api(self):

        def run(place):
            paddle.disable_static(place)
            output_box = paddle.fluid.layers.box_coder(
                paddle.to_tensor(self.prior_box),
                self.prior_box_var.tolist(),
                paddle.to_tensor(self.target_box),
                "decode_center_size",
                self.box_normalized,
                axis=self.axis)
            np.testing.assert_allclose(np.sum(self.output_ref),
                                       np.sum(output_box.numpy()),
                                       rtol=1e-05)
            paddle.enable_static()

        for place in self.place:
            run(place)


if __name__ == '__main__':
    unittest.main()
