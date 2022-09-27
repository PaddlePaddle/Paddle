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

sys.path.append("..")
import math
import paddle
from op_test import OpTest

paddle.enable_static()

np.random.seed(2021)


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
    if code_type == "decode_center_size":
        m = t_box.shape[1]
    output_box = np.zeros((n, m, 4), dtype=np.float32)
    cur_offset = 0

    for i in range(len(lod)):
        if (code_type == "encode_center_size"):
            box_encoder(t_box[cur_offset:(cur_offset + lod[i]), :], p_box, pb_v,
                        output_box[cur_offset:(cur_offset + lod[i]), :, :],
                        norm)
        elif (code_type == "decode_center_size"):
            box_decoder(t_box, p_box, pb_v, output_box, norm, axis)
        cur_offset += lod[i]
    return output_box


@unittest.skipIf(not paddle.is_compiled_with_npu(),
                 "core is not compiled with NPU")
class TestBoxCoderOp(OpTest):

    def setUp(self):
        self.op_type = "box_coder"
        self.set_npu()
        self.init_dtype()

        self.set_init_config()
        self.set_inputs()
        self.set_attrs()
        self.set_outputs()

    def set_npu(self):
        self.__class__.use_npu = True
        self.place = paddle.NPUPlace(0)

    def init_dtype(self):
        self.dtype = np.float32

    def set_init_config(self):
        self.M = 81
        self.N = 20
        self.code_type = 'decode_center_size'
        self.box_normalized = False
        self.lod = [[1, 1, 1, 1, 1]]
        self.axis = 0
        self.use_variance = False
        self.without_prior_box_var = False
        self.atol = 1e-5

    def set_inputs(self):
        self.inputs = {}
        assert (self.code_type in ['decode_center_size', 'encode_center_size'])
        assert (self.axis in [0, 1])
        if self.code_type == 'decode_center_size':
            assert (not self.use_variance or not self.without_prior_box_var)

            self.prior_box = np.random.random((self.M, 4)).astype(self.dtype)

            if self.use_variance:
                self.prior_box_var = np.random.random(4).astype(self.dtype)
            else:
                if self.without_prior_box_var:
                    self.prior_box_var = np.ones((self.M, 4)).astype(self.dtype)
                else:
                    self.prior_box_var = np.random.random(
                        (self.M, 4)).astype(self.dtype)

            if self.axis == 0:
                self.target_box = np.random.random(
                    (self.N, self.M, 4)).astype(self.dtype)
            else:
                self.target_box = np.random.random(
                    (self.M, self.N, 4)).astype(self.dtype)
            self.inputs['PriorBox'] = self.prior_box
            self.inputs['TargetBox'] = self.target_box
            if (not self.use_variance and not self.without_prior_box_var):
                self.inputs['PriorBoxVar'] = self.prior_box_var
        else:
            #encode_center_size
            self.prior_box = np.random.random((self.M, 4)).astype(self.dtype)
            if self.use_variance:
                self.prior_box_var = np.random.random(4).astype(self.dtype)
            else:
                self.prior_box_var = np.random.random(
                    (self.M, 4)).astype(self.dtype)
            self.target_box = np.random.random((self.N, 4)).astype(self.dtype)
            self.inputs['PriorBox'] = self.prior_box
            #self.inputs['PriorBoxVar'] = self.prior_box_var
            self.inputs['TargetBox'] = (self.target_box, self.lod)
            if (not self.use_variance):
                self.inputs['PriorBoxVar'] = self.prior_box_var

    def set_attrs(self):
        self.attrs = {
            'code_type': self.code_type,
            'box_normalized': self.box_normalized
        }
        if self.use_variance:
            self.attrs['variance'] = self.prior_box_var.astype(
                np.float64).flatten()
        if self.axis != 0:
            self.attrs['axis'] = self.axis

    def set_outputs(self):
        output_box = batch_box_coder(self.prior_box, self.prior_box_var,
                                     self.target_box, self.lod[0],
                                     self.code_type, self.box_normalized,
                                     self.axis)
        self.outputs = {'OutputBox': output_box.astype(self.dtype)}

    def test_check_output(self):
        self.check_output_with_place(self.place, atol=self.atol)


class TestBoxCoderOpWithoutBoxVar(TestBoxCoderOp):

    def set_init_config(self):
        super(TestBoxCoderOpWithoutBoxVar, self).set_init_config()
        self.without_prior_box_var = True
        self.lod = [[0, 1, 2, 3, 4, 5]]


class TestBoxCoderOpWithLoD(TestBoxCoderOp):

    def set_init_config(self):
        super(TestBoxCoderOpWithLoD, self).set_init_config()
        self.M = 20
        self.N = 50
        self.lod = [[10, 20, 20]]
        self.code_type = 'encode_center_size'
        self.box_normalized = True


class TestBoxCoderOpWithLoDWithVariance(TestBoxCoderOpWithLoD):

    def set_init_config(self):
        super(TestBoxCoderOpWithLoDWithVariance, self).set_init_config()
        self.use_variance = True


class TestBoxCoderOpWithAxis(TestBoxCoderOp):

    def set_init_config(self):
        super(TestBoxCoderOpWithAxis, self).set_init_config()
        self.axis = 1


class TestBoxCoderOpWithVariance(TestBoxCoderOp):

    def set_init_config(self):
        super(TestBoxCoderOpWithVariance, self).set_init_config()
        self.use_variance = True


class TestBoxCoderOpFP16(TestBoxCoderOp):

    def init_dtype(self):
        self.dtype = np.float16

    def set_init_config(self):
        super(TestBoxCoderOpFP16, self).set_init_config()
        self.atol = 1e-2


if __name__ == '__main__':
    unittest.main()
