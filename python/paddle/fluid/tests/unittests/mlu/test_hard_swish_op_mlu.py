#  Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import paddle.nn.functional as F
import paddle.fluid as fluid
import paddle
import sys

sys.path.append("..")
from op_test import OpTest

import numpy as np
import unittest

paddle.enable_static()
SEED = 2021
np.random.seed(SEED)


def scalarToType(val, data_type):
    converted_val = np.array([val]).astype(data_type)[0]
    print("converted_val type: ", type(converted_val))
    return converted_val


def ref_hard_swish_grad(x, threshold, scale, offset, data_type):
    threshold = scalarToType(threshold, data_type)
    scale = scalarToType(scale, data_type)
    offset = scalarToType(offset, data_type)
    dout = np.full_like(x, fill_value=1. / x.size)
    tmp = ((x + offset) < threshold).astype(x.dtype)
    dx = dout * (((x + offset) > 0).astype(x.dtype) *
                 (2 * x + offset) * tmp / scale + 1.0 - tmp)
    return dx


class TestHardSwishMLU(OpTest):

    def setUp(self):
        paddle.enable_static()

        self.op_type = "hard_swish"
        self.place = paddle.MLUPlace(0)
        self.init_dtype()

        x = np.random.uniform(-2, 2, [10, 12]).astype(self.dtype)
        threshold = 6.0
        scale = 6.0
        offset = 3.0

        x[np.abs(x + offset) < 0.005] = 0.02
        x[np.abs(x - threshold + offset) < 0.005] = threshold - offset + 0.02

        out = (
            x *
            (np.minimum(np.maximum(x + offset, 0.), threshold) / scale)).astype(
                self.dtype)
        self.x_grad = ref_hard_swish_grad(x, threshold, scale, offset,
                                          self.dtype)
        self.set_mlu()
        self.inputs = {'X': x}
        self.attrs = {'threshold': threshold, 'scale': scale, 'offset': offset}
        self.outputs = {'Out': out}

    def set_mlu(self):
        self.__class__.use_mlu = True

    def init_dtype(self):
        self.dtype = np.float32

    def test_check_output(self):
        self.check_output_with_place(self.place)

    def test_check_grad(self):
        self.check_grad_with_place(self.place, ['X'], 'Out')


class TestHardSwishMLUWithCPUFloat16(unittest.TestCase):

    def setUp(self):
        paddle.disable_static()

        self.place = paddle.MLUPlace(0)
        self.dtype = np.float32

        # float32
        self.float32_x = np.random.uniform(-6, 10, [8, 15]).astype(np.float32)
        paddle.set_device('cpu')
        data = paddle.to_tensor(self.float32_x, stop_gradient=False)
        self.float32_y = F.hardswish(data)
        self.float32_y.sum().backward()
        self.float32_grad = data.grad

        # float16
        self.float16_x = self.float32_x.astype(np.float16)
        threshold = 6.0
        scale = 6.0
        offset = 3.0

        threshold = scalarToType(threshold, np.float16)
        scale = scalarToType(scale, np.float16)
        offset = scalarToType(offset, np.float16)
        self.float16_y = (self.float16_x * (np.minimum(
            np.maximum(self.float16_x + offset, scalarToType(0., np.float16)),
            threshold) / scale)).astype(np.float16)
        self.float16_grad = ref_hard_swish_grad(self.float16_x, threshold,
                                                scale, offset, np.float16)

    def test_check_output_and_grad_mlu(self):
        # mlu float16
        paddle.set_device('mlu')
        data = paddle.to_tensor(self.float16_x, stop_gradient=False)
        mlu_float16_y = F.hardswish(data)
        mlu_float16_y.sum().backward()
        mlu_float16_grad = data.grad

        cpu_diff_1 = np.divide(
            np.sum(np.abs(self.float32_y.numpy() - self.float16_y)),
            np.sum(np.abs(self.float32_y.numpy())))
        mlu_diff_1 = np.divide(
            np.sum(np.abs(self.float32_y.numpy() - mlu_float16_y.numpy())),
            np.sum(np.abs(self.float32_y.numpy())))

        cpu_diff_2 = np.divide(
            np.sum(np.square(self.float32_y.numpy() - self.float16_y)),
            np.sum(np.square(self.float32_y.numpy())))
        mlu_diff_2 = np.divide(
            np.sum(np.square(self.float32_y.numpy() - mlu_float16_y.numpy())),
            np.sum(np.square(self.float32_y.numpy())))
        assert mlu_diff_1 <= cpu_diff_1
        assert mlu_diff_2 <= cpu_diff_2

        cpu_diff_1 = np.divide(
            np.sum(np.abs(self.float32_grad.numpy() - self.float16_grad)),
            np.sum(np.abs(self.float32_grad.numpy())))
        mlu_diff_1 = np.divide(
            np.sum(np.abs(self.float32_grad.numpy() -
                          mlu_float16_grad.numpy())),
            np.sum(np.abs(self.float32_grad.numpy())))

        cpu_diff_2 = np.divide(
            np.sum(np.square(self.float32_grad.numpy() - self.float16_grad)),
            np.sum(np.square(self.float32_grad.numpy())))
        mlu_diff_2 = np.divide(
            np.sum(
                np.square(self.float32_grad.numpy() -
                          mlu_float16_grad.numpy())),
            np.sum(np.square(self.float32_grad.numpy())))
        assert mlu_diff_1 <= cpu_diff_1
        assert mlu_diff_2 <= cpu_diff_2


if __name__ == '__main__':
    unittest.main()
