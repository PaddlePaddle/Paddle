#   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

import sys
import unittest

sys.path.append("../../mkldnn")
import numpy as np
from mkldnn_op_test import check_if_mkldnn_primitives_exist_in_bwd
from op_test import OpTest
from test_softmax_op import (
    TestSoftmaxOp,
    TestSoftmaxOp2,
    TestSoftmaxOp3,
    TestSoftmaxOp4,
    TestSoftmaxOp5,
    TestSoftmaxOp6,
    TestSoftmaxOp_ZeroDim1,
)
from utils import compare_legacy_with_pt

import paddle
from paddle.base import core

paddle.enable_static()


def stable_softmax(x):
    """Compute the softmax of vector x in a numerically stable way."""
    shiftx = x - np.max(x).clip(-64.0)
    exps = np.exp(shiftx)
    return exps / np.sum(exps)


class TestSoftmaxMKLDNNOp(TestSoftmaxOp):
    def get_x_shape(self):
        return [10, 10]

    def get_axis(self):
        return -1

    def setUp(self):
        self.op_type = "softmax"
        self.use_cudnn = False
        self.use_mkldnn = False
        self.dtype = np.float32
        self.init_kernel_type()
        self.shape = self.get_x_shape()
        self.axis = self.get_axis()

        x = np.random.uniform(0.1, 1, self.shape).astype(self.dtype)
        out = np.apply_along_axis(stable_softmax, self.axis, x)

        self.inputs = {'X': OpTest.np_dtype_to_base_dtype(x)}
        self.outputs = {'Out': out}
        self.attrs = {
            'axis': self.axis,
            'use_cudnn': self.use_cudnn,
            'use_mkldnn': self.use_mkldnn,
        }

    def test_check_output(self):
        # TODO(wangzhongpu): support onednn op in dygraph mode
        if self.use_cudnn:
            place = core.CUDAPlace(0)
            self.check_output_with_place(
                place, check_dygraph=False, check_pir_onednn=True
            )
        else:
            self.check_output(check_dygraph=False, check_pir_onednn=True)

    def test_check_grad(self):
        # TODO(wangzhongpu): support onednn op in dygraph mode
        if self.use_cudnn or self.dtype == np.float16:
            place = core.CUDAPlace(0)
            if core.is_float16_supported(place):
                self.check_grad_with_place(
                    place,
                    ["X"],
                    "Out",
                    max_relative_error=0.01,
                    check_dygraph=False,
                    check_pir_onednn=False,
                )
        else:
            self.check_grad(
                ["X"],
                "Out",
                max_relative_error=0.01,
                check_dygraph=False,
                check_pir_onednn=False,
            )

    def init_kernel_type(self):
        self.use_mkldnn = True


class TestSoftmaxMKLDNNOp2(TestSoftmaxOp2):
    def init_kernel_type(self):
        self.use_mkldnn = True
        # oneDNN doesn't support float64 dtype
        self.dtype = np.float32
        self.check_pir_onednn = False


class TestSoftmaxMKLDNNOp3(TestSoftmaxOp3):
    def init_kernel_type(self):
        self.use_mkldnn = True
        self.dtype = np.float32
        self.check_pir_onednn = False


class TestSoftmaxMKLDNNOp4(TestSoftmaxOp4):
    def init_kernel_type(self):
        self.use_mkldnn = True
        self.dtype = np.float32
        self.check_pir_onednn = False


class TestSoftmaxMKLDNNOp5(TestSoftmaxOp5):
    def init_kernel_type(self):
        self.use_mkldnn = True
        self.dtype = np.float32
        self.check_pir_onednn = False


class TestSoftmaxMKLDNNOp6(TestSoftmaxOp6):
    def init_kernel_type(self):
        self.use_mkldnn = True
        self.dtype = np.float32
        self.check_pir_onednn = False


class TestSoftmaxMKLDNNOp_ZeroDim(TestSoftmaxOp_ZeroDim1):
    def init_kernel_type(self):
        self.use_mkldnn = True
        self.dtype = np.float32
        self.check_pir_onednn = False


# Check if primitives already exist in backward
class TestSoftmaxMKLDNNPrimitivesAlreadyExist(unittest.TestCase):
    def setUp(self):
        super().setUp()

        np.random.seed(123)
        self.op_type = 'softmax'
        self.x = np.random.uniform(-1, 1, 2).astype(np.float32)
        self.out = stable_softmax(self.x)
        self.out_grad = np.random.random_sample(self.x.shape).astype(np.float32)
        self.x_grad = self.__softmax_bwd(self.out, self.out_grad)

    # Softmax grad calculation
    def __softmax_bwd(self, out, out_grad):
        return out * (out_grad - np.dot(out, out_grad))

    @compare_legacy_with_pt
    def test_check(self):
        check_if_mkldnn_primitives_exist_in_bwd(
            self, self.op_type, self.x, self.out, self.out_grad, self.x_grad
        )


if __name__ == '__main__':
    from paddle import enable_static

    enable_static()
    unittest.main()
