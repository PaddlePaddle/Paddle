#  Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

import sys
import unittest

import numpy as np
from op_test import skip_check_grad_ci

sys.path.append("../deprecated/legacy_test")
from test_elementwise_mul_op import ElementwiseMulOp

from paddle import enable_static


class TestOneDNNElementwiseMulOp(ElementwiseMulOp):
    def init_kernel_type(self):
        self.use_mkldnn = True
        self.check_pir_onednn = True

    def init_dtype(self):
        self.dtype = np.float32


class TestOneDNNElementwiseMulOp2(TestOneDNNElementwiseMulOp):
    def init_input_output(self):
        self.x = np.random.random((100,)).astype(self.dtype)
        self.y = np.random.random((100,)).astype(self.dtype)
        self.out = np.multiply(self.x, self.y)


class TestOneDNNElementwiseMulOp3(TestOneDNNElementwiseMulOp):
    def init_input_output(self):
        self.x = np.random.uniform(0.1, 1, [2, 3, 4, 5]).astype(self.dtype)
        self.y = np.random.uniform(0.1, 1, [2, 3, 4, 5]).astype(self.dtype)
        self.out = np.multiply(self.x, self.y)


class TestOneDNNElementwiseMulOp4(TestOneDNNElementwiseMulOp):
    def init_input_output(self):
        self.x = np.random.uniform(1, 2, [2, 3, 4, 32]).astype(self.dtype)
        self.y = np.random.uniform(1, 2, [4, 32]).astype(self.dtype)
        self.out = np.multiply(self.x, self.y)

    # TODO(jczaja): Enable when grad is ready
    def test_check_grad_normal(self):
        pass

    def test_check_grad_ignore_y(self):
        pass


class TestOneDNNElementwiseMulOp5(TestOneDNNElementwiseMulOp):
    def init_input_output(self):
        self.x = np.random.uniform(1, 2, [2, 3, 4, 100]).astype(self.dtype)
        self.y = np.random.uniform(1, 2, [100]).astype(self.dtype)
        self.out = np.multiply(self.x, self.y)

    # TODO(jczaja): Enable when grad is ready
    def test_check_grad_normal(self):
        pass

    def test_check_grad_ignore_y(self):
        pass

    def test_check_grad_ignore_x(self):
        pass


class TestOneDNNElementwiseMulOpZeroDim(TestOneDNNElementwiseMulOp):
    def init_input_output(self):
        self.x = np.random.random((100,)).astype(self.dtype)
        self.y = np.array(3.0).astype(self.dtype)
        self.out = np.multiply(self.x, self.y)

    def test_check_grad_normal(self):
        pass

    def test_check_grad_ignore_y(self):
        pass

    def test_check_grad_ignore_x(self):
        pass


class TestOneDNNElementwiseMulOpZeroDim2(TestOneDNNElementwiseMulOp):
    def init_input_output(self):
        self.x = np.array(3.0).astype(self.dtype)
        self.y = np.random.random((100,)).astype(self.dtype)
        self.out = np.multiply(self.x, self.y)

    def test_check_grad_normal(self):
        pass

    def test_check_grad_ignore_y(self):
        pass

    def test_check_grad_ignore_x(self):
        pass


class TestOneDNNElementwiseMulOpZeroDim3(TestOneDNNElementwiseMulOp):
    def init_input_output(self):
        self.x = np.array(3.0).astype(self.dtype)
        self.y = np.array(3.0).astype(self.dtype)
        self.out = np.multiply(self.x, self.y)

    def test_check_grad_normal(self):
        pass

    def test_check_grad_ignore_y(self):
        pass

    def test_check_grad_ignore_x(self):
        pass


''' INT8 Tests '''


@skip_check_grad_ci(
    reason="oneDNN's int8 elementwise_ops don't implement grad kernel."
)
class TestInt8(ElementwiseMulOp):
    def init_kernel_type(self):
        self.use_mkldnn = True
        self._cpu_only = True
        self.check_pir_onednn = True

    def init_dtype(self):
        self.dtype = np.int8

    def init_input_output(self):
        self.x = np.random.randint(0, 3, (12, 9)).astype("int8")
        self.y = np.random.randint(0, 3, (12, 9)).astype("int8")
        self.out = np.multiply(self.x, self.y)

    def init_scales(self):
        self.attrs['scale_x'] = 1.0
        self.attrs['scale_y'] = 1.0
        self.attrs['scale_out'] = 1.0

    def test_check_output(self):
        # TODO(wangzhongpu): support onednn op in dygraph mode
        self.init_scales()
        self.check_output(check_dygraph=(not self.use_mkldnn))

    def test_check_grad_normal(self):
        pass

    def test_check_grad_ignore_x(self):
        pass

    def test_check_grad_ignore_y(self):
        pass


if __name__ == '__main__':
    enable_static()
    unittest.main()
