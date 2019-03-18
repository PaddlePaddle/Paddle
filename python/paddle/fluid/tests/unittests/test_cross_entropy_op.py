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

from __future__ import print_function

import unittest
import numpy as np
import paddle.fluid.core as core
from op_test import OpTest, randomize_probability


class TestCrossEntropyOp(OpTest):
    """Test cross-entropy with discrete one-hot labels.
    """

    def setUp(self):
        self.op_type = "cross_entropy"
        self.soft_label = False
        self.ignore_index = -100
        self.dtype = np.float64
        self.batch_size = 30
        self.class_num = 10

        self.init_dtype_type()
        self.init_attr_type()
        self.init_bs_class_num()
        self.init_x()
        self.init_label()
        self.get_cross_entropy()

        self.inputs = {"X": self.x, "Label": self.label}
        self.outputs = {"Y": self.cross_entropy}
        self.attrs = {
            "soft_label": self.soft_label,
            "ignore_index": self.ignore_index
        }

    def init_x(self):
        self.x = randomize_probability(
            self.batch_size, self.class_num, dtype=self.dtype)

    def init_label(self):
        self.label = np.random.randint(
            0, self.class_num, (self.batch_size, 1), dtype="int64")

    def get_cross_entropy(self):
        self.cross_entropy = np.asmatrix(
            [[-np.log(self.x[i][self.label[i][0]])]
             for i in range(self.x.shape[0])],
            dtype="float64")

    def init_attr_type(self):
        pass

    def init_dtype_type(self):
        pass

    def init_bs_class_num(self):
        pass

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(["X"], "Y", numeric_grad_delta=0.001)


class TestCrossEntropyOp2(TestCrossEntropyOp):
    """Test cross-entropy with vectorized soft labels.
    """

    def init_label(self):
        self.label = np.random.uniform(
            0.1, 1.0, [self.batch_size, self.class_num]).astype(self.dtype)
        self.label /= self.label.sum(axis=1, keepdims=True)

    def get_cross_entropy(self):
        self.cross_entropy = (-self.label * np.log(self.x)).sum(
            axis=1, keepdims=True).astype(self.dtype)

    def init_attr_type(self):
        self.soft_label = True

    def init_dtype_type(self):
        self.dtype = np.float32

    def init_bs_class_num(self):
        self.batch_size = 5
        self.class_num = 37

    def test_check_grad(self):
        self.check_grad(
            ["X"], "Y", max_relative_error=0.05, numeric_grad_delta=0.001)


class TestCrossEntropyOp3(TestCrossEntropyOp):
    """Test cross-entropy with vectorized one-hot representation of labels.
    """

    def init_label(self):
        self.label_index = np.random.randint(0, self.class_num,
                                             (self.batch_size))
        self.label = np.zeros(self.x.shape).astype(self.dtype)
        self.label[np.arange(self.batch_size), self.label_index] = 1

    def get_cross_entropy(self):
        self.cross_entropy = np.asmatrix(
            [[-np.log(self.x[i][self.label_index[i]])]
             for i in range(self.x.shape[0])]).astype(self.dtype)

    def init_attr_type(self):
        self.soft_label = True

    def init_dtype_type(self):
        self.dtype = np.float32

    def init_bs_class_num(self):
        self.batch_size = 5
        self.class_num = 17

    def test_check_grad(self):
        self.check_grad(
            ["X"], "Y", max_relative_error=0.05, numeric_grad_delta=0.001)


class TestCrossEntropyOp4(TestCrossEntropyOp):
    """Test high rank tensor cross-entropy with discrete one-hot labels.
    """

    def init_x(self):
        self.shape = [10, 2, 4]
        self.ins_num = np.prod(np.array(self.shape))
        self.X_2d = randomize_probability(self.ins_num,
                                          self.class_num).astype(self.dtype)
        self.x = self.X_2d.reshape(self.shape + [self.class_num])

    def init_label(self):
        self.label_2d = np.random.randint(
            0, self.class_num, (self.ins_num, 1), dtype="int64")
        self.label = self.label_2d.reshape(self.shape + [1])

    def get_cross_entropy(self):
        cross_entropy_2d = np.asmatrix(
            [[-np.log(self.X_2d[i][self.label_2d[i][0]])]
             for i in range(self.X_2d.shape[0])]).astype(self.dtype)
        self.cross_entropy = np.array(cross_entropy_2d).reshape(self.shape +
                                                                [1])

    def init_attr_type(self):
        self.soft_label = False

    def init_dtype_type(self):
        self.dtype = np.float64

    def init_bs_class_num(self):
        self.class_num = 10


class TestCrossEntropyOp5(TestCrossEntropyOp):
    """Test high rank tensor cross-entropy with vectorized soft labels.
    """

    def init_x(self):
        self.shape = [4, 3]
        self.ins_num = np.prod(np.array(self.shape))
        self.X_2d = randomize_probability(self.ins_num,
                                          self.class_num).astype(self.dtype)
        self.x = self.X_2d.reshape(self.shape + [self.class_num])

    def init_label(self):
        self.label_2d = np.random.uniform(
            0.1, 1.0, [self.ins_num, self.class_num]).astype(self.dtype)
        self.label_2d /= self.label_2d.sum(axis=1, keepdims=True)
        self.label = self.label_2d.reshape(self.shape + [self.class_num])

    def get_cross_entropy(self):
        cross_entropy_2d = (-self.label_2d * np.log(self.X_2d)).sum(
            axis=1, keepdims=True).astype(self.dtype)
        self.cross_entropy = np.array(cross_entropy_2d).reshape(self.shape +
                                                                [1])

    def init_attr_type(self):
        self.soft_label = True

    def init_dtype_type(self):
        self.dtype = np.float32

    def init_bs_class_num(self):
        self.class_num = 37

    def test_check_grad(self):
        self.check_grad(
            ["X"], "Y", max_relative_error=0.05, numeric_grad_delta=0.001)


class TestCrossEntropyOp6(TestCrossEntropyOp):
    """Test high rank tensor cross-entropy with vectorized one-hot representation of labels.
    """

    def init_x(self):
        self.shape = [4, 3, 2]
        self.ins_num = np.prod(np.array(self.shape))
        self.X_2d = randomize_probability(self.ins_num,
                                          self.class_num).astype(self.dtype)
        self.x = self.X_2d.reshape(self.shape + [self.class_num])

    def init_label(self):
        self.label_index_2d = np.random.randint(
            0, self.class_num, (self.ins_num), dtype="int64")
        label_2d = np.zeros(self.X_2d.shape)
        label_2d[np.arange(self.ins_num), self.label_index_2d] = 1
        self.label = label_2d.reshape(self.shape + [self.class_num]).astype(
            self.dtype)

    def get_cross_entropy(self):
        cross_entropy_2d = np.asmatrix(
            [[-np.log(self.X_2d[i][self.label_index_2d[i]])]
             for i in range(self.X_2d.shape[0])])
        self.cross_entropy = np.array(cross_entropy_2d).reshape(
            self.shape + [1]).astype(self.dtype)

    def init_attr_type(self):
        self.soft_label = True

    def init_dtype_type(self):
        self.dtype = np.float32

    def init_bs_class_num(self):
        self.class_num = 17

    def test_check_grad(self):
        self.check_grad(
            ["X"], "Y", max_relative_error=0.05, numeric_grad_delta=0.001)


class TestCrossEntropyOp7(TestCrossEntropyOp):
    """Test cross-entropy with ignore index.
    """

    def init_label(self):
        self.label = np.random.randint(
            0, self.class_num, (self.batch_size, 1), dtype="int64")

    def get_cross_entropy(self):
        self.cross_entropy = np.asmatrix(
            [[-np.log(self.x[i][self.label[i][0]])]
             if self.label[i][0] != self.ignore_index else [0]
             for i in range(self.x.shape[0])]).astype(self.dtype)

    def init_attr_type(self):
        self.soft_label = False
        self.ignore_index = 3

    def init_dtype_type(self):
        self.dtype = np.float64

    def init_bs_class_num(self):
        self.batch_size = 30
        self.class_num = 10


# Add Fp16 test
def create_test_class(parent, cls_name):
    @unittest.skipIf(not core.is_compiled_with_cuda(),
                     "core is not compiled with CUDA")
    class TestCrossEntropyFP16Op(parent):
        def init_dtype_type(self):
            return np.float16

        def test_check_output(self):
            place = core.CUDAPlace(0)
            if core.is_float16_supported(place):
                self.check_output_with_place(place, atol=2e-1)

        def test_check_grad(self):
            place = core.CUDAPlace(0)
            if core.is_float16_supported(place):
                self.check_grad_with_place(
                    place, ['X'], 'Y', max_relative_error=0.9)

    cls_name = "{0}".format(cls_name)
    TestCrossEntropyFP16Op.__name__ = cls_name
    globals()[cls_name] = TestCrossEntropyFP16Op


create_test_class(TestCrossEntropyOp, "TestCrossEntropyF16Op")
#create_test_class(TestCrossEntropyOp2, "TestCrossEntropyF16Op2")
create_test_class(TestCrossEntropyOp3, "TestCrossEntropyF16Op3")
create_test_class(TestCrossEntropyOp4, "TestCrossEntropyF16Op4")
#create_test_class(TestCrossEntropyOp5, "TestCrossEntropyF16Op5")
create_test_class(TestCrossEntropyOp6, "TestCrossEntropyF16Op6")
create_test_class(TestCrossEntropyOp7, "TestCrossEntropyF16Op7")

if __name__ == "__main__":
    unittest.main()
