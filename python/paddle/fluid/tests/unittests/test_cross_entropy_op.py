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
from op_test import OpTest, randomize_probability

import paddle
import paddle.fluid as fluid
import paddle.fluid.core as core
=======
from __future__ import print_function

import unittest
import numpy as np
import paddle.fluid.core as core
from op_test import OpTest, randomize_probability
import paddle.fluid as fluid
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
from paddle.fluid import Program, program_guard


class TestCrossEntropyOp(OpTest):
<<<<<<< HEAD
    """Test cross-entropy with discrete one-hot labels."""
=======
    """Test cross-entropy with discrete one-hot labels.
    """
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

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
<<<<<<< HEAD
            "ignore_index": self.ignore_index,
        }

    def init_x(self):
        self.x = randomize_probability(
            self.batch_size, self.class_num, dtype=self.dtype
        )

    def init_label(self):
        self.label = np.random.randint(
            0, self.class_num, (self.batch_size, 1), dtype="int64"
        )

    def get_cross_entropy(self):
        self.cross_entropy = np.asmatrix(
            [
                [-np.log(self.x[i][self.label[i][0]])]
                for i in range(self.x.shape[0])
            ],
            dtype="float64",
        )
=======
            "ignore_index": self.ignore_index
        }

    def init_x(self):
        self.x = randomize_probability(self.batch_size,
                                       self.class_num,
                                       dtype=self.dtype)

    def init_label(self):
        self.label = np.random.randint(0,
                                       self.class_num, (self.batch_size, 1),
                                       dtype="int64")

    def get_cross_entropy(self):
        self.cross_entropy = np.asmatrix(
            [[-np.log(self.x[i][self.label[i][0]])]
             for i in range(self.x.shape[0])],
            dtype="float64")
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

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


class TestCrossEntropyOpRemoveLastDim(TestCrossEntropyOp):
<<<<<<< HEAD
    """Test cross-entropy with discrete one-hot labels with shape [batch_size]"""

    def init_label(self):
        self.label = np.random.randint(
            0, self.class_num, (self.batch_size), dtype="int64"
        )
=======
    """Test cross-entropy with discrete one-hot labels with shape [batch_size]
    """

    def init_label(self):
        self.label = np.random.randint(0,
                                       self.class_num, (self.batch_size),
                                       dtype="int64")
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

    def get_cross_entropy(self):
        self.cross_entropy = np.asmatrix(
            [-np.log(self.x[i][self.label[i]]) for i in range(self.x.shape[0])],
<<<<<<< HEAD
            dtype="float64",
        )


class TestCrossEntropyOp2(TestCrossEntropyOp):
    """Test cross-entropy with vectorized soft labels."""

    def init_label(self):
        self.label = np.random.uniform(
            0.1, 1.0, [self.batch_size, self.class_num]
        ).astype(self.dtype)
        self.label /= self.label.sum(axis=1, keepdims=True)

    def get_cross_entropy(self):
        self.cross_entropy = (
            (-self.label * np.log(self.x))
            .sum(axis=1, keepdims=True)
            .astype(self.dtype)
        )
=======
            dtype="float64")


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
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

    def init_attr_type(self):
        self.soft_label = True

    def init_dtype_type(self):
        self.dtype = np.float64

    def init_bs_class_num(self):
        self.batch_size = 5
        self.class_num = 37

    def test_check_grad(self):
<<<<<<< HEAD
        self.check_grad(
            ["X"], "Y", max_relative_error=0.05, numeric_grad_delta=0.001
        )


class TestCrossEntropyOp3(TestCrossEntropyOp):
    """Test cross-entropy with vectorized one-hot representation of labels."""

    def init_label(self):
        self.label_index = np.random.randint(
            0, self.class_num, (self.batch_size)
        )
=======
        self.check_grad(["X"],
                        "Y",
                        max_relative_error=0.05,
                        numeric_grad_delta=0.001)


class TestCrossEntropyOp3(TestCrossEntropyOp):
    """Test cross-entropy with vectorized one-hot representation of labels.
    """

    def init_label(self):
        self.label_index = np.random.randint(0, self.class_num,
                                             (self.batch_size))
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        self.label = np.zeros(self.x.shape).astype(self.dtype)
        self.label[np.arange(self.batch_size), self.label_index] = 1

    def get_cross_entropy(self):
        self.cross_entropy = np.asmatrix(
<<<<<<< HEAD
            [
                [-np.log(self.x[i][self.label_index[i]])]
                for i in range(self.x.shape[0])
            ]
        ).astype(self.dtype)
=======
            [[-np.log(self.x[i][self.label_index[i]])]
             for i in range(self.x.shape[0])]).astype(self.dtype)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

    def init_attr_type(self):
        self.soft_label = True

    def init_dtype_type(self):
        self.dtype = np.float64

    def init_bs_class_num(self):
        self.batch_size = 5
        self.class_num = 27

    def test_check_grad(self):
<<<<<<< HEAD
        self.check_grad(
            ["X"], "Y", max_relative_error=0.05, numeric_grad_delta=0.001
        )


class TestCrossEntropyOp4(TestCrossEntropyOp):
    """Test high rank tensor cross-entropy with discrete one-hot labels."""
=======
        self.check_grad(["X"],
                        "Y",
                        max_relative_error=0.05,
                        numeric_grad_delta=0.001)


class TestCrossEntropyOp4(TestCrossEntropyOp):
    """Test high rank tensor cross-entropy with discrete one-hot labels.
    """
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

    def init_x(self):
        self.shape = [10, 2, 4]
        self.ins_num = np.prod(np.array(self.shape))
<<<<<<< HEAD
        self.X_2d = randomize_probability(self.ins_num, self.class_num).astype(
            self.dtype
        )
        self.x = self.X_2d.reshape(self.shape + [self.class_num])

    def init_label(self):
        self.label_2d = np.random.randint(
            0, self.class_num, (self.ins_num, 1), dtype="int64"
        )
=======
        self.X_2d = randomize_probability(self.ins_num,
                                          self.class_num).astype(self.dtype)
        self.x = self.X_2d.reshape(self.shape + [self.class_num])

    def init_label(self):
        self.label_2d = np.random.randint(0,
                                          self.class_num, (self.ins_num, 1),
                                          dtype="int64")
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        self.label = self.label_2d.reshape(self.shape + [1])

    def get_cross_entropy(self):
        cross_entropy_2d = np.asmatrix(
<<<<<<< HEAD
            [
                [-np.log(self.X_2d[i][self.label_2d[i][0]])]
                for i in range(self.X_2d.shape[0])
            ]
        ).astype(self.dtype)
        self.cross_entropy = np.array(cross_entropy_2d).reshape(
            self.shape + [1]
        )
=======
            [[-np.log(self.X_2d[i][self.label_2d[i][0]])]
             for i in range(self.X_2d.shape[0])]).astype(self.dtype)
        self.cross_entropy = np.array(cross_entropy_2d).reshape(self.shape +
                                                                [1])
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

    def init_attr_type(self):
        self.soft_label = False

    def init_dtype_type(self):
        self.dtype = np.float64

    def init_bs_class_num(self):
        self.class_num = 10


class TestCrossEntropyOp4RemoveLastDim(TestCrossEntropyOp4):
<<<<<<< HEAD
    """Test high rank tensor cross-entropy with discrete one-hot labels with shape [batch_size]"""

    def init_label(self):
        self.label_2d = np.random.randint(
            0, self.class_num, (self.ins_num, 1), dtype="int64"
        )
=======
    """Test high rank tensor cross-entropy with discrete one-hot labels with shape [batch_size]
    """

    def init_label(self):
        self.label_2d = np.random.randint(0,
                                          self.class_num, (self.ins_num, 1),
                                          dtype="int64")
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        self.label = self.label_2d.reshape(self.shape)

    def get_cross_entropy(self):
        cross_entropy_2d = np.asmatrix(
<<<<<<< HEAD
            [
                [-np.log(self.X_2d[i][self.label_2d[i][0]])]
                for i in range(self.X_2d.shape[0])
            ]
        ).astype(self.dtype)
=======
            [[-np.log(self.X_2d[i][self.label_2d[i][0]])]
             for i in range(self.X_2d.shape[0])]).astype(self.dtype)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        self.cross_entropy = np.array(cross_entropy_2d).reshape(self.shape)


class TestCrossEntropyOp5(TestCrossEntropyOp):
<<<<<<< HEAD
    """Test high rank tensor cross-entropy with vectorized soft labels."""
=======
    """Test high rank tensor cross-entropy with vectorized soft labels.
    """
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

    def init_x(self):
        self.shape = [4, 3]
        self.ins_num = np.prod(np.array(self.shape))
<<<<<<< HEAD
        self.X_2d = randomize_probability(self.ins_num, self.class_num).astype(
            self.dtype
        )
=======
        self.X_2d = randomize_probability(self.ins_num,
                                          self.class_num).astype(self.dtype)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        self.x = self.X_2d.reshape(self.shape + [self.class_num])

    def init_label(self):
        self.label_2d = np.random.uniform(
<<<<<<< HEAD
            0.1, 1.0, [self.ins_num, self.class_num]
        ).astype(self.dtype)
=======
            0.1, 1.0, [self.ins_num, self.class_num]).astype(self.dtype)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        self.label_2d /= self.label_2d.sum(axis=1, keepdims=True)
        self.label = self.label_2d.reshape(self.shape + [self.class_num])

    def get_cross_entropy(self):
<<<<<<< HEAD
        cross_entropy_2d = (
            (-self.label_2d * np.log(self.X_2d))
            .sum(axis=1, keepdims=True)
            .astype(self.dtype)
        )
        self.cross_entropy = np.array(cross_entropy_2d).reshape(
            self.shape + [1]
        )
=======
        cross_entropy_2d = (-self.label_2d * np.log(self.X_2d)).sum(
            axis=1, keepdims=True).astype(self.dtype)
        self.cross_entropy = np.array(cross_entropy_2d).reshape(self.shape +
                                                                [1])
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

    def init_attr_type(self):
        self.soft_label = True

    def init_dtype_type(self):
        self.dtype = np.float64

    def init_bs_class_num(self):
        self.class_num = 37

    def test_check_grad(self):
<<<<<<< HEAD
        self.check_grad(
            ["X"], "Y", max_relative_error=0.05, numeric_grad_delta=0.001
        )


class TestCrossEntropyOp6(TestCrossEntropyOp):
    """Test high rank tensor cross-entropy with vectorized one-hot representation of labels."""
=======
        self.check_grad(["X"],
                        "Y",
                        max_relative_error=0.05,
                        numeric_grad_delta=0.001)


class TestCrossEntropyOp6(TestCrossEntropyOp):
    """Test high rank tensor cross-entropy with vectorized one-hot representation of labels.
    """
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

    def init_x(self):
        self.shape = [4, 3, 2]
        self.ins_num = np.prod(np.array(self.shape))
<<<<<<< HEAD
        self.X_2d = randomize_probability(self.ins_num, self.class_num).astype(
            self.dtype
        )
        self.x = self.X_2d.reshape(self.shape + [self.class_num])

    def init_label(self):
        self.label_index_2d = np.random.randint(
            0, self.class_num, (self.ins_num), dtype="int64"
        )
        label_2d = np.zeros(self.X_2d.shape)
        label_2d[np.arange(self.ins_num), self.label_index_2d] = 1
        self.label = label_2d.reshape(self.shape + [self.class_num]).astype(
            self.dtype
        )

    def get_cross_entropy(self):
        cross_entropy_2d = np.asmatrix(
            [
                [-np.log(self.X_2d[i][self.label_index_2d[i]])]
                for i in range(self.X_2d.shape[0])
            ]
        )
        self.cross_entropy = (
            np.array(cross_entropy_2d)
            .reshape(self.shape + [1])
            .astype(self.dtype)
        )
=======
        self.X_2d = randomize_probability(self.ins_num,
                                          self.class_num).astype(self.dtype)
        self.x = self.X_2d.reshape(self.shape + [self.class_num])

    def init_label(self):
        self.label_index_2d = np.random.randint(0,
                                                self.class_num, (self.ins_num),
                                                dtype="int64")
        label_2d = np.zeros(self.X_2d.shape)
        label_2d[np.arange(self.ins_num), self.label_index_2d] = 1
        self.label = label_2d.reshape(self.shape + [self.class_num]).astype(
            self.dtype)

    def get_cross_entropy(self):
        cross_entropy_2d = np.asmatrix(
            [[-np.log(self.X_2d[i][self.label_index_2d[i]])]
             for i in range(self.X_2d.shape[0])])
        self.cross_entropy = np.array(cross_entropy_2d).reshape(self.shape +
                                                                [1]).astype(
                                                                    self.dtype)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

    def init_attr_type(self):
        self.soft_label = True

    def init_dtype_type(self):
        self.dtype = np.float64

    def init_bs_class_num(self):
        self.class_num = 17

    def test_check_grad(self):
<<<<<<< HEAD
        self.check_grad(
            ["X"], "Y", max_relative_error=0.05, numeric_grad_delta=0.001
        )


class TestCrossEntropyOp7(TestCrossEntropyOp):
    """Test cross-entropy with ignore index."""

    def init_label(self):
        self.label = np.random.randint(
            0, self.class_num, (self.batch_size, 1), dtype="int64"
        )

    def get_cross_entropy(self):
        self.cross_entropy = np.asmatrix(
            [
                [-np.log(self.x[i][self.label[i][0]])]
                if self.label[i][0] != self.ignore_index
                else [0]
                for i in range(self.x.shape[0])
            ]
        ).astype(self.dtype)
=======
        self.check_grad(["X"],
                        "Y",
                        max_relative_error=0.05,
                        numeric_grad_delta=0.001)


class TestCrossEntropyOp7(TestCrossEntropyOp):
    """Test cross-entropy with ignore index.
    """

    def init_label(self):
        self.label = np.random.randint(0,
                                       self.class_num, (self.batch_size, 1),
                                       dtype="int64")

    def get_cross_entropy(self):
        self.cross_entropy = np.asmatrix(
            [[-np.log(self.x[i][self.label[i][0]])]
             if self.label[i][0] != self.ignore_index else [0]
             for i in range(self.x.shape[0])]).astype(self.dtype)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

    def init_attr_type(self):
        self.soft_label = False
        self.ignore_index = 3

    def init_dtype_type(self):
        self.dtype = np.float64

    def init_bs_class_num(self):
        self.batch_size = 30
        self.class_num = 10


class TestCrossEntropyOp7RemoveLastDim(TestCrossEntropyOp7):
<<<<<<< HEAD
    """Test cross-entropy with ignore index with shape [batch_size]."""

    def init_label(self):
        self.label = np.random.randint(
            0, self.class_num, (self.batch_size), dtype="int64"
        )

    def get_cross_entropy(self):
        self.cross_entropy = np.asmatrix(
            [
                [-np.log(self.x[i][self.label[i]])]
                if self.label[i] != self.ignore_index
                else [0]
                for i in range(self.x.shape[0])
            ]
        ).astype(self.dtype)
        self.cross_entropy = (
            np.array(self.cross_entropy)
            .reshape([self.batch_size])
            .astype(self.dtype)
        )
=======
    """Test cross-entropy with ignore index with shape [batch_size].
    """

    def init_label(self):
        self.label = np.random.randint(0,
                                       self.class_num, (self.batch_size),
                                       dtype="int64")

    def get_cross_entropy(self):
        self.cross_entropy = np.asmatrix(
            [[-np.log(self.x[i][self.label[i]])]
             if self.label[i] != self.ignore_index else [0]
             for i in range(self.x.shape[0])]).astype(self.dtype)
        self.cross_entropy = np.array(self.cross_entropy).reshape(
            [self.batch_size]).astype(self.dtype)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81


# Add Fp16 test
def create_test_class(parent, cls_name):
<<<<<<< HEAD
    @unittest.skipIf(
        not core.is_compiled_with_cuda(), "core is not compiled with CUDA"
    )
    class TestCrossEntropyFP16Op(parent):
=======

    @unittest.skipIf(not core.is_compiled_with_cuda(),
                     "core is not compiled with CUDA")
    class TestCrossEntropyFP16Op(parent):

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        def init_dtype_type(self):
            return np.float16

        def test_check_output(self):
            place = core.CUDAPlace(0)
            if core.is_float16_supported(place):
                self.check_output_with_place(place, atol=2e-1)

        def test_check_grad(self):
            place = core.CUDAPlace(0)
            if core.is_float16_supported(place):
<<<<<<< HEAD
                self.check_grad_with_place(
                    place, ['X'], 'Y', max_relative_error=0.9
                )
=======
                self.check_grad_with_place(place, ['X'],
                                           'Y',
                                           max_relative_error=0.9)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

    cls_name = "{0}".format(cls_name)
    TestCrossEntropyFP16Op.__name__ = cls_name
    globals()[cls_name] = TestCrossEntropyFP16Op


create_test_class(TestCrossEntropyOp, "TestCrossEntropyF16Op")
<<<<<<< HEAD
# create_test_class(TestCrossEntropyOp2, "TestCrossEntropyF16Op2")
create_test_class(TestCrossEntropyOp3, "TestCrossEntropyF16Op3")
create_test_class(TestCrossEntropyOp4, "TestCrossEntropyF16Op4")
create_test_class(
    TestCrossEntropyOp4RemoveLastDim, "TestCrossEntropyF16Op4RemoveLastDim"
)
# create_test_class(TestCrossEntropyOp5, "TestCrossEntropyF16Op5")
create_test_class(TestCrossEntropyOp6, "TestCrossEntropyF16Op6")
create_test_class(TestCrossEntropyOp7, "TestCrossEntropyF16Op7")
create_test_class(
    TestCrossEntropyOp7RemoveLastDim, "TestCrossEntropyF16Op7RemoveLastDim"
)


class TestCrossEntropyOpError(unittest.TestCase):
=======
#create_test_class(TestCrossEntropyOp2, "TestCrossEntropyF16Op2")
create_test_class(TestCrossEntropyOp3, "TestCrossEntropyF16Op3")
create_test_class(TestCrossEntropyOp4, "TestCrossEntropyF16Op4")
create_test_class(TestCrossEntropyOp4RemoveLastDim,
                  "TestCrossEntropyF16Op4RemoveLastDim")
#create_test_class(TestCrossEntropyOp5, "TestCrossEntropyF16Op5")
create_test_class(TestCrossEntropyOp6, "TestCrossEntropyF16Op6")
create_test_class(TestCrossEntropyOp7, "TestCrossEntropyF16Op7")
create_test_class(TestCrossEntropyOp7RemoveLastDim,
                  "TestCrossEntropyF16Op7RemoveLastDim")


class TestCrossEntropyOpError(unittest.TestCase):

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def test_errors(self):
        with program_guard(Program(), Program()):

            def test_Variable():
                # the input of cross_entropy must be Variable.
<<<<<<< HEAD
                x1 = fluid.create_lod_tensor(
                    np.array([-1, 3, 5, 5]), [[1, 1, 1, 1]], fluid.CPUPlace()
                )
                lab1 = fluid.create_lod_tensor(
                    np.array([-1, 3, 5, 5]), [[1, 1, 1, 1]], fluid.CPUPlace()
                )
                paddle.nn.functional.cross_entropy(
                    x1, lab1, reduction='none', use_softmax=False
                )
=======
                x1 = fluid.create_lod_tensor(np.array([-1, 3, 5, 5]),
                                             [[1, 1, 1, 1]], fluid.CPUPlace())
                lab1 = fluid.create_lod_tensor(np.array([-1, 3, 5, 5]),
                                               [[1, 1, 1, 1]], fluid.CPUPlace())
                fluid.layers.cross_entropy(x1, lab1)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

            self.assertRaises(TypeError, test_Variable)

            def test_dtype():
                # the input dtype of cross_entropy must be float16 or float32 or float64
                # float16 only can be set on GPU place
<<<<<<< HEAD
                x2 = paddle.static.data(
                    name='x2', shape=[-1, 3, 4, 5, 6], dtype="int32"
                )
                lab2 = paddle.static.data(
                    name='lab2', shape=[-1, 3, 4, 5, 6], dtype="int32"
                )
                paddle.nn.functional.cross_entropy(
                    x2, lab2, reduction='none', use_softmax=False
                )
=======
                x2 = fluid.layers.data(name='x2',
                                       shape=[3, 4, 5, 6],
                                       dtype="int32")
                lab2 = fluid.layers.data(name='lab2',
                                         shape=[3, 4, 5, 6],
                                         dtype="int32")
                fluid.layers.cross_entropy(x2, lab2)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

            self.assertRaises(TypeError, test_dtype)


if __name__ == "__main__":
    unittest.main()
