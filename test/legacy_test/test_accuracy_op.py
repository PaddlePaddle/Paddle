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
from op_test import OpTest, convert_float_to_uint16, paddle_static_guard

import paddle
from paddle import base
from paddle.base import Program, core, program_guard


def accuracy_wrapper(infer, indices, label):
    return paddle._C_ops.accuracy(infer, indices, label)


class TestAccuracyOp(OpTest):
    def setUp(self):
        self.op_type = "accuracy"
        self.python_api = accuracy_wrapper
        self.dtype = np.float32
        self.init_dtype()
        n = 8192
        infer = np.random.random((n, 1)).astype(self.dtype)
        indices = np.random.randint(0, 2, (n, 1)).astype('int64')
        label = np.random.randint(0, 2, (n, 1)).astype('int64')
        self.inputs = {'Out': infer, 'Indices': indices, "Label": label}
        num_correct = 0
        for rowid in range(n):
            for ele in indices[rowid]:
                if ele == label[rowid]:
                    num_correct += 1
                    break
        self.outputs = {
            'Accuracy': np.array(num_correct / float(n)).astype(self.dtype),
            'Correct': np.array(num_correct).astype("int32"),
            'Total': np.array(n).astype("int32"),
        }

    def init_dtype(self):
        pass

    def test_check_output(self):
        self.check_output(check_pir=True)


class TestAccuracyOpFp16(TestAccuracyOp):
    def init_dtype(self):
        self.dtype = np.float16

    def test_check_output(self):
        self.check_output(atol=1e-3, check_pir=True)


@unittest.skipIf(
    not core.is_compiled_with_cuda()
    or not core.is_bfloat16_supported(core.CUDAPlace(0)),
    "core is not compiled with CUDA and not support the bfloat16",
)
class TestAccuracyOpBf16(OpTest):
    def setUp(self):
        self.op_type = "accuracy"
        self.python_api = accuracy_wrapper
        self.init_dtype()
        n = 8192
        infer = np.random.random((n, 1)).astype(np.float32)
        indices = np.random.randint(0, 2, (n, 1)).astype('int64')
        label = np.random.randint(0, 2, (n, 1)).astype('int64')
        self.inputs = {
            'Out': convert_float_to_uint16(infer),
            'Indices': indices,
            "Label": label,
        }
        num_correct = 0
        for rowid in range(n):
            for ele in indices[rowid]:
                if ele == label[rowid]:
                    num_correct += 1
                    break
        self.outputs = {
            'Accuracy': convert_float_to_uint16(
                np.array(num_correct / float(n)).astype(np.float32)
            ),
            'Correct': np.array(num_correct).astype("int32"),
            'Total': np.array(n).astype("int32"),
        }

    def init_dtype(self):
        self.dtype = np.uint16

    def test_check_output(self):
        if core.is_compiled_with_cuda():
            place = core.CUDAPlace(0)
            self.check_output_with_place(place, atol=1e-2, check_pir=True)


class TestAccuracyOpError(unittest.TestCase):
    def test_type_errors(self):
        with paddle_static_guard():
            with program_guard(Program(), Program()):
                # The input type of accuracy_op must be Variable.
                x1 = base.create_lod_tensor(
                    np.array([[-1]]), [[1]], base.CPUPlace()
                )
                label = paddle.static.data(
                    name='label', shape=[-1, 1], dtype="int32"
                )
                self.assertRaises(TypeError, paddle.static.accuracy, x1, label)
                self.assertRaises(TypeError, paddle.metric.accuracy, x1, label)
                # The input dtype of accuracy_op must be float32 or float64.
                x2 = paddle.static.data(name='x2', shape=[-1, 4], dtype="int32")
                self.assertRaises(TypeError, paddle.static.accuracy, x2, label)
                self.assertRaises(TypeError, paddle.metric.accuracy, x2, label)

                x3 = paddle.static.data(
                    name='input', shape=[-1, 2], dtype="float32"
                )
                paddle.static.accuracy(input=x3, label=label)
                paddle.metric.accuracy(input=x3, label=label)

    def test_value_errors(self):
        with program_guard(Program(), Program()):
            # The input rank of accuracy_op must be 2.
            with self.assertRaises(ValueError):
                x3 = paddle.to_tensor([0.1], dtype='float32')
                label3 = paddle.to_tensor(
                    np.reshape([0], [1, 1]), dtype='int32'
                )
                paddle.metric.accuracy(x3, label3)


class TestAccuracyAPI1(unittest.TestCase):
    def run_api(self, accuracy_api):
        with paddle_static_guard():
            with paddle.static.program_guard(paddle.static.Program()):
                self.predictions = paddle.static.data(
                    shape=[2, 5], name="predictions", dtype="float32"
                )
                self.label = paddle.static.data(
                    shape=[2, 1], name="labels", dtype="int64"
                )
                self.result = accuracy_api(
                    input=self.predictions, label=self.label, k=1
                )
                self.input_predictions = np.array(
                    [[0.2, 0.1, 0.4, 0.1, 0.1], [0.2, 0.3, 0.1, 0.15, 0.25]],
                    dtype="float32",
                )
                self.input_labels = np.array([[2], [0]], dtype="int64")
                self.expect_value = np.array([0.5], dtype='float32')
                exe = paddle.static.Executor()
                (result,) = exe.run(
                    feed={
                        "predictions": self.input_predictions,
                        'labels': self.input_labels,
                    },
                    fetch_list=[self.result],
                )
                self.assertEqual((result == self.expect_value).all(), True)

    def test_api(self):
        self.run_api(accuracy_api=paddle.static.accuracy)
        self.run_api(accuracy_api=paddle.metric.accuracy)


class TestAccuracyAPI2(unittest.TestCase):
    def test_api(self):
        with base.dygraph.guard():
            predictions = paddle.to_tensor(
                [[0.2, 0.1, 0.4, 0.1, 0.1], [0.2, 0.3, 0.1, 0.15, 0.25]],
                dtype='float32',
            )
            label = paddle.to_tensor([[2], [0]], dtype="int64")
            result = paddle.static.accuracy(input=predictions, label=label, k=1)
            expect_value = np.array([0.5], dtype='float32')
            self.assertEqual((result.numpy() == expect_value).all(), True)


class TestAccuracyAPI(unittest.TestCase):
    def test_api(self):
        with base.dygraph.guard():
            predictions = paddle.to_tensor(
                [[0.2, 0.1, 0.4, 0.1, 0.1], [0.2, 0.3, 0.1, 0.15, 0.25]],
                dtype='float32',
            )
            label = paddle.to_tensor([[2], [0]], dtype="int64")
            result = paddle.metric.accuracy(input=predictions, label=label, k=1)
            expect_value = np.array([0.5], dtype='float32')

            self.assertEqual((result.numpy() == expect_value).all(), True)


if __name__ == '__main__':
    unittest.main()
