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

import unittest

import numpy as np

import paddle
from paddle import base
from paddle.base import Program, program_guard


class TestLookupTableIsSparse(unittest.TestCase):
    def init_data(self):
        self.x_data = np.array([[1, 3, 0, 4, 7]]).astype("int64")
        self.y_data = np.array([[0.1, 0.3, 0, 0.4, 0.7]]).astype("float32")

    def get_w_grad(self, is_sparse):
        paddle.enable_static()
        self.init_data()
        main_program = base.Program()
        with base.program_guard(main_program, base.Program()):
            x = paddle.static.data(name='x', shape=[-1, 5], dtype='int64')
            y_ = paddle.static.data(name='y_', shape=[-1, 5], dtype='float32')
            emb = paddle.static.nn.embedding(
                input=x,
                size=[10, 16],
                param_attr=base.ParamAttr(
                    name="emb_weight",
                    learning_rate=10,
                    initializer=paddle.nn.initializer.Assign(self.w_data),
                ),
                is_sparse=is_sparse,
            )
            y = paddle.sum(emb, axis=-1)

            loss = paddle.nn.functional.square_error_cost(input=y, label=y_)
            loss = paddle.mean(loss)

            sgd_optimizer = paddle.optimizer.SGD(learning_rate=1e-4)
            sgd_optimizer.minimize(loss)

            place = base.CPUPlace()
            exe = base.Executor(place)
            exe.run(base.default_startup_program())
            ret = exe.run(
                feed={'x': self.x_data, 'y_': self.y_data},
                fetch_list=['emb_weight'],
                return_numpy=False,
            )
            return np.array(ret[0])

    def test_w_grad(self):
        self.w_data = np.random.random(size=(10, 16)).astype("float32")
        w_grad = self.get_w_grad(False)
        w_grad_with_sparse = self.get_w_grad(True)
        self.check_grad(w_grad, w_grad_with_sparse)

    def check_grad(self, w_grad1, w_grad2, tolerance=1e-6):
        np.testing.assert_allclose(
            w_grad1, w_grad2, rtol=tolerance, atol=tolerance
        )


class TestLookupTableApi(unittest.TestCase):
    def test_api(self):
        paddle.enable_static()
        x = paddle.static.data(name='x', shape=[-1, 20], dtype='int64')
        emb = paddle.static.nn.embedding(input=x, size=[128, 64])

        place = base.CPUPlace()
        x_data = np.random.randint(0, 127, [2, 20]).astype("int64")

        exe = base.Executor(place)
        exe.run(base.default_startup_program())
        ret = exe.run(
            feed={
                'x': x_data,
            },
            fetch_list=[emb],
            return_numpy=False,
        )


class TestEmbedOpError(unittest.TestCase):
    def test_errors(self):
        paddle.enable_static()
        with program_guard(Program(), Program()):
            input_data = np.random.randint(0, 10, (4, 6)).astype("int64")

            def test_Variable():
                # the input type must be Variable
                paddle.static.nn.embedding(input=input_data, size=(10, 64))

            self.assertRaises(TypeError, test_Variable)

            def test_input_dtype():
                # the input dtype must be int64
                input = paddle.static.data(
                    name='x1', shape=[4, 6], dtype='float32'
                )
                paddle.static.nn.embedding(input=input, size=(10, 64))

            self.assertRaises(TypeError, test_input_dtype)

            def test_param_dtype():
                # dtype must be float32 or float64
                input2 = paddle.static.data(
                    name='x2', shape=[4, 6], dtype='int64'
                )
                paddle.static.nn.embedding(
                    input=input2, size=(10, 64), dtype='int64'
                )

            self.assertRaises(TypeError, test_param_dtype)
            input3 = paddle.static.data(name='x3', shape=[4, 6], dtype='int64')
            paddle.static.nn.embedding(
                input=input3, size=(10, 64), dtype='float16'
            )


if __name__ == "__main__":
    paddle.enable_static()
    unittest.main()
