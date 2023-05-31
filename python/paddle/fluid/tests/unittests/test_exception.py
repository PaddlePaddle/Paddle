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

import numpy

import paddle
from paddle import fluid
from paddle.fluid import core


class TestException(unittest.TestCase):
    def test_exception(self):
        exception = None
        try:
            core.__unittest_throw_exception__()
        except RuntimeError as ex:
            self.assertIn("This is a test of exception", str(ex))
            exception = ex

        self.assertIsNotNone(exception)

    def test_gpu_success(self):
        if not paddle.is_compiled_with_cuda():
            return

        try:
            core._test_enforce_gpu_success()
        except Exception as e:
            self.assertTrue(isinstance(e, OSError))
            self.assertIn(
                "CUDA error(35), CUDA driver version is insufficient for CUDA runtime version.",
                str(e),
            )
            self.assertIn(
                "[Hint: 'cudaErrorInsufficientDriver'. This indicates that the installed NVIDIA CUDA driver is older than the CUDA runtime library. This is not a supported configuration.Users should install an updated NVIDIA display driver to allow the application to run.]",
                str(e),
            )


class TestExceptionNoCStack(unittest.TestCase):
    def setUp(self):
        paddle.enable_static()
        # test no C++ stack format
        fluid.set_flags({'FLAGS_call_stack_level': 1})

    def test_exception_in_static_mode(self):
        x = paddle.static.data(name='X', shape=[-1, 13], dtype='float32')
        y = paddle.static.data(name='Y', shape=[-1, 1], dtype='float32')
        predict = paddle.static.nn.fc(x, size=1)
        loss = paddle.nn.functional.square_error_cost(input=predict, label=y)
        avg_loss = paddle.mean(loss)

        fluid.optimizer.SGD(learning_rate=0.01).minimize(avg_loss)

        place = fluid.CPUPlace()
        exe = fluid.Executor(place)
        exe.run(fluid.default_startup_program())

        x = numpy.random.random(size=(8, 12)).astype('float32')
        y = numpy.random.random(size=(8, 1)).astype('float32')

        with self.assertRaises(ValueError):
            exe.run(
                fluid.default_main_program(),
                feed={'X': x, 'Y': y},
                fetch_list=[avg_loss.name],
            )

    def test_exception_in_dynamic_mode(self):
        place = fluid.CPUPlace()
        with fluid.dygraph.guard(place):
            x = numpy.random.random(size=(10, 2)).astype('float32')
            linear = paddle.nn.Linear(1, 10)
            data = fluid.dygraph.to_variable(x)
            with self.assertRaises(ValueError):
                res = linear(data)


if __name__ == "__main__":
    unittest.main()
