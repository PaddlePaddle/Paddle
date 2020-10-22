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

import numpy
import unittest

import paddle
import paddle.fluid as fluid
import paddle.compat as cpt
import paddle.fluid.core as core


class TestException(unittest.TestCase):
    def test_exception(self):
        exception = None
        try:
            core.__unittest_throw_exception__()
        except RuntimeError as ex:
            self.assertIn("This is a test of exception",
                          cpt.get_exception_message(ex))
            exception = ex

        self.assertIsNotNone(exception)


class TestExceptionStatic(unittest.TestCase):
    def setUp(self):
        paddle.enable_static()
        # test no C++ stack format
        fluid.set_flags({'FLAGS_call_stack_level': 1})

    def test_exception_in_static_model(self):
        x = fluid.layers.data(name='X', shape=[-1, 13], dtype='float32')
        y = fluid.layers.data(name='Y', shape=[-1, 1], dtype='float32')
        predict = fluid.layers.fc(input=x, size=1, act=None)
        loss = fluid.layers.square_error_cost(input=predict, label=y)
        avg_loss = fluid.layers.mean(loss)

        fluid.optimizer.SGD(learning_rate=0.01).minimize(avg_loss)

        place = fluid.CPUPlace()
        exe = fluid.Executor(place)
        exe.run(fluid.default_startup_program())

        x = numpy.random.random(size=(8, 12)).astype('float32')
        y = numpy.random.random(size=(8, 1)).astype('float32')

        with self.assertRaises(ValueError):
            exe.run(fluid.default_main_program(),
                    feed={'X': x,
                          'Y': y},
                    fetch_list=[avg_loss.name])


if __name__ == "__main__":
    unittest.main()
