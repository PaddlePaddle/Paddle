# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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
import numpy as np
import paddle.fluid as fluid
from paddle.fluid import core
import paddle.compat as cpt


class TestDygraphhDataLoaderWithException(unittest.TestCase):
    def setUp(self):
        self.batch_num = 4
        self.capacity = 2

    def test_not_capacity(self):
        with fluid.dygraph.guard():
            with self.assertRaisesRegexp(ValueError,
                                         "Please give value to capacity."):
                fluid.io.DataLoader.from_generator()

    def test_single_process_with_thread_expection(self):
        def error_sample_genarator(batch_num):
            def __reader__():
                for _ in range(batch_num):
                    yield [[[1, 2], [1]]]

            return __reader__

        with fluid.dygraph.guard():
            loader = fluid.io.DataLoader.from_generator(
                capacity=self.capacity, iterable=False, use_multiprocess=False)
            loader.set_batch_generator(
                error_sample_genarator(self.batch_num), places=fluid.CPUPlace())
            exception = None
            try:
                for _ in loader():
                    print("test_single_process_with_thread_expection")
            except core.EnforceNotMet as ex:
                self.assertIn("Blocking queue is killed",
                              cpt.get_exception_message(ex))
                exception = ex
            self.assertIsNotNone(exception)

    def test_multi_process_with_thread_expection(self):
        def error_sample_genarator(batch_num):
            def __reader__():
                for _ in range(batch_num):
                    yield [[[1, 2], [1]]]

            return __reader__

        with fluid.dygraph.guard():
            loader = fluid.io.DataLoader.from_generator(
                capacity=self.capacity, use_multiprocess=True)
            loader.set_batch_generator(
                error_sample_genarator(self.batch_num), places=fluid.CPUPlace())
            exception = None
            try:
                for _ in loader():
                    print("test_multi_process_with_thread_expection")
            except core.EnforceNotMet as ex:
                self.assertIn("Blocking queue is killed",
                              cpt.get_exception_message(ex))
                exception = ex
            self.assertIsNotNone(exception)


if __name__ == '__main__':
    unittest.main()
