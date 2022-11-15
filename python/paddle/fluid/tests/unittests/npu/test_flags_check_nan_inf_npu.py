#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
import sys

sys.path.append("..")
from op_test import OpTest, skip_check_grad_ci
import paddle
import paddle.static as static
import paddle.fluid as fluid
from paddle.static import Program, program_guard

paddle.enable_static()


class TestCheckFiniteAndUnscale(unittest.TestCase):
    def setUp(self):
        fluid.set_flags({'FLAGS_check_nan_inf': True})

    def get_prog(self):
        main_program = Program()
        with program_guard(main_program):
            a = static.data(name="a", shape=[32, 32], dtype='float32')
            b = static.data(name="b", shape=[32, 32], dtype='float32')
            out = a / b
            fp16_a = a.cast(paddle.float16)
            fp16_b = b.cast(paddle.float16)
            out = fp16_a + fp16_b
        return main_program, out

    def run_prog(self, a, b):
        main_program, out = self.get_prog()
        place = paddle.set_device('npu')

        exe = static.Executor(place)
        out_ = exe.run(main_program, feed={"a": a, "b": b}, fetch_list=[out])
        return out_

    def test_contains_nan(self):
        a = np.zeros((32, 32)).astype('float32')
        b = np.zeros((32, 32)).astype('float32')

        with self.assertRaisesRegex(RuntimeError, "contains Nan/Inf"):
            out = self.run_prog(a, b)
            print(out)

    def test_contains_inf(self):
        a = np.ones((32, 32)).astype('float32')
        b = np.zeros((32, 32)).astype('float32')

        with self.assertRaisesRegex(RuntimeError, "contains Nan/Inf"):
            out = self.run_prog(a, b)
            print(out)

    def test_not_contains_nan_inf(self):
        a = np.ones((32, 32)).astype('float32')
        b = np.ones((32, 32)).astype('float32')

        out = self.run_prog(a, b)
        print(out)

    def test_fp16_overflow(self):
        a = np.ones((32, 32)).astype('float32')
        b = np.ones((32, 32)).astype('float32')
        a[0][0] = 50000
        b[0][0] = 50000

        with self.assertRaisesRegex(RuntimeError, "contains Nan/Inf"):
            out = self.run_prog(a, b)
            print(out)


if __name__ == '__main__':
    unittest.main()
