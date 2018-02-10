# Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved
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

import paddle.v2.fluid.layers as layers
import paddle.v2.fluid.framework as framework
import paddle.v2.fluid as fluid


class TestPythonOperatorOverride(unittest.TestCase):
    def check_result(self, fn, x_val, y_val, place, dtype):
        shape = [9, 10]

        x_data = np.full(shape, x_val).astype(dtype)
        y_data = np.full(shape, y_val).astype(dtype)
        python_out = fn(x_data, y_data)

        x_var = layers.create_global_var(
            shape=shape, value=x_val, dtype=dtype, persistable=True)
        y_var = layers.create_global_var(
            shape=shape, value=y_val, dtype=dtype, persistable=True)
        out = fn(x_var, y_var)

        exe = fluid.Executor(place)

        exe.run(fluid.default_startup_program())
        fluid_out = exe.run(fluid.default_main_program(),
                            feed=[],
                            fetch_list=[out])

        np.testing.assert_array_equal(python_out, fluid_out[0])

    def test_override(self):
        cpu_place = fluid.CPUPlace()
        test_data = [(lambda _a, _b: _a == _b, 0.1, 1.1, cpu_place, 'float32'),
                     (lambda _a, _b: _a == _b, 1.2, 1.1, cpu_place, 'float32'),
                     (lambda _a, _b: _a < _b, 0.1, 1.1, cpu_place, 'float32'),
                     (lambda _a, _b: _a < _b, 2.1, 1.1, cpu_place, 'float32'),
                     (lambda _a, _b: _a <= _b, 0.1, 1.1, cpu_place, 'float32'),
                     (lambda _a, _b: _a <= _b, 1.1, 1.1, cpu_place, 'float32'),
                     (lambda _a, _b: _a >= _b, 1.1, 1.1, cpu_place, 'float32')]

        main_program = framework.Program()
        startup_program = framework.Program()

        with framework.program_guard(main_program, startup_program):
            for fn, x_val, y_val, place, dtype in test_data:
                self.check_result(fn, x_val, y_val, place, dtype)


if __name__ == '__main__':
    unittest.main()
