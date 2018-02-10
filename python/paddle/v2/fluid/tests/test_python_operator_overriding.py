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

import numpy
import paddle.v2.fluid.framework as framework
import paddle.v2.fluid as fluid


class TestPythonOperatorOverride(unittest.TestCase):
    def check_result(self, fn, place, dtype='float32'):
        shape = [9, 10]

        x_data = numpy.random.random(size=shape).astype(dtype)
        y_data = numpy.random.random(size=shape).astype(dtype)
        python_out = fn(x_data, y_data)

        x_var = fluid.layers.data(name='x', shape=shape, dtype=dtype)
        y_var = fluid.layers.data(name='y', shape=shape, dtype=dtype)
        out = fn(x_var, y_var)

        exe = fluid.Executor(place)
        feeder = fluid.DataFeeder(feed_list=[x_var, y_var], place=place)

        exe.run(fluid.default_startup_program())
        fluid_out = exe.run(fluid.default_main_program(),
                            feed=feeder.feed([x_data, y_data]),
                            fetch_list=[out])

        print(python_out)
        self.assertAlmostEqual(python_out, fluid_out[0])

    def test_override(self):
        main_program = framework.Program()
        startup_program = framework.Program()
        with framework.program_guard(main_program, startup_program):
            place = fluid.CPUPlace()
            self.check_result(lambda _a, _b: _a == _b, place)


if __name__ == '__main__':
    unittest.main()
