#Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
import paddle.fluid as fluid
import paddle.fluid.layers as layers
import paddle.fluid.core as core
from paddle.fluid import compiler, Program, program_guard
from paddle.fluid.op import Operator
from paddle.fluid.backward import append_backward


class TestLoDAppendAPI(unittest.TestCase):

    def test_api(self, use_cuda=False):
        main_program = Program()
        with fluid.program_guard(main_program):
            x = fluid.layers.data(name='x', shape=[6], dtype='float32')
            level = fluid.layers.data(name='level',
                                      shape=[3],
                                      dtype='int32',
                                      lod_level=0)
            result = fluid.layers.lod_append(x, level)

            x_i = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0]).astype("float32")
            level_i = np.array([0, 2, 6]).astype("int32")

            for use_cuda in [False, True]:
                if use_cuda and not fluid.core.is_compiled_with_cuda():
                    return
                place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
                exe = fluid.Executor(place)
                [out] = exe.run(fluid.default_main_program(),
                                feed={
                                    'x': x_i,
                                    'level': level_i
                                },
                                fetch_list=[result],
                                return_numpy=False)
                self.assertEqual(out.recursive_sequence_lengths(), [[2, 4]])


class TestLodAppendOpError(unittest.TestCase):

    def test_error(self):
        # The input(x) must be Variable.
        x1 = np.array([0.9383, 0.1983, 3.2, 1.2]).astype("float64")
        level1 = [0, 2, 4]
        self.assertRaises(TypeError, fluid.layers.lod_append, x1, level1)

        #The input(level) must be Variable or list.
        x2 = fluid.layers.data(name='x2', shape=[4], dtype='float32')
        self.assertRaises(ValueError, fluid.layers.lod_append, x2, 2)

        # Input(x) dtype must be float32 or float64 or int32 or int64
        for dtype in ["bool", "float16"]:
            x3 = fluid.layers.data(name='x3_' + dtype, shape=[4], dtype=dtype)
            level3 = fluid.layers.data(name='level3' + dtype,
                                       shape=[4],
                                       dtype='int32',
                                       lod_level=2)
            self.assertRaises(TypeError, fluid.layers.lod_append, x3, level3)


if __name__ == "__main__":
    unittest.main()
