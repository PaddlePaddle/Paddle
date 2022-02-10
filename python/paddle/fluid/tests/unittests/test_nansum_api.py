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

import unittest
import numpy as np
import paddle
import paddle.fluid as fluid
import paddle.fluid.core as core
from paddle.fluid import Program, program_guard


class API_Test_Nansum(unittest.TestCase):
    def test_static_graph(self):
        paddle.enable_static()
        startup_program = fluid.Program()
        train_program = fluid.Program()
        with fluid.program_guard(train_program, startup_program):
            input = fluid.data(name='input', dtype='float32', shape=[2, 4])
            out1 = paddle.nansum(input)
            out2 = paddle.nansum(input, axis=0)
            out3 = paddle.nansum(input, axis=-1)
            out4 = paddle.nansum(input, axis=1, keepdim=True)
            place = fluid.CPUPlace()
            if fluid.core.is_compiled_with_cuda():
                place = fluid.CUDAPlace(0)
            exe = fluid.Executor(place)
            exe.run(startup_program)

            x = np.array([[float('nan'), 3, 5, 9],
                          [1, 2, float('-nan'), 7]]).astype(np.float32)
            res = exe.run(train_program,
                          feed={'input': x},
                          fetch_list=[out1, out2, out3, out4])

            out1_np = np.array(res[0])
            out2_np = np.array(res[1])
            out3_np = np.array(res[2])
            out4_np = np.array(res[3])
            out1_ref = np.array([27]).astype(np.float32)
            out2_ref = np.array([1, 5, 5, 16]).astype(np.float32)
            out3_ref = np.array([17, 10]).astype(np.float32)
            out4_ref = np.array([[17], [10]]).astype(np.float32)

            self.assertTrue(
                (out1_np == out1_ref).all(),
                msg='nansum output is wrong, out =' + str(out1_np))
            self.assertTrue(
                (out2_np == out2_ref).all(),
                msg='nansum output is wrong, out =' + str(out2_np))
            self.assertTrue(
                (out3_np == out3_ref).all(),
                msg='nansum output is wrong, out =' + str(out3_np))
            self.assertTrue(
                (out4_np == out4_ref).all(),
                msg='nansum output is wrong, out =' + str(out4_np))

    def test_error_api(self):
        paddle.enable_static()

        ## input dtype error
        def run1():
            input = fluid.data(name='input', dtype='float16', shape=[2, 3])
            output = paddle.nansum(input)

        self.assertRaises(TypeError, run1)

        ## axis type error
        def run2():
            input = fluid.data(name='input', dtype='float16', shape=[2, 3])
            output = paddle.nansum(input, axis=1.2)

        self.assertRaises(TypeError, run2)

    def test_dygraph(self):
        x = np.array([[float('nan'), 3, 5, 9],
                      [1, 2, float('-nan'), 7]]).astype(np.float32)
        with fluid.dygraph.guard():
            inputs = fluid.dygraph.to_variable(x)
            out = paddle.nansum(inputs)
            out_ref = np.array([27]).astype(np.float32)

            self.assertTrue(
                (out.numpy() == out_ref).all(),
                msg='nansum output is wrong, out =' + str(out.numpy()))


if __name__ == "__main__":
    unittest.main()
