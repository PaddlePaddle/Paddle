# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

import paddle.fluid as fluid
import unittest
import numpy as np


class TestSwap(unittest.TestCase):
    def gpu2cpu_net(self, program, block):
        helper = fluid.layer_helper.LayerHelper("swapmem_gpu2cpu")
        with fluid.program_guard(program):
            X = fluid.data(name="X", shape=(32, 64), dtype="float32")
            fluid.layers.Print(X)
            Y = helper.create_variable_for_type_inference("float32")
            block.append_op(
                type="swapmem_gpu2cpu",
                inputs={"X": X},
                outputs={"Out": Y},
                attrs={})
            fluid.layers.Print(Y)
        return Y

    def cpu2gpu_net(self, program, block):
        with fluid.program_guard(program):
            helper = fluid.layer_helper.LayerHelper("swapmem_gpu2cpu")
            X = fluid.data(name="X", shape=(32, 64), dtype="float32")
            fluid.layers.Print(X)
            Y = helper.create_variable_for_type_inference("float32")
            block.append_op(
                type="swapmem_gpu2cpu",
                inputs={"X": X},
                outputs={"Out": Y},
                attrs={})
            fluid.layers.Print(Y)
            helper = fluid.layer_helper.LayerHelper("swapmem_cpu2gpu")
            Z = helper.create_variable_for_type_inference("float32")
            block.append_op(
                type="swapmem_cpu2gpu",
                inputs={"X": Y},
                outputs={"Out": Z},
                attrs={})
            fluid.layers.Print(Z)
        return Y

    def test_gpu2cpu(self, ):
        program = fluid.Program()
        block = program.global_block()
        Y = self.gpu2cpu_net(program, block)
        place = fluid.CUDAPlace(0)
        exe = fluid.Executor(place)
        inputs = {'X': np.random.random((32, 64)).astype("float32")}
        y, = exe.run(program, feed=inputs, fetch_list=[Y])
        self.assertTrue(np.allclose(inputs['X'], y, atol=1e-5))
        print("-" * 50)

    def test_cpu2gpu(self, ):
        program = fluid.Program()
        block = program.global_block()
        Y = self.cpu2gpu_net(program, block)
        place = fluid.CUDAPlace(0)
        exe = fluid.Executor(place)
        inputs = {'X': np.random.random((32, 64)).astype("float32")}
        y, = exe.run(program, feed=inputs, fetch_list=[Y])
        self.assertTrue(np.allclose(inputs['X'], y, atol=1e-5))
        print("-" * 50)


if __name__ == '__main__':
    unittest.main()
