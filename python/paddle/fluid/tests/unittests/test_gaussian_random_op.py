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
import numpy

import paddle.fluid as fluid
import paddle.fluid.core as core
from paddle.fluid.op import Operator
from paddle.fluid.executor import Executor


class TestGaussianRandomOp(unittest.TestCase):
    def setUp(self):
        self.op_type = "gaussian_random"
        self.inputs = {}
        self.use_mkldnn = False
        self.init_kernel_type()
        self.attrs = {
            "shape": [1000, 784],
            "mean": .0,
            "std": 1.,
            "seed": 10,
            "use_mkldnn": self.use_mkldnn
        }

        self.outputs = ["Out"]

    def test_cpu(self):
        self.gaussian_random_test(place=fluid.CPUPlace())

    def test_gpu(self):
        if core.is_compiled_with_cuda():
            self.gaussian_random_test(place=fluid.CUDAPlace(0))

    def gaussian_random_test(self, place):

        program = fluid.Program()
        block = program.global_block()
        vout = block.create_var(name="Out")
        op = block.append_op(
            type=self.op_type, outputs={"Out": vout}, attrs=self.attrs)

        op.desc.infer_var_type(block.desc)
        op.desc.infer_shape(block.desc)

        fetch_list = []
        for var_name in self.outputs:
            fetch_list.append(block.var(var_name))

        exe = Executor(place)
        outs = exe.run(program, fetch_list=fetch_list)
        tensor = outs[0]

        self.assertAlmostEqual(numpy.mean(tensor), .0, delta=0.1)
        self.assertAlmostEqual(numpy.std(tensor), 1., delta=0.1)

    def init_kernel_type(self):
        pass


if __name__ == "__main__":
    unittest.main()
