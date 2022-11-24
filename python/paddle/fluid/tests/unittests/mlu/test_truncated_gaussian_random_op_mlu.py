#   Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

<<<<<<< HEAD
from __future__ import print_function

=======
>>>>>>> 43b92b633f5d2db98f45d4b9597e5389f6f9712f
import unittest
import numpy

import paddle
import paddle.fluid as fluid
import paddle.fluid.core as core
import sys

sys.path.append("..")
from op_test import OpTest
from paddle.fluid.op import Operator
from paddle.fluid.executor import Executor
from paddle.fluid.framework import _test_eager_guard

paddle.enable_static()


class TestTrunctedGaussianRandomOp(unittest.TestCase):
<<<<<<< HEAD

=======
>>>>>>> 43b92b633f5d2db98f45d4b9597e5389f6f9712f
    def setUp(self):
        self.op_type = "truncated_gaussian_random"
        self.inputs = {}
        self.attrs = {
            "shape": [10000],
<<<<<<< HEAD
            "mean": .0,
            "std": 1.,
=======
            "mean": 0.0,
            "std": 1.0,
>>>>>>> 43b92b633f5d2db98f45d4b9597e5389f6f9712f
            "seed": 10,
        }
        self.outputs = ["Out"]

    def test_cpu(self):
        self.gaussian_random_test(place=fluid.CPUPlace())
        self.gaussian_random_test_eager(place=fluid.CPUPlace())

    def test_mlu(self):
        if core.is_compiled_with_mlu():
            self.gaussian_random_test(place=fluid.MLUPlace(0))
            # self.gaussian_random_test_eager(place=fluid.MLUPlace(0))

    def gaussian_random_test(self, place):

        program = fluid.Program()
        block = program.global_block()
        vout = block.create_var(name="Out")
<<<<<<< HEAD
        op = block.append_op(type=self.op_type,
                             outputs={"Out": vout},
                             attrs=self.attrs)
=======
        op = block.append_op(
            type=self.op_type, outputs={"Out": vout}, attrs=self.attrs
        )
>>>>>>> 43b92b633f5d2db98f45d4b9597e5389f6f9712f

        op.desc.infer_var_type(block.desc)
        op.desc.infer_shape(block.desc)

        fetch_list = []
        for var_name in self.outputs:
            fetch_list.append(block.var(var_name))

        exe = Executor(place)
        outs = exe.run(program, fetch_list=fetch_list)
        tensor = outs[0]
<<<<<<< HEAD
        self.assertAlmostEqual(numpy.mean(tensor), .0, delta=0.1)
=======
        self.assertAlmostEqual(numpy.mean(tensor), 0.0, delta=0.1)
>>>>>>> 43b92b633f5d2db98f45d4b9597e5389f6f9712f
        self.assertAlmostEqual(numpy.var(tensor), 0.773, delta=0.1)

    # TruncatedNormal.__call__ has no return value, so here call _C_ops api
    # directly
    def gaussian_random_test_eager(self, place):
        with fluid.dygraph.guard(place):
            with _test_eager_guard():
<<<<<<< HEAD
                out = paddle._C_ops.final_state_truncated_gaussian_random(
                    self.attrs["shape"], self.attrs["mean"], self.attrs["std"],
                    self.attrs["seed"], core.VarDesc.VarType.FP32, place)
                self.assertAlmostEqual(numpy.mean(out.numpy()), .0, delta=0.1)
=======
                out = paddle._C_ops.truncated_gaussian_random(
                    self.attrs["shape"],
                    self.attrs["mean"],
                    self.attrs["std"],
                    self.attrs["seed"],
                    core.VarDesc.VarType.FP32,
                    place,
                )
                self.assertAlmostEqual(numpy.mean(out.numpy()), 0.0, delta=0.1)
>>>>>>> 43b92b633f5d2db98f45d4b9597e5389f6f9712f
                self.assertAlmostEqual(numpy.var(out.numpy()), 0.773, delta=0.1)


if __name__ == "__main__":
    unittest.main()
