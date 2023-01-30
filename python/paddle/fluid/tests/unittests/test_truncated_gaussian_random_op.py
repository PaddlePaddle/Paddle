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

<<<<<<< HEAD
import unittest

=======
from __future__ import print_function

import unittest
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
import numpy

import paddle
import paddle.fluid as fluid
import paddle.fluid.core as core
<<<<<<< HEAD
from paddle.fluid.executor import Executor


class TestTrunctedGaussianRandomOp(unittest.TestCase):
=======
from op_test import OpTest
from paddle.fluid.op import Operator
from paddle.fluid.executor import Executor
from paddle.fluid.framework import _test_eager_guard


class TestTrunctedGaussianRandomOp(unittest.TestCase):

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def setUp(self):
        self.op_type = "truncated_gaussian_random"
        self.inputs = {}
        self.attrs = {
            "shape": [10000],
<<<<<<< HEAD
            "mean": 0.0,
            "std": 1.0,
=======
            "mean": .0,
            "std": 1.,
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
            "seed": 10,
        }
        self.outputs = ["Out"]

    def test_cpu(self):
        self.gaussian_random_test(place=fluid.CPUPlace())
        self.gaussian_random_test_eager(place=fluid.CPUPlace())

    def test_gpu(self):
        if core.is_compiled_with_cuda():
            self.gaussian_random_test(place=fluid.CUDAPlace(0))
            self.gaussian_random_test_eager(place=fluid.CUDAPlace(0))

    def gaussian_random_test(self, place):

        program = fluid.Program()
        block = program.global_block()
        vout = block.create_var(name="Out")
<<<<<<< HEAD
        op = block.append_op(
            type=self.op_type, outputs={"Out": vout}, attrs=self.attrs
        )
=======
        op = block.append_op(type=self.op_type,
                             outputs={"Out": vout},
                             attrs=self.attrs)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

        op.desc.infer_var_type(block.desc)
        op.desc.infer_shape(block.desc)

        fetch_list = []
        for var_name in self.outputs:
            fetch_list.append(block.var(var_name))

        exe = Executor(place)
        outs = exe.run(program, fetch_list=fetch_list)
        tensor = outs[0]
<<<<<<< HEAD
        self.assertAlmostEqual(numpy.mean(tensor), 0.0, delta=0.1)
=======
        self.assertAlmostEqual(numpy.mean(tensor), .0, delta=0.1)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        self.assertAlmostEqual(numpy.var(tensor), 0.773, delta=0.1)

    # TruncatedNormal.__call__ has no return value, so here call _C_ops api
    # directly
    def gaussian_random_test_eager(self, place):
        with fluid.dygraph.guard(place):
<<<<<<< HEAD
            out = paddle._C_ops.truncated_gaussian_random(
                self.attrs["shape"],
                self.attrs["mean"],
                self.attrs["std"],
                self.attrs["seed"],
                core.VarDesc.VarType.FP32,
                place,
            )
            self.assertAlmostEqual(numpy.mean(out.numpy()), 0.0, delta=0.1)
            self.assertAlmostEqual(numpy.var(out.numpy()), 0.773, delta=0.1)
=======
            with _test_eager_guard():
                out = paddle._C_ops.truncated_gaussian_random(
                    self.attrs["shape"], self.attrs["mean"], self.attrs["std"],
                    self.attrs["seed"], core.VarDesc.VarType.FP32, place)
                self.assertAlmostEqual(numpy.mean(out.numpy()), .0, delta=0.1)
                self.assertAlmostEqual(numpy.var(out.numpy()), 0.773, delta=0.1)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81


if __name__ == "__main__":
    unittest.main()
