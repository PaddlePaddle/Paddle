# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

import os
import sys
import unittest

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(__dir__, '..')))

from semi_auto_parallel_util import SemiAutoParallelTestBase

import paddle
import paddle.distributed as dist
from paddle.distributed.auto_parallel.static.mix_to_dist_pass import (
    apply_mix2dist_pass,
)
from paddle.framework import core

import custom_relu  # pylint: disable=unused-import # isort:skip

assert core.contains_spmd_rule("custom_relu")


class TestCustomOpSemiAutoParallel(SemiAutoParallelTestBase):
    def __init__(self):
        super().__init__()
        self._backend = os.getenv("backend")
        self._seed = eval(os.getenv("seed"))

    def check_placements(self, output, expected_placements):
        assert (
            output.placements == expected_placements
        ), f"{output.placements}  vs {expected_placements}"

    def test_custom_relu(self):
        shapes = [16, 4, 4]
        specs = ['x', None, None]
        inputs, outputs = self.runfunc_and_check(
            inputs_shape=shapes,
            inputs_specs=specs,
            op_func=custom_relu.custom_relu,
            with_backward=True,
        )
        self.check_placements(outputs, [dist.Shard(0)])

    def test_custom_relu_no_spmd(self):
        shapes = [16, 4, 4]
        specs = ['x', None, None]
        inputs, outputs = self.runfunc_and_check(
            inputs_shape=shapes,
            inputs_specs=specs,
            op_func=custom_relu.custom_relu_no_spmd,
            with_backward=True,
        )
        self.check_placements(outputs, [dist.Replicate()])

    def test_custom_relu_no_shard(self):
        shapes = [16, 4, 4]
        specs = [None, None, None]
        inputs, outputs = self.runfunc_and_check(
            inputs_shape=shapes,
            inputs_specs=specs,
            op_func=custom_relu.custom_relu,
            with_backward=True,
        )
        self.check_placements(outputs, [dist.Replicate()])

    def run_test_case(self):
        if self._backend == "cpu":
            paddle.set_device("cpu")
        elif self._backend == "gpu":
            paddle.set_device("gpu:" + str(dist.get_rank()))
        else:
            raise ValueError("Only support cpu or gpu backend.")
        self.test_custom_relu_no_shard()
        self.test_custom_relu()
        self.test_custom_relu_no_spmd()


class TestBuildFakeProgramWithCustomOp(unittest.TestCase):
    def test_build_with_custom_relu(self):
        shapes = [16, 4, 4]
        paddle.enable_static()
        with paddle.pir_utils.IrGuard():
            main_program = paddle.base.Program()
            with paddle.base.program_guard(main_program):
                mesh = dist.ProcessMesh([0, 1], dim_names=['mp'])
                input = paddle.static.data(
                    name='input',
                    shape=shapes,
                )
                dist_input = dist.shard_tensor(input, mesh, [dist.Shard(0)])
                dist_out = custom_relu.custom_relu(dist_input)
        apply_mix2dist_pass(main_program)

        self.assertTrue(dist_out.is_dist_dense_tensor_type())
        self.assertEqual(dist_out._local_shape, [16 // 2, 4, 4])
        self.assertEqual(dist_out.dist_attr().dims_mapping, [0, -1, -1])
        self.assertEqual(dist_out.dist_attr().process_mesh, mesh)
        op_dist_attr = dist_out.get_defining_op().dist_attr
        self.assertEqual(op_dist_attr.process_mesh, mesh)
        self.assertEqual(
            op_dist_attr.result(0).as_tensor_dist_attr().dims_mapping,
            [0, -1, -1],
        )
        self.assertEqual(
            op_dist_attr.operand(0).as_tensor_dist_attr().dims_mapping,
            [0, -1, -1],
        )


if __name__ == '__main__':
    TestCustomOpSemiAutoParallel().run_test_case()
    TestBuildFakeProgramWithCustomOp().test_build_with_custom_relu()
