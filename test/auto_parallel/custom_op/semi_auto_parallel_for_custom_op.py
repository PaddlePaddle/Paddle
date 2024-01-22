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

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(__dir__, '..')))

from semi_auto_parallel_util import SemiAutoParallelTestBase

import paddle
import paddle.distributed as dist
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


if __name__ == '__main__':
    TestCustomOpSemiAutoParallel().run_test_case()
