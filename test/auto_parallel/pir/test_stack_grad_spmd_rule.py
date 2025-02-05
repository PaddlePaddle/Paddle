# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

import paddle
import paddle.distributed as dist
from paddle.distributed.auto_parallel.static.mix_to_dist_pass import (
    apply_mix2dist_pass,
)

paddle.enable_static()


class TestStackGradSpmdRule(unittest.TestCase):
    def test_build_replicated_program(self):
        main_program = paddle.base.Program()
        with paddle.base.program_guard(main_program):
            mesh = dist.ProcessMesh([0, 1])
            x0 = paddle.static.data(name='x0', shape=[64, 36])
            x1 = paddle.static.data(name='x1', shape=[64, 36])
            x0.stop_gradient = False
            x1.stop_gradient = False
            y = paddle.static.data(name='y', shape=[2, 64, 36])
            dist_x0 = dist.shard_tensor(x0, mesh, [dist.Shard(0)])
            dist_x1 = dist.shard_tensor(x1, mesh, [dist.Shard(0)])
            dist_out = paddle.stack([dist_x0, dist_x1], axis=0)
            loss = paddle.mean(dist_out - y)

        dist_program = main_program.clone()
        apply_mix2dist_pass(dist_program)
        dist_loss_value = dist_program.global_block().ops[-1].result(0)

        with paddle.static.program_guard(dist_program):
            params_grads = paddle.autograd.ir_backward.append_backward(
                dist_loss_value
            )

        stack_grad_op = [
            op
            for op in dist_program.global_block().ops
            if op.name() == "pd_op.stack_grad"
        ]
        stack_grad_op = stack_grad_op[0]
        out_grad = stack_grad_op.operand_source(1)
        x0_grad = dist_program.global_block().ops[-1].result(0)
        x1_grad = dist_program.global_block().ops[-1].result(1)
        self.assertEqual(out_grad.dist_attr().dims_mapping, [-1, 0, -1])
        self.assertEqual(x0_grad.dist_attr().dims_mapping, [0, -1])
        self.assertEqual(x1_grad.dist_attr().dims_mapping, [0, -1])


if __name__ == "__main__":
    unittest.main()
