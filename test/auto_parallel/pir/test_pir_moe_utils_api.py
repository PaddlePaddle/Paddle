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


class TestDistReshape(unittest.TestCase):
    def build_program(
        self,
        src_shape,
        dst_shape,
        src_mesh,
        dst_mesh,
        src_placements,
        dst_placements,
    ):
        main_program = paddle.base.Program()
        with paddle.base.program_guard(main_program):
            x = paddle.static.data(name='x', shape=src_shape)
            x.stop_gradient = False
            labels = paddle.static.data(name='labels', shape=dst_shape)
            dist_x = dist.shard_tensor(x, src_mesh, src_placements)
            dist_labels = dist.shard_tensor(labels, dst_mesh, dst_placements)

            dist_y = dist.auto_parallel.moe_utils._dist_reshape(
                dist_x, dst_shape, dst_mesh, dst_placements
            )
            loss = dist_y - dist_labels

        dist_program = main_program.clone()
        apply_mix2dist_pass(dist_program)
        dist_loss_value = dist_program.global_block().ops[-1].result(0)

        with paddle.static.program_guard(dist_program):
            params_grads = paddle.autograd.ir_backward.append_backward(
                dist_loss_value
            )

        return dist_program

    def check_placements(self, fwd_op, bwd_op, x_placements, out_placements):
        assert fwd_op.name() == "dist_op.dist_reshape"
        assert bwd_op.name() == "dist_op.dist_reshape"

        out = fwd_op.result(0)
        assert out.dist_attr().placements == out_placements
        x_grad = bwd_op.result(0)
        assert x_grad.dist_attr().placements == x_placements

    def test_case0(self):
        src_shape = [64, 32]
        dst_shape = [32, 64]
        mesh = dist.ProcessMesh([[0, 1], [2, 3]])
        x_placements = [dist.Shard(1), dist.Replicate()]
        out_placements = [dist.Shard(1), dist.Replicate()]
        dist_program = self.build_program(
            src_shape, dst_shape, mesh, mesh, x_placements, out_placements
        )

        ops = dist_program.global_block().ops
        fwd_op = ops[2]
        bwd_op = ops[-1]
        self.check_placements(fwd_op, bwd_op, x_placements, out_placements)

        x = ops[0].result(0)
        assert x.dist_attr().placements_attr == x_placements

        out = fwd_op.result(0)
        assert out.shape == dst_shape
        assert out._local_shape == [32, 32]

        x_grad = bwd_op.result(0)
        assert x_grad.shape == src_shape
        assert x_grad._local_shape == [64, 16]

    def test_shard_on_multi_dim(self):
        src_shape = [2, 64, 32]
        dst_shape = [-1, 32]
        src_mesh = dist.ProcessMesh([[0, 1], [2, 3]])
        x_placements = [dist.Shard(0), dist.Shard(1)]
        dst_mesh = dist.ProcessMesh([0, 1, 2, 3])
        dst_placements = [dist.Shard(0)]

        dist_program = self.build_program(
            src_shape,
            dst_shape,
            src_mesh,
            dst_mesh,
            x_placements,
            dst_placements,
        )

        ops = dist_program.global_block().ops
        fwd_op = ops[2]
        bwd_op = ops[-1]
        self.check_placements(fwd_op, bwd_op, x_placements, dst_placements)

        out = fwd_op.result(0)
        assert out.shape == [128, 32]
        assert out._local_shape == [32, 32]

        x_grad = bwd_op.result(0)
        assert x_grad.shape == src_shape
        assert x_grad._local_shape == [1, 32, 32]


if __name__ == "__main__":
    unittest.main()
