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

import pathlib
import sys
import unittest

import paddle
import paddle.distributed as dist
from paddle import nn
from paddle.distributed.auto_parallel.static.mix_to_dist_pass import (
    apply_mix2dist_pass,
)

sys.path.append(str(pathlib.Path(__file__).resolve().parents[0]))
from test_to_static_pir_program import (
    BATCH_SIZE,
    CLASS_NUM,
    DemoNet,
    create_data_loader,
)


class ReshardDemoNet(DemoNet):
    def __init__(self, mesh, shard=True):
        super().__init__(mesh, shard=True)

    def forward(self, x):
        out = DemoNet.forward(self, x)
        out = dist.reshard(out, self._mesh, [dist.Shard(0)])
        return out


class TestToStaticPirProgramTrain(unittest.TestCase):
    def test_to_static_program(self):
        mesh = dist.ProcessMesh([0, 1], dim_names=["x"])
        layer = ReshardDemoNet(mesh)
        opt = paddle.optimizer.SGD(
            learning_rate=0.1, parameters=layer.parameters()
        )
        loss_fn = nn.MSELoss()
        loader = create_data_loader()
        dist_loader = dist.shard_dataloader(loader, meshes=[mesh])
        dist_model = dist.to_static(layer, dist_loader, loss_fn, opt)
        engine = dist_model._engine
        engine._build("train")
        dist_program = engine._fwd_main_progs["train"]
        apply_mix2dist_pass(dist_program)
        loss = dist_program.get_output_value_by_name(engine._loss_names[0])
        with paddle.static.program_guard(dist_program):
            params_grads = paddle.autograd.ir_backward.append_backward(loss)
            engine._optimizer._apply_optimize(
                loss, startup_program=None, params_grads=params_grads
            )

        index = 0
        for op in dist_program.global_block().ops:
            if op.name() == 'dist_op.reshard':
                if index == 0:
                    # forward reshard op
                    self.fwd_input = op.operand_source(0)
                    self.assertEqual(
                        self.fwd_input.dist_attr().dims_mapping, [-1, -1]
                    )
                    self.assertEqual(
                        self.fwd_input.dist_attr().partial_dims, set()
                    )
                    self.assertEqual(
                        self.fwd_input._local_shape,
                        [BATCH_SIZE, CLASS_NUM],
                    )
                    self.fwd_output = op.result(0)
                    self.assertEqual(
                        self.fwd_output.dist_attr().dims_mapping, [0, -1]
                    )
                    self.assertEqual(
                        self.fwd_output.dist_attr().partial_dims, set()
                    )
                    self.assertEqual(
                        self.fwd_output._local_shape,
                        [BATCH_SIZE / 2, CLASS_NUM],
                    )
                elif index == 1:
                    # backward reshard op
                    self.assertEqual(op.result(0).type(), self.fwd_input.type())
                index += 1
        self.assertEqual(index, 2)


if __name__ == "__main__":
    unittest.main()
