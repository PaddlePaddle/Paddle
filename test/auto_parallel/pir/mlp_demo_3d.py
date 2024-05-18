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

import numpy as np
from test_to_static_pir_program import create_data_loader

import paddle
import paddle.distributed as dist
from paddle import nn
from paddle.distributed.auto_parallel.static.mix_to_dist_pass import (
    apply_mix2dist_pass,
)

BATCH_SIZE = 4
BATCH_NUM = 40
IMAGE_SIZE = 16
CLASS_NUM = 8
np.random.seed(2024)
paddle.seed(2024)


class HybridParallelDemoNet(nn.Layer):
    def __init__(self, mesh1, mesh2):
        super().__init__()
        self._mesh1 = mesh1
        self._mesh2 = mesh2
        self.linear_0 = nn.Linear(IMAGE_SIZE, IMAGE_SIZE, bias_attr=False)
        self.linear_1 = nn.Linear(IMAGE_SIZE, CLASS_NUM, bias_attr=False)
        self.relu_0 = nn.ReLU()
        self.relu_1 = nn.ReLU()
        self.relu_2 = nn.ReLU()
        # shard the weights of this layer
        self.linear_0.weight = dist.shard_tensor(
            self.linear_0.weight,
            self._mesh1,
            [dist.Replicate(), dist.Shard(1)],
            stop_gradient=False,
        )
        self.linear_1.weight = dist.shard_tensor(
            self.linear_1.weight,
            self._mesh2,
            [dist.Replicate(), dist.Shard(0)],
            stop_gradient=False,
        )

    def forward(self, x):
        x.stop_gradient = False
        out = self.relu_0(x)  # triggle backward partial allreduce
        out = self.linear_0(out)
        out = self.relu_1(out)
        out = dist.reshard(out, self._mesh2, [dist.Shard(0), dist.Shard(1)])
        out = self.linear_1(out)
        out = self.relu_2(out)  # triggle forward partial allreduce
        return out


class PPDemoNet(nn.Layer):
    def __init__(self, mesh1, mesh2):
        super().__init__()
        self._mesh1 = mesh1
        self._mesh2 = mesh2
        self.linear_0 = nn.Linear(IMAGE_SIZE, IMAGE_SIZE, bias_attr=False)
        self.linear_1 = nn.Linear(IMAGE_SIZE, CLASS_NUM, bias_attr=False)
        self.relu_0 = nn.ReLU()
        self.relu_1 = nn.ReLU()
        self.relu_2 = nn.ReLU()
        # shard the weights of this layer
        self.linear_0.weight = dist.shard_tensor(
            self.linear_0.weight,
            self._mesh1,
            [dist.Replicate()],
            stop_gradient=False,
        )
        self.linear_1.weight = dist.shard_tensor(
            self.linear_1.weight,
            self._mesh2,
            [dist.Replicate()],
            stop_gradient=False,
        )

    def forward(self, x):
        x.stop_gradient = False
        out = self.relu_0(x)  # triggle backward partial allreduce
        out = self.linear_0(out)
        out = self.relu_1(out)
        out = dist.reshard(out, self._mesh2, [dist.Replicate()])
        out = self.linear_1(out)
        out = self.relu_2(out)  # triggle forward partial allreduce
        return out


class TestML3DParallel(unittest.TestCase):
    def test_to_static_program(self):
        paddle.base.set_flags({'FLAGS_enable_pir_api': 1})
        mesh1 = dist.ProcessMesh([[0, 1], [2, 3]], dim_names=["x", "y"])
        mesh2 = dist.ProcessMesh([[4, 5], [6, 7]], dim_names=["x", "y"])
        threeD_layer = HybridParallelDemoNet(mesh1, mesh2)
        opt = paddle.optimizer.SGD(
            learning_rate=0.1, parameters=threeD_layer.parameters()
        )
        loss_fn = nn.MSELoss()
        loader = create_data_loader()
        dist_loader = dist.shard_dataloader(
            loader, shard_dims="x", meshes=[mesh1, mesh2]
        )
        dist_model = dist.to_static(threeD_layer, dist_loader, loss_fn, opt)

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
        from paddle.distributed.auto_parallel.static.pir_pass import (
            apply_partition_pass,
        )

        apply_partition_pass(dist_program)
        rank = paddle.distributed.get_rank()
        ops = dist_program.global_block().ops
        op_names = [op.name() for op in ops]
        if rank < 4:
            std_ops = [
                'pd_op.data',
                'builtin.parameter',
                'pd_op.data',
                'pd_op.relu',
                'pd_op.matmul',
                'pd_op.relu',
                'dist_op.reshard',
                'dist_op.reshard',
                'pd_op.relu_grad',
                'pd_op.matmul_grad',
                'dist_op.reshard',
                'dist_op.reshard',
                'pd_op.relu_grad',
                'pd_op.sgd_',
            ]
        else:
            std_ops = [
                'pd_op.data',
                'builtin.parameter',
                'pd_op.data',
                'dist_op.reshard',
                'pd_op.matmul',
                'dist_op.reshard',
                'pd_op.relu',
                'pd_op.subtract',
                'pd_op.square',
                'pd_op.mean',
                'builtin.shadow_output',
                'pd_op.full',
                'pd_op.full_like',
                'dist_op.reshard',
                'pd_op.mean_grad',
                'dist_op.reshard',
                'pd_op.square_grad',
                'pd_op.subtract_grad',
                'pd_op.relu_grad',
                'pd_op.matmul_grad',
                'dist_op.reshard',
                'dist_op.reshard',
                'pd_op.sgd_',
            ]

        self.assertEqual(op_names, std_ops)

        # TODO(zyc)
        # dist_model.train()
        # mode = "train"

        # # TODO(2024-Q2) hack for engine api
        # dist_model._engine._has_prepared[mode] = True
        # dist_model._mode = mode
        # dist_model._engine._mode = mode
        # paddle.disable_static()
        # dist_model._engine._initialize(mode)
        # dist_model._engine._executor = paddle.static.Executor(
        #     _current_expected_place()
        # )
        # dist_model._engine._init_comm()

        # for batch_id, (image, label) in enumerate(dist_loader()):
        #     loss = dist_model(image, label)


if __name__ == "__main__":
    unittest.main()
