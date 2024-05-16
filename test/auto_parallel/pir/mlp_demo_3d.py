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

import os
import random
import unittest

import numpy as np
from test_to_static_pir_program import create_data_loader

import paddle
import paddle.distributed as dist
from paddle import nn
from paddle.distributed.auto_parallel.static.mix_to_dist_pass import (
    apply_mix2dist_pass,
)
from paddle.framework import _current_expected_place

BATCH_SIZE = 4
BATCH_NUM = 2
IMAGE_SIZE = 4
CLASS_NUM = 2


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
        out = self.relu_0(x)  # trigger backward partial allreduce
        out = self.linear_0(out)
        out = self.relu_1(out)
        out = dist.reshard(out, self._mesh2, [dist.Shard(0), dist.Shard(1)])
        out = self.linear_1(out)
        out = self.relu_2(out)  # trigger forward partial allreduce
        return out


class TestML3DParallel(unittest.TestCase):
    def __init__(self):
        self._seed = eval(os.getenv("seed"))
        self.mesh1 = dist.ProcessMesh([[0, 1], [2, 3]], dim_names=["x", "y"])
        self.mesh2 = dist.ProcessMesh([[4, 5], [6, 7]], dim_names=["x", "y"])

    def set_random_seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        paddle.seed(seed)

    def test_to_static_program(self):
        paddle.base.set_flags({'FLAGS_enable_pir_api': 1})
        mesh1 = self.mesh1
        mesh2 = self.mesh2
        threeD_layer = HybridParallelDemoNet(mesh1, mesh2)
        opt = paddle.optimizer.SGD(
            learning_rate=0.1, parameters=threeD_layer.parameters()
        )
        loss_fn = nn.MSELoss()
        loader = create_data_loader(
            batch_size=4, batch_num=1, image_size=4, class_num=2
        )
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

    def test_loss_value(self):
        paddle.disable_static()
        paddle.base.set_flags({'FLAGS_enable_pir_api': 1})
        self.set_random_seed(self._seed)
        data_loader = create_data_loader(
            batch_size=BATCH_SIZE,
            batch_num=BATCH_NUM,
            image_size=IMAGE_SIZE,
            class_num=CLASS_NUM,
        )

        self.set_random_seed(self._seed)
        dy_layer = HybridParallelDemoNet(self.mesh1, self.mesh2)
        dy_opt = paddle.optimizer.SGD(
            learning_rate=0.1, parameters=dy_layer.parameters()
        )

        self.set_random_seed(self._seed)
        loss_fn = nn.MSELoss()
        dy2st_layer = HybridParallelDemoNet(self.mesh1, self.mesh2)
        dy2st_opt = paddle.optimizer.SGD(
            learning_rate=0.1, parameters=dy2st_layer.parameters()
        )

        dist_loader = dist.shard_dataloader(
            data_loader, shard_dims="x", meshes=[self.mesh1, self.mesh2]
        )

        dy_losses = self.run_dynamic(dy_layer, dy_opt, dist_loader)
        dy2st_losses = self.run_dy2static(dy2st_layer, dy2st_opt, dist_loader)

        # for i in range(len(dy_losses)):
        #     print("===== batch %d ====="%i)
        #     if "out" in self.dy_fetch_values[i]:
        #         print("==== dy_out ====")
        #         print(self.dy_fetch_values[i]["out"])
        #         print("==== dy2st_out ====")
        #         print(self.dy2st_fetch_values[i]["out"])

        paddle.disable_static()
        if dist.get_rank() in self.mesh2.process_ids:
            pd_loss_dy2st = paddle.to_tensor(dy2st_losses)
            pg = dist.new_group(self.mesh2.process_ids)
            paddle.distributed.all_reduce(pd_loss_dy2st, group=pg)
            pd_loss_dy2st = pd_loss_dy2st / 4
            dy2st_losses = pd_loss_dy2st.numpy()
            # print("dy_losses:", dy_losses)
            # print("dy2st_losses:", dy2st_losses)
            np.testing.assert_equal(dy_losses, dy2st_losses)

    def run_dy2static(self, layer, opt, dist_loader):
        loss_fn = nn.MSELoss()
        dist_model = dist.to_static(layer, dist_loader, loss_fn, opt)
        dist_model.train()
        mode = "train"

        # TODO(2024-Q2) hack for engine api
        dist_model._engine._has_prepared[mode] = True
        dist_model._mode = mode
        dist_model._engine._mode = mode
        paddle.disable_static()
        dist_model._engine._initialize(mode)
        dist_model._engine._executor = paddle.static.Executor(
            _current_expected_place()
        )
        dist_model._engine._init_comm()
        # print("==== dist program ====")
        # print(dist_model._engine._pir_dist_main_progs[mode])
        # print("==== dense program ====")
        # print(dist_model._engine._pir_dense_main_progs[mode])
        # print("==== dense program ops ====")
        # ops = dist_model._engine._pir_dense_main_progs[mode].global_block().ops
        # for i, op in enumerate(dist_model._engine._pir_dense_main_progs[mode].global_block().ops):
        #     print(i, op)
        # if dist.get_rank() < 4:
        #     pass
        #     # dist_model.fetch_value(ops[10].result(0), "linear_0.weight.grad")
        # else:
        #     dist_model.fetch_value(ops[6].result(0), "out")
        #     # dist_model.fetch_value(ops[21].result(0), "linear_1.weight.grad")

        loss_list = []
        self.dy2st_fetch_values = []
        for batch_id, data in enumerate(dist_loader()):
            if isinstance(data, dict):
                image = data['image']
                label = data['label']
            else:
                image, label = data
            # print("==== dy2st label ====")
            # print(label)
            loss = dist_model(image, label)
            loss_list.append(loss)
            self.dy2st_fetch_values.append(dist_model.outs)

        return np.array(loss_list)

    def run_dynamic(self, layer, opt, dist_loader, is_recompute=False):
        # create loss
        loss_fn = nn.MSELoss()
        loss_list = []
        self.dy_fetch_values = []
        for batch_id, data in enumerate(dist_loader()):
            values = {}
            if isinstance(data, dict):
                image = data['image']
                label = data['label']
            else:
                image, label = data
            if is_recompute:
                image.stop_gradient = False
            out = layer(image)
            # print("==== image ====")
            # print(image)
            # print("==== dy label ====")
            # print(label)
            loss = loss_fn(out, label)
            # print("==== loss ====")
            # print(loss)
            loss_list.append(loss.numpy())
            loss.backward()
            if dist.get_rank() < 4:
                pass
                # values["linear_0.weight.grad"] = layer.linear_0.weight.grad.numpy()
            else:
                # values["linear_1.weight.grad"] = layer.linear_1.weight.grad.numpy()
                values["out"] = out.numpy()
            self.dy_fetch_values.append(values)

            opt.step()
            opt.clear_grad()
        return np.array(loss_list)


if __name__ == "__main__":
    TestML3DParallel().test_loss_value()
