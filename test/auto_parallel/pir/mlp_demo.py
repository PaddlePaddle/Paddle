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
import unittest

import numpy as np
from test_to_static_pir_program import (
    DemoNet,
    RandomDataset,
    create_data_loader,
)

import paddle
import paddle.distributed as dist
from paddle import nn
from paddle.distributed import ParallelEnv
from paddle.distributed.fleet import auto

BATCH_SIZE = 4
BATCH_NUM = 2
IMAGE_SIZE = 3
CLASS_NUM = 4
np.random.seed(2024)
paddle.seed(2024)


def apply_pass(schedule_mode="FThenB", enable_send_recv_overlap=False):
    strategy = auto.Strategy()
    strategy.auto_mode = "semi"
    strategy.reinit = True

    pipeline = strategy.pipeline
    pipeline.enable = True
    pipeline.schedule_mode = schedule_mode
    pipeline.accumulate_steps = 4
    pipeline.enable_send_recv_overlap = enable_send_recv_overlap

    return strategy


def reset_prog():
    paddle.base.framework.switch_main_program(paddle.static.Program())
    paddle.base.framework.switch_startup_program(paddle.static.Program())


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
        # out = self.relu_0(x)  # triggle backward partial allreduce
        out = self.linear_0(x)
        # out = self.relu_1(out)
        out = dist.reshard(out, self._mesh2, [dist.Replicate()])
        out = self.linear_1(out)
        # out = self.relu_2(out)  # triggle forward partial allreduce
        return out


class DPDemoNet(nn.Layer):
    def __init__(
        self,
        mesh,
    ):
        super().__init__()
        self._mesh = mesh
        self.linear_0 = nn.Linear(IMAGE_SIZE, IMAGE_SIZE, bias_attr=False)
        self.linear_1 = nn.Linear(IMAGE_SIZE, CLASS_NUM, bias_attr=False)
        self.linear_0.weight = dist.shard_tensor(
            self.linear_0.weight,
            self._mesh,
            [dist.Replicate()],
            stop_gradient=False,
        )
        self.linear_1.weight = dist.shard_tensor(
            self.linear_1.weight,
            self._mesh,
            [dist.Replicate()],
            stop_gradient=False,
        )
        self.relu_0 = nn.ReLU()
        self.relu_1 = nn.ReLU()
        self.relu_2 = nn.ReLU()

    def forward(self, x):
        out = self.relu_0(x)
        out = self.linear_0(out)
        out = self.relu_1(out)
        out = self.linear_1(out)
        out = self.relu_2(out)
        return out


class TestMLPTensorParallel(unittest.TestCase):
    def test_to_static_program(self):
        paddle.base.set_flags({'FLAGS_enable_pir_api': 1})
        mesh = dist.ProcessMesh([0, 1], dim_names=["x"])
        mp_layer = DemoNet(mesh, True)
        opt = paddle.optimizer.SGD(
            learning_rate=0.1, parameters=mp_layer.parameters()
        )
        loss_fn = nn.MSELoss()
        loader = create_data_loader()
        dist_loader = dist.shard_dataloader(loader, meshes=[mesh])
        dist_model = dist.to_static(mp_layer, dist_loader, loss_fn, opt)

        dist_model.train()
        for batch_id, (image, label) in enumerate(dist_loader()):
            loss = dist_model(image, label)


class TestMLPReplicated(unittest.TestCase):
    def test_to_static_program(self):
        paddle.base.set_flags({'FLAGS_enable_pir_api': 1})
        mesh = dist.ProcessMesh([0, 1], dim_names=["x"])
        replicated_layer = DemoNet(mesh, False)
        opt = paddle.optimizer.SGD(
            learning_rate=0.1, parameters=replicated_layer.parameters()
        )
        loss_fn = nn.MSELoss()
        loader = create_data_loader()
        dist_loader = dist.shard_dataloader(loader, meshes=[mesh])
        dist_model = dist.to_static(replicated_layer, dist_loader, loss_fn, opt)

        dist_model.train()
        for batch_id, (image, label) in enumerate(dist_loader()):
            loss = dist_model(image, label)


class TestMLPPipelineParallel(unittest.TestCase):
    def test_to_static_program(self):
        paddle.base.set_flags({'FLAGS_enable_pir_api': 1})
        mesh1 = dist.ProcessMesh([0], dim_names=["x"])
        mesh2 = dist.ProcessMesh([1], dim_names=["y"])
        pp_layer = PPDemoNet(mesh1, mesh2)
        opt = paddle.optimizer.SGD(
            learning_rate=0.1, parameters=pp_layer.parameters()
        )
        loss_fn = nn.MSELoss()
        loader = create_data_loader(
            BATCH_SIZE, BATCH_NUM, IMAGE_SIZE, CLASS_NUM
        )
        dist_loader = dist.shard_dataloader(loader, meshes=[mesh1, mesh2])
        dist_model = dist.to_static(pp_layer, dist_loader, loss_fn, opt)
        dist_model.train()
        mode = "train"

        print("==== program ====")
        print(dist_model._engine._pir_dense_main_progs[mode])
        ops = dist_model._engine._pir_dense_main_progs[mode].global_block().ops
        if dist.get_rank() == 1:
            dist_model._fetch_value(ops[11].result(0), "mean_grad_out")
            dist_model._fetch_value(ops[14].result(1), "matmul_grad_out1")
        for batch_id, (image, label) in enumerate(dist_loader()):
            loss = dist_model(image, label)
            if dist.get_rank() == 1:
                print("==== mean_grad_out ====")
                print(dist_model.outs["mean_grad_out"])
                print("==== matmul_grad_out1 ====")
                print(dist_model.outs["matmul_grad_out1"] * 4)
            print("==== step%d loss ====" % batch_id)
            print(loss)

    def test_split_program(self):
        paddle.set_flags({'FLAGS_enable_pir_api': 1})
        mesh1 = dist.ProcessMesh([0], dim_names=["x"])
        mesh2 = dist.ProcessMesh([1], dim_names=["y"])
        pp_layer = PPDemoNet(mesh1, mesh2)
        opt = paddle.optimizer.SGD(
            learning_rate=0.1, parameters=pp_layer.parameters()
        )
        loss_fn = nn.MSELoss()
        loader = create_data_loader(
            BATCH_SIZE, BATCH_NUM, IMAGE_SIZE, CLASS_NUM
        )
        strategy = dist.Strategy()
        strategy.pipeline.enable = True
        strategy.pipeline.schedule_mode = "FThenB"
        strategy.pipeline.accumulate_steps = 4
        dist_loader = dist.shard_dataloader(loader, meshes=[mesh1, mesh2])
        dist_model = dist.to_static(
            pp_layer, dist_loader, loss_fn, opt, strategy
        )
        dist_model.train()
        mode = "train"

        # bwd_program = dist_model._engine._pir_dense_bwd_progs[mode]
        # bwd_ops = bwd_program.global_block().ops
        # if strategy.pipeline.enable:
        #     print("==== whole program ====")
        #     print(dist_model._engine._pir_dense_main_progs[mode])
        #     print("==== fwd program ====")
        #     print(dist_model._engine._pir_dense_fwd_progs[mode])
        #     print("==== bwd program ====")
        #     print(dist_model._engine._pir_dense_bwd_progs[mode])
        #     print("==== opt program ====")
        #     print(dist_model._engine._pir_dense_opt_progs[mode])

        # if dist.get_rank() == 1:
        #     loss_in_fwd = dist_model._engine._pir_dense_fwd_progs[mode].global_block().ops[-1].result(0)
        # else:
        #     loss_in_fwd = None
        for batch_id, (image, label) in enumerate(dist_loader()):
            loss = dist_model(image, label)
            print("==== step%d loss ====" % batch_id)
            print(loss)

    def get_engine(
        self, schedule_mode="FThenB", enable_send_recv_overlap=False
    ):
        reset_prog()

        mesh1 = dist.ProcessMesh([0], dim_names=["x"])
        mesh2 = dist.ProcessMesh([1], dim_names=["y"])
        strategy = apply_pass(schedule_mode, enable_send_recv_overlap)
        opt = paddle.optimizer.SGD(learning_rate=0.1)
        pp_layer = PPDemoNet(mesh1, mesh2)
        loss_fn = nn.MSELoss()

        engine = auto.Engine(pp_layer, loss_fn, opt, strategy=strategy)
        paddle.distributed.fleet.init(is_collective=True)
        place = paddle.base.CUDAPlace(ParallelEnv().dev_id)
        engine._executor = paddle.static.Executor(place)
        return engine

    def test_pp_pass(self):
        paddle.enable_static()
        # pp2 fthenb training with standalone executor
        os.environ['FLAGS_new_executor_micro_batching'] = 'True'
        engine_fthenb = self.get_engine(schedule_mode="FThenB")
        nsamples = BATCH_SIZE * BATCH_NUM
        images = np.random.rand(nsamples, IMAGE_SIZE).astype('float32')
        labels = np.random.rand(nsamples, CLASS_NUM).astype('float32')
        dataset = RandomDataset(images, labels, nsamples)
        history_fthenb = engine_fthenb.fit(
            dataset, batch_size=BATCH_SIZE, log_freq=1
        )
        print("===== old ir dist program ====")
        print(engine_fthenb.main_program)
        assert engine_fthenb._strategy.pipeline.schedule_mode == "FThenB"
        assert os.environ.get('FLAGS_new_executor_micro_batching') == "True"


if __name__ == "__main__":
    # unittest.main()
    # TestMLPPipelineParallel().test_to_static_program()
    TestMLPPipelineParallel().test_split_program()
    # TestMLPPipelineParallel().test_pp_pass()
