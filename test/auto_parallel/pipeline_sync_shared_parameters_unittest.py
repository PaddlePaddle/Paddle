# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

import random

import numpy as np

import paddle
import paddle.distributed as dist
from paddle import nn
from paddle.distributed import fleet
from paddle.distributed.auto_parallel.pipelining.schedules import (
    Schedule1F1B,
    ScheduleFThenB,
    ScheduleVPP,
)
from paddle.distributed.auto_parallel.pipelining.stage import PipelineStage
from paddle.io import DataLoader, Dataset


def fix_seeds(seed=2025):
    """Fix random seeds to ensure reproducibility"""
    paddle.seed(seed)
    random.seed(seed)
    np.random.seed(seed)


class PPModel(nn.Layer):
    def __init__(self, name_prefix="", schedule="FThenB", shared_parameters={}):
        super().__init__(name_scope=name_prefix)
        self.name_prefix = name_prefix
        self.mesh = paddle.distributed.ProcessMesh(
            [0, 1, 2, 3], dim_names=["pp"]
        )
        self.num_layers = 8
        self.num_layers_per_card = self.num_layers // 4
        # Store the names of each pair of shared parameters.
        self.shared_parameters = shared_parameters

        self.linears = nn.LayerList()
        for i in range(self.num_layers):
            linear = nn.Linear(8, 8, bias_attr=False)

            # Different models have distinct parameter name spaces to avoid naming conflicts.
            linear.weight.name = f"{self.name_prefix}_linear_{i}_weight"

            # Mark network parameters
            linear.weight = dist.shard_tensor(
                linear.weight,
                (
                    self.get_pp_mesh(i)
                    if schedule != "VPP"
                    else self.get_vpp_mesh(i)
                ),
                [dist.Replicate()],
            )

            self.linears.append(linear)

        # Store the parameters to be shared under different model names.
        self.model_shared_param_mp = {}

        # Build `model_shared_param_mp`.
        self.set_shared_param()

    def set_shared_param(self):
        for pair in self.shared_parameters:
            assert len(pair) == 2
            ori_name = pair[0]
            sync_name = pair[1]
            ori_param = None
            for _, linear in enumerate(self.linears):
                if ori_name == linear.weight.name:
                    ori_param = linear.weight
            assert ori_param is not None
            self.model_shared_param_mp[sync_name] = ori_param

    def get_pp_mesh(self, layer_index):
        mesh_idx = int(layer_index / (self.num_layers / 4))
        return self.mesh[mesh_idx]

    def get_vpp_mesh(self, layer_index):
        mesh_idx = int(layer_index % 4)
        return self.mesh[mesh_idx]

    def forward(self, x):
        x.stop_gradient = False
        out = x
        for i in range(self.num_layers):
            # Mark intermediate variables, reshard when switching devices
            cur_mesh = self.get_pp_mesh(i)
            if i % self.num_layers_per_card == 0 and i > 0:
                out = dist.reshard(out, cur_mesh, [dist.Replicate()])
            weight = self.linears[i].weight
            if weight.name in self.model_shared_param_mp:
                weight = dist.reshard(
                    self.model_shared_param_mp[weight.name],
                    cur_mesh,
                    [dist.Replicate()],
                )
                out = paddle.matmul(out, weight)
            else:
                out = self.linears[i](out)
        return paddle.cast(out, 'float32')


class SingleStage(nn.Layer):
    def __init__(self, layers):
        super().__init__()
        self.layers = layers

    def forward(self, x):
        x.stop_gradient = False
        out = x
        for i in range(len(self.layers)):
            out = self.layers[i](out)
        return paddle.cast(out, 'float32')


class RandomDataset(Dataset):
    def __init__(self, image_size, output_size, num_samples=1):
        super().__init__()
        self.image_size = image_size
        self.num_samples = num_samples
        self.output_size = output_size

    def __getitem__(self, index):
        input = paddle.rand([self.image_size], dtype='float32')
        label = paddle.rand([self.output_size], dtype='float32')
        return input, label

    def __len__(self):
        return self.num_samples


def _get_param_from_name(param_name, model):
    for param in model.parameters():
        if param.name == param_name:
            return param
    return None


def build_shared_parameters(shared_params_names, model):
    # Find the two shared parameters and build shared parameter information.
    shared_mp = []
    for pair in shared_params_names:
        assert len(pair) == 2
        ori_name = pair[0]
        sync_name = pair[1]
        ori_param = _get_param_from_name(ori_name, model)
        sync_param = _get_param_from_name(sync_name, model)
        # Note: Users must strictly maintain the format of the data structure here.
        shared_mp.append({"params": [ori_param, sync_param]})
    return shared_mp


rtol = 1e-5


class TestSharedParameters:
    @classmethod
    def setUpClass(cls):
        """Initialize test class setup"""
        paddle.distributed.init_parallel_env()
        cls.group = paddle.distributed.new_group([0, 1, 2, 3])
        cls.rank = dist.get_rank()
        cls.mesh = paddle.distributed.ProcessMesh(
            [0, 1, 2, 3], dim_names=["pp"]
        )
        fleet.auto.set_mesh(cls.mesh)

    def test_single_schedule(self, sing_schedule="FThenB"):
        """Test pipeline parallel model with shared parameters using FThenB/1F1B strategy"""
        fix_seeds()
        name_prefix = "pp_" + sing_schedule
        self.model = PPModel(name_prefix=name_prefix)

        self.micro_batches = 8
        shared_params_names = [
            [
                f"{name_prefix}_linear_0_weight.dist",
                f"{name_prefix}_linear_7_weight.dist",
            ]
        ]
        # Pre-build shared parameter information.
        shared_mp = build_shared_parameters(shared_params_names, self.model)

        num_layers_per_card = 2
        cur_rank = dist.get_rank()
        stage_layers = SingleStage(
            self.model.linears[
                cur_rank
                * num_layers_per_card : (cur_rank + 1)
                * num_layers_per_card
            ]
        )

        self.stage = PipelineStage(
            stage_layers,
            self.rank,
            4,
            group=self.group,
            shared_parameters=shared_mp,
        )

        self.stage.has_backward = True
        loss_fn_ = nn.MSELoss()
        if sing_schedule == "FThenB":
            schedule = ScheduleFThenB(
                self.stage, self.micro_batches, loss_fn=loss_fn_
            )
        elif sing_schedule == "1F1B":
            schedule = Schedule1F1B(
                self.stage, self.micro_batches, loss_fn=loss_fn_
            )
        else:
            raise ValueError(
                f"Unknown schedule type: {sing_schedule}. "
                f"Currently `test_single_schedule` supported types are 'FThenB' and '1F1B'."
            )
        opt = paddle.optimizer.AdamW(
            learning_rate=0.001, parameters=self.model.parameters()
        )
        dataset = RandomDataset(image_size=8, output_size=8, num_samples=8)
        loader = DataLoader(dataset, batch_size=8)
        losses_by_step = []
        num_iterations = 20

        for _ in range(num_iterations):
            losses_by_micro_batch = []
            for _, (data, label) in enumerate(loader):
                schedule.step(data, target=label, losses=losses_by_micro_batch)
                if self.rank == 3:
                    losses_by_step.append(
                        np.array(losses_by_micro_batch, dtype=np.float32).mean()
                    )
            opt.step()
            opt.clear_grad()
        return losses_by_step

    def test_multi_schedule(self, multi_schedule="VPP"):
        """Test pipeline parallel with shared parameters model using VPP strategy"""
        fix_seeds()
        name_prefix = "pp_" + multi_schedule
        self.model = PPModel(name_prefix=name_prefix, schedule="VPP")
        self.local_stages = 2
        self.micro_batches = 8
        self.stage_list = []

        shared_params_names = [
            [
                f"{name_prefix}_linear_0_weight.dist",
                f"{name_prefix}_linear_7_weight.dist",
            ]
        ]
        # Pre-build shared parameter information.
        shared_mp = build_shared_parameters(shared_params_names, self.model)

        cur_rank = dist.get_rank()
        for i in range(self.local_stages):
            stage_layers = SingleStage(
                self.model.linears[cur_rank + i * 4 : cur_rank + i * 4 + 1]
            )
            # Note: In VPP mode, the same `shared_mp` is used for building multiple
            # stages to avoid redundant group creation.
            self.stage_list.append(
                PipelineStage(
                    stage_layers,
                    cur_rank + i * 4,
                    8,
                    group=self.group,
                    shared_parameters=shared_mp,
                )
            )
            self.stage_list[i].has_backward = True

        loss_fn_ = nn.MSELoss()
        schedule = ScheduleVPP(
            self.stage_list, self.micro_batches, loss_fn=loss_fn_
        )
        opt = paddle.optimizer.AdamW(
            learning_rate=0.001, parameters=self.model.parameters()
        )
        dataset = RandomDataset(image_size=8, output_size=8, num_samples=8)
        loader = DataLoader(dataset, batch_size=8)
        losses_by_micro_batch = []
        losses_by_step = []
        num_iterations = 20

        for _ in range(num_iterations):
            for _, (data, label) in enumerate(loader):
                schedule.step(data, target=label, losses=losses_by_micro_batch)
                if self.rank == 3:
                    losses_by_step.append(
                        np.array(losses_by_micro_batch, dtype=np.float32).mean()
                    )
            opt.step()
            opt.clear_grad()
        return losses_by_step

    def test_pp_model(self):
        """Test pipeline parallel model using PPModel as the baseline"""
        fix_seeds()
        name_prefix = "pp_model"
        shared_params_names = [
            [
                f"{name_prefix}_linear_0_weight.dist",
                f"{name_prefix}_linear_7_weight.dist",
            ]
        ]
        pp_model = PPModel(
            name_prefix=name_prefix, shared_parameters=shared_params_names
        )
        opt = paddle.optimizer.AdamW(
            learning_rate=0.001, parameters=pp_model.parameters()
        )
        loss_fn = nn.MSELoss()
        dataset = RandomDataset(image_size=8, output_size=8, num_samples=8)
        loader = DataLoader(dataset, batch_size=1)
        pp_losses_step = []
        num_iterations = 20

        for _ in range(num_iterations):
            pp_losses_micro_batch = []
            for _, (data, label) in enumerate(loader):
                output = pp_model(data)
                loss = loss_fn(output, label)
                pp_losses_micro_batch.append(loss.item())
                loss.backward()
            pp_losses_step.append(
                np.array(pp_losses_micro_batch, dtype=np.float32).mean()
            )
            opt.step()
            opt.clear_grad()
        return pp_losses_step

    def run_test(self):
        """Compare shared params losses between three training methods"""
        self.setUpClass()
        pp_losses = self.test_pp_model()
        pp_FThenB_losses = self.test_single_schedule(sing_schedule="FThenB")
        pp_1F1B_losses = self.test_single_schedule(sing_schedule="1F1B")
        pp_vpp_losses = self.test_multi_schedule(multi_schedule="VPP")

        if self.rank == 3:
            np.testing.assert_allclose(
                pp_losses,
                pp_FThenB_losses,
                rtol=rtol,
            )
            np.testing.assert_allclose(
                pp_losses,
                pp_1F1B_losses,
                rtol=rtol,
            )
            np.testing.assert_allclose(
                pp_losses,
                pp_vpp_losses,
                rtol=rtol,
            )


if __name__ == '__main__':
    TestSharedParameters().run_test()
