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
    ScheduleFThenB,
)
from paddle.distributed.auto_parallel.pipelining.stage import PipelineStage
from paddle.io import DataLoader, Dataset


def fix_seeds(seed=2025):
    """Fix random seeds to ensure reproducibility"""
    paddle.seed(seed)
    random.seed(seed)
    np.random.seed(seed)


class PPMyModel(nn.Layer):
    def __init__(self, name_prefix="", shared_param_map={}):
        super().__init__(name_scope=name_prefix)
        self.name_prefix = name_prefix
        self.mesh = paddle.distributed.ProcessMesh(
            [0, 1, 2, 3], dim_names=["pp"]
        )
        self.num_layers = 8
        self.num_layers_per_card = self.num_layers // 4
        self.shared_param_map = shared_param_map

        self.linears = nn.LayerList()
        for i in range(self.num_layers):
            linear = nn.Linear(8, 8, bias_attr=False)

            linear.weight.name = f"{self.name_prefix}_linear_{i}_weight"

            # Mark network parameters
            linear.weight = dist.shard_tensor(
                linear.weight,
                self.get_pp_mesh(i),
                [dist.Replicate()],
            )

            self.linears.append(linear)

        self.set_shared_param()

    def set_shared_param(self):
        for _, pair in self.shared_param_map.items():
            assert len(pair) == 2
            ori_name = pair[0]
            sync_name = pair[1]
            ori_weight_idx = -1
            sync_weight_idx = -1
            for idx, linear in enumerate(self.linears):
                if ori_name == linear.weight.name:
                    ori_weight_idx = idx
                elif sync_name == linear.weight.name:
                    sync_weight_idx = idx
            assert ori_weight_idx != -1 and sync_weight_idx != -1
            self.linears[sync_weight_idx].weight = self.linears[
                ori_weight_idx
            ].weight

    def get_pp_mesh(self, layer_index):
        # layer_index=0-3 corresponds to mesh_idx 0,0,1,1,2,2,3,3
        mesh_idx = int(layer_index / (self.num_layers / 4))
        return self.mesh[mesh_idx]

    def forward(self, x):
        x.stop_gradient = False
        out = x
        for i in range(self.num_layers):
            # Mark intermediate variables, reshard when switching devices
            cur_mesh = self.get_pp_mesh(i)
            if i % self.num_layers_per_card == 0 and i > 0:
                out = dist.reshard(out, cur_mesh, [dist.Replicate()])
            if self.linears[i].weight.process_mesh != cur_mesh:
                y = dist.reshard(
                    self.linears[i].weight, cur_mesh, [dist.Replicate()]
                )
                out = paddle.matmul(out, y)
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


def _build_current_sync_commm_group(ranks_1, ranks_2, get_group_from_ranks):
    cur_rank = paddle.distributed.get_rank()
    cur_group = None
    assert len(ranks_1) == len(ranks_2)
    for idx in range(len(ranks_1)):
        group_ranks = tuple(sorted([ranks_1[idx], ranks_2[idx]]))
        if group_ranks not in get_group_from_ranks:
            new_group = dist.new_group(ranks=list(group_ranks))
            get_group_from_ranks[group_ranks] = new_group
        if cur_rank in group_ranks:
            cur_group = get_group_from_ranks[group_ranks]
    return cur_group


def build_shared_param_map(shared_params_names, model):
    shared_mp = {}
    get_group_from_ranks = {}
    cur_rank = paddle.distributed.get_rank()
    for key, pair in shared_params_names.items():
        assert len(pair) == 2
        ori_name = pair[0]
        sync_name = pair[1]
        ori_param = _get_param_from_name(ori_name, model)
        sync_param = _get_param_from_name(sync_name, model)
        assert ori_param is not None and sync_param is not None
        ori_process_ids = ori_param.process_mesh.process_ids
        sync_process_ids = sync_param.process_mesh.process_ids
        cur_group = _build_current_sync_commm_group(
            ori_process_ids, sync_process_ids, get_group_from_ranks
        )
        cur_param = None
        if cur_rank in ori_process_ids:
            cur_param = ori_param
        elif cur_rank in sync_process_ids:
            cur_param = sync_param
        if cur_param is not None and cur_group is not None:
            shared_mp[key] = {
                "param": cur_param,
                "group": cur_group,
            }
    return shared_mp


rtol = 1e-2


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

    def test_ScheduleFThenB(self):
        fix_seeds()
        name_prefix = "ScheduleFThenB"
        self.model = PPMyModel(name_prefix=name_prefix)

        self.micro_batches = 8
        shared_params_names = {
            "gpt_shared_weight": [
                f"{name_prefix}_linear_0_weight.dist",
                f"{name_prefix}_linear_7_weight.dist",
            ]
        }
        shared_mp = build_shared_param_map(shared_params_names, self.model)

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
            shared_param_map=shared_mp,
        )

        self.stage.has_backward = True
        loss_fn_ = nn.MSELoss()
        schedule = ScheduleFThenB(
            self.stage, self.micro_batches, loss_fn=loss_fn_
        )
        opt = paddle.optimizer.AdamW(
            learning_rate=0.001, parameters=self.model.parameters()
        )
        dataset = RandomDataset(image_size=8, output_size=8, num_samples=8)
        loader = DataLoader(dataset, batch_size=8)
        losses_by_step = []
        num_iterations = 4

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

    def test_pp_model(self):
        """Test pipeline parallel model using PPMyModel as the baseline"""
        fix_seeds()
        name_prefix = "pp_model"
        shared_params_names = {
            "gpt_shared_weight": [
                f"{name_prefix}_linear_0_weight.dist",
                f"{name_prefix}_linear_7_weight.dist",
            ]
        }
        pp_model = PPMyModel(
            name_prefix=name_prefix, shared_param_map=shared_params_names
        )
        opt = paddle.optimizer.AdamW(
            learning_rate=0.001, parameters=pp_model.parameters()
        )
        loss_fn = nn.MSELoss()
        dataset = RandomDataset(image_size=8, output_size=8, num_samples=8)
        loader = DataLoader(dataset, batch_size=1)
        pp_losses_step = []
        num_iterations = 4

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
        """Compare losses between three training methods"""
        self.setUpClass()
        scheduleFThenB_losses = self.test_ScheduleFThenB()
        pp_losses = self.test_pp_model()

        if self.rank == 3:
            np.testing.assert_allclose(
                pp_losses,
                scheduleFThenB_losses,
                rtol=rtol,
            )


if __name__ == '__main__':
    TestSharedParameters().run_test()
