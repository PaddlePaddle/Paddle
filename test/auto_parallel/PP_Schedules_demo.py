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
from paddle.distributed.auto_parallel.pipelining.stage import (
    PipelineStage,
)
from paddle.io import DataLoader, Dataset


def fix_seeds(seed=2025):
    """Fix random seeds to ensure reproducibility"""
    paddle.seed(seed)
    random.seed(seed)
    np.random.seed(seed)


class PPMyModel(nn.Layer):
    def __init__(self):
        super().__init__()
        self.mesh = paddle.distributed.ProcessMesh(
            [0, 1, 2, 3], dim_names=["pp"]
        )
        self.num_layers = 8
        self.num_layers_per_card = self.num_layers // 4

        self.linears = nn.LayerList()
        for i in range(self.num_layers):
            linear = nn.Linear(8, 8, bias_attr=False)

            # Mark network parameters
            linear.weight = dist.shard_tensor(
                linear.weight,
                self.get_pp_mesh(i),
                [dist.Replicate()],
            )

            self.linears.append(linear)

    def get_pp_mesh(self, layer_index):
        # layer_index=0-3 corresponds to mesh_idx 0,0,1,1,2,2,3,3
        mesh_idx = int(layer_index / (self.num_layers / 4))
        return self.mesh[mesh_idx]

    def forward(self, x):
        x.stop_gradient = False
        out = x
        for i in range(self.num_layers):
            # Mark intermediate variables, reshard when switching devices
            if i % self.num_layers_per_card == 0 and i > 0:
                out = dist.reshard(out, self.get_pp_mesh(i), [dist.Replicate()])

            out = self.linears[i](out)
        return paddle.cast(out, 'float32')


class PPMyModel_SingleStage(nn.Layer):
    def __init__(self):
        super().__init__()
        self.mesh = paddle.distributed.ProcessMesh(
            [0, 1, 2, 3], dim_names=["pp"]
        )
        self.num_layers = 8
        self.num_layers_per_card = 2
        self.linears = nn.LayerList()
        for i in range(self.num_layers):
            linear = nn.Linear(8, 8, bias_attr=False)

            linear.weight = dist.shard_tensor(
                linear.weight,
                self.get_pp_mesh(i),
                [dist.Replicate()],
            )

            self.linears.append(linear)

    def get_pp_mesh(self, layer_index):
        # layer_index=0-7 maps to mesh_idx as 0,0,1,1,2,2,3,3
        mesh_idx = int(layer_index // self.num_layers_per_card)
        return self.mesh[mesh_idx]

    def forward(self, x):
        x.stop_gradient = False
        out = x
        device_id = dist.get_rank()
        for i in range(self.num_layers):
            if int(i // self.num_layers_per_card) == device_id:
                out = self.linears[i](out)

        return paddle.cast(out, 'float32')


class PPMyModel_MultiStage(nn.Layer):
    def __init__(self):
        super().__init__()
        self.mesh = paddle.distributed.ProcessMesh(
            [0, 1, 2, 3], dim_names=["pp"]
        )
        self.num_layers = 8
        self.linears = nn.LayerList()
        for i in range(self.num_layers):
            linear = nn.Linear(8, 8, bias_attr=False)

            linear.weight = dist.shard_tensor(
                linear.weight,
                self.get_pp_mesh(i),
                [dist.Replicate()],
            )

            self.linears.append(linear)

    def get_pp_mesh(self, layer_index):
        mesh_idx = int(layer_index % 4)
        return self.mesh[mesh_idx]

    def forward(self, x):
        # For MultiStage, we shard model layers, so forward calls _Pipeline_model_chunk's forward
        pass


class _Pipeline_model_chunk(nn.Layer):
    def __init__(self, layers):
        super().__init__()
        self.layers = layers

    def forward(self, x):
        out = x
        for layer in self.layers:
            out = layer(out)
        return out


class PP_DP_MyModel(nn.Layer):
    def __init__(self):
        super().__init__()
        pp_mesh0 = paddle.distributed.ProcessMesh([0, 2], dim_names=["dp"])
        pp_mesh1 = paddle.distributed.ProcessMesh([1, 3], dim_names=["dp"])
        self.num_layers = 8
        self.linears = nn.LayerList()
        for i in range(self.num_layers):
            linear = nn.Linear(8, 8, bias_attr=False)
            if i < 4:
                linear.weight = dist.shard_tensor(
                    linear.weight, pp_mesh0, [dist.Replicate()]
                )
            else:
                linear.weight = dist.shard_tensor(
                    linear.weight, pp_mesh1, [dist.Replicate()]
                )

            self.linears.append(linear)

    def forward(self, x):
        x.stop_gradient = False
        out = x
        # Get current rank's position in pp group (0 or 1)
        pp_rank = dist.get_rank() % 2

        # Only process layers belonging to current rank
        start_layer = (
            4 * pp_rank
        )  # rank 0/2 processes layers 0-3, rank 1/3 processes layers 4-7
        end_layer = start_layer + 4

        for i in range(start_layer, end_layer):
            out = self.linears[i](out)

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


class Test_Schedules:
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
        self.model = PPMyModel_SingleStage()
        self.micro_batches = 8
        self.stage = PipelineStage(self.model, self.rank, 4, group=self.group)
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
        num_iterations = 20

        for iter_idx in range(num_iterations):
            losses_by_micro_batch = []
            for i, (data, label) in enumerate(loader):
                schedule.step(data, target=label, losses=losses_by_micro_batch)
                if self.rank == 3:
                    losses_by_step.append(
                        np.array(losses_by_micro_batch, dtype=np.float32).mean()
                    )
            opt.step()
            opt.clear_grad()
        return losses_by_step

    def test_Schedule1F1B(self):
        fix_seeds()
        self.model = PPMyModel_SingleStage()
        self.micro_batches = 8
        self.stage = PipelineStage(self.model, self.rank, 4, group=self.group)
        self.stage.has_backward = True
        loss_fn_ = nn.MSELoss()
        schedule = Schedule1F1B(
            self.stage, self.micro_batches, loss_fn=loss_fn_
        )
        opt = paddle.optimizer.AdamW(
            learning_rate=0.001, parameters=self.model.parameters()
        )
        dataset = RandomDataset(image_size=8, output_size=8, num_samples=8)
        loader = DataLoader(dataset, batch_size=8)
        losses_by_step = []
        num_iterations = 20

        for iter_idx in range(num_iterations):
            losses_by_micro_batch = []
            for i, (data, label) in enumerate(loader):
                schedule.step(data, target=label, losses=losses_by_micro_batch)
                if self.rank == 3:
                    losses_by_step.append(
                        np.array(losses_by_micro_batch, dtype=np.float32).mean()
                    )
            opt.step()
            opt.clear_grad()
        return losses_by_step

    def test_ScheduleVPP(self):
        fix_seeds()
        self.model = PPMyModel_MultiStage()
        self.local_stages = 2
        self.micro_batches = 8
        self.stage_list = []
        for i in range(self.local_stages):
            stage_model = _Pipeline_model_chunk(
                self.model.linears[self.rank + i * 4 : self.rank + i * 4 + 1]
            )
            self.stage_list.append(
                PipelineStage(
                    stage_model, self.rank + i * 4, 8, group=self.group
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

        for iter_idx in range(num_iterations):
            for i, (data, label) in enumerate(loader):
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
        pp_model = PPMyModel()
        opt = paddle.optimizer.AdamW(
            learning_rate=0.001, parameters=pp_model.parameters()
        )
        loss_fn = nn.MSELoss()
        dataset = RandomDataset(image_size=8, output_size=8, num_samples=8)
        loader = DataLoader(dataset, batch_size=1)
        pp_losses_step = []
        num_iterations = 20

        for iter_idx in range(num_iterations):
            pp_losses_micro_batch = []
            for i, (data, label) in enumerate(loader):
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

    def test_dp_pp(self):
        fix_seeds()
        global_mesh = paddle.distributed.ProcessMesh(
            [[0, 2], [1, 3]], dim_names=["pp", "dp"]
        )
        fleet.auto.set_mesh(global_mesh)
        self.model = PP_DP_MyModel()
        pp_mesh0 = paddle.distributed.ProcessMesh([0, 2], dim_names=["dp"])
        pp_mesh1 = paddle.distributed.ProcessMesh([1, 3], dim_names=["dp"])
        dp_pp_pleacement = [dist.Shard(0)]
        pp_group_1 = paddle.distributed.new_group([0, 1])
        pp_group_2 = paddle.distributed.new_group([2, 3])
        dp_group = paddle.distributed.new_group([1, 3])
        self.micro_batches = 4
        if self.rank < 2:
            self.stage = PipelineStage(
                self.model, self.rank % 2, 2, group=pp_group_1
            )
        else:
            self.stage = PipelineStage(
                self.model, self.rank % 2, 2, group=pp_group_2
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
        num_iterations = 20
        all_losses_in_one_step_md5sum = []
        for iter_idx in range(num_iterations):
            losses_by_micro_batch = []
            for i, (data, label) in enumerate(loader):
                dist_data = dist.shard_tensor(data, pp_mesh0, dp_pp_pleacement)
                dist_label = dist.shard_tensor(
                    label, pp_mesh1, dp_pp_pleacement
                )
                schedule.step(
                    dist_data, target=dist_label, losses=losses_by_micro_batch
                )
                # Losses from two dp paths are in Partial(AVG) state, need to do all_reduce
                if self.rank == 1 or self.rank == 3:
                    reduced_losses = []
                    for item in losses_by_micro_batch:
                        local_loss = item._local_value()
                        dist.all_reduce(
                            local_loss, op=dist.ReduceOp.AVG, group=dp_group
                        )
                        reduced_losses.append(local_loss)
                        if iter_idx == 0:
                            all_losses_in_one_step_md5sum.append(
                                local_loss._md5sum()
                            )

                if self.rank == 3:
                    # Calculate mean using reduced losses
                    losses_by_step.append(
                        np.array(reduced_losses, dtype=np.float32).mean()
                    )
            opt.step()
            opt.clear_grad()
        return losses_by_step, all_losses_in_one_step_md5sum

    def test_pp_model_with_ClipGradByGlobalNorm(self):
        """Test pipeline parallel model with ClipGradByGlobalNorm using PPMyModel as the baseline"""
        fix_seeds()
        pp_model = PPMyModel()
        opt = paddle.optimizer.AdamW(
            learning_rate=0.001,
            parameters=pp_model.parameters(),
            grad_clip=paddle.nn.ClipGradByGlobalNorm(1.0),
        )
        loss_fn = nn.MSELoss()
        dataset = RandomDataset(image_size=8, output_size=8, num_samples=8)
        loader = DataLoader(dataset, batch_size=1)
        pp_losses_step = []
        num_iterations = 20

        for iter_idx in range(num_iterations):
            pp_losses_micro_batch = []
            for i, (data, label) in enumerate(loader):
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

    def test_ScheduleFThenB_with_ClipGradByGlobalNorm(self):
        fix_seeds()
        self.model = PPMyModel_SingleStage()
        self.micro_batches = 8
        self.stage = PipelineStage(self.model, self.rank, 4, group=self.group)
        self.stage.has_backward = True
        loss_fn_ = nn.MSELoss()
        schedule = ScheduleFThenB(
            self.stage, self.micro_batches, loss_fn=loss_fn_
        )
        opt = paddle.optimizer.AdamW(
            learning_rate=0.001,
            parameters=self.model.parameters(),
            grad_clip=paddle.nn.ClipGradByGlobalNorm(1.0),
        )
        dataset = RandomDataset(image_size=8, output_size=8, num_samples=8)
        loader = DataLoader(dataset, batch_size=8)
        losses_by_step = []
        num_iterations = 20

        for iter_idx in range(num_iterations):
            losses_by_micro_batch = []
            for i, (data, label) in enumerate(loader):
                schedule.step(data, target=label, losses=losses_by_micro_batch)
                if self.rank == 3:
                    losses_by_step.append(
                        np.array(losses_by_micro_batch, dtype=np.float32).mean()
                    )
            opt.step()
            opt.clear_grad()
        return losses_by_step

    def test_dp_pp_align_mode(self):
        fix_seeds()
        paddle.set_flags({'FLAGS_enable_auto_parallel_align_mode': True})
        global_mesh = paddle.distributed.ProcessMesh(
            [[0, 2], [1, 3]], dim_names=["pp", "dp"]
        )
        fleet.auto.set_mesh(global_mesh)
        self.model = PP_DP_MyModel()
        pp_mesh0 = paddle.distributed.ProcessMesh([0, 2], dim_names=["dp"])
        pp_mesh1 = paddle.distributed.ProcessMesh([1, 3], dim_names=["dp"])
        dp_pp_pleacement = [dist.Shard(0)]
        pp_group_1 = paddle.distributed.new_group([0, 1])
        pp_group_2 = paddle.distributed.new_group([2, 3])
        dp_group = paddle.distributed.new_group([1, 3])
        self.micro_batches = 4
        if self.rank < 2:
            self.stage = PipelineStage(
                self.model, self.rank % 2, 2, group=pp_group_1
            )
        else:
            self.stage = PipelineStage(
                self.model, self.rank % 2, 2, group=pp_group_2
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
        all_losses_in_one_step_md5sum = []
        num_iterations = 20
        for iter_idx in range(num_iterations):
            losses_by_micro_batch = []
            for i, (data, label) in enumerate(loader):
                dist_data = dist.shard_tensor(data, pp_mesh0, dp_pp_pleacement)
                dist_label = dist.shard_tensor(
                    label, pp_mesh1, dp_pp_pleacement
                )
                schedule.step(
                    dist_data, target=dist_label, losses=losses_by_micro_batch
                )
                # Losses from two dp paths are in Partial(AVG) state, need to do all_reduce
                if self.rank == 1 or self.rank == 3:
                    reduced_losses = []
                    for item in losses_by_micro_batch:
                        local_loss = item._local_value()
                        dist.all_reduce(
                            local_loss, op=dist.ReduceOp.AVG, group=dp_group
                        )
                        reduced_losses.append(local_loss)
                        if iter_idx == 0:
                            all_losses_in_one_step_md5sum.append(
                                local_loss._md5sum()
                            )

                if self.rank == 3:
                    # Calculate mean using reduced losses
                    losses_by_step.append(
                        np.array(reduced_losses, dtype=np.float32).mean()
                    )
            opt.step()
            opt.clear_grad()
        return losses_by_step, all_losses_in_one_step_md5sum

    def run_test(self):
        """Compare losses between three training methods"""
        self.setUpClass()
        pp_losses = self.test_pp_model()
        scheduleFThenB_losses = self.test_ScheduleFThenB()
        schedule1f1b_losses = self.test_Schedule1F1B()
        schedulevpp_losses = self.test_ScheduleVPP()
        pp_model_with_ClipGradByGlobalNorm_losses = (
            self.test_pp_model_with_ClipGradByGlobalNorm()
        )
        scheduleFThenB_with_ClipGradByGlobalNorm_losses = (
            self.test_ScheduleFThenB_with_ClipGradByGlobalNorm()
        )
        dp_pp_losses, dp_pp_losses_md5sum = self.test_dp_pp()
        dp_pp_align_mode_losses, dp_pp_align_mode_losses_md5sum = (
            self.test_dp_pp_align_mode()
        )

        if self.rank == 3:
            np.testing.assert_allclose(
                pp_losses,
                scheduleFThenB_losses,
                rtol=1e-5,
            )

            np.testing.assert_allclose(
                schedule1f1b_losses,
                scheduleFThenB_losses,
                rtol=1e-5,
            )

            np.testing.assert_allclose(
                schedulevpp_losses,
                scheduleFThenB_losses,
                rtol=1e-5,
            )

            np.testing.assert_allclose(
                dp_pp_losses,
                scheduleFThenB_losses,
                rtol=1e-5,
            )

            np.testing.assert_allclose(
                pp_model_with_ClipGradByGlobalNorm_losses,
                scheduleFThenB_with_ClipGradByGlobalNorm_losses,
                rtol=1e-5,
            )

            np.testing.assert_allclose(
                dp_pp_align_mode_losses,
                dp_pp_losses,
                rtol=1e-5,
            )

            assert dp_pp_losses_md5sum == dp_pp_align_mode_losses_md5sum


if __name__ == '__main__':
    Test_Schedules().run_test()
