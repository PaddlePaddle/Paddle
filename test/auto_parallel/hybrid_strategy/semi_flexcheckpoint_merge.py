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

import os

import numpy as np

import paddle
import paddle.distributed as dist
from paddle.io import BatchSampler, DataLoader, Dataset


class RandomDataset(Dataset):
    def __init__(self, seq_len, hidden, num_samples=100):
        super().__init__()
        self.seq_len = seq_len
        self.hidden = hidden
        self.num_samples = num_samples

    def __getitem__(self, index):
        input = np.random.uniform(size=[self.seq_len, self.hidden]).astype(
            "float32"
        )
        return input

    def __len__(self):
        return self.num_samples


class DistMlpModel(paddle.nn.Layer):
    def __init__(self, mesh):
        super().__init__()
        self.w0 = self.create_parameter(shape=[1024, 4096])
        self.w1 = self.create_parameter(shape=[4096, 1024])
        self.mesh = mesh
        self.w0 = dist.shard_tensor(
            self.w0, mesh, [dist.Replicate(), dist.Shard(1)]
        )
        self.w1 = dist.shard_tensor(
            self.w1, mesh, [dist.Replicate(), dist.Shard(0)]
        )

    def forward(self, x):
        x = dist.shard_tensor(x, self.mesh, [dist.Shard(0), dist.Replicate()])
        y = paddle.matmul(x, self.w0)
        z = paddle.matmul(y, self.w1)
        return z


class MultiMlpModel(paddle.nn.Layer):
    def __init__(self, mesh):
        super().__init__()
        self.layer1 = DistMlpModel(mesh)
        self.layer2 = DistMlpModel(mesh)

    def forward(self, x):
        y = self.layer1(x)
        z = self.layer2(x)
        return z


class SingleMlpModel(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
        self.w0 = self.create_parameter(shape=[1024, 4096])
        self.w1 = self.create_parameter(shape=[4096, 1024])

    def forward(self, x):
        y = paddle.matmul(x, self.w0)
        z = paddle.matmul(y, self.w1)
        return z


class TestDistCheckpoint:
    def __init__(self):
        np.random.seed(42)
        self.mesh = dist.ProcessMesh([[0, 1], [2, 3]], dim_names=['dp', 'mp'])
        self.temp_dir = os.getenv("ckpt_path")

    def _get_single_loss(self, dataloader, unsharded_state_dict):
        with paddle.LazyGuard():
            model = SingleMlpModel()
        model.w0.set_value(unsharded_state_dict['w0'])
        model.w1.set_value(unsharded_state_dict['w1'])
        opt = paddle.optimizer.AdamW(
            learning_rate=0.001, parameters=model.parameters()
        )

        losses = []
        for step, inputs in enumerate(dataloader):
            data = inputs
            logits = model(data)
            loss = paddle.mean(logits)
            losses.append(float(loss))
            loss.backward()
            opt.step()
            opt.clear_grad()

        return losses[0]

    def _get_dist_loss(self, dataloader, sharded_state_dict):
        with paddle.LazyGuard():
            model = DistMlpModel(self.mesh)
        model.w0.set_value(sharded_state_dict['w0'])
        model.w1.set_value(sharded_state_dict['w1'])

        opt = paddle.optimizer.AdamW(
            learning_rate=0.001, parameters=model.parameters()
        )

        losses = []
        for step, inputs in enumerate(dataloader):
            data = inputs
            logits = model(data)
            loss = paddle.mean(logits)
            loss.backward()
            opt.step()
            opt.clear_grad()
            losses.append(float(loss))

        return losses[0]

    def dist_checkpoint(self, offload=False, safetensors=True):
        model_path = os.path.join(self.temp_dir, '/model')
        opt_path = os.path.join(self.temp_dir, '/opt')

        # Test checkpoint saving
        with paddle.LazyGuard():
            model = DistMlpModel(self.mesh)
        for p in model.parameters():
            p.initialize()

        dataset = RandomDataset(128, 1024)
        sampler = BatchSampler(
            dataset,
            batch_size=4,
        )
        dataloader = DataLoader(
            dataset,
            batch_sampler=sampler,
        )
        opt = paddle.optimizer.AdamW(
            learning_rate=0.001, parameters=model.parameters()
        )
        opt = dist.shard_optimizer(opt)

        for step, inputs in enumerate(dataloader):
            data = inputs
            logits = model(data)
            loss = paddle.mean(logits)
            loss.backward()
            opt.step()
            opt.clear_grad()

        dist.save_state_dict(
            model.state_dict(), model_path, safetensors=safetensors
        )
        dist.save_state_dict(
            opt.state_dict(), opt_path, safetensors=safetensors
        )

        unsharded_state_dict = dist.load_merged_state_dict(
            model_path, offload=offload, safetensors=safetensors
        )
        # Get single loss
        single_loss = self._get_single_loss(dataloader, unsharded_state_dict)

        shard_state_dict = model.state_dict()
        dist.load_state_dict(
            shard_state_dict, model_path, safetensors=safetensors
        )

        # Get distributed loss
        dist_loss = self._get_dist_loss(dataloader, shard_state_dict)
        np.testing.assert_array_equal(
            unsharded_state_dict['w0'].numpy(), shard_state_dict['w0'].numpy()
        )
        np.testing.assert_array_equal(
            unsharded_state_dict['w1'].numpy(), shard_state_dict['w1'].numpy()
        )

    def test_dist_checkpoint(self):
        self.dist_checkpoint(True, True)
        self.dist_checkpoint(False, True)
        self.dist_checkpoint(True, False)
        self.dist_checkpoint(False, False)

    def count_files_in_temp_dir(self, single_path):
        if not os.path.exists(single_path):
            return 0
        files = [
            f
            for f in os.listdir(single_path)
            if os.path.isfile(os.path.join(single_path, f))
        ]
        return len(files)

    def test_checkpoint_load_merge_save(self):
        model_path = os.path.join(self.temp_dir, 'model')
        single_path = os.path.join(self.temp_dir, 'single_model')

        # Test checkpoint saving
        with paddle.LazyGuard():
            model = MultiMlpModel(self.mesh)
        for p in model.parameters():
            p.initialize()

        dataset = RandomDataset(128, 1024)
        sampler = BatchSampler(
            dataset,
            batch_size=4,
        )
        dataloader = DataLoader(
            dataset,
            batch_sampler=sampler,
        )
        opt = paddle.optimizer.AdamW(
            learning_rate=0.001, parameters=model.parameters()
        )
        opt = dist.shard_optimizer(opt)

        for step, inputs in enumerate(dataloader):
            data = inputs
            logits = model(data)
            loss = paddle.mean(logits)
            loss.backward()
            opt.step()
            opt.clear_grad()

        dist.save_state_dict(model.state_dict(), model_path, safetensors=False)

        dist.flex_checkpoint.dcp.load_state_dict.merge_sharded_state_dict(
            model_path, single_path, offload=True, safetensors=False, file_num=2
        )
        # assert self.count_files_in_temp_dir(single_path) == 5, (
        #     f"Expected 5 files in temp dir, but got {self.count_files_in_temp_dir(single_path)}"
        # )


if __name__ == '__main__':
    TestDistCheckpoint().test_dist_checkpoint()
    TestDistCheckpoint().test_checkpoint_load_merge_save()
