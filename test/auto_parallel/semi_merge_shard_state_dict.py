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


class SingleMlpModel(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
        self.w0 = self.create_parameter(shape=[1024, 4096])
        self.w1 = self.create_parameter(shape=[4096, 1024])

    def forward(self, x):
        y = paddle.matmul(x, self.w0)
        z = paddle.matmul(y, self.w1)
        return z


class MultiMlpModel(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
        self.layer1 = SingleMlpModel()
        self.layer2 = SingleMlpModel()

    def forward(self, x):
        y = self.layer1(x)
        z = self.layer2(y)
        return z


class TestDistCheckpoint:
    def __init__(self):
        np.random.seed(42)
        self.temp_dir = os.getenv("ckpt_path")

    def test_checkpoint_load_merge_save(self):
        model_path = os.path.join(self.temp_dir, 'model')
        single_path = os.path.join(self.temp_dir, 'single_model')

        # Test checkpoint saving
        with paddle.LazyGuard():
            model = MultiMlpModel()
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

        state_dict = model.state_dict()
        for key, value in opt.state_dict().items():
            state_dict[key] = value

        assert len(state_dict) == 20
        dist.save_state_dict(state_dict, model_path, safetensors=False)

        dist.flex_checkpoint.dcp.load_state_dict.merge_sharded_state_dict(
            model_path,
            single_path,
            skip_postfix_list=[
                "moment1_0",
                "moment2_0",
                "beta1_pow_acc_0",
                "beta2_pow_acc_0",
            ],
            offload=True,
            safetensors=False,
        )
        import safetensors

        load_result = safetensors.paddle.load_file(
            f"{single_path}/model-00001-of-00001.safetensors"
        )
        assert len(load_result) == 4


if __name__ == '__main__':
    # TestDistCheckpoint().test_dist_checkpoint()
    TestDistCheckpoint().test_checkpoint_load_merge_save()
