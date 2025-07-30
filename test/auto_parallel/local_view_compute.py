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

import random

import numpy as np

import paddle
import paddle.distributed as dist
from paddle.distributed import ProcessMesh, fleet, get_rank, shard_dataloader
from paddle.io import BatchSampler, DataLoader, DistributedBatchSampler

base_lr = 0.001  # Learning rate
l2_decay = 1e-5  # Weight decay

epoch = 5  # Number of training epochs
batch_num = 100  # Number of batches per epoch
batch_size = 32  # Batch size for training
class_dim = 10
global_local_loss_list = []


class RandomDataset(paddle.io.Dataset):
    def __init__(self, images, labels):
        self.num_samples = len(images)
        self.images = images
        self.labels = labels

    def __getitem__(self, idx):
        # image = np.random.random([256]).astype('float32')
        # label = np.random.randint(0, class_dim - 1, (1, )).astype('int64')
        image = self.images[idx]
        label = self.labels[idx]
        return image, label

    def __len__(self):
        return self.num_samples


class SimpleNet(paddle.nn.Layer):
    def __init__(self, input_size, inner_size, output_size):
        super().__init__()
        self.linear1 = paddle.nn.Linear(input_size, inner_size)
        self.linear2 = paddle.nn.Linear(inner_size, input_size)
        self.linear3 = paddle.nn.Linear(input_size, output_size)
        self.relu = paddle.nn.ReLU()

    def forward(self, x):
        x = self.linear1(x)
        x = self.linear2(x)
        x = self.linear3(x)
        x = self.relu(x)
        return x


def masked_lm_loss_func(pred, label, global_local_loss_list_item=None):
    """自定义损失函数，基于rank进行掩码"""
    lossmask = paddle.zeros_like(label).astype('float32')
    if dist.get_rank() == 0:
        lossmask[:8] = 1
    else:
        lossmask[8:16] = 1

    pred_sub = pred[:, 0:1]  # shape [B,1]
    label_float = paddle.cast(label, 'float32')  # shape [B,1]
    raw_loss = paddle.abs(pred_sub - label_float)
    lossmask_ = lossmask.reshape([-1]).cast('float32')
    raw_loss_flat = raw_loss.reshape([-1]).cast('float32')

    masked_lm_loss_sum = paddle.sum(raw_loss_flat * lossmask_)
    valid_count = paddle.sum(lossmask_)

    loss = masked_lm_loss_sum / (valid_count + 1e-8)
    if global_local_loss_list_item is not None:
        np.testing.assert_allclose(
            global_local_loss_list_item,
            loss.numpy(),
            rtol=1e-8,
        )
    return loss


class TestLocalViewCompute:
    def __init__(self):
        self._mesh = dist.ProcessMesh([0, 1], dim_names=["x"])

    def set_random_seed(self):
        np.random.seed(2025)
        paddle.seed(2025)
        random.seed(2025)

    def create_dataset(self):
        images = np.random.rand(batch_num * batch_size * 2, 256).astype(
            'float32'
        )
        labels = np.random.randint(
            0, class_dim - 1, (batch_num * batch_size * 2, 1)
        ).astype('int64')
        datasets = RandomDataset(images, labels)
        return datasets

    def run_test_cases(self):
        # run_dy_hand_get_local_loss
        self.set_random_seed()
        dataset = self.create_dataset()
        dist_strategy = fleet.DistributedStrategy()
        dist_strategy.hybrid_configs = {
            "dp_degree": 2,
            "mp_degree": 1,
            "pp_degree": 1,
        }

        fleet.init(is_collective=True, strategy=dist_strategy)
        model = SimpleNet(
            input_size=256, inner_size=102400, output_size=class_dim
        )
        clip = paddle.nn.ClipGradByGlobalNorm(clip_norm=1.0)
        optimizer = paddle.optimizer.AdamW(
            learning_rate=base_lr,
            weight_decay=l2_decay,
            parameters=model.parameters(),
            grad_clip=clip,
        )

        model = fleet.distributed_model(model)
        optimizer = fleet.distributed_optimizer(optimizer)

        sampler = DistributedBatchSampler(
            dataset,
            rank=get_rank(),
            batch_size=batch_size // 2,
            shuffle=False,
            drop_last=True,
        )
        train_loader = DataLoader(
            dataset, batch_sampler=sampler, num_workers=1, shuffle=False
        )

        model.train()
        for batch_id, data in enumerate(train_loader()):
            if batch_id > 10:
                break

            img, label = data

            out = model(img)

            avg_loss = masked_lm_loss_func(out, label)
            avg_loss.backward()
            optimizer.step()
            model.clear_gradients()
            global_local_loss_list.append(avg_loss.numpy())

        # run_dy_semi_auto
        self.set_random_seed()
        dataset = self.create_dataset()
        world_process_mesh = ProcessMesh([0, 1], dim_names=["dp"])
        model = SimpleNet(
            input_size=256, inner_size=102400, output_size=class_dim
        )
        optimizer = paddle.optimizer.AdamW(
            learning_rate=base_lr,
            weight_decay=l2_decay,
            parameters=model.parameters(),
            grad_clip=clip,
        )

        sampler = BatchSampler(
            dataset, batch_size=batch_size, shuffle=False, drop_last=True
        )
        train_loader = DataLoader(
            dataset, batch_sampler=sampler, num_workers=1, shuffle=False
        )

        dist_dataloader = shard_dataloader(
            dataloader=train_loader, meshes=world_process_mesh, shard_dims="dp"
        )

        model.train()
        process_mesh = ProcessMesh([0, 1], dim_names=["dp"])
        out_placements = [dist.Partial(dist.ReduceType.kRedAvg)]

        for batch_id, data in enumerate(dist_dataloader()):
            if batch_id > 10:
                break

            img, label = data

            out = model(img)
            loss_func = dist.local_map(
                masked_lm_loss_func,
                out_placements=out_placements,
                in_placements=[None, None],
                process_mesh=process_mesh,
            )
            avg_loss = loss_func(
                out,
                label,
                global_local_loss_list_item=global_local_loss_list[batch_id],
            )
            avg_loss.backward()
            optimizer.step()
            model.clear_gradients()


if __name__ == '__main__':
    TestLocalViewCompute().run_test_cases()
