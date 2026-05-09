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

import hashlib
import os
import random

import numpy as np

import paddle
import paddle.distributed as dist
from paddle import nn
from paddle.distributed import fleet
from paddle.io import DataLoader

BATCH_SIZE = 2
SEQ_LEN = 50
VOCAB_SIZE = 200
HIDDEN_SIZE = 100


class CEmbeddingNet(nn.Layer):
    def __init__(self, mesh):
        super().__init__()
        self.embedding = fleet.meta_parallel.VocabParallelEmbedding(
            VOCAB_SIZE,
            HIDDEN_SIZE,
            weight_attr=paddle.nn.initializer.Constant(value=0.5),
        )

    def forward(self, x):
        x = paddle.to_tensor(x, dtype="int32")
        out = self.embedding(x)
        out = out.astype(self.embedding.weight.dtype)
        out = paddle.transpose(out, [1, 0, 2])
        t = paddle.randn([SEQ_LEN, BATCH_SIZE, HIDDEN_SIZE])
        out = out * t
        out = paddle.transpose(out, [1, 0, 2])
        return out


class EmbeddingNet(nn.Layer):
    def __init__(self, mesh):
        super().__init__()
        self.embedding = paddle.nn.Embedding(
            VOCAB_SIZE,
            HIDDEN_SIZE,
            weight_attr=paddle.nn.initializer.Constant(value=0.5),
        )
        self.mesh_ = mesh
        self.embedding.weight = dist.shard_tensor(
            self.embedding.weight,
            mesh,
            [dist.Replicate(), dist.Shard(1)],
            stop_gradient=False,
        )

    def forward(self, x):
        out = self.embedding(x)
        out = out.astype(self.embedding.weight.dtype)
        out = paddle.transpose(out, [1, 0, 2])
        out = dist.reshard(
            out, self.mesh_, [dist.Replicate(), dist.Replicate()]
        )
        t = paddle.randn([SEQ_LEN, BATCH_SIZE, HIDDEN_SIZE])
        out = out * t
        out = paddle.transpose(out, [1, 0, 2])
        return out


class RandomDataset(paddle.io.Dataset):
    def __init__(self, inputs, labels, num_samples):
        self.inputs = inputs
        self.labels = labels
        self.num_samples = num_samples

    def __getitem__(self, idx):
        return self.inputs[idx], self.labels[idx]

    def __len__(self):
        return self.num_samples


class TestSimpleNetForSemiAutoParallel:
    def __init__(self):
        self._seed = eval(os.getenv("seed"))
        self.mesh = dist.ProcessMesh([[0, 1]])
        strategy = fleet.DistributedStrategy()
        strategy.hybrid_configs = {
            "dp_degree": 1,
            "mp_degree": 2,
            "pp_degree": 1,
        }
        fleet.init(is_collective=True, strategy=strategy)

    def set_random_seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        paddle.seed(seed)

    def create_data_loader(self):
        inputs = np.random.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN))
        labels = np.random.rand(BATCH_SIZE, SEQ_LEN, HIDDEN_SIZE).astype(
            'float32'
        )
        dataset = RandomDataset(inputs, labels, BATCH_SIZE)
        loader = DataLoader(dataset, batch_size=BATCH_SIZE)
        return loader

    def run_dy2static(self, layer, opt, dist_loader, use_pass):
        loss_fn = nn.MSELoss()
        strategy = dist.Strategy()
        strategy._mp_optimization.replace_with_c_embedding = use_pass
        dist_model = dist.to_static(
            layer, dist_loader, loss_fn, opt, strategy=strategy
        )
        loss_list = []
        dist_model._engine._mode = "train"
        dist_model.train()
        dist_program = dist_model._engine._pir_dist_main_progs["train"]
        op_name = dist_program.global_block().ops[8].name()
        expected_op = 'pd_op.c_embedding' if use_pass else 'pd_op.embedding'
        np.testing.assert_equal(op_name, expected_op)
        for epoch in range(3):
            for batch_id, data in enumerate(dist_loader()):
                x, label = data
                loss = dist_model(x, label)
                loss_list.append(loss)
        return np.array(loss_list), dist_model

    def run_dynamic(self, layer, opt, dist_loader):
        loss_fn = nn.MSELoss()
        loss_list = []
        for epoch in range(3):
            for batch_id, data in enumerate(dist_loader()):
                x, label = data
                out = layer(x)
                loss = loss_fn(out, label)
                loss_list.append(loss.numpy())
                loss.backward()
                opt.step()
                opt.clear_grad()
        return np.array(loss_list)

    def test_mp_demo_net(self):
        paddle.disable_static()
        paddle.base.set_flags({'FLAGS_enable_pir_api': 1})
        self.set_random_seed(self._seed)
        data_loader = self.create_data_loader()
        dist_dataloader = dist.shard_dataloader(
            dataloader=data_loader,
            meshes=[self.mesh],
        )
        self.set_random_seed(self._seed)
        dy2static_layer_use_pass = EmbeddingNet(self.mesh)
        dy2static_opt_use_pass = paddle.optimizer.AdamW(
            learning_rate=0.1, parameters=dy2static_layer_use_pass.parameters()
        )
        loss_pass, dist_model_use_pass = self.run_dy2static(
            dy2static_layer_use_pass,
            dy2static_opt_use_pass,
            dist_dataloader,
            True,
        )
        self.set_random_seed(self._seed)
        dy2static_layer = EmbeddingNet(self.mesh)
        dy2static_opt = paddle.optimizer.AdamW(
            learning_rate=0.1, parameters=dy2static_layer.parameters()
        )
        loss_st, dist_model = self.run_dy2static(
            dy2static_layer, dy2static_opt, dist_dataloader, False
        )
        self.set_random_seed(self._seed)
        dy_layer = CEmbeddingNet(self.mesh)
        dy_opt = paddle.optimizer.AdamW(
            learning_rate=0.1, parameters=dy_layer.parameters()
        )
        loss_dy = self.run_dynamic(dy_layer, dy_opt, data_loader)
        md5_pass = hashlib.md5(loss_pass.tobytes()).hexdigest()
        md5_st = hashlib.md5(loss_st.tobytes()).hexdigest()
        md5_dy = hashlib.md5(loss_dy.tobytes()).hexdigest()
        np.testing.assert_equal(md5_pass, md5_st)
        np.testing.assert_equal(md5_pass, md5_dy)

    def test_c_embedding_with_pir_fp16(self):
        paddle.disable_static()
        data_loader = self.create_data_loader()
        dist_loader = dist.shard_dataloader(
            dataloader=data_loader,
            meshes=[self.mesh],
        )
        paddle.set_default_dtype('float16')
        layer = EmbeddingNet(self.mesh)
        paddle.set_default_dtype('float32')
        opt = paddle.optimizer.AdamW(
            learning_rate=0.1, parameters=layer.parameters()
        )
        loss_fn = nn.MSELoss()
        strategy = dist.Strategy()
        strategy._mp_optimization.replace_with_c_embedding = True
        dist_model = dist.to_static(
            layer, dist_loader, loss_fn, opt, strategy=strategy
        )
        dist_model._engine._mode = "train"
        dist_model.train()
        dist_program = dist_model._engine._pir_dist_main_progs["train"]
        # check the dtype of c_embedding_grad is float16, consistent with c_embedding.
        op_check = dist_program.global_block().ops[-5]
        np.testing.assert_equal(op_check.name(), "pd_op.c_embedding_grad")
        np.testing.assert_equal(op_check.result(0).dtype.name, "FLOAT16")

    def run_test_case(self):
        self.test_mp_demo_net()
        self.test_c_embedding_with_pir_fp16()


if __name__ == '__main__':
    TestSimpleNetForSemiAutoParallel().run_test_case()
