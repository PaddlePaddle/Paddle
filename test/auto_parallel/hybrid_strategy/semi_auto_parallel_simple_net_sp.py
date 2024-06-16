# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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
from auto_parallel.semi_auto_parallel_simple_net import (
    TestSimpleNetForSemiAutoParallel,
    create_numpy_like_random,
)

import paddle
import paddle.distributed as dist
from paddle import nn
from paddle.distributed import Replicate, Shard

BATCH_SIZE = 8
SEQUENCE_LEN = 512
HIDDEN_SIZE = 1024
NUM_HEAD = 64
HEAD_DIM = 16
CLASS_NUM = 10

np.set_printoptions(threshold=np.inf)


class DemoNet(nn.Layer):
    def __init__(self, param_prefix="", is_sp=False, is_dp=False):
        super().__init__()

        if is_dp:
            self.pp0_mesh = dist.ProcessMesh(
                [[0, 1], [2, 3]], dim_names=["dp", "mp"]
            )
            self.pp1_mesh = dist.ProcessMesh(
                [[4, 5], [6, 7]], dim_names=["dp", "mp"]
            )
            self.placement0 = [Replicate(), Shard(1)]
            self.placement1 = [Replicate(), Shard(0)]
            self.sp_reshard_placement0 = [Shard(1), Shard(0)]
            self.sp_reshard_placement1 = [Shard(1), Replicate()]
        else:
            self.pp0_mesh = dist.ProcessMesh([0, 1], dim_names=["mp"])
            self.pp1_mesh = dist.ProcessMesh([2, 3], dim_names=["mp"])
            self.placement0 = [Shard(1)]
            self.placement1 = [Shard(0)]
            self.sp_reshard_placement0 = [Shard(0)]
            self.sp_reshard_placement1 = [Replicate()]

        self.is_sp = is_sp
        self.is_dp = is_dp

        self.norm = nn.LayerNorm(HIDDEN_SIZE, epsilon=1e-4)
        self.linear_0_weight = dist.shard_tensor(
            self.create_parameter(
                shape=[HEAD_DIM, 4 * HIDDEN_SIZE],
                attr=create_numpy_like_random(param_prefix + "w_0"),
                dtype=paddle.float32,
                is_bias=False,
            ),
            self.pp0_mesh,
            self.placement0,
        )

        self.linear_1_weight = dist.shard_tensor(
            self.create_parameter(
                shape=[4 * HIDDEN_SIZE, HEAD_DIM],
                attr=create_numpy_like_random(param_prefix + "w_1"),
                dtype=paddle.float32,
                is_bias=False,
            ),
            self.pp0_mesh,
            self.placement1,
        )

        self.linear_2_weight = dist.shard_tensor(
            self.create_parameter(
                shape=[HIDDEN_SIZE, 4 * HIDDEN_SIZE],
                attr=create_numpy_like_random(param_prefix + "w_2"),
                dtype=paddle.float32,
                is_bias=False,
            ),
            self.pp1_mesh,
            self.placement0,
        )

        self.linear_3_weight = dist.shard_tensor(
            self.create_parameter(
                shape=[4 * HIDDEN_SIZE, CLASS_NUM],
                attr=create_numpy_like_random(param_prefix + "w_3"),
                dtype=paddle.float32,
                is_bias=False,
            ),
            self.pp1_mesh,
            self.placement1,
        )

    def forward(self, x):
        # Layer 0
        tgt = paddle.transpose(x, [1, 0, 2])
        out = paddle.reshape(x, [BATCH_SIZE, SEQUENCE_LEN, NUM_HEAD, HEAD_DIM])
        # [BATCH_SIZE, SEQUENCE_LEN, NUM_HEAD, HEAD_DIM] -> [BATCH_SIZE, NUM_HEAD, SEQUENCE_LEN, HEAD_DIM]
        out = paddle.transpose(out, [0, 2, 1, 3])
        out = paddle.matmul(out, self.linear_0_weight)
        out = paddle.matmul(out, self.linear_1_weight)
        out = paddle.transpose(out, [2, 0, 1, 3])
        out = paddle.reshape(out, [SEQUENCE_LEN, BATCH_SIZE, HIDDEN_SIZE])

        # SP Region, should be reduce_scatter
        if self.is_sp:
            out = dist.reshard(out, self.pp0_mesh, self.sp_reshard_placement0)

        # out = out + tgt
        out = self.norm(out)

        out = dist.reshard(out, self.pp1_mesh, self.sp_reshard_placement1)

        out = paddle.matmul(out, self.linear_2_weight)
        out = paddle.matmul(out, self.linear_3_weight)
        out = paddle.transpose(out, [1, 0, 2])

        return out


class TestSimpleNetHybridStrategyForSemiAutoParallel(
    TestSimpleNetForSemiAutoParallel
):
    def __init__(self):
        self._dtype = os.getenv("dtype")
        self._backend = os.getenv("backend")
        self._seed = eval(os.getenv("seed"))
        self._is_dp = os.getenv("is_dp") == "true"
        if self._is_dp:
            self.pp0_mesh = dist.ProcessMesh(
                [[0, 1], [2, 3]], dim_names=["dp", "mp"]
            )

        paddle.set_device(self._backend)

        self.set_random_seed(self._seed)
        self.init_single_card_net_result()

    def init_single_card_net_result(self):
        self.set_random_seed(self._seed)
        self.base_loss, self.base_parameters = self.run_dynamic(
            DemoNet("demo_weight", is_sp=False, is_dp=self._is_dp), is_sp=False
        )

    def init_input_data(self):
        image = np.random.randn(BATCH_SIZE, SEQUENCE_LEN, HIDDEN_SIZE).astype(
            'float32'
        )
        label = np.random.randn(BATCH_SIZE, SEQUENCE_LEN, CLASS_NUM).astype(
            'float32'
        )

        return paddle.to_tensor(image), paddle.to_tensor(label)

    def check_tensor_eq(self, a, b, rtol=1e-7, atol=0, verbose=True):
        np1 = a.astype("float32").numpy()
        np2 = b.astype("float32").numpy()
        np.testing.assert_allclose(
            np1, np2, rtol=rtol, atol=atol, verbose=verbose
        )

    def run_dynamic(self, layer, is_sp=False):
        # create loss
        loss_fn = nn.MSELoss()
        # run forward and backward
        opt = paddle.optimizer.AdamW(
            learning_rate=0.001, parameters=layer.parameters()
        )
        for _ in range(5):
            image, label = self.init_input_data()
            if self._is_dp:
                image = dist.shard_tensor(
                    image, self.pp0_mesh, [Shard(0), Replicate()]
                )

            out = layer(image)

            loss = loss_fn(out, label)
            loss.backward()
            opt.step()
        return loss, layer.parameters()

    def test_dp_mp_sp_demo_net(self):
        self.set_random_seed(self._seed)
        model = DemoNet("dp_mp_hybrid_strategy", is_sp=True, is_dp=self._is_dp)

        (
            self.dp_mp_sp_loss,
            self.dp_mp_sp_parameters,
        ) = self.run_dynamic(model, is_sp=True)
        if dist.get_rank() in model.pp1_mesh.process_ids:
            self.check_tensor_eq(self.dp_mp_sp_loss, self.base_loss, rtol=1e-3)

    def run_test_case(self):
        self.test_dp_mp_sp_demo_net()


if __name__ == '__main__':
    TestSimpleNetHybridStrategyForSemiAutoParallel().run_test_case()
