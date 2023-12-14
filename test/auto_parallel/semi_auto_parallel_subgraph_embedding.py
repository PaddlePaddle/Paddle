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

import paddle
import paddle.distributed as dist


class Layer(paddle.nn.Layer):
    def __init__(self, vocab_size, hidden_size):
        super().__init__()
        self.embedding = paddle.nn.Embedding(vocab_size, hidden_size)

    def forward(self, x):
        return self.embedding(x)


class TestEmbeddingSubgraphSemiAutoParallel:
    def __init__(self):
        self._dtype = os.getenv("dtype")
        self._backend = os.getenv("backend")
        self._seed = eval(os.getenv("seed"))
        self._mesh = dist.ProcessMesh([0, 1], dim_names=["x"])
        self._batch_size = 17
        self._seq_length = 23
        self._vocab_size = 48
        self._hidden_size = 16

    def test_dp(self):
        paddle.seed(self._seed)
        np.random.seed(self._seed)
        self._input = np.random.randint(
            0, self._vocab_size, size=(self._batch_size, self._seq_length)
        )
        x = paddle.to_tensor(self._input)
        paddle.seed(self._seed)
        np.random.seed(self._seed)
        layer = Layer(self._vocab_size, self._hidden_size)
        desired_out = layer(x)
        desired_out.backward()
        desired_grad = layer.embedding.weight.grad

        paddle.seed(self._seed)
        np.random.seed(self._seed)
        dist_x = dist.shard_tensor(x, self._mesh, placements=(dist.Shard(0),))
        layer = Layer(self._vocab_size, self._hidden_size)
        actual_out = layer(x)
        actual_out.backward()
        actual_grad = layer.embedding.weight.grad

        np.testing.assert_allclose(actual_out, desired_out, rtol=1e-6, atol=0)
        np.testing.assert_allclose(actual_grad, desired_grad, rtol=1e-6, atol=0)
        # The threshold setting refers to Megatron-LM
        assert (
            np.max(np.abs(actual_out.numpy() - desired_out.numpy())) < 1.0e-12
        ), f'embedding dp forward error. actual: {actual_out}, desired: {desired_out}'
        assert (
            np.max(np.abs(actual_grad.numpy() - desired_grad.numpy())) < 1.0e-12
        ), f'embedding dp backward error. actual: {actual_out}, desired: {desired_out}'

    def test_mp(self):
        paddle.seed(self._seed)
        np.random.seed(self._seed)
        self._input = np.random.randint(
            0, self._vocab_size, size=(self._batch_size, self._seq_length)
        )

        x = paddle.to_tensor(self._input)
        paddle.seed(self._seed)
        np.random.seed(self._seed)
        layer = Layer(self._vocab_size, self._hidden_size)
        desired_out = layer(x)
        desired_out.backward()
        desired_grad = layer.embedding.weight.grad

        paddle.seed(self._seed)
        np.random.seed(self._seed)
        dist_x = dist.shard_tensor(
            x, self._mesh, placements=(dist.Replicate(),)
        )

        def shard_fn(layer_name, layer, process_mesh):
            if 'embedding' in layer_name:
                layer.weight = dist.shard_tensor(
                    layer.weight, process_mesh, (dist.Shard(1),)
                )

        layer = dist.shard_layer(
            Layer(self._vocab_size, self._hidden_size), self._mesh, shard_fn
        )
        actual_out = layer(x)
        actual_out.backward()
        actual_grad = layer.embedding.weight.grad

        # The threshold setting refers to Megatron-LM
        assert (
            np.max(np.abs(actual_out.numpy() - desired_out.numpy())) < 1.0e-12
        ), f'embedding mp forward error. actual: {actual_out}, desired: {desired_out}'
        assert (
            np.max(np.abs(actual_grad.numpy() - desired_grad.numpy())) < 1.0e-12
        ), f'embedding mp backward error. actual: {actual_out}, desired: {desired_out}'

    def run_test_case(self):
        if self._backend == "cpu":
            paddle.set_device("cpu")
        elif self._backend == "gpu":
            paddle.set_device("gpu:" + str(dist.get_rank()))
        else:
            raise ValueError("Only support cpu or gpu backend.")

        self.test_dp()
        self.test_mp()


if __name__ == '__main__':
    TestEmbeddingSubgraphSemiAutoParallel().run_test_case()
