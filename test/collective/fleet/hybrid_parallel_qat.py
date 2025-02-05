# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
import unittest

import numpy as np

import paddle
import paddle.distributed as dist
from paddle import base, nn
from paddle.distributed import fleet
from paddle.distributed.utils.launch_utils import find_free_ports, get_cluster
from paddle.quantization import ImperativeQuantAware


def set_random_seed(seed, dp_id, rank_id):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed + dp_id)
    paddle.seed(seed + rank_id)


vocab_size = 20
hidden_size = 10
inner_size = 8
output_size = 10
seq_length = 2
batch_size = 4


def get_attr(layer, name):
    if getattr(layer, name, None) is not None:
        return getattr(layer, name, None)
    else:
        return get_attr(layer._layer, name)


def get_gpus(selected_gpus):
    selected_gpus = [x.strip() for x in selected_gpus.split(',')]
    return selected_gpus


def get_cluster_from_args(selected_gpus):
    cluster_node_ips = '127.0.0.1'
    node_ip = '127.0.0.1'

    node_ips = [x.strip() for x in cluster_node_ips.split(',')]

    node_ips.index(node_ip)

    free_ports = None

    free_ports = find_free_ports(len(selected_gpus))
    if free_ports is not None:
        free_ports = list(free_ports)

    trainer_endpoints = []
    for ip in node_ips:
        trainer_endpoints.append([f"{ip}:{port}" for port in free_ports])
    return get_cluster(node_ips, node_ip, trainer_endpoints, selected_gpus)


def parallel_matmul(lm_output, logit_weights, parallel_output):
    hcg = fleet.get_hybrid_communicate_group()
    model_parallel_group = hcg.get_model_parallel_group()
    world_size = hcg.get_model_parallel_world_size()
    rank = hcg.get_model_parallel_rank()

    if world_size > 1:
        input_parallel = paddle.distributed.collective._c_identity(
            lm_output, group=model_parallel_group
        )

        logits = paddle.matmul(input_parallel, logit_weights, transpose_y=True)

        if parallel_output:
            return logits

        return paddle.distributed.collective._c_concat(
            logits, group=model_parallel_group
        )
    else:
        logits = paddle.matmul(lm_output, logit_weights, transpose_y=True)
        return logits


class PACT(nn.Layer):
    def __init__(self, init_value=20):
        super().__init__()
        alpha_attr = paddle.ParamAttr(
            name=self.full_name() + ".pact",
            initializer=paddle.nn.initializer.Constant(value=init_value),
        )
        self.alpha = self.create_parameter(
            shape=[1], attr=alpha_attr, dtype='float32'
        )

    def forward(self, x):
        out_left = paddle.nn.functional.relu(x - self.alpha)
        out_right = paddle.nn.functional.relu(-self.alpha - x)
        x = x - out_left + out_right
        return x


class SimpleMPNet(nn.Layer):
    def __init__(
        self,
        vocab_size,
        hidden_size,
        inner_size,
        output_size,
        np_fc1,
        np_fc2,
        mp_id,
    ):
        super().__init__()

        if mp_id == 0:
            init_fc1_data = np_fc1[:, : (inner_size // 2)]
            init_fc2_data = np_fc2[: (inner_size // 2), :]
        else:
            init_fc1_data = np_fc1[:, (inner_size // 2) :]
            init_fc2_data = np_fc2[(inner_size // 2) :, :]

        self.linear1 = fleet.meta_parallel.ColumnParallelLinear(
            hidden_size,
            inner_size,
            weight_attr=paddle.framework.ParamAttr(
                initializer=paddle.nn.initializer.Assign(init_fc1_data)
            ),
            gather_output=False,
            has_bias=True,
        )

        self.linear2 = fleet.meta_parallel.RowParallelLinear(
            inner_size,
            hidden_size,
            weight_attr=paddle.framework.ParamAttr(
                initializer=paddle.nn.initializer.Assign(init_fc2_data)
            ),
            input_is_parallel=True,
            has_bias=True,
        )

        self.linear3 = paddle.nn.Linear(
            hidden_size,
            output_size,
            weight_attr=paddle.framework.ParamAttr(
                initializer=paddle.nn.initializer.Constant(0.0)
            ),
            bias_attr=paddle.framework.ParamAttr(
                initializer=paddle.nn.initializer.Constant(0.0)
            ),
        )

        self.embedding = fleet.meta_parallel.VocabParallelEmbedding(
            vocab_size,
            hidden_size,
            weight_attr=paddle.nn.initializer.Constant(value=1.0),
        )

    def forward(self, x):
        x = self.embedding(x)
        x = self.linear1(x)
        x = self.linear2(x)
        x = self.linear3(x)
        x = parallel_matmul(x, get_attr(self.embedding, "weight"), False)
        return x


class SimpleDPNet(nn.Layer):
    def __init__(
        self, vocab_size, hidden_size, inner_size, output_size, np_fc1, np_fc2
    ):
        super().__init__()
        self.linear1 = paddle.nn.Linear(
            hidden_size,
            inner_size,
            weight_attr=paddle.framework.ParamAttr(
                initializer=paddle.nn.initializer.Assign(np_fc1)
            ),
            bias_attr=paddle.framework.ParamAttr(
                initializer=paddle.nn.initializer.Constant(0.0)
            ),
        )

        self.linear2 = paddle.nn.Linear(
            inner_size,
            hidden_size,
            weight_attr=paddle.framework.ParamAttr(
                initializer=paddle.nn.initializer.Assign(np_fc2)
            ),
            bias_attr=paddle.framework.ParamAttr(
                initializer=paddle.nn.initializer.Constant(0.0)
            ),
        )

        self.linear3 = paddle.nn.Linear(
            hidden_size,
            output_size,
            weight_attr=paddle.framework.ParamAttr(
                initializer=paddle.nn.initializer.Constant(0.0)
            ),
            bias_attr=paddle.framework.ParamAttr(
                initializer=paddle.nn.initializer.Constant(0.0)
            ),
        )

        self.embedding = paddle.nn.Embedding(
            vocab_size,
            hidden_size,
            weight_attr=paddle.nn.initializer.Constant(value=1.0),
        )

    def forward(self, x):
        x = self.embedding(x)
        x = self.linear1(x)
        x = self.linear2(x)
        x = self.linear3(x)
        x = paddle.matmul(
            x, get_attr(self.embedding, "weight"), transpose_y=True
        )
        return x


class TestDistMPTraining(unittest.TestCase):
    def setUp(self):
        strategy = fleet.DistributedStrategy()
        self.model_parallel_size = 2
        self.data_parallel_size = 1
        strategy.hybrid_configs = {
            "dp_degree": self.data_parallel_size,
            "mp_degree": self.model_parallel_size,
            "pp_degree": 1,
        }
        fleet.init(is_collective=True, strategy=strategy)
        self.onnx_format = False
        self.check_export_model_accuracy = True
        self.diff_threshold = 0.01
        self.fuse_conv_bn = False

    def train_batch(self, batch, model, optimizer, is_mp):
        output = model(batch)
        loss = output.mean()
        loss.backward()  # do backward
        optimizer.step()  # update parameters
        optimizer.clear_grad()
        return loss

    def build_optimizer(self, model):
        optimizer = paddle.optimizer.SGD(
            learning_rate=0.001, parameters=model.parameters()
        )
        return optimizer

    def build_model_optimizer(
        self, weight_quantize_type, activation_quantize_type, use_pact=False
    ):
        hcg = fleet.get_hybrid_communicate_group()
        word_size = hcg.get_model_parallel_world_size()
        mp_id = hcg.get_model_parallel_rank()
        dp_id = hcg.get_data_parallel_rank()
        rank_id = dist.get_rank()
        imperative_qat = ImperativeQuantAware(
            weight_quantize_type=weight_quantize_type,
            activation_quantize_type=activation_quantize_type,
            fuse_conv_bn=self.fuse_conv_bn,
            act_preprocess_layer=PACT if use_pact else None,
        )
        set_random_seed(1024, dp_id, rank_id)

        np_fc1 = np.ones((hidden_size, inner_size))
        np_fc2 = np.ones(
            (inner_size, hidden_size)
        )  # np.random.random_sample((inner_size, hidden_size))

        model_a = SimpleMPNet(
            vocab_size,
            hidden_size,
            inner_size,
            output_size,
            np_fc1,
            np_fc2,
            mp_id,
        )
        model_a = imperative_qat.quantize(model_a)
        optimizer_a = self.build_optimizer(model_a)
        model_a = fleet.distributed_model(model_a)
        optimizer_a = fleet.distributed_optimizer(optimizer_a)

        model_b = SimpleDPNet(
            vocab_size, hidden_size, inner_size, output_size, np_fc1, np_fc2
        )
        model_b = imperative_qat.quantize(model_b)
        optimizer_b = self.build_optimizer(model_b)

        return model_a, optimizer_a, model_b, optimizer_b

    def train(self, model_a, optimizer_a, model_b, optimizer_b):
        for epoch in range(5):
            np_data = np.random.randint(
                0,
                vocab_size,
                (
                    batch_size,
                    seq_length,
                ),
            )
            batch = paddle.to_tensor(np_data)
            loss_a = self.train_batch(batch, model_a, optimizer_a, True)
            loss_b = self.train_batch(batch, model_b, optimizer_b, False)

            np.testing.assert_allclose(
                loss_a.numpy(), loss_b.numpy(), rtol=1e-6
            )

    def test_mp_model_1(self):
        if (
            not base.core.is_compiled_with_cuda()
            or base.core.get_cuda_device_count() == 0
        ):
            return
        selected_gpus = get_gpus('0,1')
        cluster = None
        pod = None

        model_a, optimizer_a, model_b, optimizer_b = self.build_model_optimizer(
            weight_quantize_type='abs_max',
            activation_quantize_type='moving_average_abs_max',
        )
        self.train(model_a, optimizer_a, model_b, optimizer_b)

    def test_mp_model_2(self):
        if (
            not base.core.is_compiled_with_cuda()
            or base.core.get_cuda_device_count() == 0
        ):
            return
        selected_gpus = get_gpus('0,1')
        cluster = None
        pod = None

        model_a, optimizer_a, model_b, optimizer_b = self.build_model_optimizer(
            weight_quantize_type='channel_wise_abs_max',
            activation_quantize_type='moving_average_abs_max',
            use_pact=True,
        )
        self.train(model_a, optimizer_a, model_b, optimizer_b)


if __name__ == "__main__":
    unittest.main()
