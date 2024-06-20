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

import os
import random

import numpy as np

import paddle
import paddle.distributed as dist
import paddle.nn.functional as F
from paddle import nn
from paddle.distributed import fleet
from paddle.distributed.fleet.meta_parallel import LayerDesc, PipelineLayer
from paddle.nn import Layer


def set_random_seed(seed, dp_id, rank_id):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed + dp_id)
    paddle.seed(seed + dp_id)


batch_size = 24
length = 8
micro_batch_size = 4
num_virtual_pipeline_stages = 2
vocab_size = 128
hidden_size = 16
d_model = hidden_size
dim_feedforward = 4 * d_model


class EmbeddingNet(Layer):
    def __init__(self):
        super().__init__()
        self.word_embeddings = nn.Embedding(vocab_size, hidden_size)
        self.position_embeddings = nn.Embedding(vocab_size, hidden_size)

    def forward(self, x):
        attention_mask = paddle.tensor.triu(
            (paddle.ones((length, length), dtype="float32") * -1e9), 1
        )

        no_used = paddle.ones((3, 3), dtype="int32")

        w_emb = self.word_embeddings(x)
        p_emb = self.position_embeddings(x)
        w_emb = w_emb + p_emb

        attention_mask.stop_gradient = True
        no_used.stop_gradient = True
        return w_emb, attention_mask, no_used, p_emb


class TransformerNet(Layer):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)

        self.norm1 = nn.LayerNorm(d_model, epsilon=1e-5)

    def forward(self, x, mask):
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        product = paddle.matmul(x=q, y=k, transpose_y=True)
        product = paddle.scale(product, scale=d_model**-0.5)

        weights = F.softmax(product + mask)
        tgt = paddle.matmul(weights, v)
        residual = tgt
        tgt = self.norm1(tgt)
        tgt = residual + tgt

        out = self.linear2(F.gelu(self.linear1(tgt), approximate=True))
        return out


class EmbeddingPipe(EmbeddingNet):
    def forward(self, x):
        w_emb, attention_mask, no_used, p_emb = super().forward(x)
        ret = {
            'x': w_emb,
            'mask': attention_mask,
            'no_used': no_used,
            'p_emb': p_emb,
        }
        return ret


class TransformerNetPipe(TransformerNet):
    def forward(self, kwargs):
        x = kwargs['x']
        mask = kwargs['mask']
        no_used = kwargs['no_used']
        p_emb = kwargs['p_emb']

        output = super().forward(x, mask)
        output = output + p_emb
        mask.stop_gradient = True
        p_emb.stop_gradient = True

        ret = {'x': output, 'mask': mask, 'no_used': no_used, 'p_emb': p_emb}
        return ret


class CriterionPipe(Layer):
    def __init__(self):
        super().__init__()

    def forward(self, out, label):
        loss = out.mean()
        return loss


class SimpleNet(Layer):
    def __init__(self):
        super().__init__()
        self.word_embeddings = nn.Embedding(
            vocab_size, hidden_size, name="simple_net_word_embeddings"
        )

        self.softmax_weight = self.create_parameter(
            shape=[hidden_size, vocab_size],
            attr=paddle.ParamAttr(name="simple_net_softmax_weight"),
        )
        self.softmax_bias = self.create_parameter(
            shape=[vocab_size],
            is_bias=False,
            attr=paddle.ParamAttr(name="simple_net_softmax_bias"),
        )

    def forward(self, x1, x2, y1):
        x_emb = self.word_embeddings(x1)
        fc = paddle.matmul(x_emb, self.softmax_weight)
        fc = paddle.add(fc, self.softmax_bias)
        projection = paddle.reshape(fc, shape=[-1, vocab_size])

        loss = paddle.nn.functional.softmax_with_cross_entropy(
            logits=projection, label=y1, soft_label=False
        )
        return loss.mean()


class EmbeddingNet_V2(Layer):
    def __init__(self):
        super().__init__()
        self.word_embeddings = nn.Embedding(
            vocab_size, hidden_size, name="single_word_embeddings"
        )

    @property
    def embedding_weight(self):
        return self.word_embeddings.weight

    def forward(self, args):
        x1, x2 = args
        x_emb = self.word_embeddings(x1)
        x2.stop_gradient = True
        ret = {
            "x1": x_emb,
            "x2": x2,
        }
        return ret


class MatmulNet(Layer):
    def __init__(self):
        super().__init__()
        self.softmax_weight = self.create_parameter(
            shape=[hidden_size, vocab_size],
            attr=paddle.ParamAttr(name="single_softmax_weight"),
        )

    def forward(self, kwargs):
        x1, x2 = kwargs["x1"], kwargs["x2"]
        fc = paddle.matmul(x1, self.softmax_weight)
        ret = {"fc": fc, "x2": x2}
        return ret


class BiasNet(Layer):
    def __init__(self):
        super().__init__()
        self.softmax_bias = self.create_parameter(
            shape=[vocab_size],
            is_bias=False,
            attr=paddle.ParamAttr(name="single_softmax_bias"),
        )

    def forward(self, kwargs):
        fc, x2 = kwargs["fc"], kwargs["x2"]
        fc = paddle.add(fc, self.softmax_bias)
        projection = paddle.reshape(fc, shape=[-1, vocab_size])
        return projection, x2


class LossNet(Layer):
    def __init__(self):
        super().__init__()

    def forward(self, args, y1):
        projection, x2 = args
        loss = paddle.nn.functional.softmax_with_cross_entropy(
            logits=projection, label=y1[0], soft_label=False
        )
        return loss.mean()


class SimpleNetPipe(PipelineLayer):
    def __init__(self, **kwargs):
        self.descs = []
        self.descs.append(LayerDesc(EmbeddingNet_V2))
        self.descs.append(LayerDesc(MatmulNet))
        self.descs.append(LayerDesc(BiasNet))

        super().__init__(layers=self.descs, loss_fn=LossNet(), **kwargs)


class ModelPipe(PipelineLayer):
    def __init__(self, topology, transformer_layer_num: int = 6):
        self.descs = []
        self.descs.append(LayerDesc(EmbeddingPipe))

        for x in range(transformer_layer_num):
            self.descs.append(LayerDesc(TransformerNetPipe))

        self.descs.append(lambda ret_dict: ret_dict['x'])

        super().__init__(
            layers=self.descs,
            loss_fn=CriterionPipe(),
            topology=topology,
            seg_method="layer:TransformerNetPipe",
        )


class ModelPipeWithInterleave(PipelineLayer):
    def __init__(self, topology, transformer_layer_num: int = 6):
        self.descs = []
        self.descs.append(LayerDesc(EmbeddingPipe))

        for x in range(transformer_layer_num):
            self.descs.append(LayerDesc(TransformerNetPipe))

        self.descs.append(lambda ret_dict: ret_dict['x'])

        super().__init__(
            layers=self.descs,
            loss_fn=CriterionPipe(),
            topology=topology,
            num_virtual_pipeline_stages=num_virtual_pipeline_stages,
            seg_method="layer:TransformerNetPipe",
        )


class TestDistPPTraining:
    def __init__(self):
        self._backend = os.getenv("backend")
        if self._backend not in ["nccl", "gloo"]:
            raise NotImplementedError(
                "Only support nccl and gloo as the backend for now."
            )
        os.environ["PADDLE_DISTRI_BACKEND"] = self._backend

        self.model_parallel_size = 1
        self.data_parallel_size = 1
        self.pipeline_parallel_size = 3

        strategy = fleet.DistributedStrategy()
        strategy.hybrid_configs = {
            "dp_degree": self.data_parallel_size,
            "mp_degree": self.model_parallel_size,
            "pp_degree": self.pipeline_parallel_size,
        }
        strategy.pipeline_configs = {
            "accumulate_steps": batch_size // micro_batch_size,
            "micro_batch_size": micro_batch_size,
        }
        fleet.init(is_collective=True, strategy=strategy)

    def run_test_cases(self):
        self.test_pp_model()
        self.test_pp_model_with_interleaved()
        self.test_pp_model_backward()

    def test_pp_model(self):
        hcg = fleet.get_hybrid_communicate_group()
        word_size = hcg.get_model_parallel_world_size()
        dp_id = hcg.get_data_parallel_rank()
        pp_id = hcg.get_stage_id()
        rank_id = dist.get_rank()
        topology = hcg.topology()
        set_random_seed(1024, dp_id, rank_id)

        model = ModelPipe(topology)
        scheduler = paddle.optimizer.lr.PiecewiseDecay(
            boundaries=[2], values=[0.001, 0.002], verbose=True
        )
        optimizer = paddle.optimizer.SGD(
            learning_rate=scheduler, parameters=model.parameters()
        )

        model = fleet.distributed_model(model)
        optimizer = fleet.distributed_optimizer(optimizer)

        for step_id in range(5):
            x_data = np.random.randint(0, vocab_size, size=[batch_size, length])
            x = paddle.to_tensor(x_data)
            x.stop_gradient = True

            e_loss = model.eval_batch([x, x], True)
            loss = model.train_batch([x, x], optimizer, scheduler)

            if pp_id != 0:
                np.testing.assert_allclose(loss.numpy(), e_loss.numpy())

    def test_pp_model_with_interleaved(self):
        hcg = fleet.get_hybrid_communicate_group()
        word_size = hcg.get_model_parallel_world_size()
        dp_id = hcg.get_data_parallel_rank()
        pp_id = hcg.get_stage_id()
        rank_id = dist.get_rank()
        topology = hcg.topology()
        set_random_seed(1024, dp_id, rank_id)

        model = ModelPipeWithInterleave(topology)
        scheduler = paddle.optimizer.lr.PiecewiseDecay(
            boundaries=[2], values=[0.001, 0.002], verbose=True
        )
        optimizer = paddle.optimizer.SGD(
            learning_rate=scheduler, parameters=model.parameters()
        )

        model = fleet.distributed_model(model)
        optimizer = fleet.distributed_optimizer(optimizer)

        for step_id in range(5):
            x_data = np.random.randint(0, vocab_size, size=[batch_size, length])
            x = paddle.to_tensor(x_data)
            x.stop_gradient = True

            e_loss = model.eval_batch([x, x], True)
            loss = model.train_batch([x, x], optimizer, scheduler)

            if pp_id != 0:
                np.testing.assert_allclose(loss.numpy(), e_loss.numpy())

    def test_pp_model_backward(self):
        hcg = fleet.get_hybrid_communicate_group()
        word_size = hcg.get_model_parallel_world_size()
        dp_id = hcg.get_data_parallel_rank()
        pp_id = hcg.get_stage_id()
        rank_id = dist.get_rank()
        set_random_seed(1024, dp_id, rank_id)

        # construct model a
        model_a = SimpleNet()
        scheduler_a = paddle.optimizer.lr.PiecewiseDecay(
            boundaries=[2, 3, 4], values=[0.01, 0.02, 0.03, 0.04], verbose=True
        )
        optimizer_a = paddle.optimizer.SGD(
            learning_rate=scheduler_a, parameters=model_a.parameters()
        )

        model_b = SimpleNetPipe(topology=hcg.topology())

        scheduler_b = paddle.optimizer.lr.PiecewiseDecay(
            boundaries=[2, 3, 4], values=[0.01, 0.02, 0.03, 0.04], verbose=True
        )
        optimizer_b = paddle.optimizer.SGD(
            learning_rate=scheduler_b, parameters=model_b.parameters()
        )
        model_b = fleet.distributed_model(model_b)
        optimizer_b = fleet.distributed_optimizer(optimizer_b)

        param_len = len(model_a.parameters())

        parameters = []
        for param in model_a.parameters():
            parameters.append(param.numpy())

        model_b_params = model_b.parameters()

        def _get_matched_parameters(model_params, substr):
            for param in model_params:
                if substr in param.name:
                    return param.numpy()

        if pp_id == 0:
            model_b_params[0].set_value(
                _get_matched_parameters(model_a.parameters(), "embedding")
            )
        elif pp_id == 1:
            # single_softmax_weight <----> simple_net_softmax_weight
            model_b_params[0].set_value(
                _get_matched_parameters(model_a.parameters(), "softmax_weight")
            )
        else:
            # single_softmax_bias <----> simple_net_softmax_bias
            model_b_params[0].set_value(
                _get_matched_parameters(model_a.parameters(), "softmax_bias")
            )

        for step in range(5):
            x1_data = np.random.randint(0, vocab_size, size=[batch_size, 1])
            x2_data = np.random.randint(0, vocab_size, size=[batch_size, 1])
            y1_data = np.random.randint(0, hidden_size, size=[batch_size, 1])

            x1 = paddle.to_tensor(x1_data)
            x2 = paddle.to_tensor(x2_data)
            y1 = paddle.to_tensor(y1_data)

            x1.stop_gradient = True
            x2.stop_gradient = True
            y1.stop_gradient = True

            loss_a = model_a(x1, x2, y1)
            loss_a.backward()

            optimizer_a.step()
            optimizer_a.clear_grad()
            scheduler_a.step()

            loss_b = model_b.train_batch(
                [(x1, x2), (y1,)], optimizer_b, scheduler_b
            )
            np.testing.assert_allclose(
                loss_a.numpy(), loss_b.numpy(), rtol=1e-6, atol=1e-6
            )


if __name__ == "__main__":
    testcases = TestDistPPTraining()
    testcases.run_test_cases()
