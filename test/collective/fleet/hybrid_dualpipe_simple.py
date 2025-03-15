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
import unittest

import numpy as np

import paddle
import paddle.distributed as dist
from paddle import nn
from paddle.distributed import fleet
from paddle.distributed.fleet.meta_parallel import (
    LayerDesc,
    LocalSharedLayerDesc,
    PipelineLayer,
    ScheduleNode,
)
from paddle.distributed.fleet.utils.mix_precision_utils import (
    MixPrecisionLayer,
    MixPrecisionOptimizer,
)
from paddle.io import DataLoader, Dataset
from paddle.nn import Layer, Sequential


def set_random_seed(seed, dp_id, rank_id):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed + dp_id)
    paddle.seed(seed + dp_id)


batch_size = 10
micro_batch_size = 2


class RandomDataset(Dataset):
    def __init__(self, num_samples):
        self.num_samples = num_samples

    def __getitem__(self, idx):
        image = np.random.random([5, 5]).astype('float32')
        label = np.random.randint(0, 5, (5)).astype('int64')
        return image, label

    def __len__(self):
        return self.num_samples


class SharedLinear(Layer):
    def __init__(self, weight):
        super().__init__()
        self.weight = weight

    def forward(self, input):
        if isinstance(input, list):
            input = input[0]
        return paddle.matmul(input, self.weight)

    def build_schedule_node(self):
        return ScheduleNode(self.forward)


class LinearPipe(nn.Linear):
    def forward(self, input):
        if isinstance(input, list):
            input = input[0]
        return paddle.matmul(input, self.weight)

    def build_schedule_node(self):
        return ScheduleNode(self.forward)


class CrossEntropyLossPipe(nn.loss.CrossEntropyLoss):
    def forward(self, logits, label):
        if isinstance(logits, list):
            logits = logits[0]
        return super().forward(logits, label)

    def build_schedule_node(self):
        return ScheduleNode(self.forward)


class SimpleNet(Layer):
    def __init__(self):
        super().__init__()
        self.features = Sequential(
            nn.Linear(5, 5, bias_attr=False),
            nn.Linear(5, 5, bias_attr=False),
            nn.Linear(5, 5, bias_attr=False),
        )
        self.shared_linear = SharedLinear(self.features[0].weight)
        self.loss_fn = nn.loss.CrossEntropyLoss()

    def forward(self, x, y):
        x = self.features(x)
        x = self.shared_linear(x)
        return self.loss_fn(x, y)


class SimpleNetPipeDesc(PipelineLayer):
    def __init__(self, **kwargs):
        decs = [
            LocalSharedLayerDesc(
                "shared_linear",
                LinearPipe,
                shared_weight_attr="weight",
                in_features=5,
                out_features=5,
                bias_attr=False,
            ),
            LayerDesc(LinearPipe, 5, 5, bias_attr=False),
            LayerDesc(LinearPipe, 5, 5, bias_attr=False),
            LocalSharedLayerDesc(
                "shared_linear", SharedLinear, shared_weight_attr="weight"
            ),
        ]
        super().__init__(layers=decs, loss_fn=CrossEntropyLossPipe(), **kwargs)


class SimpleNetPipeScheduleNodeDesc(SimpleNetPipeDesc):
    def overlapped_forward_backward(
        self,
        forward_chunk,  # the module of the forward chunk
        forward_inputs,
        forward_loss_fn_node,
        backward_chunk,  # the module of the backward chunk, maybe not used
        backward_loss_fn_node,
        backward_input_grads,
        scaler,
    ):
        forward_outputs = forward_chunk.forward(forward_inputs)
        forward_outputs = (
            [forward_outputs]
            if isinstance(forward_outputs, paddle.Tensor)
            else forward_outputs
        )

        if forward_loss_fn_node is not None:
            forward_loss = forward_loss_fn_node.forward(forward_outputs)
        else:
            forward_loss = None

        if backward_loss_fn_node is not None:
            if scaler:
                backward_input_grads = backward_loss_fn_node.backward(
                    scaler=scaler
                )
            else:
                backward_input_grads = backward_loss_fn_node.backward()
        backward_input_grads = backward_chunk.backward(backward_input_grads)
        return forward_outputs, forward_loss, backward_input_grads


class TestDistPPTraining(unittest.TestCase):
    def setUp(self):
        strategy = fleet.DistributedStrategy()
        self.model_parallel_size = 1
        self.data_parallel_size = 1
        self.pipeline_parallel_size = 2
        strategy.hybrid_configs = {
            "dp_degree": self.data_parallel_size,
            "mp_degree": self.model_parallel_size,
            "pp_degree": self.pipeline_parallel_size,
        }
        strategy.pipeline_configs = {
            "accumulate_steps": batch_size // micro_batch_size,
            "micro_batch_size": micro_batch_size,
        }
        strategy.hybrid_configs["pp_configs"].use_dualpipev = True
        fleet.init(is_collective=True, strategy=strategy)

    def build_optimizer(self, model):
        scheduler = paddle.optimizer.lr.PiecewiseDecay(
            boundaries=[2], values=[0.001, 0.002], verbose=True
        )
        optimizer = paddle.optimizer.SGD(
            learning_rate=scheduler, parameters=model.parameters()
        )
        return scheduler, optimizer

    def wrapper_mix_precision(self, model, optimizer):
        return model, optimizer

    def test_pp_model(self):
        hcg = fleet.get_hybrid_communicate_group()
        word_size = hcg.get_pipe_parallel_world_size()
        dp_id = hcg.get_data_parallel_rank()
        pp_id = hcg.get_stage_id()
        rank_id = dist.get_rank()
        set_random_seed(1024, dp_id, rank_id)

        # construct model a
        model_a = SimpleNet()
        scheduler_a, optimizer_a = self.build_optimizer(model_a)

        param_len = len(model_a.parameters())

        parameters = []
        for param in model_a.parameters():
            parameters.append(param.numpy())

        # construct model b
        model_b = SimpleNetPipeDesc(
            num_stages=self.pipeline_parallel_size, use_dualpipev=True
        )
        scheduler_b, optimizer_b = self.build_optimizer(model_b)
        model_b, optimizer_b = self.wrapper_mix_precision(model_b, optimizer_b)
        model_b = fleet.distributed_model(model_b)
        optimizer_b = fleet.distributed_optimizer(optimizer_b)

        # construct model c
        model_c = SimpleNetPipeScheduleNodeDesc(
            num_stages=self.pipeline_parallel_size, use_dualpipev=True
        )
        scheduler_c, optimizer_c = self.build_optimizer(model_c)
        model_c, optimizer_c = self.wrapper_mix_precision(model_c, optimizer_c)
        model_c = fleet.distributed_model(model_c)
        optimizer_c = fleet.distributed_optimizer(optimizer_c)

        if 0 == pp_id:
            model_b.parameters()[0].set_value(parameters[0])
            model_c.parameters()[0].set_value(parameters[0])
        else:
            model_b.parameters()[0].set_value(parameters[1])
            model_b.parameters()[1].set_value(parameters[2])
            model_c.parameters()[0].set_value(parameters[1])
            model_c.parameters()[1].set_value(parameters[2])

        dataset = RandomDataset(5 * batch_size)

        # construct reader
        train_reader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=2,
        )

        for i, (img, label) in enumerate(train_reader()):
            if i >= 5:
                return True

            loss_a = model_a(img, label)
            loss_a.backward()
            optimizer_a.step()
            optimizer_a.clear_grad()
            scheduler_a.step()

            loss_b = model_b.train_batch([img, label], optimizer_b, scheduler_b)

            loss_c = model_c.train_batch([img, label], optimizer_c, scheduler_c)

            print("loss: ", loss_a.numpy(), loss_b.numpy(), loss_c.numpy())
            np.testing.assert_allclose(
                loss_a.numpy(), loss_b.numpy(), rtol=5e-5
            )
            np.testing.assert_equal(loss_b.numpy(), loss_c.numpy())


class TestDistPPDelayScaleLoss(TestDistPPTraining):
    def setUp(self):
        strategy = fleet.DistributedStrategy()
        self.model_parallel_size = 1
        self.data_parallel_size = 1
        self.pipeline_parallel_size = 2
        strategy.hybrid_configs = {
            "dp_degree": self.data_parallel_size,
            "mp_degree": self.model_parallel_size,
            "pp_degree": self.pipeline_parallel_size,
            "pp_configs": {
                "delay_scale_loss": True,
                "enable_timer": True,
            },
        }
        strategy.pipeline_configs = {
            "accumulate_steps": batch_size // micro_batch_size,
            "micro_batch_size": micro_batch_size,
        }
        strategy.hybrid_configs["pp_configs"].use_dualpipev = True
        fleet.init(is_collective=True, strategy=strategy)


class TestDistPPMainGrad(TestDistPPTraining):
    def wrapper_mix_precision(self, model, optimizer):
        model = MixPrecisionLayer(model, dtype="float16")
        optimizer = MixPrecisionOptimizer(optimizer)
        return model._layers, optimizer

    def build_optimizer(self, model):
        scheduler = paddle.optimizer.lr.PiecewiseDecay(
            boundaries=[2], values=[0.001, 0.002], verbose=True
        )
        optimizer = paddle.optimizer.SGD(
            learning_rate=scheduler,
            parameters=model.parameters(),
            grad_clip=paddle.nn.ClipGradByGlobalNorm(clip_norm=1.0),
        )
        return scheduler, optimizer


if __name__ == "__main__":
    unittest.main()
