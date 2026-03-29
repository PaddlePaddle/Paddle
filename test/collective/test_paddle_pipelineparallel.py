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

import random
import unittest

import numpy as np

import paddle
import paddle.distributed as dist
import paddle.nn.functional as F
from paddle import nn
from paddle.distributed import fleet
from paddle.distributed.fleet.meta_parallel import LayerDesc, PipelineLayer
from paddle.distributed.utils.nccl_utils import check_nccl_version_for_bf16
from paddle.nn import Layer


class ReshapeHelp(Layer):
    def __init__(self, shape):
        super().__init__()
        self.shape = shape

    def forward(self, x):
        return x.reshape(shape=self.shape)


class AlexNetPipeDesc(PipelineLayer):
    def __init__(self, num_classes=10, **kwargs):
        self.num_classes = num_classes
        decs = [
            LayerDesc(nn.Conv2D, 1, 64, kernel_size=11, stride=4, padding=5),
            LayerDesc(nn.ReLU),
            LayerDesc(nn.MaxPool2D, kernel_size=2, stride=2),
            LayerDesc(nn.Conv2D, 64, 192, kernel_size=5, padding=2),
            F.relu,
            LayerDesc(nn.MaxPool2D, kernel_size=2, stride=2),
            LayerDesc(nn.Conv2D, 192, 384, kernel_size=3, padding=1),
            F.relu,
            LayerDesc(nn.Conv2D, 384, 256, kernel_size=3, padding=1),
            F.relu,
            LayerDesc(nn.Conv2D, 256, 256, kernel_size=3, padding=1),
            F.relu,
            LayerDesc(nn.MaxPool2D, kernel_size=2, stride=2),
            LayerDesc(ReshapeHelp, shape=[-1, 256]),
            LayerDesc(nn.Linear, 256, self.num_classes),  # classifier
        ]
        super().__init__(layers=decs, loss_fn=nn.CrossEntropyLoss(), **kwargs)


def set_random_seed(seed, dp_id, rank_id):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed + dp_id)
    paddle.seed(seed + dp_id)


batch_size = 4
micro_batch_size = 2


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
        fleet.init(is_collective=True, strategy=strategy)

    def test_pp_model(self):
        def forward_step_func(model, input):
            rank = dist.get_rank()
            output = model.forward_function(0, len(model.run_function))(input)

            def loss_func(output_tensor, labels):
                loss_fn = nn.CrossEntropyLoss()
                output = loss_fn(output_tensor, labels)
                return output, None

            return output, loss_func

        hcg = fleet.get_hybrid_communicate_group()
        word_size = hcg.get_model_parallel_world_size()
        dp_id = hcg.get_data_parallel_rank()
        pp_id = hcg.get_stage_id()
        rank_id = dist.get_rank()
        set_random_seed(1024, dp_id, rank_id)

        grad_clip = paddle.nn.ClipGradByGlobalNorm(1.0)

        # construct model a
        model_a = AlexNetPipeDesc(num_stages=self.pipeline_parallel_size)
        scheduler_a = paddle.optimizer.lr.PiecewiseDecay(
            boundaries=[2], values=[0.001, 0.002], verbose=True
        )
        optimizer_a = paddle.optimizer.SGD(
            learning_rate=scheduler_a,
            grad_clip=grad_clip,
            parameters=model_a.parameters(),
        )

        model_a, optimizer_a = paddle.amp.decorate(
            models=model_a,
            optimizers=optimizer_a,
            level='O2',
            dtype='bfloat16',
            save_dtype='float32',
        )

        model_a = fleet.distributed_model(
            model_a, forward_func=forward_step_func
        )
        optimizer_a = fleet.distributed_optimizer(optimizer_a)
        scaler_a = paddle.amp.GradScaler(
            init_loss_scaling=1, use_dynamic_loss_scaling=False
        )
        scaler_a = fleet.distributed_scaler(scaler_a)
        train_reader = paddle.batch(
            paddle.dataset.mnist.train(), batch_size=batch_size, drop_last=True
        )

        for step_id, data in enumerate(train_reader()):
            x_data = (
                np.array([x[0] for x in data])
                .astype('float32')
                .reshape(batch_size, 1, 28, 28)
            )
            y_data = (
                np.array([x[1] for x in data])
                .astype('int64')
                .reshape(batch_size, 1)
            )
            img = paddle.to_tensor(x_data)
            label = paddle.to_tensor(y_data)
            img.stop_gradient = True
            label.stop_gradient = True

            if step_id >= 5:
                return True

            with paddle.amp.auto_cast(
                enable=True,
                dtype='bfloat16',
                level='O2',
                custom_black_list=['softmax_with_cross_entropy'],
            ):
                loss_a = model_a.train_batch(
                    [img, label], optimizer_a, scheduler_a, scaler=scaler_a
                )
            print("loss: ", loss_a.numpy())


if __name__ == "__main__":
    if (
        check_nccl_version_for_bf16()
        and paddle.device.cuda.get_device_properties().major >= 8
    ):
        unittest.main()
