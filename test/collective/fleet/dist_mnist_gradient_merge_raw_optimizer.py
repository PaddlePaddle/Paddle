# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
import sys

sys.path.append("../../legacy_test")

import numpy as np
from dist_mnist import cnn_model
from test_dist_base import TestDistRunnerBase, runtime_main

import paddle
from paddle import base, nn
from paddle.distributed import fleet


class TestDistMnistGradientMergeRawOptimizer(TestDistRunnerBase):
    def get_model(self, batch_size=2, single_device=False):
        paddle.enable_static()
        paddle.seed(1)
        np.random.seed(1)

        assert base.core.globals()['FLAGS_apply_pass_to_program']
        strategy = fleet.DistributedStrategy()
        build_strategy = paddle.static.BuildStrategy()
        settings = {
            "fuse_relu_depthwise_conv": True,
            "fuse_bn_act_ops": True,
            "fuse_bn_add_act_ops": True,
            "fuse_elewise_add_act_ops": True,
            "fuse_all_optimizer_ops": True,
            "enable_addto": True,
            "enable_inplace": True,
        }
        for k, v in settings.items():
            setattr(build_strategy, k, v)
        strategy.build_strategy = build_strategy

        strategy.gradient_merge = True
        avg = os.environ['enable_gm_avg'] == "True"
        strategy.gradient_merge_configs = {
            "k_steps": 2,
            "avg": avg,
        }
        strategy.without_graph_optimization = True

        fleet.init(is_collective=True, strategy=strategy)
        image = paddle.static.data(
            name='image', shape=[None, 1, 28, 28], dtype="float32"
        )
        label = paddle.static.data(name='label', shape=[None, 1], dtype='int64')
        predict = cnn_model(image)
        acc = paddle.metric.accuracy(predict, label)
        loss_fn = nn.CrossEntropyLoss(use_softmax=False)
        cost = loss_fn(predict, label)
        test_program = paddle.static.default_main_program().clone(for_test=True)
        optimizer = paddle.optimizer.Adam(learning_rate=1e-3)
        if single_device:
            optimizer = paddle.incubate.optimizer.GradientMergeOptimizer(
                optimizer,
                k_steps=strategy.gradient_merge_configs["k_steps"],
                avg=strategy.gradient_merge_configs["avg"],
            )
            world_size = 1
        else:
            optimizer = fleet.distributed_optimizer(optimizer)
            world_size = fleet.world_size()
        optimizer.minimize(cost)
        if world_size > 1:
            assert paddle.static.default_main_program().num_blocks == 2
            gm_block = paddle.static.default_main_program().block(1)
            start_allreduce_idx = None
            for i, op in enumerate(gm_block.ops):
                if op.type == "c_allreduce_sum":
                    start_allreduce_idx = i
                    break
            # the magic number 1 below means skip the c_sync_calc_stream op
            if avg:
                assert start_allreduce_idx > 1
            else:
                assert start_allreduce_idx == 1

        train_reader = paddle.batch(
            paddle.dataset.mnist.test(), batch_size=batch_size
        )
        test_reader = paddle.batch(
            paddle.dataset.mnist.test(), batch_size=batch_size
        )
        return test_program, cost, train_reader, test_reader, acc, predict


if __name__ == "__main__":
    runtime_main(TestDistMnistGradientMergeRawOptimizer)
