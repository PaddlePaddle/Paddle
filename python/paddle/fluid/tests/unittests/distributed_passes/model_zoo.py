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
import paddle
import paddle.distributed.fleet as fleet
from paddle.vision.models import resnet50 as resnet
import numpy as np
import paddle.nn as nn

__all__ = ['resnet_model', ]


def get_seed_from_env():
    return int(os.environ.get("SEED", 0))


def resnet_model(place, batch_size, image_shape=[3, 224, 224],
                 num_classes=1000):
    image = paddle.static.data(
        shape=[batch_size] + image_shape, dtype='float32', name='image')
    label = paddle.static.data(
        shape=[batch_size, 1], dtype='int64', name='label')
    model = resnet(pretrained=False)
    loss_fn = nn.loss.CrossEntropyLoss()
    pred_out = model(image)
    loss = loss_fn(pred_out, label)
    optimizer = paddle.optimizer.Adam(learning_rate=1e-3)

    dist_strategy = fleet.DistributedStrategy()
    dist_strategy.fuse_all_reduce_ops = False
    dist_strategy.without_graph_optimization = True
    fleet.init(is_collective=True, strategy=dist_strategy)
    optimizer = fleet.distributed_optimizer(optimizer)
    optimizer.minimize(loss)

    rank = paddle.distributed.get_rank()

    def reader():
        seed = get_seed_from_env()
        np.random.seed(seed + rank)
        for _ in range(10):
            image_np = np.random.random(size=image.shape).astype('float32')
            label_np = np.random.randint(
                low=0, high=num_classes, size=label.shape).astype('int64')
            yield image_np, label_np

    main_program = paddle.static.default_main_program()
    startup_program = paddle.static.default_startup_program()
    return main_program, startup_program, [image, label], [loss], reader
