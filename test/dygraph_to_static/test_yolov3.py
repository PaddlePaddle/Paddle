# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
import time
import unittest

import numpy as np
from dygraph_to_static_utils import (
    Dy2StTestBase,
    enable_to_static_guard,
    test_default_and_pir,
)
from yolov3 import YOLOv3, cfg

import paddle

if paddle.is_compiled_with_cuda():
    paddle.base.set_flags({'FLAGS_cudnn_deterministic': True})

random.seed(0)
np.random.seed(0)
paddle.seed(0)


class SmoothedValue:
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self):
        self.loss_sum = 0.0
        self.iter_cnt = 0

    def add_value(self, value):
        self.loss_sum += np.mean(value)
        self.iter_cnt += 1

    def get_mean_value(self):
        return self.loss_sum / self.iter_cnt


class FakeDataReader:
    def __init__(self):
        self.generator_out = []
        self.total_iter = cfg.max_iter
        for i in range(self.total_iter):
            batch_out = []
            for j in range(cfg.batch_size):
                img = np.random.normal(
                    0.485, 0.229, [3, cfg.input_size, cfg.input_size]
                )
                point1 = 1 / 4
                point2 = 1 / 2
                gt_boxes = np.array([[point1, point1, point2, point2]])
                gt_labels = np.random.randint(
                    low=0, high=cfg.class_num, size=[1]
                )
                gt_scores = np.zeros([1])
                batch_out.append([img, gt_boxes, gt_labels, gt_scores])
            self.generator_out.append(batch_out)

    def reader(self):
        def generator():
            for i in range(self.total_iter):
                yield self.generator_out[i]

        return generator


fake_data_reader = FakeDataReader()


def train():
    random.seed(0)
    np.random.seed(0)
    paddle.seed(1000)

    model = paddle.jit.to_static(YOLOv3(3, is_train=True))

    boundaries = cfg.lr_steps
    gamma = cfg.lr_gamma
    step_num = len(cfg.lr_steps)
    learning_rate = cfg.learning_rate
    values = [learning_rate * (gamma**i) for i in range(step_num + 1)]

    lr = paddle.optimizer.lr.PiecewiseDecay(
        boundaries=boundaries, values=values
    )

    lr = paddle.optimizer.lr.LinearWarmup(
        learning_rate=lr,
        warmup_steps=cfg.warm_up_iter,
        start_lr=0.0,
        end_lr=cfg.learning_rate,
    )
    optimizer = paddle.optimizer.Momentum(
        learning_rate=lr,
        weight_decay=paddle.regularizer.L2Decay(cfg.weight_decay),
        momentum=cfg.momentum,
        parameters=model.parameters(),
    )

    start_time = time.time()
    snapshot_loss = 0
    snapshot_time = 0
    total_sample = 0

    input_size = cfg.input_size
    shuffle = True
    shuffle_seed = None
    total_iter = cfg.max_iter
    mixup_iter = total_iter - cfg.no_mixup_iter

    train_reader = FakeDataReader().reader()

    smoothed_loss = SmoothedValue()
    ret = []
    for iter_id, data in enumerate(train_reader()):
        prev_start_time = start_time
        start_time = time.time()
        img = np.array([x[0] for x in data]).astype('float32')
        img = paddle.to_tensor(img)

        gt_box = np.array([x[1] for x in data]).astype('float32')
        gt_box = paddle.to_tensor(gt_box)

        gt_label = np.array([x[2] for x in data]).astype('int32')
        gt_label = paddle.to_tensor(gt_label)

        gt_score = np.array([x[3] for x in data]).astype('float32')
        gt_score = paddle.to_tensor(gt_score)

        loss = model(img, gt_box, gt_label, gt_score, None, None)
        smoothed_loss.add_value(np.mean(loss.numpy()))
        snapshot_loss += loss.numpy()
        snapshot_time += start_time - prev_start_time
        total_sample += 1

        print(
            f"Iter {iter_id:d}, loss {smoothed_loss.get_mean_value():.6f}, time {start_time - prev_start_time:.5f}"
        )
        ret.append(smoothed_loss.get_mean_value())

        loss.backward()

        optimizer.minimize(loss)
        model.clear_gradients()

    return np.array(ret)


class TestYolov3(Dy2StTestBase):
    @test_default_and_pir
    def test_dygraph_static_same_loss(self):
        with enable_to_static_guard(False):
            dygraph_loss = train()

        static_loss = train()
        np.testing.assert_allclose(
            dygraph_loss, static_loss, rtol=0.001, atol=1e-05
        )


if __name__ == '__main__':
    unittest.main()
