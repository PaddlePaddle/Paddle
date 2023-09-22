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
from dygraph_to_static_util import test_and_compare_with_new_ir
from yolov3 import YOLOv3, cfg

import paddle
from paddle import base
from paddle.base.dygraph import to_variable

paddle.enable_static()
random.seed(0)
np.random.seed(0)


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


def train(to_static):
    paddle.jit.enable_to_static(to_static)

    random.seed(0)
    np.random.seed(0)

    place = base.CUDAPlace(0) if cfg.use_gpu else base.CPUPlace()
    with base.dygraph.guard(place):
        base.default_startup_program().random_seed = 1000
        base.default_main_program().random_seed = 1000
        model = YOLOv3(3, is_train=True)

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
            img = to_variable(img)

            gt_box = np.array([x[1] for x in data]).astype('float32')
            gt_box = to_variable(gt_box)

            gt_label = np.array([x[2] for x in data]).astype('int32')
            gt_label = to_variable(gt_label)

            gt_score = np.array([x[3] for x in data]).astype('float32')
            gt_score = to_variable(gt_score)

            loss = model(img, gt_box, gt_label, gt_score, None, None)
            smoothed_loss.add_value(np.mean(loss.numpy()))
            snapshot_loss += loss.numpy()
            snapshot_time += start_time - prev_start_time
            total_sample += 1

            print(
                "Iter {:d}, loss {:.6f}, time {:.5f}".format(
                    iter_id,
                    smoothed_loss.get_mean_value(),
                    start_time - prev_start_time,
                )
            )
            ret.append(smoothed_loss.get_mean_value())

            loss.backward()

            optimizer.minimize(loss)
            model.clear_gradients()

        return np.array(ret)


class TestYolov3(unittest.TestCase):
    @test_and_compare_with_new_ir(False)
    def test_dygraph_static_same_loss(self):
        dygraph_loss = train(to_static=False)
        static_loss = train(to_static=True)
        np.testing.assert_allclose(
            dygraph_loss, static_loss, rtol=0.001, atol=1e-05
        )


if __name__ == '__main__':
    unittest.main()
