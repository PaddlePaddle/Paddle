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

from __future__ import print_function

import math
import time
import unittest

import numpy as np

import paddle
import paddle.fluid as fluid
from paddle.fluid.dygraph import declarative, ProgramTranslator
from paddle.fluid.dygraph.nn import BatchNorm, Conv2D, Linear, Pool2D
from test_resnet import ResNet, optimizer_setting, SEED

# NOTE: Reduce batch_size from 8 to 2 to avoid unittest timeout.
batch_size = 2
epoch_num = 1

program_translator = ProgramTranslator()

if fluid.is_compiled_with_cuda():
    fluid.set_flags({'FLAGS_cudnn_deterministic': True})


def train(to_static, build_strategy=None):
    """
    Tests model decorated by `dygraph_to_static_output` in static mode. For users, the model is defined in dygraph mode and trained in static mode.
    """
    np.random.seed(SEED)
    paddle.seed(SEED)
    paddle.framework.random._manual_program_seed(SEED)

    resnet = ResNet()
    if to_static:
        resnet = paddle.jit.to_static(resnet, build_strategy=build_strategy)
    optimizer = optimizer_setting(parameter_list=resnet.parameters())
    scaler = paddle.amp.GradScaler(init_loss_scaling=1024)

    resnet, optimizer = paddle.amp.decorate(
        models=resnet, optimizers=optimizer, level='O2', save_dtype='float32')

    for epoch in range(epoch_num):
        loss_data = []
        total_loss = 0.0
        total_acc1 = 0.0
        total_acc5 = 0.0
        total_sample = 0

        for batch_id in range(100):
            start_time = time.time()
            img = paddle.to_tensor(
                np.random.random([batch_size, 3, 224, 224]).astype('float32'))
            label = paddle.to_tensor(
                np.random.randint(
                    0, 100, [batch_size, 1], dtype='int64'))
            img.stop_gradient = True
            label.stop_gradient = True

            with paddle.amp.auto_cast(
                    enable=True,
                    custom_white_list=None,
                    custom_black_list=None,
                    level='O2'):
                pred = resnet(img)
                loss = fluid.layers.cross_entropy(input=pred, label=label)
            avg_loss = fluid.layers.mean(x=pred)
            acc_top1 = fluid.layers.accuracy(input=pred, label=label, k=1)
            acc_top5 = fluid.layers.accuracy(input=pred, label=label, k=5)

            scaled = scaler.scale(avg_loss)
            scaled.backward()
            scaler.minimize(optimizer, scaled)
            resnet.clear_gradients()

            loss_data.append(avg_loss.numpy()[0])
            total_loss += avg_loss
            total_acc1 += acc_top1
            total_acc5 += acc_top5
            total_sample += 1

            end_time = time.time()
            if batch_id % 2 == 0:
                print( "epoch %d | batch step %d, loss %0.3f, acc1 %0.3f, acc5 %0.3f, time %f" % \
                    ( epoch, batch_id, total_loss.numpy() / total_sample, \
                        total_acc1.numpy() / total_sample, total_acc5.numpy() / total_sample, end_time-start_time))
            if batch_id == 10:
                break

    return loss_data


class TestResnet(unittest.TestCase):
    def train(self, to_static):
        program_translator.enable(to_static)
        build_strategy = paddle.static.BuildStrategy()
        # Why set `build_strategy.enable_inplace = False` here?
        # Because we find that this PASS strategy of PE makes dy2st training loss unstable.
        build_strategy.enable_inplace = False
        return train(to_static, build_strategy)

    def test_resnet(self):
        if fluid.is_compiled_with_cuda():
            static_loss = self.train(to_static=True)
            dygraph_loss = self.train(to_static=False)
            # NOTE: In pure fp16 training, loss is not stable, so we enlarge atol here.
            self.assertTrue(
                np.allclose(
                    static_loss, dygraph_loss, atol=1e-3),
                msg="static_loss: {} \n dygraph_loss: {}".format(static_loss,
                                                                 dygraph_loss))


if __name__ == '__main__':
    with fluid.framework._test_eager_guard():
        unittest.main()
