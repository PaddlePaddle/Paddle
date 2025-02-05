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

import time
import unittest

import numpy as np
from dygraph_to_static_utils import (
    Dy2StTestBase,
    enable_to_static_guard,
    test_default_and_pir,
)
from test_resnet import SEED, ResNet, optimizer_setting

import paddle
from paddle import base
from paddle.base import core

# NOTE: Reduce batch_size from 8 to 2 to avoid unittest timeout.
batch_size = 2
epoch_num = 1


if paddle.is_compiled_with_cuda():
    paddle.set_flags({'FLAGS_cudnn_deterministic': True})


def train(build_strategy=None):
    """
    Tests model decorated by `dygraph_to_static_output` in static graph mode. For users, the model is defined in dygraph mode and trained in static graph mode.
    """
    np.random.seed(SEED)
    paddle.seed(SEED)
    paddle.framework.random._manual_program_seed(SEED)

    resnet = paddle.jit.to_static(ResNet(), build_strategy=build_strategy)
    optimizer = optimizer_setting(parameter_list=resnet.parameters())
    scaler = paddle.amp.GradScaler(init_loss_scaling=1024)

    resnet, optimizer = paddle.amp.decorate(
        models=resnet, optimizers=optimizer, level='O2', save_dtype='float32'
    )

    for epoch in range(epoch_num):
        loss_data = []
        total_loss = 0.0
        total_acc1 = 0.0
        total_acc5 = 0.0
        total_sample = 0

        for batch_id in range(100):
            start_time = time.time()
            img = paddle.to_tensor(
                np.random.random([batch_size, 3, 224, 224]).astype('float32')
            )
            label = paddle.to_tensor(
                np.random.randint(0, 100, [batch_size, 1], dtype='int64')
            )
            img.stop_gradient = True
            label.stop_gradient = True

            with paddle.amp.auto_cast(
                enable=True,
                custom_white_list=None,
                custom_black_list=None,
                level='O2',
            ):
                pred = resnet(img)
                loss = paddle.nn.functional.cross_entropy(
                    input=pred, label=label, reduction='none', use_softmax=False
                )
            avg_loss = paddle.mean(x=pred)
            acc_top1 = paddle.static.accuracy(input=pred, label=label, k=1)
            acc_top5 = paddle.static.accuracy(input=pred, label=label, k=5)

            scaled = scaler.scale(avg_loss)
            scaled.backward()
            scaler.minimize(optimizer, scaled)
            resnet.clear_gradients()

            loss_data.append(float(avg_loss))
            total_loss += avg_loss
            total_acc1 += acc_top1
            total_acc5 += acc_top5
            total_sample += 1

            end_time = time.time()
            if batch_id % 2 == 0:
                print(
                    f"epoch {epoch} | batch step {batch_id}, "
                    f"loss {total_loss.numpy() / total_sample:0.3f}, "
                    f"acc1 {total_acc1.numpy() / total_sample:0.3f}, "
                    f"acc5 {total_acc5.numpy() / total_sample:0.3f}, "
                    f"time {end_time - start_time:f}"
                )
            if batch_id == 10:
                break

    return loss_data


class TestResnet(Dy2StTestBase):
    def train(self, to_static: bool):
        with enable_to_static_guard(to_static):
            build_strategy = paddle.static.BuildStrategy()
            # Why set `build_strategy.enable_inplace = False` here?
            # Because we find that this PASS strategy of PE makes dy2st training loss unstable.
            build_strategy.enable_inplace = False
            return train(build_strategy)

    @test_default_and_pir
    def test_resnet(self):
        if base.is_compiled_with_cuda():
            static_loss = self.train(to_static=True)
            dygraph_loss = self.train(to_static=False)
            # NOTE: In pure fp16 training, loss is not stable, so we enlarge atol here.
            np.testing.assert_allclose(
                static_loss,
                dygraph_loss,
                rtol=1e-05,
                atol=0.001,
                err_msg=f'static_loss: {static_loss} \n dygraph_loss: {dygraph_loss}',
            )

    @test_default_and_pir
    def test_resnet_composite(self):
        if base.is_compiled_with_cuda():
            core._set_prim_backward_enabled(True)
            static_loss = self.train(to_static=True)
            core._set_prim_backward_enabled(False)
            dygraph_loss = self.train(to_static=False)
            # NOTE: In pure fp16 training, loss is not stable, so we enlarge atol here.
            np.testing.assert_allclose(
                static_loss,
                dygraph_loss,
                rtol=1e-05,
                atol=0.001,
                err_msg=f'static_loss: {static_loss} \n dygraph_loss: {dygraph_loss}',
            )


if __name__ == '__main__':
    unittest.main()
