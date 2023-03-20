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

import time
import unittest

import numpy as np

import paddle
import paddle.fluid as fluid
import paddle.fluid.core as core
from paddle.vision.models import resnet50

SEED = 2020
base_lr = 0.001
momentum_rate = 0.9
l2_decay = 1e-4
batch_size = 2
epoch_num = 1

if core.is_compiled_with_cuda():
    paddle.set_flags({'FLAGS_cudnn_deterministic': True})


def reader_decorator(reader):
    def __reader__():
        for item in reader():
            img = np.array(item[0]).astype('float32').reshape(3, 224, 224)
            label = np.array(item[1]).astype('int64').reshape(1)
            yield img, label

    return __reader__


def optimizer_setting(parameter_list=None):
    optimizer = fluid.optimizer.Momentum(
        learning_rate=base_lr,
        momentum=momentum_rate,
        regularization=fluid.regularizer.L2Decay(l2_decay),
        parameter_list=parameter_list,
    )

    return optimizer


def train(to_static, enable_prim, enable_cinn):
    if core.is_compiled_with_cuda():
        paddle.set_device('gpu')
    else:
        paddle.set_device('cpu')
    np.random.seed(SEED)
    paddle.seed(SEED)
    paddle.framework.random._manual_program_seed(SEED)
    fluid.core._set_prim_all_enabled(enable_prim)

    train_reader = paddle.batch(
        reader_decorator(paddle.dataset.flowers.train(use_xmap=False)),
        batch_size=batch_size,
        drop_last=True,
    )
    data_loader = fluid.io.DataLoader.from_generator(capacity=5, iterable=True)
    data_loader.set_sample_list_generator(train_reader)

    resnet = resnet50(False)
    if to_static:
        build_strategy = paddle.static.BuildStrategy()
        if enable_cinn:
            build_strategy.build_cinn_pass = True
        resnet = paddle.jit.to_static(resnet, build_strategy=build_strategy)
    optimizer = optimizer_setting(parameter_list=resnet.parameters())

    for epoch in range(epoch_num):
        total_acc1 = 0.0
        total_acc5 = 0.0
        total_sample = 0
        losses = []

        for batch_id, data in enumerate(data_loader()):
            start_time = time.time()
            img, label = data

            pred = resnet(img)
            avg_loss = paddle.nn.functional.cross_entropy(
                input=pred,
                label=label,
                soft_label=False,
                reduction='mean',
                use_softmax=True,
            )

            acc_top1 = paddle.static.accuracy(input=pred, label=label, k=1)
            acc_top5 = paddle.static.accuracy(input=pred, label=label, k=5)

            avg_loss.backward()
            optimizer.minimize(avg_loss)
            resnet.clear_gradients()

            total_acc1 += acc_top1
            total_acc5 += acc_top5
            total_sample += 1
            losses.append(avg_loss.numpy())

            end_time = time.time()
            print(
                "epoch %d | batch step %d, loss %0.8f, acc1 %0.3f, acc5 %0.3f, time %f"
                % (
                    epoch,
                    batch_id,
                    avg_loss,
                    total_acc1.numpy() / total_sample,
                    total_acc5.numpy() / total_sample,
                    end_time - start_time,
                )
            )
            if batch_id == 10:
                # avoid dataloader throw abort signaal
                data_loader._reset()
                break

    return losses


class TestResnet(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dy2st = train(to_static=True, enable_prim=False, enable_cinn=False)

    def test_prim(self):
        # todo: to be removed after adjust of rtol
        core._set_prim_forward_blacklist("batch_norm")
        core._add_skip_comp_ops("batch_norm")
        dy2st_prim = train(to_static=True, enable_prim=True, enable_cinn=False)
        # NOTE: Now dy2st is equal to dy2st_prim. With the splitting of kernels, the threshold here may need to be adjusted
        np.testing.assert_allclose(self.dy2st, dy2st_prim, rtol=1e-6)

    @unittest.skipIf(
        not paddle.is_compiled_with_cinn(), "padle is not compiled with CINN"
    )
    def test_cinn(self):
        dy2st_cinn = train(to_static=True, enable_prim=False, enable_cinn=True)
        # TODO(0x45f): The following is only temporary thresholds, and the final thresholds needs to be discussed
        np.testing.assert_allclose(self.dy2st[0:2], dy2st_cinn[0:2], rtol=1e-3)
        np.testing.assert_allclose(self.dy2st, dy2st_cinn, rtol=1e-1)

    @unittest.skipIf(
        not paddle.is_compiled_with_cinn(), "padle is not compiled with CINN"
    )
    def test_prim_cinn(self):
        core._set_prim_forward_blacklist("flatten_contiguous_range")
        dy2st_prim_cinn = train(
            to_static=True, enable_prim=True, enable_cinn=True
        )
        # TODO(0x45f): The following is only temporary thresholds, and the final thresholds need to be discussed
        np.testing.assert_allclose(
            self.dy2st[0:2], dy2st_prim_cinn[0:2], rtol=1e-2
        )
        np.testing.assert_allclose(self.dy2st, dy2st_prim_cinn, rtol=1e-1)


if __name__ == '__main__':
    unittest.main()
