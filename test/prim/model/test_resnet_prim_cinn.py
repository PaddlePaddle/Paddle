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
from paddle import fluid
from paddle.fluid import core
from paddle.vision.models import resnet50

SEED = 2020
base_lr = 0.001
momentum_rate = 0.9
l2_decay = 1e-4
batch_size = 2
epoch_num = 1

# In V100, 16G, CUDA 11.2, the results are as follows:
# DY2ST_PRIM_GT = [
#     5.8473358154296875,
#     8.354944229125977,
#     5.098367691040039,
#     8.533346176147461,
#     8.179085731506348,
#     7.285282135009766,
#     9.824585914611816,
#     8.56928825378418,
#     8.539499282836914,
#     10.256929397583008,
# ]
# DY2ST_CINN_GT = [
#     5.847336769104004,
#     8.336246490478516,
#     5.108744144439697,
#     8.316713333129883,
#     8.175262451171875,
#     7.590441703796387,
#     9.895681381225586,
#     8.196207046508789,
#     8.438933372497559,
#     10.305074691772461,
# ]
# DY2ST_PRIM_CINN_GT = [
#     5.8473358154296875,
#     8.322463989257812,
#     5.169863700866699,
#     8.399882316589355,
#     7.859550476074219,
#     7.4672698974609375,
#     9.828727722167969,
#     8.270355224609375,
#     8.456792831420898,
#     9.919631958007812,
# ]

# The results in ci as as follows:
DY2ST_PRIM_GT = [
    5.82879114151001,
    8.333706855773926,
    5.07769250869751,
    8.66937255859375,
    8.411705017089844,
    7.252340793609619,
    9.683248519897461,
    8.177335739135742,
    8.195427894592285,
    10.219732284545898,
]
DY2ST_CINN_GT = [
    5.828789710998535,
    8.340764999389648,
    4.998944282531738,
    8.474305152893066,
    8.09157943725586,
    7.440057754516602,
    9.907357215881348,
    8.304681777954102,
    8.383116722106934,
    10.120304107666016,
]
DY2ST_PRIM_CINN_GT = [
    5.828784942626953,
    8.341737747192383,
    5.113619327545166,
    8.625601768493652,
    8.082450866699219,
    7.4913249015808105,
    9.858025550842285,
    8.287693977355957,
    8.435812950134277,
    10.372406005859375,
]

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
            losses.append(avg_loss.numpy().item())

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
            if batch_id >= 9:
                # avoid dataloader throw abort signaal
                data_loader._reset()
                break
    print(losses)
    return losses


class TestResnet(unittest.TestCase):
    @unittest.skipIf(
        not (paddle.is_compiled_with_cinn() and paddle.is_compiled_with_cuda()),
        "paddle is not compiled with CINN and CUDA",
    )
    def test_prim(self):
        dy2st_prim = train(to_static=True, enable_prim=True, enable_cinn=False)
        np.testing.assert_allclose(dy2st_prim, DY2ST_PRIM_GT, rtol=1e-5)

    @unittest.skipIf(
        not (paddle.is_compiled_with_cinn() and paddle.is_compiled_with_cuda()),
        "paddle is not compiled with CINN and CUDA",
    )
    def test_cinn(self):
        dy2st_cinn = train(to_static=True, enable_prim=False, enable_cinn=True)
        np.testing.assert_allclose(dy2st_cinn, DY2ST_CINN_GT, rtol=1e-5)

    @unittest.skipIf(
        not (paddle.is_compiled_with_cinn() and paddle.is_compiled_with_cuda()),
        "paddle is not compiled with CINN and CUDA",
    )
    def test_prim_cinn(self):
        dy2st_prim_cinn = train(
            to_static=True, enable_prim=True, enable_cinn=True
        )
        np.testing.assert_allclose(
            dy2st_prim_cinn, DY2ST_PRIM_CINN_GT, rtol=1e-5
        )


if __name__ == '__main__':
    unittest.main()
