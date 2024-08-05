#   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

import sys
import unittest

# TODO: remove sys.path.append
sys.path.append("../../legacy_test")
import nets

import paddle
from paddle.framework import in_pir_mode
from paddle.static.amp import decorate

paddle.enable_static()


def vgg16_bn_drop(input):
    def conv_block(input, num_filter, groups, dropouts):
        return nets.img_conv_group(
            input=input,
            pool_size=2,
            pool_stride=2,
            conv_num_filter=[num_filter] * groups,
            conv_filter_size=3,
            conv_act='relu',
            conv_with_batchnorm=True,
            conv_batchnorm_drop_rate=dropouts,
            pool_type='max',
        )

    conv1 = conv_block(input, 64, 2, [0.3, 0])
    conv2 = conv_block(conv1, 128, 2, [0.4, 0])
    conv3 = conv_block(conv2, 256, 3, [0.4, 0.4, 0])
    conv4 = conv_block(conv3, 512, 3, [0.4, 0.4, 0])
    conv5 = conv_block(conv4, 512, 3, [0.4, 0.4, 0])

    drop = paddle.nn.functional.dropout(x=conv5, p=0.5)
    fc1 = paddle.static.nn.fc(x=drop, size=4096, activation=None)
    if in_pir_mode():
        batch_norm = paddle.nn.BatchNorm(4096)
        bn = batch_norm(fc1)
    else:
        bn = paddle.static.nn.batch_norm(input=fc1, act='relu')
    drop2 = paddle.nn.functional.dropout(x=bn, p=0.5)
    fc2 = paddle.static.nn.fc(x=drop2, size=4096, activation=None)
    return fc2


class TestAmpWithNonIterableDataLoader(unittest.TestCase):
    def decorate_with_data_loader(self):
        main_prog = paddle.static.Program()
        start_prog = paddle.static.Program()
        with paddle.static.program_guard(main_prog, start_prog):
            with paddle.base.unique_name.guard():
                image = paddle.static.data(
                    name='image', shape=[-1, 3, 224, 224], dtype='float32'
                )
                label = paddle.static.data(
                    name='label', shape=[-1, 1], dtype='int64'
                )

                net = vgg16_bn_drop(image)
                logits = paddle.static.nn.fc(
                    x=net, size=10, activation="softmax"
                )
                cost, predict = paddle.nn.functional.softmax_with_cross_entropy(
                    logits, label, return_softmax=True
                )
                avg_cost = paddle.mean(cost)

                optimizer = paddle.optimizer.Lamb(learning_rate=0.001)
                amp_lists = paddle.static.amp.AutoMixedPrecisionLists(
                    custom_black_varnames={"loss", "conv2d_0.w_0"}
                )
                mp_optimizer = decorate(
                    optimizer=optimizer,
                    amp_lists=amp_lists,
                    init_loss_scaling=8.0,
                    use_dynamic_loss_scaling=True,
                )

                mp_optimizer.minimize(avg_cost)

    def test_non_iterable_dataloader(self):
        self.decorate_with_data_loader()


if __name__ == '__main__':
    unittest.main()
