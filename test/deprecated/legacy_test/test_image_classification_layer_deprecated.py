#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

sys.path.append("../../legacy_test")
import nets

import paddle
from paddle import base
from paddle.base.framework import Program

paddle.enable_static()


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


class TestLayer(unittest.TestCase):
    def test_batch_norm_layer(self):
        main_program = Program()
        startup_program = Program()
        with base.program_guard(main_program, startup_program):
            images = paddle.static.data(
                name='pixel', shape=[-1, 3, 48, 48], dtype='float32'
            )
            hidden1 = paddle.static.nn.batch_norm(input=images)
            hidden2 = paddle.static.nn.fc(
                x=hidden1, size=128, activation='relu'
            )
            paddle.static.nn.batch_norm(input=hidden2)

        print(str(main_program))


if __name__ == '__main__':
    unittest.main()
