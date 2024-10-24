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

import unittest

import nets

import paddle


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
    def test_dropout_layer(self):
        main_program = paddle.static.Program()
        startup_program = paddle.static.Program()
        with paddle.static.program_guard(main_program, startup_program):
            images = paddle.static.data(
                name='pixel', shape=[-1, 3, 48, 48], dtype='float32'
            )
            paddle.nn.functional.dropout(x=images, p=0.5)

        print(str(main_program))

    def test_img_conv_group(self):
        main_program = paddle.static.Program()
        startup_program = paddle.static.Program()

        with paddle.static.program_guard(main_program, startup_program):
            images = paddle.static.data(
                name='pixel', shape=[-1, 3, 48, 48], dtype='float32'
            )
            conv1 = conv_block(images, 64, 2, [0.3, 0])
            conv_block(conv1, 256, 3, [0.4, 0.4, 0])

        print(str(main_program))

    def test_elementwise_add_with_act(self):
        main_program = paddle.static.Program()
        startup_program = paddle.static.Program()
        with paddle.static.program_guard(main_program, startup_program):
            image1 = paddle.static.data(
                name='pixel1', shape=[-1, 3, 48, 48], dtype='float32'
            )
            image2 = paddle.static.data(
                name='pixel2', shape=[-1, 3, 48, 48], dtype='float32'
            )
            paddle.nn.functional.relu(paddle.add(x=image1, y=image2))
        print(main_program)


if __name__ == '__main__':
    unittest.main()
