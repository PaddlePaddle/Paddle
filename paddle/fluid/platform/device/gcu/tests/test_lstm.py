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

import numpy as np
import pytest
from api_base import ApiBase

import paddle

# test1 = ApiBase(func=paddle.fluid.layers.lstm,
#                 feed_names=['input', 'init_h', 'init_c'],
#                 feed_shapes=[[3, 2, 8], [1, 2, 7], [1, 2, 7]], is_train=False, threshold=1.0e-5)
# @pytest.mark.lstm
# @pytest.mark.filterwarnings('ignore::UserWarning')
# def test_lstm_1():
#     np.random.seed(1)
#     batch_size = 2
#     seq_len = 3
#     input_dim = 8
#     num_layers = 1
#     max_len = 10
#     hidden_size = 7
#     input = np.random.random(size=[seq_len, batch_size, input_dim]).astype('float32')
#     init_h = np.random.random(size=[num_layers, batch_size, hidden_size]).astype('float32')
#     init_c = np.random.random(size=[num_layers, batch_size, hidden_size]).astype('float32')
#     test1.run(feed=[input, init_h, init_c], max_len=max_len, hidden_size=hidden_size, num_layers=num_layers)

# test1 = ApiBase(func=paddle.nn.LSTM,
#                 feed_names=['inputs', ('initial_states')],
#                 feed_shapes=[[4, 3, 120], ([2, 4, 50], [2, 4, 50])], is_train=False, threshold=1.0e-5)
# @pytest.mark.lstm
# @pytest.mark.filterwarnings('ignore::UserWarning')
# def test_lstm_1():
#     np.random.seed(1)
#     batch_size = 4
#     time_steps = 3
#     input_size = 120
#     hidden_size = 50
#     num_layers = 1
#     direction = "bidirect"
#     num_directions = 2 if direction == "bidirect" else 1
#     inputs = np.random.random(size=[batch_size, time_steps, input_size]).astype('float32')
#     prev_h = np.random.random(size=[num_layers*num_directions, batch_size, hidden_size]).astype('float32')
#     prev_c = np.random.random(size=[num_layers*num_directions, batch_size, hidden_size]).astype('float32')
#     initial_states = (prev_h, prev_c)
#     test1.run(feed=[inputs, initial_states], input_size=input_size, hidden_size=hidden_size, direction=direction)


test1 = ApiBase(
    func=paddle.nn.LSTM,
    feed_names=['inputs'],
    feed_shapes=[[4, 3, 120]],
    is_train=False,
    threshold=1.0e-5,
)


@pytest.mark.lstm
@pytest.mark.filterwarnings('ignore::UserWarning')
def test_lstm_1():
    np.random.seed(1)
    batch_size = 4
    time_steps = 3
    input_size = 120
    hidden_size = 50
    num_layers = 2
    direction = "bidirect"
    # direction = "forward"
    num_directions = 2 if direction == "bidirect" else 1
    inputs = np.random.random(size=[batch_size, time_steps, input_size]).astype(
        'float32'
    )
    # inputs = np.ones(shape=[batch_size, time_steps, input_size]).astype('float32')
    test1.run(
        feed=[inputs],
        input_size=input_size,
        hidden_size=hidden_size,
        direction=direction,
        num_layers=num_layers,
        weight_ih_attr=paddle.nn.initializer.Constant(value=2.0),
        weight_hh_attr=paddle.nn.initializer.Constant(value=2.0),
    )


# def eager():
#     batch_size = 4
#     time_steps = 3
#     input_size = 120
#     hidden_size = 50
#     num_layers = 1
#     # direction = "bidirect"
#     direction = "forward"
#     num_directions = 2 if direction == "bidirect" else 1
#     rnn = paddle.nn.LSTM(input_size, hidden_size, num_layers)
#     x = paddle.ones(shape=(batch_size, time_steps, input_size))
#     # prev_h = paddle.randn((num_layers*num_directions, batch_size, hidden_size))
#     # prev_c = paddle.randn((num_layers*num_directions, batch_size, hidden_size))
#     y, (h, c) = rnn(x)
#     print(y.shape)
#     print(y)

# def eager2():
#     import paddle

#     # weight_ih_attr = paddle.ParamAttr(name="weight_ih_attr", trainable=False,)
#     rnn = paddle.nn.LSTM(16, 32, 2,
#                          weight_ih_attr=paddle.nn.initializer.Constant(value=2.0),
#                          weight_hh_attr=paddle.nn.initializer.Constant(value=2.0),)

#     x = paddle.ones((4, 23, 16))
#     prev_h = paddle.zeros((2, 4, 32))
#     prev_c = paddle.zeros((2, 4, 32))
#     y, (h, c) = rnn(x, (prev_h, prev_c))
#     print(y.shape)
#     print(y)


# eager()
# eager2()
