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

test = ApiBase(func=paddle.nn.Swish, feed_names=['data'], feed_shapes=[[3]])


@pytest.mark.swish
@pytest.mark.filterwarnings('ignore::UserWarning')
def test_swish():
    data = np.array([-1, 0, 1], dtype=np.float32)
    test.run(feed=[data])


def swish_test_with_random_data(func, shape, dtype='float32'):
    test = ApiBase(
        func=func, feed_names=['data'], feed_shapes=[shape], feed_dtypes=[dtype]
    )
    data = np.random.random(size=shape).astype(dtype)
    test.run(feed=[data])


@pytest.mark.swish
@pytest.mark.filterwarnings('ignore::UserWarning')
def test_swish_1():
    swish_test_with_random_data(paddle.nn.Swish, shape=[128, 240, 3, 15, 15])


@pytest.mark.swish
@pytest.mark.filterwarnings('ignore::UserWarning')
def test_swish_2():
    swish_test_with_random_data(paddle.nn.Swish, shape=[128, 96, 60, 60])


@pytest.mark.swish
@pytest.mark.filterwarnings('ignore::UserWarning')
def test_swish_3():
    swish_test_with_random_data(paddle.nn.Swish, shape=[128, 8, 1, 1])


@pytest.mark.swish
@pytest.mark.filterwarnings('ignore::UserWarning')
def test_swish_4():
    swish_test_with_random_data(
        paddle.nn.functional.swish, shape=[128, 672, 15, 15]
    )


@pytest.mark.swish
@pytest.mark.filterwarnings('ignore::UserWarning')
def test_swish_5():
    swish_test_with_random_data(
        paddle.nn.functional.swish, shape=[128, 240, 30, 30]
    )


@pytest.mark.swish
@pytest.mark.filterwarnings('ignore::UserWarning')
def test_swish_6():
    swish_test_with_random_data(
        paddle.nn.functional.swish, shape=[128, 10, 1, 1]
    )


@pytest.mark.swish
@pytest.mark.filterwarnings('ignore::UserWarning')
def test_swish_7():
    swish_test_with_random_data(
        paddle.nn.Swish, shape=[128, 8, 1, 1], dtype='float64'
    )
