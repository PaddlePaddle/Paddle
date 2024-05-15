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

test0 = ApiBase(
    func=paddle.nn.Hardswish,
    feed_names=['data'],
    feed_shapes=[[2, 4, 3, 640, 640]],
)


@pytest.mark.hard_swish
@pytest.mark.filterwarnings('ignore::UserWarning')
def test_hard_swish_0():
    data = np.random.uniform(-4, 8, [2, 4, 3, 640, 640]).astype('float32')
    test0.run(feed=[data])


test1 = ApiBase(
    func=paddle.nn.Hardswish,
    feed_names=['data'],
    feed_shapes=[[1, 1, 640, 640]],
)


@pytest.mark.hard_swish
@pytest.mark.filterwarnings('ignore::UserWarning')
def test_hard_swish_1():
    data = np.random.uniform(-4, 8, [1, 1, 640, 640]).astype('float32')
    test1.run(feed=[data])


test2 = ApiBase(
    func=paddle.nn.Hardswish,
    feed_names=['data'],
    feed_shapes=[[2, 1, 640, 640]],
)


@pytest.mark.hard_swish
@pytest.mark.filterwarnings('ignore::UserWarning')
def test_hard_swish_2():
    data = np.random.uniform(-4, 8, [2, 1, 640, 640]).astype('float32')
    test2.run(feed=[data])


test3 = ApiBase(
    func=paddle.nn.Hardswish,
    feed_names=['data'],
    feed_shapes=[[2, 8, 352, 640]],
)


@pytest.mark.hard_swish
@pytest.mark.filterwarnings('ignore::UserWarning')
def test_hard_swish_3():
    data = np.random.uniform(-4, 8, [2, 8, 352, 640]).astype('float32')
    test3.run(feed=[data])


test4 = ApiBase(
    func=paddle.nn.Hardswish,
    feed_names=['data'],
    feed_shapes=[[2, 120, 44, 80]],
)


@pytest.mark.hard_swish
@pytest.mark.filterwarnings('ignore::UserWarning')
def test_hard_swish_4():
    data = np.random.uniform(-4, 8, [2, 120, 44, 80]).astype('float32')
    test4.run(feed=[data])


test5 = ApiBase(
    func=paddle.nn.Hardswish,
    feed_names=['data'],
    feed_shapes=[[16, 120, 88, 160]],
)


@pytest.mark.hard_swish
@pytest.mark.filterwarnings('ignore::UserWarning')
def test_hard_swish_5():
    data = np.random.uniform(-4, 8, [16, 120, 88, 160]).astype('float32')
    test5.run(feed=[data])


test6 = ApiBase(
    func=paddle.nn.Hardswish,
    feed_names=['data'],
    feed_shapes=[[16, 120, 44, 80]],
)


@pytest.mark.hard_swish
@pytest.mark.filterwarnings('ignore::UserWarning')
def test_hard_swish_6():
    data = np.random.uniform(-4, 8, [16, 120, 44, 80]).astype('float32')
    test6.run(feed=[data])


test7 = ApiBase(
    func=paddle.nn.Hardswish,
    feed_names=['data'],
    feed_shapes=[[16, 104, 44, 80]],
)


@pytest.mark.hard_swish
@pytest.mark.filterwarnings('ignore::UserWarning')
def test_hard_swish_7():
    data = np.random.uniform(-4, 8, [16, 104, 44, 80]).astype('float32')
    test7.run(feed=[data])


test8 = ApiBase(
    func=paddle.nn.Hardswish,
    feed_names=['data'],
    feed_shapes=[[16, 96, 44, 80]],
)


@pytest.mark.hard_swish
@pytest.mark.filterwarnings('ignore::UserWarning')
def test_hard_swish_8():
    data = np.random.uniform(-4, 8, [16, 96, 44, 80]).astype('float32')
    test8.run(feed=[data])


test9 = ApiBase(
    func=paddle.nn.Hardswish,
    feed_names=['data'],
    feed_shapes=[[16, 240, 44, 80]],
)


@pytest.mark.hard_swish
@pytest.mark.filterwarnings('ignore::UserWarning')
def test_hard_swish_9():
    data = np.random.uniform(-4, 8, [16, 240, 44, 80]).astype('float32')
    test9.run(feed=[data])


test10 = ApiBase(
    func=paddle.nn.Hardswish,
    feed_names=['data'],
    feed_shapes=[[16, 336, 44, 80]],
)


@pytest.mark.hard_swish
@pytest.mark.filterwarnings('ignore::UserWarning')
def test_hard_swish_10():
    data = np.random.uniform(-4, 8, [16, 336, 44, 80]).astype('float32')
    test10.run(feed=[data])


test11 = ApiBase(
    func=paddle.nn.Hardswish,
    feed_names=['data'],
    feed_shapes=[[16, 336, 22, 40]],
)


@pytest.mark.hard_swish
@pytest.mark.filterwarnings('ignore::UserWarning')
def test_hard_swish_11():
    data = np.random.uniform(-4, 8, [16, 336, 22, 40]).astype('float32')
    test11.run(feed=[data])


test12 = ApiBase(
    func=paddle.nn.Hardswish,
    feed_names=['data'],
    feed_shapes=[[16, 480, 22, 40]],
)


@pytest.mark.hard_swish
@pytest.mark.filterwarnings('ignore::UserWarning')
def test_hard_swish_12():
    data = np.random.uniform(-4, 8, [16, 480, 22, 40]).astype('float32')
    test12.run(feed=[data])


test13 = ApiBase(
    func=paddle.nn.Hardswish,
    feed_names=['data'],
    feed_shapes=[[16, 8, 352, 640]],
)


@pytest.mark.hard_swish
@pytest.mark.filterwarnings('ignore::UserWarning')
def test_hard_swish_13():
    data = np.random.uniform(-4, 8, [16, 8, 352, 640]).astype('float32')
    test13.run(feed=[data])
