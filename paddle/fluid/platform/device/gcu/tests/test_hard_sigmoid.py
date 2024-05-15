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

test1 = ApiBase(
    func=paddle.nn.Hardsigmoid,
    feed_names=['data'],
    feed_shapes=[[1, 1, 640, 640]],
)


@pytest.mark.hard_sigmoid
@pytest.mark.filterwarnings('ignore::UserWarning')
def test_hard_sigmoid_b1():
    data = np.random.random(size=[1, 1, 640, 640]).astype('float32')
    test1.run(feed=[data])


test2 = ApiBase(
    func=paddle.nn.Hardsigmoid,
    feed_names=['data'],
    feed_shapes=[[2, 1, 640, 640]],
)


@pytest.mark.hard_sigmoid
@pytest.mark.filterwarnings('ignore::UserWarning')
def test_hard_sigmoid_b2():
    data = np.random.random(size=[2, 1, 640, 640]).astype('float32')
    test2.run(feed=[data])


test3 = ApiBase(
    func=paddle.nn.Hardsigmoid,
    feed_names=['data'],
    feed_shapes=[[2, 336, 1, 1]],
)


@pytest.mark.hard_sigmoid
@pytest.mark.filterwarnings('ignore::UserWarning')
def test_hard_sigmoid_b3():
    data = np.random.random(size=[2, 336, 1, 1]).astype('float32')
    test3.run(feed=[data])


test4 = ApiBase(
    func=paddle.nn.functional.hardsigmoid,
    feed_names=['data'],
    feed_shapes=[[2, 336, 1, 1]],
)


@pytest.mark.hard_sigmoid
@pytest.mark.filterwarnings('ignore::UserWarning')
def test_hard_sigmoid_b4():
    data = np.random.random(size=[2, 336, 1, 1]).astype('float32')
    test4.run(feed=[data], offset=0.5, slope=0.20000000298023224)


test5 = ApiBase(
    func=paddle.nn.functional.hardsigmoid,
    feed_names=['data'],
    feed_shapes=[[16, 64, 1, 1]],
)


@pytest.mark.hard_sigmoid
@pytest.mark.filterwarnings('ignore::UserWarning')
def test_hard_sigmoid_b5():
    data = np.random.random(size=[16, 64, 1, 1]).astype('float32')
    test5.run(feed=[data], offset=0.5, slope=0.20000000298023224)


test6 = ApiBase(
    func=paddle.nn.functional.hardsigmoid,
    feed_names=['data'],
    feed_shapes=[[16, 240, 1, 1]],
)


@pytest.mark.hard_sigmoid
@pytest.mark.filterwarnings('ignore::UserWarning')
def test_hard_sigmoid_b6():
    data = np.random.random(size=[16, 240, 1, 1]).astype('float32')
    test6.run(feed=[data], offset=0.5, slope=0.20000000298023224)


test7 = ApiBase(
    func=paddle.nn.functional.hardsigmoid,
    feed_names=['data'],
    feed_shapes=[[16, 336, 1, 1]],
)


@pytest.mark.hard_sigmoid
@pytest.mark.filterwarnings('ignore::UserWarning')
def test_hard_sigmoid_b7():
    data = np.random.random(size=[16, 336, 1, 1]).astype('float32')
    test7.run(feed=[data], offset=0.5, slope=0.20000000298023224)


test8 = ApiBase(
    func=paddle.nn.functional.hardsigmoid,
    feed_names=['data'],
    feed_shapes=[[16, 480, 1, 1]],
)


@pytest.mark.hard_sigmoid
@pytest.mark.filterwarnings('ignore::UserWarning')
def test_hard_sigmoid_b8():
    data = np.random.random(size=[16, 480, 1, 1]).astype('float32')
    test8.run(feed=[data], offset=0.5, slope=0.20000000298023224)
