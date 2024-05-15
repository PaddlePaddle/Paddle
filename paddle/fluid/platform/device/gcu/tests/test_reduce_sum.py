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

# reduce_all = false
test1 = ApiBase(
    func=paddle.sum,
    feed_names=['data'],
    feed_shapes=[[1, 640, 640]],
    threshold=1.0e-3,
)


@pytest.mark.reduce_sum
@pytest.mark.filterwarnings('ignore::UserWarning')
def test_reduce_sum_1():
    data = np.random.random(size=[1, 640, 640]).astype('float32')
    test1.run(feed=[data], axis=[-1], keepdim=False)


# reduce_all = true
test2 = ApiBase(
    func=paddle.sum,
    feed_names=['data'],
    feed_shapes=[[1, 640, 640]],
    threshold=1.0e-2,
)


@pytest.mark.reduce_sum
@pytest.mark.filterwarnings('ignore::UserWarning')
def test_reduce_sum_2():
    data = np.random.random(size=[1, 640, 640]).astype('float32')
    test2.run(feed=[data], axis=list(range(len(data.shape))), keepdim=False)


# reduce_all = false
test3 = ApiBase(
    func=paddle.sum,
    feed_names=['data'],
    feed_shapes=[[1, 640, 640]],
    threshold=1.0e-3,
)


@pytest.mark.reduce_sum
@pytest.mark.filterwarnings('ignore::UserWarning')
def test_reduce_sum_3():
    data = np.random.random(size=[1, 640, 640]).astype('float32')
    test3.run(feed=[data], axis=[0], keepdim=False)


test4 = ApiBase(
    func=paddle.sum, feed_names=['data'], feed_shapes=[[1, 704, 1280]]
)


@pytest.mark.reduce_sum
@pytest.mark.filterwarnings('ignore::UserWarning')
def test_reduce_sum_4():
    data = np.random.random(size=[1, 704, 1280]).astype('float32')
    test4.run(feed=[data], axis=[0, 1, 2], keepdim=False)


test5 = ApiBase(func=paddle.sum, feed_names=['data'], feed_shapes=[[1]])


@pytest.mark.reduce_sum
@pytest.mark.filterwarnings('ignore::UserWarning')
def test_reduce_sum_5():
    data = np.random.random(size=[1]).astype('float32')
    test5.run(feed=[data], axis=[0], keepdim=False)
