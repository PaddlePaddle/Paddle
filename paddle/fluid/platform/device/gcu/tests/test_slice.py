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
    func=paddle.slice, feed_names=['data'], feed_shapes=[[1, 3, 640, 640]]
)


@pytest.mark.slice
@pytest.mark.filterwarnings('ignore::UserWarning')
def test_slice_1():
    data = np.random.random(size=[1, 3, 640, 640]).astype('float32')
    test1.run(feed=[data], axes=[1], starts=[1], ends=[2])


test2 = ApiBase(
    func=paddle.slice, feed_names=['data'], feed_shapes=[[1, 3, 640, 640]]
)


@pytest.mark.slice
@pytest.mark.filterwarnings('ignore::UserWarning')
def test_slice_2():
    data = np.random.random(size=[1, 3, 640, 640]).astype('float32')
    test2.run(feed=[data], axes=[1], starts=[0], ends=[1])


test3 = ApiBase(
    func=paddle.slice, feed_names=['data'], feed_shapes=[[1, 3, 704, 1280]]
)


@pytest.mark.slice
@pytest.mark.filterwarnings('ignore::UserWarning')
def test_slice_3():
    data = np.random.random(size=[1, 3, 704, 1280]).astype('float32')
    test3.run(feed=[data], axes=[1], starts=[0], ends=[1])


test4 = ApiBase(
    func=paddle.slice, feed_names=['data'], feed_shapes=[[1, 3, 704, 1280]]
)


@pytest.mark.slice
@pytest.mark.filterwarnings('ignore::UserWarning')
def test_slice_4():
    data = np.random.random(size=[1, 3, 704, 1280]).astype('float32')
    test4.run(feed=[data], axes=[1], starts=[2], ends=[3])


test5 = ApiBase(
    func=paddle.slice, feed_names=['data'], feed_shapes=[[1, 3, 704, 1280]]
)


@pytest.mark.slice
@pytest.mark.filterwarnings('ignore::UserWarning')
def test_slice_5():
    data = np.random.random(size=[1, 3, 704, 1280]).astype('float32')
    test5.run(feed=[data], axes=[1], starts=[1], ends=[2])


test6 = ApiBase(
    func=paddle.slice, feed_names=['data'], feed_shapes=[[1, 3, 640, 640]]
)


@pytest.mark.slice
@pytest.mark.filterwarnings('ignore::UserWarning')
def test_slice_6():
    data = np.random.random(size=[1, 3, 640, 640]).astype('float32')
    test6.run(feed=[data], axes=[1], starts=[-2147483647], ends=[2147483647])
