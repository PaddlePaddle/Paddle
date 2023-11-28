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

test = ApiBase(
    func=paddle.full,
    feed_names=[],
    feed_shapes=[],
    feed_dtypes=[],
    is_train=False,
)


@pytest.mark.fill_constant
@pytest.mark.filterwarnings('ignore::UserWarning')
def test_fill_constant_f32():
    np.random.seed(1)
    test.run(feed=[], shape=[2, 3], fill_value=0, dtype='float32')


test2 = ApiBase(
    func=paddle.full,
    feed_names=[],
    feed_shapes=[],
    feed_dtypes=[],
    is_train=False,
)


@pytest.mark.fill_constant
@pytest.mark.filterwarnings('ignore::UserWarning')
def test_fill_constant_f32_2():
    np.random.seed(1)
    test2.run(feed=[], shape=[1], fill_value=1, dtype='float32')


test3 = ApiBase(
    func=paddle.fluid.layers.fill_constant_batch_size_like,
    feed_names=['input'],
    feed_shapes=[[2, 2]],
    is_train=False,
)


@pytest.mark.fill_constant
@pytest.mark.filterwarnings('ignore::UserWarning')
def test_fill_constant_batch_size_like_1():
    like = np.random.random(size=[2, 2]).astype('float32')
    test3.run(feed=[like], shape=[2], value=0, dtype='int32')


test4 = ApiBase(
    func=paddle.fluid.layers.fill_constant_batch_size_like,
    feed_names=['input'],
    feed_shapes=[[1, 2]],
    is_train=False,
)


@pytest.mark.fill_constant
@pytest.mark.filterwarnings('ignore::UserWarning')
def test_fill_constant_batch_size_like_2():
    like = np.random.random(size=[1, 2]).astype('float32')
    test4.run(feed=[like], shape=[1], value=10.5, dtype='float32')
