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
    func=paddle.mean,
    feed_names=['data'],
    feed_shapes=[[1, 6, 6]],
    threshold=1.0e-5,
)


@pytest.mark.reduce_mean
@pytest.mark.filterwarnings('ignore::UserWarning')
def test_reduce_mean_1():
    np.random.seed(1)
    data = np.random.random(size=[1, 6, 6]).astype('float32')
    test1.run(feed=[data], axis=[-1], keepdim=True)


# reduce_all = true
test2 = ApiBase(
    func=paddle.mean,
    feed_names=['data'],
    feed_shapes=[[1, 6, 6]],
    threshold=1.0e-5,
)


@pytest.mark.reduce_mean
@pytest.mark.filterwarnings('ignore::UserWarning')
def test_reduce_mean_2():
    np.random.seed(1)
    data = np.random.random(size=[1, 6, 6]).astype('float32')
    test2.run(feed=[data], keepdim=False)


# reduce_all = true
test3 = ApiBase(func=paddle.mean, feed_names=['data'], feed_shapes=[[1]])


@pytest.mark.reduce_mean
@pytest.mark.filterwarnings('ignore::UserWarning')
def test_reduce_mean_3():
    np.random.seed(1)
    data = np.random.random(size=[1]).astype('float32')
    test3.run(feed=[data], keepdim=False)


# reduce_all = true
test4 = ApiBase(func=paddle.mean, feed_names=['data'], feed_shapes=[[1]])


@pytest.mark.reduce_mean
@pytest.mark.filterwarnings('ignore::UserWarning')
def test_reduce_mean_4():
    np.random.seed(1)
    data = np.random.random(size=[1]).astype('float32')
    test4.run(feed=[data], keepdim=False, axis=[-1])
