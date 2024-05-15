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
    func=paddle.static.nn.batch_norm,  # paddle.add
    feed_names=['data'],
    feed_shapes=[[2, 3, 2, 4]],
)


@pytest.mark.batch_norm
@pytest.mark.filterwarnings('ignore::UserWarning')
def test_batch_norm_1():
    data = np.random.random(size=[2, 3, 2, 4]).astype('float32')
    test1.run(feed=[data])


test2 = ApiBase(
    func=paddle.static.nn.batch_norm,  # paddle.add
    feed_names=['data'],
    feed_shapes=[[1, 5, 7, 9]],
)


@pytest.mark.batch_norm
@pytest.mark.filterwarnings('ignore::UserWarning')
def test_batch_norm_2():
    data = np.random.random(size=[1, 5, 7, 9]).astype('float32')
    test2.run(feed=[data])


test3 = ApiBase(
    func=paddle.static.nn.batch_norm,  # paddle.add
    feed_names=['data'],
    feed_shapes=[[1, 8, 352, 640]],
)


@pytest.mark.batch_norm
@pytest.mark.filterwarnings('ignore::UserWarning')
def test_batch_norm_3():
    data = np.random.uniform(-1, 1, (1, 8, 352, 640)).astype('float32')
    test3.run(feed=[data])


test4 = ApiBase(
    func=paddle.static.nn.batch_norm,  # paddle.add
    feed_names=['data'],
    feed_shapes=[[8, 3, 224, 224]],
    is_train=False,
)


@pytest.mark.batch_norm
@pytest.mark.filterwarnings('ignore::UserWarning')
def test_batch_norm_4():
    data = np.random.uniform(-1, 1, (8, 3, 224, 224)).astype('float32')
    test4.run(feed=[data], is_test=True)
