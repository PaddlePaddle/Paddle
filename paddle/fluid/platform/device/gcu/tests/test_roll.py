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

test1 = ApiBase(func=paddle.roll, feed_names=['data'], feed_shapes=[[3, 4, 5]])


@pytest.mark.roll
@pytest.mark.filterwarnings('ignore::UserWarning')
def test_roll_1():
    # data = np.arange(9).reshape(3,3).astype('float32')
    data = np.random.random(size=[3, 4, 5]).astype('float32')
    test1.run(feed=[data], shifts=[2, 2, 2], axis=[0, 1, 2])


test2 = ApiBase(
    func=paddle.roll, feed_names=['data'], feed_shapes=[[31, 34, 12, 7, 43]]
)


@pytest.mark.roll
@pytest.mark.filterwarnings('ignore::UserWarning')
def test_roll_2():
    data = np.random.random(size=[31, 34, 12, 7, 43]).astype('float32')
    test2.run(feed=[data], shifts=[-5, 324, -433, 31], axis=[0, 2, 1, 3])


test3 = ApiBase(
    func=paddle.roll, feed_names=['data'], feed_shapes=[[31, 34, 43]]
)


@pytest.mark.roll
@pytest.mark.filterwarnings('ignore::UserWarning')
def test_roll_3():
    data = np.random.random(size=[31, 34, 43]).astype('float32')
    test3.run(feed=[data], shifts=[1234])
