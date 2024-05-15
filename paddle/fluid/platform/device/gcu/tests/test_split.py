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

test1 = ApiBase(func=paddle.split, feed_names=['data'], feed_shapes=[[2, 3]])


@pytest.mark.split
@pytest.mark.filterwarnings('ignore::UserWarning')
def test_split_1():
    data = np.arange(6).reshape(2, 3).astype('float32')
    test1.run(feed=[data], axis=1, num_or_sections=3)


test2 = ApiBase(func=paddle.split, feed_names=['data'], feed_shapes=[[2, 3]])


@pytest.mark.split
@pytest.mark.filterwarnings('ignore::UserWarning')
def test_split_2():
    data = np.arange(6).reshape(2, 3).astype('float32')
    test2.run(feed=[data], axis=-1, num_or_sections=[2, -1])


test3 = ApiBase(func=paddle.split, feed_names=['data'], feed_shapes=[[2, 3]])


@pytest.mark.split
@pytest.mark.filterwarnings('ignore::UserWarning')
def test_split_3():
    data = np.arange(6).reshape(2, 3).astype('float32')
    test3.run(feed=[data], axis=1, num_or_sections=[2, 1])


test4 = ApiBase(func=paddle.split, feed_names=['data'], feed_shapes=[[2, 3]])


@pytest.mark.split
@pytest.mark.filterwarnings('ignore::UserWarning')
def test_split_4():
    data = np.arange(6).reshape(2, 3).astype('float32')
    test4.run(feed=[data], num_or_sections=2)
