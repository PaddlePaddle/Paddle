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
    func=paddle.tril,
    feed_names=['x'],
    feed_shapes=[[1, 9, 9, 4]],
    is_train=True,
)


@pytest.mark.tril_triu
@pytest.mark.filterwarnings('ignore::UserWarning')
def test_tril_0():
    data = np.random.random([1, 9, 9, 4]).astype('float32')
    test0.run(feed=[data], diagonal=0)


@pytest.mark.tril_triu
@pytest.mark.filterwarnings('ignore::UserWarning')
def test_tril_1():
    data = np.random.random([1, 9, 9, 4]).astype('float32')
    test0.run(feed=[data], diagonal=1)


@pytest.mark.tril_triu
@pytest.mark.filterwarnings('ignore::UserWarning')
def test_tril_2():
    data = np.random.random([1, 9, 9, 4]).astype('float32')
    test0.run(feed=[data], diagonal=2)


test1 = ApiBase(
    func=paddle.triu,
    feed_names=['x'],
    feed_shapes=[[1, 9, 9, 4]],
    is_train=True,
)


@pytest.mark.tril_triu
@pytest.mark.filterwarnings('ignore::UserWarning')
def test_triu_0():
    data = np.random.random([1, 9, 9, 4]).astype('float32')
    test1.run(feed=[data], diagonal=0)


@pytest.mark.tril_triu
@pytest.mark.filterwarnings('ignore::UserWarning')
def test_triu_1():
    data = np.random.random([1, 9, 9, 4]).astype('float32')
    test1.run(feed=[data], diagonal=1)


@pytest.mark.tril_triu
@pytest.mark.filterwarnings('ignore::UserWarning')
def test_triu_2():
    data = np.random.random([1, 9, 9, 4]).astype('float32')
    test1.run(feed=[data], diagonal=2)
