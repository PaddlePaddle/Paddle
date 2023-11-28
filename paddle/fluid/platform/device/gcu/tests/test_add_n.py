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


@pytest.mark.add_n
@pytest.mark.filterwarnings('ignore::UserWarning')
def test_add_n():
    test = ApiBase(
        func=paddle.add_n,
        feed_names=['data_0', 'data_1', 'data_2'],
        feed_shapes=[[2, 2], [2, 2], [2, 2]],
        input_is_list=True,
        is_train=False,
    )
    np.random.seed(1)
    data_0 = np.random.uniform(0, 1, (2, 2)).astype("float32")
    data_1 = np.random.uniform(0, 1, (2, 2)).astype("float32")
    data_2 = np.random.uniform(0, 1, (2, 2)).astype("float32")
    test.run(feed=[data_0, data_1, data_2])


@pytest.mark.add_n
@pytest.mark.filterwarnings('ignore::UserWarning')
def test_add_n_1():
    test = ApiBase(
        func=paddle.add_n,
        feed_names=['data_0', 'data_1'],
        feed_shapes=[[1], [1]],
        input_is_list=True,
        is_train=False,
    )
    data_0 = np.random.uniform(-5, 5, (1)).astype("float32")
    data_1 = np.random.uniform(-5, 5, (1)).astype("float32")
    test.run(feed=[data_0, data_1])


@pytest.mark.add_n
@pytest.mark.filterwarnings('ignore::UserWarning')
def test_add_n_2():
    test = ApiBase(
        func=paddle.add_n,
        feed_names=['data_0', 'data_1'],
        feed_shapes=[[16, 40, 44, 80], [16, 40, 44, 80]],
        input_is_list=True,
        is_train=False,
    )
    data_0 = np.random.uniform(-5, 5, (16, 40, 44, 80)).astype("float32")
    data_1 = np.random.uniform(-5, 5, (16, 40, 44, 80)).astype("float32")
    test.run(feed=[data_0, data_1])


@pytest.mark.add_n
@pytest.mark.filterwarnings('ignore::UserWarning')
def test_add_n_3():
    test = ApiBase(
        func=paddle.add_n,
        feed_names=['data_0', 'data_1'],
        feed_shapes=[[40], [40]],
        input_is_list=True,
        is_train=False,
    )
    data_0 = np.random.uniform(-5, 5, (40)).astype("float32")
    data_1 = np.random.uniform(-5, 5, (40)).astype("float32")
    test.run(feed=[data_0, data_1])


@pytest.mark.add_n
@pytest.mark.filterwarnings('ignore::UserWarning')
def test_add_n_4():
    test = ApiBase(
        func=paddle.add_n,
        feed_names=['data_0', 'data_1'],
        feed_shapes=[[40, 120, 1, 1], [40, 120, 1, 1]],
        input_is_list=True,
        is_train=False,
    )
    data_0 = np.random.uniform(-5, 5, (40, 120, 1, 1)).astype("float32")
    data_1 = np.random.uniform(-5, 5, (40, 120, 1, 1)).astype("float32")
    test.run(feed=[data_0, data_1])


@pytest.mark.add_n
@pytest.mark.filterwarnings('ignore::UserWarning')
def test_add_n_5():
    test = ApiBase(
        func=paddle.add_n,
        feed_names=['data_0', 'data_1'],
        feed_shapes=[[120, 1, 3, 3], [120, 1, 3, 3]],
        input_is_list=True,
        is_train=False,
    )
    data_0 = np.random.uniform(-5, 5, (120, 1, 3, 3)).astype("float32")
    data_1 = np.random.uniform(-5, 5, (120, 1, 3, 3)).astype("float32")
    test.run(feed=[data_0, data_1])
