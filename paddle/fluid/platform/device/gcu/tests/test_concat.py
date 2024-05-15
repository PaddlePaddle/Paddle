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
    func=paddle.concat,
    feed_names=['data_0', 'data_1', 'data_2'],
    feed_shapes=[[1, 1, 2, 2], [1, 1, 2, 2], [1, 2, 2, 2]],
    input_is_list=True,
)


@pytest.mark.concat
@pytest.mark.filterwarnings('ignore::UserWarning')
def test_concat():
    np.random.seed(1)
    data_0 = np.random.uniform(0, 1, (1, 1, 2, 2)).astype("float32")
    data_1 = np.random.uniform(0, 1, (1, 1, 2, 2)).astype("float32")
    data_2 = np.random.uniform(0, 1, (1, 2, 2, 2)).astype("float32")
    test.run(feed=[data_0, data_1, data_2], axis=1)


test1 = ApiBase(
    func=paddle.concat,
    feed_names=['data_0', 'data_1', 'data_2', 'data_3'],
    feed_shapes=[
        [16, 24, 176, 320],
        [16, 24, 176, 320],
        [16, 24, 176, 320],
        [16, 24, 176, 320],
    ],
    input_is_list=True,
)


@pytest.mark.concat
@pytest.mark.filterwarnings('ignore::UserWarning')
def test_concat1():
    np.random.seed(1)
    data_0 = np.random.uniform(0, 1, (16, 24, 176, 320)).astype("float32")
    data_1 = np.random.uniform(0, 1, (16, 24, 176, 320)).astype("float32")
    data_2 = np.random.uniform(0, 1, (16, 24, 176, 320)).astype("float32")
    data_3 = np.random.uniform(0, 1, (16, 24, 176, 320)).astype("float32")
    test1.run(feed=[data_0, data_1, data_2, data_3], axis=1)


test2 = ApiBase(
    func=paddle.concat,
    feed_names=['data_0', 'data_1', 'data_2'],
    feed_shapes=[[16, 1, 704, 1280], [16, 1, 704, 1280], [16, 1, 704, 1280]],
    input_is_list=True,
)


@pytest.mark.concat
@pytest.mark.filterwarnings('ignore::UserWarning')
def test_concat2():
    np.random.seed(1)
    data_0 = np.random.uniform(0, 1, (16, 1, 704, 1280)).astype("float32")
    data_1 = np.random.uniform(0, 1, (16, 1, 704, 1280)).astype("float32")
    data_2 = np.random.uniform(0, 1, (16, 1, 704, 1280)).astype("float32")
    test2.run(feed=[data_0, data_1, data_2], axis=1)
