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
    func=paddle.cast,
    feed_names=['data'],
    feed_shapes=[[2, 3, 4]],
    feed_dtypes=['float32'],
    is_train=False,
)

test1 = ApiBase(
    func=paddle.cast,
    feed_names=['data'],
    feed_shapes=[[2, 3, 4]],
    feed_dtypes=['int64'],
    is_train=False,
)


@pytest.mark.cast
@pytest.mark.filterwarnings('ignore::UserWarning')
def test_cast_f32_to_i32():
    np.random.seed(1)
    data = np.random.uniform(0, 5, (2, 3, 4)).astype("float32")
    test.run(feed=[data], dtype='int32')


@pytest.mark.cast
@pytest.mark.filterwarnings('ignore::UserWarning')
def test_cast_i64_to_f32():
    data = np.array(list(range(24)), dtype=np.int64).reshape([2, 3, 4])
    test1.run(feed=[data], dtype='float32')


test2 = ApiBase(
    func=paddle.cast,
    feed_names=['data'],
    feed_shapes=[[1]],
    feed_dtypes=['float32'],
    is_train=False,
)


@pytest.mark.cast
@pytest.mark.filterwarnings('ignore::UserWarning')
def test_cast_f32_to_i32_2():
    np.random.seed(1)
    data = np.random.uniform(0, 1000, (1)).astype("float32")
    test2.run(feed=[data], dtype='int32')
