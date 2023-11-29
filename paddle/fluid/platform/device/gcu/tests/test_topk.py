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
    func=paddle.topk, feed_names=['data'], feed_shapes=[[2, 10]], is_train=False
)


@pytest.mark.topk
@pytest.mark.filterwarnings('ignore::UserWarning')
def test_topk():
    np.random.seed(1)
    data = np.random.uniform(0, 1, (2, 10)).astype("float32")
    test.run(feed=[data], k=3)


test1 = ApiBase(
    func=paddle.topk,
    feed_names=['data'],
    feed_shapes=[[901120]],
    is_train=False,
)


@pytest.mark.topk
@pytest.mark.filterwarnings('ignore::UserWarning')
def test_topk1():
    np.random.seed(1)
    data = np.random.uniform(0, 1, (901120)).astype("float32")
    test1.run(feed=[data], k=1)


test2 = ApiBase(
    func=paddle.topk, feed_names=['data'], feed_shapes=[[3, 4]], is_train=False
)


@pytest.mark.topk
@pytest.mark.filterwarnings('ignore::UserWarning')
def test_topk_median():
    data = np.array(
        [[2, 3, 1, 1], [10, 1, 15, float('inf')], [4, 8, float('inf'), 7]]
    ).astype('float32')
    test2.run(feed=[data], k=2, axis=0, largest=True)
