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


@pytest.mark.log_softmax
@pytest.mark.filterwarnings('ignore::UserWarning')
def test_log_softmax():
    test = ApiBase(
        func=paddle.nn.functional.log_softmax,
        feed_names=['data'],
        feed_shapes=[[2, 3, 4]],
        is_train=True,
    )
    data = np.array(
        [
            [[5, 8, 9, 5], [0, 0, 1, 7], [6, 9, 2, 4]],
            [[5, 2, 4, 2], [4, 7, 7, 9], [1, 7, 0, 6]],
        ],
        dtype=np.float32,
    )
    test.run(feed=[data])


def test_log_softmax_axis0():
    test = ApiBase(
        func=paddle.nn.functional.log_softmax,
        feed_names=['data'],
        feed_shapes=[[2, 3, 4]],
        is_train=True,
    )
    data = np.array(
        [
            [[5, 8, 9, 5], [0, 0, 1, 7], [6, 9, 2, 4]],
            [[5, 2, 4, 2], [4, 7, 7, 9], [1, 7, 0, 6]],
        ],
        dtype=np.float32,
    )
    test.run(feed=[data], axis=0)


def test_log_softmax_axis1():
    test = ApiBase(
        func=paddle.nn.functional.log_softmax,
        feed_names=['data'],
        feed_shapes=[[2, 3, 4]],
        is_train=True,
    )
    data = np.array(
        [
            [[5, 8, 9, 5], [0, 0, 1, 7], [6, 9, 2, 4]],
            [[5, 2, 4, 2], [4, 7, 7, 9], [1, 7, 0, 6]],
        ],
        dtype=np.float32,
    )
    test.run(feed=[data], axis=1)


def test_log_softmax_axis2():
    test = ApiBase(
        func=paddle.nn.functional.log_softmax,
        feed_names=['data'],
        feed_shapes=[[2, 3, 4]],
        is_train=True,
    )
    data = np.array(
        [
            [[5, 8, 9, 5], [0, 0, 1, 7], [6, 9, 2, 4]],
            [[5, 2, 4, 2], [4, 7, 7, 9], [1, 7, 0, 6]],
        ],
        dtype=np.float32,
    )
    test.run(feed=[data], axis=2)
