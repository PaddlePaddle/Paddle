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


@pytest.mark.conv2d_transpose
@pytest.mark.filterwarnings('ignore::UserWarning')
def test_conv2d_transpose():
    test = ApiBase(
        func=paddle.nn.functional.conv2d_transpose,
        feed_names=['data', 'kernel'],
        feed_shapes=[[1, 2, 4, 4], [2, 4, 3, 3]],
    )
    np.random.seed(1)
    data = np.random.uniform(-1, 1, (1, 2, 4, 4)).astype('float32')
    kernel = np.random.uniform(-1, 1, (2, 4, 3, 3)).astype('float32')
    test.run(feed=[data, kernel], bias=None, stride=2)


@pytest.mark.conv2d_transpose
@pytest.mark.filterwarnings('ignore::UserWarning')
def test_conv2d_transpose_1():
    test = ApiBase(
        func=paddle.nn.functional.conv2d_transpose,
        feed_names=['data', 'kernel'],
        feed_shapes=[[1, 24, 176, 320], [24, 24, 2, 2]],
    )
    np.random.seed(1)
    data = np.random.uniform(-1, 1, (1, 24, 176, 320)).astype('float32')
    kernel = np.random.uniform(-1, 1, (24, 24, 2, 2)).astype('float32')
    test.run(feed=[data, kernel], bias=None, stride=2)


@pytest.mark.conv2d_transpose
@pytest.mark.filterwarnings('ignore::UserWarning')
def test_conv2d_transpose_2():
    test = ApiBase(
        func=paddle.nn.functional.conv2d_transpose,
        feed_names=['data', 'kernel'],
        feed_shapes=[[16, 24, 352, 240], [24, 1, 2, 2]],
    )
    np.random.seed(1)
    data = np.random.uniform(-1, 1, (16, 24, 352, 240)).astype('float32')
    kernel = np.random.uniform(-1, 1, (24, 1, 2, 2)).astype('float32')
    test.run(feed=[data, kernel], bias=None, stride=2)


@pytest.mark.conv2d_transpose
@pytest.mark.filterwarnings('ignore::UserWarning')
def test_conv2d_transpose_3():
    test = ApiBase(
        func=paddle.nn.functional.conv2d_transpose,
        feed_names=['data', 'kernel'],
        feed_shapes=[[2, 3, 2, 2], [3, 1, 3, 3]],
    )
    np.random.seed(1)
    data = np.random.uniform(-1, 1, (2, 3, 2, 2)).astype('float32')
    kernel = np.random.uniform(-1, 1, (3, 1, 3, 3)).astype('float32')
    test.run(feed=[data, kernel], bias=None, stride=1, dilation=2)


@pytest.mark.conv2d_transpose
@pytest.mark.filterwarnings('ignore::UserWarning')
def test_conv2d_transpose_4():
    test = ApiBase(
        func=paddle.nn.functional.conv2d_transpose,
        feed_names=['data', 'kernel'],
        feed_shapes=[[2, 3, 2, 2], [3, 1, 3, 3]],
    )
    np.random.seed(1)
    data = np.random.uniform(-1, 1, (2, 3, 2, 2)).astype('float32')
    kernel = np.random.uniform(-1, 1, (3, 1, 3, 3)).astype('float32')
    test.run(feed=[data, kernel], bias=None, stride=2, dilation=2)
