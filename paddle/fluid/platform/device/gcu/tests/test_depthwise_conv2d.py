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


@pytest.mark.depthwise_conv2d
@pytest.mark.filterwarnings('ignore::UserWarning')
def test_depthwise_conv2d():
    test = ApiBase(
        func=paddle.nn.functional.conv2d,
        feed_names=['data', 'kernel'],
        feed_shapes=[[1, 2, 8, 8], [2, 1, 3, 3]],
    )
    np.random.seed(1)
    data = np.random.uniform(-1, 1, (1, 2, 8, 8)).astype('float32')
    kernel = np.random.uniform(-1, 1, (2, 1, 3, 3)).astype('float32')
    test.run(feed=[data, kernel], bias=None, stride=2, padding=1, groups=2)


@pytest.mark.depthwise_conv2d
@pytest.mark.filterwarnings('ignore::UserWarning')
def test_depthwise_conv2d_1():
    test = ApiBase(
        func=paddle.nn.functional.conv2d,
        feed_names=['data', 'kernel'],
        feed_shapes=[[1, 8, 352, 640], [8, 1, 3, 3]],
    )
    np.random.seed(1)
    data = np.random.uniform(-1, 1, (1, 8, 352, 640)).astype('float32')
    kernel = np.random.uniform(-1, 1, (8, 1, 3, 3)).astype('float32')
    test.run(feed=[data, kernel], bias=None, stride=1, padding=1, groups=8)


@pytest.mark.depthwise_conv2d
@pytest.mark.filterwarnings('ignore::UserWarning')
def test_depthwise_conv2d_2():
    test = ApiBase(
        func=paddle.nn.functional.conv2d,
        feed_names=['data', 'kernel'],
        feed_shapes=[[16, 40, 172, 320], [40, 1, 5, 5]],
    )
    np.random.seed(1)
    data = np.random.uniform(-1, 1, (16, 40, 172, 320)).astype('float32')
    kernel = np.random.uniform(-1, 1, (40, 1, 5, 5)).astype('float32')
    test.run(feed=[data, kernel], bias=None, stride=2, padding=2, groups=40)
