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


@pytest.mark.nearest_interp
@pytest.mark.filterwarnings('ignore::UserWarning')
def test_nearest_interp_1():
    test = ApiBase(
        func=paddle.nn.functional.interpolate,
        feed_names=['data'],
        feed_shapes=[[16, 96, 22, 40]],
    )
    data = np.random.random(size=[16, 96, 22, 40]).astype('float32')
    test.run(feed=[data], scale_factor=2.0, mode='nearest')


@pytest.mark.nearest_interp
@pytest.mark.filterwarnings('ignore::UserWarning')
def test_nearest_interp_2():
    test = ApiBase(
        func=paddle.nn.functional.interpolate,
        feed_names=['data'],
        feed_shapes=[[2, 96, 22, 40]],
    )
    data = np.random.random(size=[2, 96, 22, 40]).astype('float32')
    test.run(feed=[data], size=[44, 80], mode='nearest')


@pytest.mark.nearest_interp
@pytest.mark.filterwarnings('ignore::UserWarning')
def test_nearest_interp_3():
    test = ApiBase(
        func=paddle.nn.functional.interpolate,
        feed_names=['data'],
        feed_shapes=[[16, 96, 44, 80]],
    )
    data = np.random.random(size=[16, 96, 44, 80]).astype('float32')
    test.run(feed=[data], scale_factor=2.0, mode='nearest')


@pytest.mark.nearest_interp
@pytest.mark.filterwarnings('ignore::UserWarning')
def test_nearest_interp_4():
    test = ApiBase(
        func=paddle.nn.functional.interpolate,
        feed_names=['data'],
        feed_shapes=[[16, 96, 88, 160]],
    )
    data = np.random.random(size=[16, 96, 88, 160]).astype('float32')
    test.run(feed=[data], scale_factor=2.0, mode='nearest')


@pytest.mark.nearest_interp
@pytest.mark.filterwarnings('ignore::UserWarning')
def test_nearest_interp_5():
    test = ApiBase(
        func=paddle.nn.functional.interpolate,
        feed_names=['data'],
        feed_shapes=[[16, 24, 22, 40]],
    )
    data = np.random.random(size=[16, 24, 22, 40]).astype('float32')
    test.run(feed=[data], scale_factor=8.0, mode='nearest')


@pytest.mark.nearest_interp
@pytest.mark.filterwarnings('ignore::UserWarning')
def test_nearest_interp_6():
    test = ApiBase(
        func=paddle.nn.functional.interpolate,
        feed_names=['data'],
        feed_shapes=[[16, 24, 44, 80]],
    )
    data = np.random.random(size=[16, 24, 44, 80]).astype('float32')
    test.run(feed=[data], scale_factor=4.0, mode='nearest')


@pytest.mark.nearest_interp
@pytest.mark.filterwarnings('ignore::UserWarning')
def test_nearest_interp_7():
    test = ApiBase(
        func=paddle.nn.functional.interpolate,
        feed_names=['data'],
        feed_shapes=[[16, 24, 88, 160]],
    )
    data = np.random.random(size=[16, 24, 88, 160]).astype('float32')
    test.run(feed=[data], scale_factor=2.0, mode='nearest')
