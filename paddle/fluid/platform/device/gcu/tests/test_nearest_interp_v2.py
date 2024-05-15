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

test1 = ApiBase(
    func=paddle.nn.functional.interpolate,
    feed_names=['data'],
    feed_shapes=[[2, 96, 30, 30]],
)


@pytest.mark.nearest_interp_v2
@pytest.mark.filterwarnings('ignore::UserWarning')
def test_nearest_interp_v2_1():
    data = np.random.random(size=[2, 96, 30, 30]).astype('float32')
    test1.run(feed=[data], scale_factor=2.0)


test2 = ApiBase(
    func=paddle.nn.functional.interpolate,
    feed_names=['data'],
    feed_shapes=[[2, 96, 60, 60]],
)


@pytest.mark.nearest_interp_v2
@pytest.mark.filterwarnings('ignore::UserWarning')
def test_nearest_interp_v2_2():
    data = np.random.random(size=[2, 96, 60, 60]).astype('float32')
    test2.run(feed=[data], size=[120, 120])


test3 = ApiBase(
    func=paddle.nn.functional.interpolate,
    feed_names=['data'],
    feed_shapes=[[2, 96, 120, 120]],
)


@pytest.mark.nearest_interp_v2
@pytest.mark.filterwarnings('ignore::UserWarning')
def test_nearest_interp_v2_3():
    data = np.random.random(size=[2, 96, 120, 120]).astype('float32')
    test3.run(feed=[data], size=[240, 240])
