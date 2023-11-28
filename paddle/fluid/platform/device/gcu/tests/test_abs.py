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


@pytest.mark.abs
@pytest.mark.filterwarnings('ignore::UserWarning')
def test_abs():
    test = ApiBase(func=paddle.abs, feed_names=['data'], feed_shapes=[[3]])
    data = np.array([-1, 0, 1], dtype=np.float32)
    test.run(feed=[data])


@pytest.mark.abs
@pytest.mark.filterwarnings('ignore::UserWarning')
def test_relu_1():
    test = ApiBase(
        func=paddle.abs, feed_names=['data'], feed_shapes=[[16, 704, 1280]]
    )
    data = np.random.uniform(-5, 5, (16, 704, 1280)).astype('float32')
    test.run(feed=[data])
