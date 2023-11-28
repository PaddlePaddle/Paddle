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
    func=paddle.floor, feed_names=['data'], feed_shapes=[[3]], is_train=True
)


@pytest.mark.floor_sub
@pytest.mark.filterwarnings('ignore::UserWarning')
def test_floor_sub_1():
    data = np.array([-0.4, 0.6, 2.3], dtype=np.float32)
    test1.run(feed=[data])


test2 = ApiBase(
    func=paddle.floor, feed_names=['data'], feed_shapes=[[2, 3]], is_train=True
)


@pytest.mark.floor_sub
@pytest.mark.filterwarnings('ignore::UserWarning')
def test_floor_sub_2():
    data = np.array([[-3.5, 2.1, 1.6], [-1, -3.1, 6.5]], dtype=np.float32)
    test2.run(feed=[data])
