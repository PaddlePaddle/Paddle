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
    func=paddle.flip,
    feed_names=['data'],
    feed_shapes=[[3, 4, 5]],
    is_train=False,
)


@pytest.mark.flip
@pytest.mark.filterwarnings('ignore::UserWarning')
def test_flip_1():
    data = np.random.random(size=[3, 4, 5]).astype('float32')
    test1.run(feed=[data], axis=[-1, 1])


test2 = ApiBase(
    func=paddle.flip,
    feed_names=['data'],
    feed_shapes=[[6, 6, 6]],
    is_train=False,
)


@pytest.mark.flip
@pytest.mark.filterwarnings('ignore::UserWarning')
def test_flip_2():
    data = np.random.random(size=[6, 6, 6]).astype('float32')
    test2.run(feed=[data], axis=[1])
