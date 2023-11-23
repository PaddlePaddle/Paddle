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
    func=paddle.nn.functional.grid_sample,
    feed_names=['data', 'grid'],
    feed_shapes=[[1, 1, 3, 2], [1, 2, 4, 2]],
    is_train=False,
)


@pytest.mark.grid_sampler
@pytest.mark.filterwarnings('ignore::UserWarning')
def test_grid_sampler_1():
    data = np.array([[[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]]], dtype=np.float32)
    grid = np.array(
        [
            [
                [[-1, -1], [-0.5, -0.5], [-0.2, -0.2], [0.1, 0.1]],
                [[0.1, 0.1], [-0.2, -0.2], [0.5, 0.5], [1, 1]],
            ]
        ],
        dtype=np.float32,
    )
    test1.run(
        feed=[data, grid],
        mode="bilinear",
        align_corners=False,
        padding_mode="zeros",
    )


test2 = ApiBase(
    func=paddle.nn.functional.grid_sample,
    feed_names=['data', 'grid'],
    feed_shapes=[[1, 1, 3, 2], [1, 2, 4, 2]],
    is_train=False,
)


@pytest.mark.grid_sampler
@pytest.mark.filterwarnings('ignore::UserWarning')
def test_grid_sampler_2():
    np.random.seed(1)
    data = np.random.random(size=[1, 1, 3, 2]).astype('float32')
    grid = np.random.random(size=[1, 2, 4, 2]).astype('float32')
    test2.run(
        feed=[data, grid],
        mode="bilinear",
        align_corners=True,
        padding_mode="zeros",
    )
