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
    func=paddle.vision.ops.prior_box,
    feed_names=['input', 'image'],
    is_train=False,
    feed_shapes=[[64, 512, 5, 5], [64, 3, 300, 300]],
    threshold=1.0e-5,
)


@pytest.mark.prior_box
@pytest.mark.filterwarnings('ignore::UserWarning')
def test_prior_box():
    np.random.seed(1)
    input = np.random.random(size=[64, 512, 5, 5]).astype('float32')
    image = np.random.random(size=[64, 3, 300, 300]).astype('float32')
    test1.run(
        feed=[input, image],
        min_sizes=[100.0],
        max_sizes=[200.0],
        clip=True,
        flip=True,
    )


test_prior_box()
