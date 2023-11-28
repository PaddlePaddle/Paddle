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

test = ApiBase(
    func=paddle.equal,
    feed_names=['X', 'Y'],
    feed_shapes=[[3], [3]],
    is_train=False,
)


@pytest.mark.equal
@pytest.mark.filterwarnings('ignore::UserWarning')
def test_equal():
    x = np.array([-0.4, -0.2, 1], dtype=np.float32)
    y = np.array([-0.4, -0.2, 1], dtype=np.float32)
    test.run(feed=[x, y])


test_equal()
