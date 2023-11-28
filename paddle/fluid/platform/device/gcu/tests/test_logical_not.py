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


@pytest.mark.logical_not
@pytest.mark.filterwarnings('ignore::UserWarning')
def test_logical_not():
    support_types = [
        'bool',
        'int8',
        'int16',
        'int32',
        'int64',
        'float32',
        'float64',
    ]
    for ty in support_types:
        rd_sz_h = np.random.randint(1, 64)
        rd_sz_w = np.random.randint(1, 64)
        test = ApiBase(
            func=paddle.logical_not,
            feed_names=['X'],
            feed_shapes=[[1, 3, rd_sz_h, rd_sz_w]],
            is_train=False,
            feed_dtypes=[ty],
        )
        x = np.random.randint(2, size=(1, 3, rd_sz_h, rd_sz_w)) - 1
        print(x)
        x = x.astype(ty)
        test.run(feed=[x])
