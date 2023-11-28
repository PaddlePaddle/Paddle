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
    func=paddle.pow,  # paddle.pow API use "elementwise_pow" operator when y is a Tensor.
    feed_names=['x', 'y'],
    # is_train=False,
    feed_shapes=[[2, 2, 3], [2, 3]],
    feed_dtypes=["float32", "float32"],
)


@pytest.mark.elementwise_pow
@pytest.mark.filterwarning('ignore::UserWarning')
def test_elementwise_pow():
    x = np.array(
        [[[2.1, 3.2, 2.3], [4.4, 6.5, 3.1]], [[7.7, 8.8, 6.6], [5.5, 9.9, 3.3]]]
    ).astype('float32')
    y = np.array([[2.2, 3.3, 2.2], [2.2, 3.3, 2.2]]).astype('float32')
    test.run(feed=[x, y])
