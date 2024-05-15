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

from paddle.fluid.data_feeder import check_variable_and_dtype
from paddle.fluid.layer_helper import LayerHelper


def my_squared_l2_norm(input, name=None):
    helper = LayerHelper("squared_l2_norm", **locals())
    check_variable_and_dtype(
        input, 'x', ['float32', 'float64'], 'squared_l2_norm'
    )
    out = helper.create_variable_for_type_inference(dtype=input.dtype)
    helper.append_op(
        type="squared_l2_norm", inputs={"X": input}, outputs={"Out": out}
    )
    return out


test1 = ApiBase(
    func=my_squared_l2_norm,
    feed_names=['data'],
    feed_shapes=[[1, 6, 6]],
    threshold=1.0e-5,
    is_train=False,
)


@pytest.mark.my_squared_l2_norm
@pytest.mark.filterwarnings('ignore::UserWarning')
def test_reduce_mean_1():
    np.random.seed(1)
    data = np.random.random(size=[1, 6, 6]).astype('float32')
    test1.run(feed=[data])


test2 = ApiBase(
    func=my_squared_l2_norm,
    feed_names=['data'],
    feed_shapes=[[1]],
    is_train=False,
)


@pytest.mark.my_squared_l2_norm
@pytest.mark.filterwarnings('ignore::UserWarning')
def test_squared_l2_norm_2():
    np.random.seed(1)
    data = np.random.random(size=[1]).astype('float32')
    test2.run(feed=[data])
