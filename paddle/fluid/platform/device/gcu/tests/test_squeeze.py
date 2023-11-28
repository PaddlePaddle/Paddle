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

from paddle.fluid.data_feeder import check_type, check_variable_and_dtype
from paddle.fluid.layer_helper import LayerHelper


def my_squeeze(input, axes, name=None):
    helper = LayerHelper("squeeze", **locals())
    check_variable_and_dtype(
        input,
        'input',
        [
            'float16',
            'float32',
            'float64',
            'bool',
            'int8',
            'int32',
            'int64',
            'complex64',
            'complex128',
        ],
        'squeeze',
    )
    check_type(axes, 'axis/axes', (list, tuple), 'squeeze')
    out = helper.create_variable_for_type_inference(dtype=input.dtype)
    x_shape = helper.create_variable_for_type_inference(dtype=input.dtype)
    helper.append_op(
        type="squeeze",
        inputs={"X": input},
        attrs={"axes": axes},
        outputs={"Out": out},
    )

    return out


test = ApiBase(
    func=my_squeeze,
    feed_names=['data'],
    feed_shapes=[[2, 1, 4]],
    input_is_list=False,
    is_train=True,
)


@pytest.mark.squeeze
@pytest.mark.filterwarnings('ignore::UserWarning')
def test_squeeze():
    np.random.seed(1)
    data = np.random.uniform(0, 1, (2, 1, 4)).astype("float32")
    test.run(feed=[data], axes=[1])


test2 = ApiBase(
    func=my_squeeze,
    feed_names=['data'],
    feed_shapes=[[2, 1, 3, 1, 4]],
    input_is_list=False,
    is_train=True,
)


@pytest.mark.squeeze
@pytest.mark.filterwarnings('ignore::UserWarning')
def test_squeeze2():
    np.random.seed(1)
    data = np.random.uniform(0, 1, (2, 1, 3, 1, 4)).astype("float32")
    test2.run(feed=[data], axes=[1])
