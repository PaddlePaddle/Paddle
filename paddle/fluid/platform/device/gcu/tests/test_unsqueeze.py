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
from paddle.fluid.data_feeder import check_variable_and_dtype
from paddle.fluid.layer_helper import LayerHelper

test = ApiBase(
    func=paddle.unsqueeze,
    feed_names=['data'],
    feed_shapes=[[3, 4, 5]],
    input_is_list=False,
    is_train=True,
)


@pytest.mark.unsqueeze
@pytest.mark.filterwarnings('ignore::UserWarning')
def test_unsqueeze2():
    np.random.seed(1)
    data = np.random.uniform(0, 1, (3, 4, 5)).astype("float32")
    test.run(feed=[data], axis=[1, 2])


def my_unsqueeze(input, axes, name=None):
    check_variable_and_dtype(
        input,
        'input',
        [
            'float16',
            'float32',
            'float64',
            'bool',
            'int8',
            'int16',
            'int32',
            'int64',
            'complex64',
            'complex128',
        ],
        'unsqueeze',
    )
    helper = LayerHelper("unsqueeze", **locals())
    inputs = {"X": input}
    attrs = {}

    if isinstance(axes, int):
        axes = [axes]
    if isinstance(axes, (list)):
        attrs["axes"] = axes

    out = helper.create_variable_for_type_inference(dtype=input.dtype)
    helper.append_op(
        type="unsqueeze", inputs=inputs, attrs=attrs, outputs={"Out": out}
    )

    return out


test2 = ApiBase(
    func=my_unsqueeze,
    feed_names=['data'],
    feed_shapes=[[3, 4, 5]],
    input_is_list=False,
    is_train=True,
)


@pytest.mark.unsqueeze
@pytest.mark.filterwarnings('ignore::UserWarning')
def test_unsqueeze():
    np.random.seed(1)
    data = np.random.uniform(0, 1, (3, 4, 5)).astype("float32")
    test2.run(feed=[data], axes=[1, 2])
