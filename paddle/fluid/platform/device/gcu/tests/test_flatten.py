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
    func=paddle.flatten,
    feed_names=['data'],
    feed_shapes=[[2, 3, 4, 5]],
    input_is_list=False,
    is_train=True,
)
np.random.seed(1)


@pytest.mark.flatten
@pytest.mark.filterwarnings('ignore::UserWarning')
def test_flatten_contiguous_range_1():
    data = np.random.uniform(0, 1, (2, 3, 4, 5)).astype("float32")
    test.run(feed=[data], start_axis=2, stop_axis=3)


@pytest.mark.flatten
@pytest.mark.filterwarnings('ignore::UserWarning')
def test_flatten_contiguous_range_2():
    data = np.random.uniform(0, 1, (2, 3, 4, 5)).astype("float32")
    test.run(feed=[data], start_axis=0, stop_axis=-1)


@pytest.mark.flatten
@pytest.mark.filterwarnings('ignore::UserWarning')
def test_flatten_contiguous_range_3():
    data = np.random.uniform(0, 1, (2, 3, 4, 5)).astype("float32")
    test.run(feed=[data], start_axis=2, stop_axis=3)


def my_flatten(x, axis=1, name=None):
    check_variable_and_dtype(
        x,
        'x',
        ['float32', 'float64', 'int8', 'int32', 'int64', 'uint8'],
        'flatten',
    )
    helper = LayerHelper('flatten', **locals())

    if not (isinstance(axis, int)) or axis > len(x.shape) or axis < 0:
        raise ValueError("The axis should be a int, and in range [0, rank(x)]")

    out = helper.create_variable_for_type_inference(x.dtype)
    # x_shape = helper.create_variable_for_type_inference(x.dtype)
    helper.append_op(
        type='flatten',
        inputs={"X": x},
        outputs={'Out': out},
        attrs={"axis": axis},
    )
    return out


test2 = ApiBase(
    func=my_flatten,
    feed_names=['data'],
    feed_shapes=[[2, 3, 4, 5]],
    input_is_list=False,
    is_train=True,
)


@pytest.mark.flatten
@pytest.mark.filterwarnings('ignore::UserWarning')
def test_flatten_0():
    data = np.random.uniform(0, 1, (2, 3, 4, 5)).astype("float32")
    test2.run(feed=[data], axis=0)


@pytest.mark.flatten
@pytest.mark.filterwarnings('ignore::UserWarning')
def test_flatten_1():
    data = np.random.uniform(0, 1, (2, 3, 4, 5)).astype("float32")
    test2.run(feed=[data], axis=1)


@pytest.mark.flatten
@pytest.mark.filterwarnings('ignore::UserWarning')
def test_flatten_2():
    data = np.random.uniform(0, 1, (2, 3, 4, 5)).astype("float32")
    test2.run(feed=[data], axis=2)


@pytest.mark.flatten
@pytest.mark.filterwarnings('ignore::UserWarning')
def test_flatten_3():
    data = np.random.uniform(0, 1, (2, 3, 4, 5)).astype("float32")
    test2.run(feed=[data], axis=3)
