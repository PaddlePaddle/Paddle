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
import six
from api_base import ApiBase

from paddle.fluid.data_feeder import check_type, check_variable_and_dtype
from paddle.fluid.framework import Variable
from paddle.fluid.layer_helper import LayerHelper


def my_one_hot_v2(input, depth, allow_out_of_range=False):
    helper = LayerHelper("one_hot", **locals())
    check_variable_and_dtype(input, 'input', ['int32', 'int64'], 'one_hot')
    check_type(depth, 'depth', (six.integer_types, Variable), 'one_hot')
    one_hot_out = helper.create_variable_for_type_inference(dtype='float32')

    if not isinstance(depth, Variable):
        # user attribute
        inputs = {'X': input}
        attrs = {'depth': depth, 'allow_out_of_range': allow_out_of_range}
    else:
        depth.stop_gradient = True
        inputs = {'X': input, 'depth_tensor': depth}
        attrs = {'allow_out_of_range': allow_out_of_range}
    helper.append_op(
        type="one_hot_v2",
        inputs=inputs,
        attrs=attrs,
        outputs={'Out': one_hot_out},
    )
    one_hot_out.stop_gradient = True
    return one_hot_out


test_0 = ApiBase(
    func=my_one_hot_v2,
    feed_names=["data"],
    feed_shapes=[[4, 2]],
    feed_dtypes=["int64"],
)


@pytest.mark.one_hot_v2
@pytest.mark.filterwarnings('ignore::UserWarning')
def test_one_hot_v2_0():
    data = np.array([[1, 0], [1, 1], [3, 0], [0, 2]]).astype('int64')
    test_0.run(feed=[data], depth=4)


test_1 = ApiBase(
    func=my_one_hot_v2,
    feed_names=["data"],
    feed_shapes=[[3]],
    feed_dtypes=["int64"],
)


@pytest.mark.one_hot_v2
@pytest.mark.filterwarnings('ignore::UserWarning')
def test_one_hot_v2_1():
    data = np.array([1, 1, 3]).astype('int64')
    test_1.run(feed=[data], depth=4, allow_out_of_range=True)


test_2 = ApiBase(
    func=my_one_hot_v2,
    feed_names=["data"],
    feed_shapes=[[4]],
    feed_dtypes=["int64"],
)


@pytest.mark.one_hot_v2
@pytest.mark.filterwarnings('ignore::UserWarning')
def test_one_hot_v2_2():
    data = np.array([1, 1, 3, 0]).astype('int64')
    test_2.run(feed=[data], depth=4, allow_out_of_range=False)


test_3 = ApiBase(
    func=my_one_hot_v2,
    feed_names=["data"],
    feed_shapes=[[4, 1]],
    feed_dtypes=["int64"],
)


@pytest.mark.one_hot_v2
@pytest.mark.filterwarnings('ignore::UserWarning')
def test_one_hot_v2_3():
    data = np.array([[1], [1], [3], [2]]).astype('int64')
    test_3.run(feed=[data], depth=4, allow_out_of_range=False)


test_4 = ApiBase(
    func=my_one_hot_v2,
    feed_names=["data"],
    feed_shapes=[[3]],
    feed_dtypes=["int32"],
)


@pytest.mark.one_hot_v2
@pytest.mark.filterwarnings('ignore::UserWarning')
def test_one_hot_v2_4():
    data = np.array([1, 1, 3]).astype('int32')
    test_4.run(feed=[data], depth=4, allow_out_of_range=True)
