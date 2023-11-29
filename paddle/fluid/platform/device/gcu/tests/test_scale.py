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

test_after_scale = ApiBase(
    func=paddle.scale, feed_names=["data"], feed_shapes=[[2, 3]], is_train=False
)


@pytest.mark.scale
@pytest.mark.filterwarnings('ignore::UserWarning')
def test_scale_bias_after_scale():
    np.random.seed(1)
    data = np.random.random(size=[2, 3]).astype('float32')
    test_after_scale.run(feed=[data], scale=2, bias=0.5)


test_afte_bias = ApiBase(
    func=paddle.scale, feed_names=["data"], feed_shapes=[[2, 3]], is_train=False
)


@pytest.mark.scale
@pytest.mark.filterwarnings('ignore::UserWarning')
def test_scale_scale_after_bias():
    np.random.seed(1)
    data = np.random.random(size=[2, 3]).astype('float32')
    test_afte_bias.run(feed=[data], scale=2, bias=0.5, bias_after_scale=False)


test1 = ApiBase(
    func=paddle.scale,
    feed_names=["data"],
    feed_shapes=[[1, 1, 704, 1280]],
    is_train=False,
)


@pytest.mark.scale
@pytest.mark.filterwarnings('ignore::UserWarning')
def test_scale_1():
    np.random.seed(1)
    data = np.random.random(size=[1, 1, 704, 1280]).astype('float32')
    test1.run(feed=[data], scale=-50, bias=0, bias_after_scale=True)


test2 = ApiBase(
    func=paddle.scale,
    feed_names=["data"],
    feed_shapes=[[1, 1, 704, 1280]],
    is_train=False,
)


@pytest.mark.scale
@pytest.mark.filterwarnings('ignore::UserWarning')
def test_scale_2():
    np.random.seed(1)
    data = np.random.random(size=[1, 1, 704, 1280]).astype('float32')
    test2.run(feed=[data], scale=1, bias=1, bias_after_scale=True)


test3 = ApiBase(
    func=paddle.scale, feed_names=["data"], feed_shapes=[[1]], is_train=False
)


@pytest.mark.scale
@pytest.mark.filterwarnings('ignore::UserWarning')
def test_scale_3():
    np.random.seed(1)
    data = np.random.random(size=[1]).astype('float32')
    test3.run(feed=[data], scale=3, bias=0, bias_after_scale=True)


test4 = ApiBase(
    func=paddle.scale, feed_names=["data"], feed_shapes=[[1]], is_train=False
)


@pytest.mark.scale
@pytest.mark.filterwarnings('ignore::UserWarning')
def test_scale_4():
    np.random.seed(1)
    data = np.random.random(size=[1]).astype('float32')
    test4.run(
        feed=[data], scale=1, bias=9.999999974752427e-7, bias_after_scale=True
    )


# if bias_after_scale=True:
#   Out = scale*X + bias
# else:
#   Out = scale*(X + bias)

# 480, 80, 1, 1 bias=0, bias_after_scale=True, scale=0
# 480 bias=0, bias_after_scale=True, scale=0
# 16, 1, 704, 1280 bias=0, bias_after_scale=True, scale=-50
# 16, 1, 704, 1280 bias=0, bias_after_scale=True, scale=1
# 16, 1, 704, 1280 bias=1, bias_after_scale=True, scale=1
# 16, 704, 1280 bias=1, bias_after_scale=True, scale=-1
# 1 bias=0, bias_after_scale=True, scale=-1
# 1 bias=0, bias_after_scale=True, scale=1
# 1 bias=0, bias_after_scale=True, scale=2
# 1 bias=0, bias_after_scale=True, scale=3
# 1 bias=0, bias_after_scale=True, scale=5
# 1 bias=0, bias_after_scale=True, scale=10
# 1 bias=1, bias_after_scale=True, scale=-1
