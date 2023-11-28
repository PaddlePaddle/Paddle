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

test0 = ApiBase(
    func=paddle.strided_slice,
    feed_names=['data'],
    feed_shapes=[[2, 56, 56, 96]],
    is_train=True,
)


@pytest.mark.strided_slice
@pytest.mark.filterwarnings('ignore::UserWarning')
def test_slice_0():
    data = np.random.random(size=[2, 56, 56, 96]).astype('float32')
    test0.run(
        feed=[data],
        axes=[1, 2],
        starts=[0, 0],
        ends=[2147483647, 2147483647],
        strides=[2, 2],
    )


@pytest.mark.strided_slice
@pytest.mark.filterwarnings('ignore::UserWarning')
def test_slice_1():
    data = np.random.random(size=[2, 56, 56, 96]).astype('float32')
    test0.run(
        feed=[data],
        axes=[1, 2],
        starts=[1, 0],
        ends=[2147483647, 2147483647],
        strides=[2, 2],
    )


@pytest.mark.strided_slice
@pytest.mark.filterwarnings('ignore::UserWarning')
def test_slice_2():
    data = np.random.random(size=[2, 56, 56, 96]).astype('float32')
    test0.run(
        feed=[data],
        axes=[1, 2],
        starts=[0, 1],
        ends=[2147483647, 2147483647],
        strides=[2, 2],
    )


@pytest.mark.strided_slice
@pytest.mark.filterwarnings('ignore::UserWarning')
def test_slice_3():
    data = np.random.random(size=[2, 56, 56, 96]).astype('float32')
    test0.run(
        feed=[data],
        axes=[1, 2],
        starts=[1, 1],
        ends=[2147483647, 2147483647],
        strides=[2, 2],
    )


test1 = ApiBase(
    func=paddle.strided_slice,
    feed_names=['data'],
    feed_shapes=[[2, 28, 28, 192]],
    is_train=True,
)


@pytest.mark.strided_slice
@pytest.mark.filterwarnings('ignore::UserWarning')
def test_slice_4():
    data = np.random.random(size=[2, 28, 28, 192]).astype('float32')
    test1.run(
        feed=[data],
        axes=[1, 2],
        starts=[0, 0],
        ends=[2147483647, 2147483647],
        strides=[2, 2],
    )


@pytest.mark.strided_slice
@pytest.mark.filterwarnings('ignore::UserWarning')
def test_slice_5():
    data = np.random.random(size=[2, 28, 28, 192]).astype('float32')
    test1.run(
        feed=[data],
        axes=[1, 2],
        starts=[1, 0],
        ends=[2147483647, 2147483647],
        strides=[2, 2],
    )


@pytest.mark.strided_slice
@pytest.mark.filterwarnings('ignore::UserWarning')
def test_slice_6():
    data = np.random.random(size=[2, 28, 28, 192]).astype('float32')
    test1.run(
        feed=[data],
        axes=[1, 2],
        starts=[0, 1],
        ends=[2147483647, 2147483647],
        strides=[2, 2],
    )


@pytest.mark.strided_slice
@pytest.mark.filterwarnings('ignore::UserWarning')
def test_slice_7():
    data = np.random.random(size=[2, 28, 28, 192]).astype('float32')
    test1.run(
        feed=[data],
        axes=[1, 2],
        starts=[1, 1],
        ends=[2147483647, 2147483647],
        strides=[2, 2],
    )


test2 = ApiBase(
    func=paddle.strided_slice,
    feed_names=['data'],
    feed_shapes=[[2, 14, 14, 384]],
    is_train=True,
)


@pytest.mark.strided_slice
@pytest.mark.filterwarnings('ignore::UserWarning')
def test_slice_8():
    data = np.random.random(size=[2, 14, 14, 384]).astype('float32')
    test2.run(
        feed=[data],
        axes=[1, 2],
        starts=[0, 0],
        ends=[2147483647, 2147483647],
        strides=[2, 2],
    )


@pytest.mark.strided_slice
@pytest.mark.filterwarnings('ignore::UserWarning')
def test_slice_9():
    data = np.random.random(size=[2, 14, 14, 384]).astype('float32')
    test2.run(
        feed=[data],
        axes=[1, 2],
        starts=[1, 0],
        ends=[2147483647, 2147483647],
        strides=[2, 2],
    )


@pytest.mark.strided_slice
@pytest.mark.filterwarnings('ignore::UserWarning')
def test_slice_10():
    data = np.random.random(size=[2, 14, 14, 384]).astype('float32')
    test2.run(
        feed=[data],
        axes=[1, 2],
        starts=[0, 1],
        ends=[2147483647, 2147483647],
        strides=[2, 2],
    )


@pytest.mark.strided_slice
@pytest.mark.filterwarnings('ignore::UserWarning')
def test_slice_11():
    data = np.random.random(size=[2, 14, 14, 384]).astype('float32')
    test2.run(
        feed=[data],
        axes=[1, 2],
        starts=[1, 1],
        ends=[2147483647, 2147483647],
        strides=[2, 2],
    )
