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


@pytest.mark.matmul
@pytest.mark.filterwarnings('ignore::UserWarning')
def test_matmul1():
    test = ApiBase(
        func=paddle.matmul,
        feed_names=['data1', 'data2'],
        feed_shapes=[[4], [4]],
        is_train=True,
    )
    data1 = np.random.random(size=[4]).astype('float32')
    data2 = np.random.random(size=[4]).astype('float32')
    test.run(feed=[data1, data2], transpose_x=True, transpose_y=True)


@pytest.mark.matmul
@pytest.mark.filterwarnings('ignore::UserWarning')
def test_matmul2():
    test = ApiBase(
        func=paddle.matmul,
        feed_names=['data1', 'data2'],
        feed_shapes=[[3, 4], [4]],
        is_train=True,
    )
    data1 = np.random.random(size=[3, 4]).astype('float32')
    data2 = np.random.random(size=[4]).astype('float32')
    test.run(feed=[data1, data2], transpose_x=False)


@pytest.mark.matmul
@pytest.mark.filterwarnings('ignore::UserWarning')
def test_matmul3():
    test = ApiBase(
        func=paddle.matmul,
        feed_names=['data1', 'data2'],
        feed_shapes=[[4], [4, 5]],
        is_train=True,
    )
    data1 = np.random.random(size=[4]).astype('float32')
    data2 = np.random.random(size=[4, 5]).astype('float32')
    test.run(feed=[data1, data2], transpose_x=False)


@pytest.mark.matmul
@pytest.mark.filterwarnings('ignore::UserWarning')
def test_matmul4():
    test = ApiBase(
        func=paddle.matmul,
        feed_names=['data1', 'data2'],
        feed_shapes=[[2, 3, 5, 4], [5]],
        is_train=True,
    )
    data1 = np.random.random(size=[2, 3, 5, 4]).astype('float32')
    data2 = np.random.random(size=[5]).astype('float32')
    test.run(feed=[data1, data2], transpose_x=True, transpose_y=False)


@pytest.mark.matmul
@pytest.mark.filterwarnings('ignore::UserWarning')
def test_matmul5():
    test = ApiBase(
        func=paddle.matmul,
        feed_names=['data1', 'data2'],
        feed_shapes=[[4], [4, 3]],
        is_train=True,
    )
    data1 = np.random.random(size=[4]).astype('float32')
    data2 = np.random.random(size=[4, 3]).astype('float32')
    test.run(feed=[data1, data2], transpose_x=False, transpose_y=False)


@pytest.mark.matmul
@pytest.mark.filterwarnings('ignore::UserWarning')
def test_matmul6():
    test = ApiBase(
        func=paddle.matmul,
        feed_names=['data1', 'data2'],
        feed_shapes=[[4], [2, 3, 5, 4]],
        is_train=True,
    )
    data1 = np.random.random(size=[4]).astype('float32')
    data2 = np.random.random(size=[2, 3, 5, 4]).astype('float32')
    test.run(feed=[data1, data2], transpose_x=False, transpose_y=True)


@pytest.mark.matmul
@pytest.mark.filterwarnings('ignore::UserWarning')
def test_matmul7():
    test = ApiBase(
        func=paddle.matmul,
        feed_names=['data1', 'data2'],
        feed_shapes=[[4, 5], [5, 6]],
        is_train=True,
    )
    data1 = np.random.random(size=[4, 5]).astype('float32')
    data2 = np.random.random(size=[5, 6]).astype('float32')
    test.run(feed=[data1, data2])


@pytest.mark.matmul
@pytest.mark.filterwarnings('ignore::UserWarning')
def test_matmul8():
    test = ApiBase(
        func=paddle.matmul,
        feed_names=['data1', 'data2'],
        feed_shapes=[[2, 3, 5, 4], [6, 5]],
        is_train=True,
    )
    data1 = np.random.random(size=[2, 3, 5, 4]).astype('float32')
    data2 = np.random.random(size=[6, 5]).astype('float32')
    test.run(feed=[data1, data2], transpose_x=True, transpose_y=True)


@pytest.mark.matmul
@pytest.mark.filterwarnings('ignore::UserWarning')
def test_matmul9():
    test = ApiBase(
        func=paddle.matmul,
        feed_names=['data1', 'data2'],
        feed_shapes=[[4, 3], [2, 3, 5, 4]],
        is_train=True,
    )
    data1 = np.random.random(size=[4, 3]).astype('float32')
    data2 = np.random.random(size=[2, 3, 5, 4]).astype('float32')
    test.run(feed=[data1, data2], transpose_x=True, transpose_y=True)


@pytest.mark.matmul
@pytest.mark.filterwarnings('ignore::UserWarning')
def test_matmul10():
    test = ApiBase(
        func=paddle.matmul,
        feed_names=['data1', 'data2'],
        feed_shapes=[[3, 4, 5], [3, 5, 6]],
        is_train=True,
    )
    data1 = np.random.random(size=[3, 4, 5]).astype('float32')
    data2 = np.random.random(size=[3, 5, 6]).astype('float32')
    test.run(feed=[data1, data2])


@pytest.mark.matmul
@pytest.mark.filterwarnings('ignore::UserWarning')
def test_matmul11():
    test = ApiBase(
        func=paddle.matmul,
        feed_names=['data1', 'data2'],
        feed_shapes=[[2, 3, 5, 4], [2, 3, 6, 5]],
        is_train=True,
    )
    data1 = np.random.random(size=[2, 3, 5, 4]).astype('float32')
    data2 = np.random.random(size=[2, 3, 6, 5]).astype('float32')
    test.run(feed=[data1, data2], transpose_x=True, transpose_y=True)


@pytest.mark.matmul
@pytest.mark.filterwarnings('ignore::UserWarning')
def test_matmul12():
    test = ApiBase(
        func=paddle.matmul,
        feed_names=['data1', 'data2'],
        feed_shapes=[[32, 128, 1024], [1024, 1024]],
        is_train=True,
    )
    data1 = np.random.random(size=[32, 128, 1024]).astype('float32')
    data2 = np.random.random(size=[1024, 1024]).astype('float32')
    test.run(feed=[data1, data2], transpose_x=False, transpose_y=False)
