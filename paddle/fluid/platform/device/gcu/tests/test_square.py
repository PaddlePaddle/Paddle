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

test = ApiBase(func=paddle.square, feed_names=['data'], feed_shapes=[[3]])


@pytest.mark.square
@pytest.mark.filterwarnings('ignore::UserWarning')
def test_square():
    data = np.array([-1, 0, 1], dtype=np.float32)
    test.run(feed=[data])


test1 = ApiBase(
    func=paddle.square, feed_names=['data'], feed_shapes=[[1, 24, 352, 640]]
)


@pytest.mark.square
@pytest.mark.filterwarnings('ignore::UserWarning')
def test_square1():
    data = np.random.random(size=[1, 24, 352, 640]).astype('float32')
    test1.run(feed=[data])


test2 = ApiBase(
    func=paddle.square, feed_names=['data'], feed_shapes=[[16, 8, 352, 640]]
)


@pytest.mark.square
@pytest.mark.filterwarnings('ignore::UserWarning')
def test_square2():
    data = np.random.random(size=[16, 8, 352, 640]).astype('float32')
    test2.run(feed=[data])


test3 = ApiBase(
    func=paddle.square, feed_names=['data'], feed_shapes=[[16, 32, 352, 640]]
)


@pytest.mark.square
@pytest.mark.filterwarnings('ignore::UserWarning')
def test_square3():
    data = np.random.random(size=[16, 32, 352, 640]).astype('float32')
    test3.run(feed=[data])


test4 = ApiBase(
    func=paddle.square, feed_names=['data'], feed_shapes=[[16, 32, 176, 320]]
)


@pytest.mark.square
@pytest.mark.filterwarnings('ignore::UserWarning')
def test_square4():
    data = np.random.random(size=[16, 32, 176, 320]).astype('float32')
    test4.run(feed=[data])


test5 = ApiBase(
    func=paddle.square, feed_names=['data'], feed_shapes=[[16, 40, 176, 320]]
)


@pytest.mark.square
@pytest.mark.filterwarnings('ignore::UserWarning')
def test_square5():
    data = np.random.random(size=[16, 40, 176, 320]).astype('float32')
    test5.run(feed=[data])


test6 = ApiBase(
    func=paddle.square, feed_names=['data'], feed_shapes=[[16, 40, 88, 160]]
)


@pytest.mark.square
@pytest.mark.filterwarnings('ignore::UserWarning')
def test_square6():
    data = np.random.random(size=[16, 40, 88, 160]).astype('float32')
    test6.run(feed=[data])


test7 = ApiBase(
    func=paddle.square, feed_names=['data'], feed_shapes=[[16, 8, 352, 640]]
)


@pytest.mark.square
@pytest.mark.filterwarnings('ignore::UserWarning')
def test_square7():
    data = np.random.random(size=[16, 8, 352, 640]).astype('float32')
    test7.run(feed=[data])


test8 = ApiBase(
    func=paddle.square, feed_names=['data'], feed_shapes=[[16, 64, 88, 160]]
)


@pytest.mark.square
@pytest.mark.filterwarnings('ignore::UserWarning')
def test_square8():
    data = np.random.random(size=[16, 64, 88, 160]).astype('float32')
    test8.run(feed=[data])


test9 = ApiBase(
    func=paddle.square, feed_names=['data'], feed_shapes=[[16, 10, 1, 1]]
)


@pytest.mark.square
@pytest.mark.filterwarnings('ignore::UserWarning')
def test_square9():
    data = np.random.random(size=[16, 10, 1, 1]).astype('float32')
    test9.run(feed=[data])


test10 = ApiBase(
    func=paddle.square, feed_names=['data'], feed_shapes=[[16, 16, 1, 1]]
)


@pytest.mark.square
@pytest.mark.filterwarnings('ignore::UserWarning')
def test_square10():
    data = np.random.random(size=[16, 16, 1, 1]).astype('float32')
    test10.run(feed=[data])


test11 = ApiBase(
    func=paddle.square, feed_names=['data'], feed_shapes=[[16, 60, 1, 1]]
)


@pytest.mark.square
@pytest.mark.filterwarnings('ignore::UserWarning')
def test_square11():
    data = np.random.random(size=[16, 60, 1, 1]).astype('float32')
    test11.run(feed=[data])


test12 = ApiBase(
    func=paddle.square, feed_names=['data'], feed_shapes=[[16, 84, 1, 1]]
)


@pytest.mark.square
@pytest.mark.filterwarnings('ignore::UserWarning')
def test_square12():
    data = np.random.random(size=[16, 84, 1, 1]).astype('float32')
    test12.run(feed=[data])


test13 = ApiBase(
    func=paddle.square, feed_names=['data'], feed_shapes=[[16, 120, 1, 1]]
)


@pytest.mark.square
@pytest.mark.filterwarnings('ignore::UserWarning')
def test_square13():
    data = np.random.random(size=[16, 120, 1, 1]).astype('float32')
    test13.run(feed=[data])


test14 = ApiBase(
    func=paddle.square, feed_names=['data'], feed_shapes=[[16, 24, 176, 320]]
)


@pytest.mark.square
@pytest.mark.filterwarnings('ignore::UserWarning')
def test_square14():
    data = np.random.random(size=[16, 24, 176, 320]).astype('float32')
    test14.run(feed=[data])


test15 = ApiBase(
    func=paddle.square, feed_names=['data'], feed_shapes=[[16, 24, 352, 640]]
)


@pytest.mark.square
@pytest.mark.filterwarnings('ignore::UserWarning')
def test_square15():
    data = np.random.random(size=[16, 24, 352, 640]).astype('float32')
    test15.run(feed=[data])


test16 = ApiBase(
    func=paddle.square, feed_names=['data'], feed_shapes=[[16, 24, 3, 352, 640]]
)


@pytest.mark.square
@pytest.mark.filterwarnings('ignore::UserWarning')
def test_square16():
    data = np.random.random(size=[16, 24, 3, 352, 640]).astype('float32')
    test16.run(feed=[data])
