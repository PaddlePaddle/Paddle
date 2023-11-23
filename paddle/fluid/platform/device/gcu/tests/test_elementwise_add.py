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


@pytest.mark.elementwise_add
@pytest.mark.filterwarnings('ignore::UserWarning')
def test_elementwise_add_1():
    test1 = ApiBase(
        func=paddle.add,  # paddle.add
        feed_names=['X', 'Y'],
        feed_shapes=[[1, 64, 160, 160], [1, 64, 160, 160]],
    )
    x = np.random.random(size=[1, 64, 160, 160]).astype('float32')
    y = np.random.random(size=[1, 64, 160, 160]).astype('float32')
    test1.run(feed=[x, y])


@pytest.mark.elementwise_add
@pytest.mark.filterwarnings('ignore::UserWarning')
def test_elementwise_add_2():
    test2 = ApiBase(
        func=paddle.add,
        feed_names=['X', 'Y'],
        feed_shapes=[[1, 2, 160, 160], [2, 1, 1, 1]],
        threshold=1.0e-3,
    )
    x = np.random.random(size=[1, 2, 160, 160]).astype('float32')
    y = np.random.random(size=[2, 1, 1, 1]).astype('float32')
    test2.run(feed=[x, y])


@pytest.mark.elementwise_add
@pytest.mark.filterwarnings('ignore::UserWarning')
def test_elementwise_add_3():
    test3 = ApiBase(
        func=paddle.add, feed_names=['X', 'Y'], feed_shapes=[[1], [1]]
    )
    x = np.random.random(size=[1]).astype('float32')
    y = np.random.random(size=[1]).astype('float32')
    test3.run(feed=[x, y])


test4 = ApiBase(
    func=paddle.add,
    feed_names=['X', 'Y'],
    feed_shapes=[[16, 8, 352, 640], [16, 8, 352, 640]],
)


@pytest.mark.elementwise_add
@pytest.mark.filterwarnings('ignore::UserWarning')
def test_elementwise_add_4():
    x = np.random.random(size=[16, 8, 352, 640]).astype('float32')
    y = np.random.random(size=[16, 8, 352, 640]).astype('float32')
    test4.run(feed=[x, y])


test5 = ApiBase(
    func=paddle.add,
    feed_names=['X', 'Y'],
    feed_shapes=[[16, 16, 176, 320], [16, 16, 176, 320]],
)


@pytest.mark.elementwise_add
@pytest.mark.filterwarnings('ignore::UserWarning')
def test_elementwise_add_5():
    x = np.random.random(size=[16, 16, 176, 320]).astype('float32')
    y = np.random.random(size=[16, 16, 176, 320]).astype('float32')
    test5.run(feed=[x, y])


test6 = ApiBase(
    func=paddle.add,
    feed_names=['X', 'Y'],
    feed_shapes=[[16, 24, 88, 160], [16, 24, 88, 160]],
)


@pytest.mark.elementwise_add
@pytest.mark.filterwarnings('ignore::UserWarning')
def test_elementwise_add_6():
    x = np.random.random(size=[16, 24, 88, 160]).astype('float32')
    y = np.random.random(size=[16, 24, 88, 160]).astype('float32')
    test6.run(feed=[x, y])


test7 = ApiBase(
    func=paddle.add,
    feed_names=['X', 'Y'],
    feed_shapes=[[16, 40, 44, 80], [16, 40, 44, 80]],
)


@pytest.mark.elementwise_add
@pytest.mark.filterwarnings('ignore::UserWarning')
def test_elementwise_add_7():
    x = np.random.random(size=[16, 40, 44, 80]).astype('float32')
    y = np.random.random(size=[16, 40, 44, 80]).astype('float32')
    test7.run(feed=[x, y])


test8 = ApiBase(
    func=paddle.add,
    feed_names=['X', 'Y'],
    feed_shapes=[[16, 56, 44, 80], [16, 56, 44, 80]],
)


@pytest.mark.elementwise_add
@pytest.mark.filterwarnings('ignore::UserWarning')
def test_elementwise_add_8():
    x = np.random.random(size=[16, 56, 44, 80]).astype('float32')
    y = np.random.random(size=[16, 56, 44, 80]).astype('float32')
    test8.run(feed=[x, y])


test9 = ApiBase(
    func=paddle.add,
    feed_names=['X', 'Y'],
    feed_shapes=[[16, 96, 44, 80], [16, 96, 44, 80]],
)


@pytest.mark.elementwise_add
@pytest.mark.filterwarnings('ignore::UserWarning')
def test_elementwise_add_9():
    x = np.random.random(size=[16, 96, 44, 80]).astype('float32')
    y = np.random.random(size=[16, 96, 44, 80]).astype('float32')
    test9.run(feed=[x, y])


test10 = ApiBase(
    func=paddle.add,
    feed_names=['X', 'Y'],
    feed_shapes=[[16, 96, 88, 160], [16, 96, 88, 160]],
)


@pytest.mark.elementwise_add
@pytest.mark.filterwarnings('ignore::UserWarning')
def test_elementwise_add_10():
    x = np.random.random(size=[16, 96, 88, 160]).astype('float32')
    y = np.random.random(size=[16, 96, 88, 160]).astype('float32')
    test10.run(feed=[x, y])


test11 = ApiBase(
    func=paddle.add,
    feed_names=['X', 'Y'],
    feed_shapes=[[16, 96, 176, 320], [16, 96, 176, 320]],
)


@pytest.mark.elementwise_add
@pytest.mark.filterwarnings('ignore::UserWarning')
def test_elementwise_add_11():
    x = np.random.random(size=[16, 96, 176, 320]).astype('float32')
    y = np.random.random(size=[16, 96, 176, 320]).astype('float32')
    test11.run(feed=[x, y])


test12 = ApiBase(
    func=paddle.add,
    feed_names=['X', 'Y'],
    feed_shapes=[[16, 80, 22, 40], [16, 80, 22, 40]],
)


@pytest.mark.elementwise_add
@pytest.mark.filterwarnings('ignore::UserWarning')
def test_elementwise_add_12():
    x = np.random.random(size=[16, 80, 22, 40]).astype('float32')
    y = np.random.random(size=[16, 80, 22, 40]).astype('float32')
    test12.run(feed=[x, y])


test13 = ApiBase(
    func=paddle.add, feed_names=['X', 'Y'], feed_shapes=[[16, 10, 1, 1], [10]]
)


@pytest.mark.elementwise_add
@pytest.mark.filterwarnings('ignore::UserWarning')
def test_elementwise_add_13():
    x = np.random.random(size=[16, 10, 1, 1]).astype('float32')
    y = np.random.random(size=[10]).astype('float32')
    test13.run(feed=[x, y])


test14 = ApiBase(
    func=paddle.add, feed_names=['X', 'Y'], feed_shapes=[[16, 40, 1, 1], [40]]
)


@pytest.mark.elementwise_add
@pytest.mark.filterwarnings('ignore::UserWarning')
def test_elementwise_add_14():
    x = np.random.random(size=[16, 40, 1, 1]).astype('float32')
    y = np.random.random(size=[40]).astype('float32')
    test14.run(feed=[x, y])


test15 = ApiBase(
    func=paddle.add, feed_names=['X', 'Y'], feed_shapes=[[16, 16, 1, 1], [16]]
)


@pytest.mark.elementwise_add
@pytest.mark.filterwarnings('ignore::UserWarning')
def test_elementwise_add_15():
    x = np.random.random(size=[16, 16, 1, 1]).astype('float32')
    y = np.random.random(size=[16]).astype('float32')
    test15.run(feed=[x, y])


test16 = ApiBase(
    func=paddle.add, feed_names=['X', 'Y'], feed_shapes=[[16, 24, 1, 1], [24]]
)


@pytest.mark.elementwise_add
@pytest.mark.filterwarnings('ignore::UserWarning')
def test_elementwise_add_16():
    x = np.random.random(size=[16, 24, 1, 1]).astype('float32')
    y = np.random.random(size=[24]).astype('float32')
    test16.run(feed=[x, y])


test17 = ApiBase(
    func=paddle.add, feed_names=['X', 'Y'], feed_shapes=[[16, 60, 1, 1], [60]]
)


@pytest.mark.elementwise_add
@pytest.mark.filterwarnings('ignore::UserWarning')
def test_elementwise_add_17():
    x = np.random.random(size=[16, 60, 1, 1]).astype('float32')
    y = np.random.random(size=[60]).astype('float32')
    test17.run(feed=[x, y])


test18 = ApiBase(
    func=paddle.add, feed_names=['X', 'Y'], feed_shapes=[[16, 64, 1, 1], [64]]
)


@pytest.mark.elementwise_add
@pytest.mark.filterwarnings('ignore::UserWarning')
def test_elementwise_add_18():
    x = np.random.random(size=[16, 64, 1, 1]).astype('float32')
    y = np.random.random(size=[64]).astype('float32')
    test18.run(feed=[x, y])


test19 = ApiBase(
    func=paddle.add, feed_names=['X', 'Y'], feed_shapes=[[16, 84, 1, 1], [84]]
)


@pytest.mark.elementwise_add
@pytest.mark.filterwarnings('ignore::UserWarning')
def test_elementwise_add_19():
    x = np.random.random(size=[16, 84, 1, 1]).astype('float32')
    y = np.random.random(size=[84]).astype('float32')
    test19.run(feed=[x, y])


test20 = ApiBase(
    func=paddle.add, feed_names=['X', 'Y'], feed_shapes=[[16, 120, 1, 1], [120]]
)


@pytest.mark.elementwise_add
@pytest.mark.filterwarnings('ignore::UserWarning')
def test_elementwise_add_20():
    x = np.random.random(size=[16, 120, 1, 1]).astype('float32')
    y = np.random.random(size=[120]).astype('float32')
    test20.run(feed=[x, y])


test21 = ApiBase(
    func=paddle.add, feed_names=['X', 'Y'], feed_shapes=[[16, 240, 1, 1], [240]]
)


@pytest.mark.elementwise_add
@pytest.mark.filterwarnings('ignore::UserWarning')
def test_elementwise_add_21():
    x = np.random.random(size=[16, 240, 1, 1]).astype('float32')
    y = np.random.random(size=[240]).astype('float32')
    test21.run(feed=[x, y])


test22 = ApiBase(
    func=paddle.add, feed_names=['X', 'Y'], feed_shapes=[[16, 336, 1, 1], [336]]
)


@pytest.mark.elementwise_add
@pytest.mark.filterwarnings('ignore::UserWarning')
def test_elementwise_add_22():
    x = np.random.random(size=[16, 336, 1, 1]).astype('float32')
    y = np.random.random(size=[336]).astype('float32')
    test22.run(feed=[x, y])


test23 = ApiBase(
    func=paddle.add, feed_names=['X', 'Y'], feed_shapes=[[16, 480, 1, 1], [480]]
)


@pytest.mark.elementwise_add
@pytest.mark.filterwarnings('ignore::UserWarning')
def test_elementwise_add_23():
    x = np.random.random(size=[16, 480, 1, 1]).astype('float32')
    y = np.random.random(size=[480]).astype('float32')
    test23.run(feed=[x, y])


# dy should euqal to dense<[0.04166672]> : <24xf32>, cpu result is wrong
# test24 = ApiBase(func=paddle.add,
#                feed_names=['X', 'Y'],
#                feed_shapes=[[16, 24, 352, 640], [24]])
# @pytest.mark.elementwise_add
# @pytest.mark.filterwarnings('ignore::UserWarning')
# def test_elementwise_add_24():
#     x = np.random.random(size=[16, 24, 352, 640]).astype('float32')
#     y = np.random.random(size=[24]).astype('float32')
#     test24.run(feed=[x, y])

# dy should equal to 1, cpu result is wrong
# test25 = ApiBase(func=paddle.add,
#                feed_names=['X', 'Y'],
#                feed_shapes=[[16, 1, 704, 1280], [1]])
# @pytest.mark.elementwise_add
# @pytest.mark.filterwarnings('ignore::UserWarning')
# def test_elementwise_add_25():
#     x = np.random.random(size=[16, 1, 704, 1280]).astype('float32')
#     y = np.random.random(size=[1]).astype('float32')
#     test25.run(feed=[x, y])
