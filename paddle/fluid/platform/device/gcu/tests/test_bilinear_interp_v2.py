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

test1 = ApiBase(
    func=paddle.nn.functional.interpolate,
    feed_names=['data'],
    feed_shapes=[[4, 1, 7, 8]],
    is_train=True,
)


@pytest.mark.bilinear_interp_v2
@pytest.mark.filterwarnings('ignore::UserWarning')
def test_bilinear_interp_v2_1():
    data = np.random.random(size=[4, 1, 7, 8]).astype('float32')
    test1.run(
        feed=[data],
        size=[1, 1],
        mode='bilinear',
        align_corners=True,
        align_mode=1,
    )


test2 = ApiBase(
    func=paddle.nn.functional.interpolate,
    feed_names=['data'],
    feed_shapes=[[3, 3, 9, 6]],
    is_train=True,
)


@pytest.mark.bilinear_interp_v2
@pytest.mark.filterwarnings('ignore::UserWarning')
def test_bilinear_interp_v2_2():
    data = np.random.random(size=[3, 3, 9, 6]).astype('float32')
    test2.run(
        feed=[data],
        size=[12, 12],
        mode='bilinear',
        align_corners=True,
        align_mode=1,
    )


test3 = ApiBase(
    func=paddle.nn.functional.interpolate,
    feed_names=['data'],
    feed_shapes=[[1, 1, 32, 64]],
    rel_tol=1e-5,
    is_train=True,
)


@pytest.mark.bilinear_interp_v2
@pytest.mark.filterwarnings('ignore::UserWarning')
def test_bilinear_interp_v2_3():
    data = np.random.random(size=[1, 1, 32, 64]).astype('float32')
    test3.run(
        feed=[data],
        size=[64, 32],
        mode='bilinear',
        align_corners=True,
        align_mode=1,
    )


test4 = ApiBase(
    func=paddle.nn.functional.interpolate,
    feed_names=['data'],
    feed_shapes=[[4, 512, 32, 64]],
    threshold=1e-4,
    rel_tol=1e-4,
    is_train=True,
)


@pytest.mark.bilinear_interp_v2
@pytest.mark.filterwarnings('ignore::UserWarning')
def test_bilinear_interp_v2_4():
    data = np.random.random(size=[4, 512, 32, 64]).astype('float32')
    test4.run(feed=[data], size=[64, 128], mode='bilinear', align_corners=False)


test5 = ApiBase(
    func=paddle.nn.functional.interpolate,
    feed_names=['data'],
    feed_shapes=[[4, 256, 64, 128]],
    threshold=1e-4,
    rel_tol=1e-4,
    is_train=True,
)


@pytest.mark.bilinear_interp_v2
@pytest.mark.filterwarnings('ignore::UserWarning')
def test_bilinear_interp_v2_5():
    data = np.random.random(size=[4, 256, 64, 128]).astype('float32')
    test5.run(
        feed=[data], size=[128, 256], mode='bilinear', align_corners=False
    )


test6 = ApiBase(
    func=paddle.nn.functional.interpolate,
    feed_names=['data'],
    feed_shapes=[[4, 128, 128, 256]],
    threshold=1e-4,
    rel_tol=1e-4,
    is_train=True,
)


@pytest.mark.bilinear_interp_v2
@pytest.mark.filterwarnings('ignore::UserWarning')
def test_bilinear_interp_v2_6():
    data = np.random.random(size=[4, 128, 128, 256]).astype('float32')
    test6.run(
        feed=[data], size=[256, 512], mode='bilinear', align_corners=False
    )


test7 = ApiBase(
    func=paddle.nn.functional.interpolate,
    feed_names=['data'],
    feed_shapes=[[4, 64, 256, 512]],
    threshold=1e-4,
    rel_tol=1e-4,
    is_train=True,
)


@pytest.mark.bilinear_interp_v2
@pytest.mark.filterwarnings('ignore::UserWarning')
def test_bilinear_interp_v2_7():
    data = np.random.random(size=[4, 64, 256, 512]).astype('float32')
    test7.run(
        feed=[data], size=[512, 1024], mode='bilinear', align_corners=False
    )
