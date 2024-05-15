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


class TestStackOpBase:
    def initDefaultParameters(self):
        self.num_inputs = 4
        self.input_dim = (5, 6, 7)
        self.axis = 0
        self.dtype = 'float64'

    def initParameters(self):
        pass

    def get_x_names(self):
        x_names = []
        for i in range(self.num_inputs):
            x_names.append(f'x{i}')
        return x_names

    def __init__(self):
        support_types = ['int32', 'int64', 'float32', 'float64']
        self.initDefaultParameters()
        self.initParameters()
        self.op_type = 'stack'
        self.x = []
        for i in range(self.num_inputs):
            self.x.append(
                np.random.random(size=self.input_dim).astype(self.dtype)
            )

        self.attrs = {'axis': self.axis}
        for ty in support_types:
            tmp = []
            tmp_shapes = []
            tys = []
            names = []
            for i in range(self.num_inputs):
                tmp.append(self.x[i].astype(ty))
                tmp_shapes.append(self.x[i].shape)
                tys.append(ty)
                names.append("x%d" % i)
            print(len(names), len(tys), len(tmp_shapes))
            self.test = ApiBase(
                func=paddle.stack,
                feed_names=names,
                feed_shapes=tmp_shapes,
                is_train=False,
                feed_dtypes=tys,
                input_is_list=True,
            )

            self.test.run(feed=tmp, **self.attrs)


class TestStackOp1(TestStackOpBase):
    def initParameters(self):
        self.num_inputs = 8


class TestStackOp2(TestStackOpBase):
    def initParameters(self):
        self.num_inputs = 10


class TestStackOp3(TestStackOpBase):
    def initParameters(self):
        self.axis = -1


class TestStackOp4(TestStackOpBase):
    def initParameters(self):
        self.axis = -4


class TestStackOp5(TestStackOpBase):
    def initParameters(self):
        self.axis = 1


class TestStackOp6(TestStackOpBase):
    def initParameters(self):
        self.axis = 3


@pytest.mark.stack
@pytest.mark.filterwarnings('ignore::UserWarning')
def test_stack():
    TestStackOp3()
    TestStackOpBase()
    TestStackOp1()
    TestStackOp2()
    #

    TestStackOp3()
    TestStackOp4()
    TestStackOp5()
    TestStackOp6()


test = ApiBase(
    func=paddle.stack,
    feed_names=['data0', 'data1', 'data2'],
    feed_shapes=[[2, 3, 4], [2, 3, 4], [2, 3, 4]],
    input_is_list=True,
)


@pytest.mark.stack
@pytest.mark.filterwarnings('ignore::UserWarning')
def test_stack_grad_0():
    data0 = np.random.random(size=[2, 3, 4]).astype('float32')
    data1 = np.random.random(size=[2, 3, 4]).astype('float32')
    data2 = np.random.random(size=[2, 3, 4]).astype('float32')
    test.run(feed=[data0, data1, data2], axis=1)


@pytest.mark.stack
@pytest.mark.filterwarnings('ignore::UserWarning')
def test_stack_grad_1():
    data0 = np.random.random(size=[2, 3, 4]).astype('float32')
    data1 = np.random.random(size=[2, 3, 4]).astype('float32')
    data2 = np.random.random(size=[2, 3, 4]).astype('float32')
    test.run(feed=[data0, data1, data2], axis=-2)
