#  Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import unittest

import numpy as np
import paddle
import paddle.static
from paddle.fluid.tests.unittests.ipu.op_test_ipu import IPUOpTest


class TestBase(IPUOpTest):

    def setUp(self):
        self.set_atol()
        self.set_training()
        self.set_data_feed()
        self.set_feed_attr()
        self.set_op_attrs()

    def set_atol(self):
        self.atol = 1e-6
        self.rtol = 1e-5
        self.atol_fp16 = 1e-2
        self.rtol_fp16 = 1e-3

    def set_data_feed(self):
        x = np.random.uniform(size=[1, 3, 10, 10])
        self.feed_fp32 = {"x": x.astype(np.float32)}
        self.feed_fp16 = {"x": x.astype(np.float16)}

    def set_feed_attr(self):
        self.feed_shape = [x.shape for x in self.feed_fp32.values()]
        self.feed_list = list(self.feed_fp32.keys())
        self.feed_dtype = [x.dtype for x in self.feed_fp32.values()]

    def set_op_attrs(self):
        self.attrs = {
            "scale": True,
            "shift": True,
            "begin_norm_axis": 1,
            "epsilon": 1e-05,
        }
        self.optimizer = None

    @IPUOpTest.static_graph
    def build_model(self):
        x = paddle.static.data(name=self.feed_list[0],
                               shape=self.feed_shape[0],
                               dtype='float32')
        if self.is_training:
            ch = self.feed_shape[0][1]
            conv1 = paddle.static.nn.conv2d(x,
                                            num_filters=ch,
                                            filter_size=3,
                                            bias_attr=False)
            scale = paddle.ParamAttr(trainable=True)
            bias = paddle.ParamAttr(trainable=True)
            out = paddle.fluid.layers.nn.layer_norm(conv1,
                                                    param_attr=scale,
                                                    bias_attr=bias,
                                                    **self.attrs)
            loss = paddle.mean(out)
            self.fetch_list = [loss.name]
        else:
            scale = self.attrs['scale']
            bias = self.attrs['shift']
            out = paddle.fluid.layers.nn.layer_norm(x,
                                                    param_attr=scale,
                                                    bias_attr=bias,
                                                    **self.attrs)
            self.fetch_list = [out.name]

        if self.is_training:
            optimizer = None
            if self.optimizer == 'sgd':
                optimizer = paddle.optimizer.SGD(learning_rate=1e-2)
            elif self.optimizer == 'adam':
                optimizer = paddle.optimizer.Adam(learning_rate=1e-2)
            elif self.optimizer == 'lamb':
                optimizer = paddle.optimizer.Lamb(learning_rate=1e-2,
                                                  lamb_weight_decay=0.0)
            if optimizer is not None:
                optimizer.minimize(loss)

    def run_model(self, exec_mode):
        self.run_op_test(exec_mode)

    def test(self):
        for m in IPUOpTest.ExecutionMode:
            if not self.skip_mode(m):
                self.build_model()
                self.run_model(m)
        self.check()


@unittest.skip('raise error')
class TestCase1(TestBase):

    def set_op_attrs(self):
        self.attrs = {
            "scale": False,
            "shift": True,
            "begin_norm_axis": 1,
            "epsilon": 1e-05,
        }


@unittest.skip('raise error')
class TestCase2(TestBase):

    def set_op_attrs(self):
        self.attrs = {
            "scale": True,
            "shift": False,
            "begin_norm_axis": 1,
            "epsilon": 1e-05,
        }


class TestCase3(TestBase):

    def set_op_attrs(self):
        self.attrs = {
            "scale": True,
            "shift": True,
            "begin_norm_axis": 2,
            "epsilon": 1e-05,
        }
        self.optimizer = None


class TestTrainCase1(TestBase):

    def set_op_attrs(self):
        self.attrs = {
            "scale": True,
            "shift": True,
            "begin_norm_axis": 1,
            "epsilon": 1e-05
        }
        self.optimizer = 'sgd'

    def set_atol(self):
        super().set_atol()
        self.atol = 1e-6

    def set_training(self):
        self.is_training = True
        self.epoch = 20


class TestTrainCase3(TestBase):

    def set_atol(self):
        super().set_atol()
        self.atol = 5e-3

    def set_op_attrs(self):
        self.attrs = {
            "scale": True,
            "shift": True,
            "begin_norm_axis": 2,
            "epsilon": 1e-05
        }
        self.optimizer = 'lamb'

    def set_training(self):
        self.is_training = True
        self.epoch = 20


# not support `layer_norm(x, param_attr=False, bias_attr=False, **self.attrs)`

if __name__ == "__main__":
    unittest.main()
