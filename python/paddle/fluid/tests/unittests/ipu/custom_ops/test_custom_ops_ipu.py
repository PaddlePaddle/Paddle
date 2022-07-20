#  Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

import os
import unittest
import sys

import numpy as np
import paddle
import paddle.static
from paddle.utils.cpp_extension import load

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from op_test_ipu import IPUOpTest


# just load one custom-op for the data race issue under parallel mode
def load_custom_detach():
    cur_dir = os.path.dirname(os.path.realpath(__file__))
    custom_ops = load(name=f"custom_detach",
                      sources=[
                          f"{cur_dir}/custom_detach.cc",
                      ],
                      extra_cxx_cflags=['-DONNX_NAMESPACE=onnx'],
                      extra_ldflags=['-lpopfloat'])
    return custom_ops


def load_custom_identity():
    cur_dir = os.path.dirname(os.path.realpath(__file__))
    custom_ops = load(name=f"custom_identity",
                      sources=[
                          f"{cur_dir}/custom_identity.cc",
                      ],
                      extra_cxx_cflags=['-DONNX_NAMESPACE=onnx'],
                      extra_ldflags=['-lpopfloat'])
    return custom_ops


def load_custom_nll():
    cur_dir = os.path.dirname(os.path.realpath(__file__))
    custom_ops = load(name=f"custom_nll",
                      sources=[
                          f"{cur_dir}/custom_nll.cc",
                      ],
                      extra_cxx_cflags=['-DONNX_NAMESPACE=onnx'],
                      extra_ldflags=['-lpopfloat'])
    return custom_ops


def build_ipu_strategy():
    ipu_strategy = paddle.static.IpuStrategy()
    ipu_strategy.add_custom_op(paddle_op="custom_detach",
                               popart_op="Detach",
                               domain="ai.graphcore",
                               version=1)
    ipu_strategy.add_custom_op(paddle_op="custom_identity",
                               popart_op="Identity",
                               domain="ai.onnx",
                               version=11)
    ipu_strategy.add_custom_op(paddle_op="custom_nll",
                               popart_op="Nll",
                               domain="ai.graphcore",
                               version=1)
    return ipu_strategy


class TestBase(IPUOpTest):

    def setUp(self):
        self.load_custom_ops()
        self.set_atol()
        self.set_test_op()
        self.set_training()
        self.set_data_feed()
        self.set_feed_attr()

    @property
    def fp16_enabled(self):
        return False

    def load_custom_ops(self):
        self.custom_ops = load_custom_detach()

    def set_test_op(self):
        self.op = self.custom_ops.custom_detach
        self.op_attrs = {}

    def set_data_feed(self):
        data = np.random.uniform(size=[1, 3, 10, 10])
        self.feed_fp32 = {'in_0': data.astype(np.float32)}

    def set_feed_attr(self):
        self.feed_shape = [x.shape for x in self.feed_fp32.values()]
        self.feed_list = list(self.feed_fp32.keys())

    @IPUOpTest.static_graph
    def build_model(self):
        x = paddle.static.data(name=self.feed_list[0],
                               shape=self.feed_shape[0],
                               dtype='float32')
        out = self.op(x, **self.op_attrs)
        out = paddle.mean(out)
        self.fetch_list = [out.name]

    def run_model(self, exec_mode):
        ipu_strategy = build_ipu_strategy()
        ipu_strategy.set_graph_config(is_training=self.is_training)
        self.run_op_test(exec_mode, ipu_strategy=ipu_strategy)

    def test(self):
        self.build_model()
        # only test IPU_FP32
        self.run_model(IPUOpTest.ExecutionMode.IPU_FP32)
        print(self.output_dict)


class TestIdentity(TestBase):

    def load_custom_ops(self):
        self.custom_ops = load_custom_identity()

    def set_test_op(self):
        self.op = self.custom_ops.custom_identity
        self.op_attrs = {}


class TestNll(TestBase):

    def load_custom_ops(self):
        self.custom_ops = load_custom_nll()

    def set_data_feed(self):
        x = np.random.rand(16, 20, 256).astype('float32')
        label = np.random.uniform(0, 256, size=[16, 20]).astype('int32')
        self.feed_fp32 = {
            'x': x,
            'label': label,
        }

    def set_test_op(self):
        self.op = self.custom_ops.custom_nll
        self.op_attrs = {
            "reduction": "Sum",
            "ignoreindex": 0,
            "inputislogprobability": False,
        }

    @IPUOpTest.static_graph
    def build_model(self):
        x = paddle.static.data(name=self.feed_list[0],
                               shape=self.feed_shape[0],
                               dtype='float32')
        label = paddle.static.data(name=self.feed_list[1],
                                   shape=self.feed_shape[1],
                                   dtype='int32')
        out = self.op(x, label, **self.op_attrs)
        out = paddle.mean(out)
        self.fetch_list = [out.name]


if __name__ == "__main__":
    unittest.main()
