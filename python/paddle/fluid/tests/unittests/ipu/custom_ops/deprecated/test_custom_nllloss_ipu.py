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

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))))
from op_test_ipu import IPUOpTest


def load_custom_ops():
    cur_dir = os.path.dirname(os.path.realpath(__file__))
    custom_ops = load(name="custom_nll_loss",
                      sources=[f"{cur_dir}/custom_nllloss.cc"],
                      extra_cxx_cflags=['-DONNX_NAMESPACE=onnx'],
                      extra_ldflags=['-lpopfloat'])
    return custom_ops


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
        self.custom_ops = load_custom_ops()

    def set_data_feed(self):
        x = np.random.rand(16, 20, 256).astype('float32')
        label = np.random.uniform(0, 256, size=[16, 20]).astype('int32')
        self.feed_fp32 = {
            'x': x,
            'label': label,
        }

    def set_test_op(self):
        self.op = self.custom_ops.custom_nll_loss
        self.op_attrs = {
            "reduction": 0,
            "ignoreindex": "0",
            "inputislogprobability": False,
        }

    def set_feed_attr(self):
        self.feed_shape = [x.shape for x in self.feed_fp32.values()]
        self.feed_list = list(self.feed_fp32.keys())

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

    def run_model(self, exec_mode):
        self.run_op_test(exec_mode)

    def test(self):
        self.build_model()
        # only test IPU_FP32
        self.run_model(IPUOpTest.ExecutionMode.IPU_FP32)
        print(self.output_dict)


class TestCase1(TestBase):

    def set_test_op(self):
        self.op = self.custom_ops.custom_nll_loss
        self.op_attrs = {
            "reduction": 0,
            "ignoreindex": "None",
            "inputislogprobability": False,
        }


if __name__ == "__main__":
    unittest.main()
