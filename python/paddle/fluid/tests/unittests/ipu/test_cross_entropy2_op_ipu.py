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

    def set_data_feed(self):
        x = np.random.uniform(size=[3, 7])
        label = np.arange(3).reshape([3, 1])
        self.feed_fp32 = {
            "x": x.astype(np.float32),
            "label": label.astype(np.int64),
        }
        self.feed_fp16 = {
            "x": x.astype(np.float16),
            "label": label.astype(np.int32),
        }

    def set_feed_attr(self):
        self.feed_shape = [x.shape for x in self.feed_fp32.values()]
        self.feed_list = list(self.feed_fp32.keys())

    def set_op_attrs(self):
        self.attrs = {
            'soft_label': False,
        }

    @IPUOpTest.static_graph
    def build_model(self, on_ipu):
        x = paddle.static.data(
            name=self.feed_list[0], shape=self.feed_shape[0], dtype="float32"
        )
        if on_ipu:
            label = paddle.static.data(
                name=self.feed_list[1], shape=self.feed_shape[1], dtype='int32'
            )
        else:
            label = paddle.static.data(
                name=self.feed_list[1], shape=self.feed_shape[1], dtype='int64'
            )
        out = paddle.nn.functional.cross_entropy(
            input=x,
            label=label,
            reduction='none',
            use_softmax=False,
            **self.attrs
        )
        self.fetch_list = [out.name]

    def run_model(self, exec_mode):
        if self.is_ipu_mode(exec_mode):
            self.feed_fp32['label'] = self.feed_fp32['label'].astype(np.int32)
        self.run_op_test(exec_mode)

    def test(self):
        for m in IPUOpTest.ExecutionMode:
            if not self.skip_mode(m):
                self.build_model(self.is_ipu_mode(m))
                self.run_model(m)
        self.check()


class TestCase1(TestBase):
    def set_op_attrs(self):
        self.attrs = {
            'soft_label': False,
            'ignore_index': 1,
        }


class TestCase2(TestBase):
    def set_data_feed(self):
        x = np.random.uniform(size=[30, 70])
        label = np.arange(30).reshape([30, 1])
        self.feed_fp32 = {
            "x": x.astype(np.float32),
            "label": label.astype(np.int64),
        }
        self.feed_fp16 = {
            "x": x.astype(np.float16),
            "label": label.astype(np.int32),
        }


@unittest.skip("soft_label=True is not supported")
class TestCase3(TestBase):
    def set_op_attrs(self):
        self.attrs = {
            'soft_label': True,
        }


class TestCase4(TestBase):
    def set_data_feed(self):
        x = np.random.uniform(size=[3, 5, 7])
        label = np.random.randint(0, 7, [3, 5, 1], dtype='int64')
        self.feed_fp32 = {
            "x": x.astype(np.float32),
            "label": label.astype(np.int64),
        }
        self.feed_fp16 = {
            "x": x.astype(np.float16),
            "label": label.astype(np.int32),
        }


class TestCase5(TestBase):
    def set_data_feed(self):
        x = np.random.uniform(size=[3, 5, 6, 7])
        label = np.random.randint(0, 7, [3, 5, 6], dtype='int64')
        self.feed_fp32 = {
            "x": x.astype(np.float32),
            "label": label.astype(np.int64),
        }
        self.feed_fp16 = {
            "x": x.astype(np.float16),
            "label": label.astype(np.int32),
        }


if __name__ == "__main__":
    unittest.main()
