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
        x = np.array([[[1], [3]], [[2], [4]], [[4], [127]]])
        self.feed_fp32 = {"x": x.astype(np.int64)}
        self.feed_fp16 = {"x": x.astype(np.int32)}

    def set_feed_attr(self):
        self.feed_shape = [x.shape for x in self.feed_fp32.values()]
        self.feed_list = list(self.feed_fp32.keys())
        self.feed_dtype = [x.dtype for x in self.feed_fp32.values()]

    def set_op_attrs(self):
        self.attrs = {
            "num_embeddings": 128,
            "embedding_dim": 16,
            "sparse": False,
            "padding_idx": -1,
            "weight_attr": None
        }

    @IPUOpTest.static_graph
    def build_model(self):
        x = paddle.static.data(name=self.feed_list[0],
                               shape=self.feed_shape[0],
                               dtype='int64')
        embedding = paddle.nn.Embedding(**self.attrs)
        out = embedding(x)
        if self.is_training:
            loss = paddle.mean(out)
            adam = paddle.optimizer.Adam(learning_rate=1e-2)
            adam.minimize(loss)
            self.fetch_list = [loss.name]
        else:
            self.fetch_list = [out.name]

    def run_model(self, exec_mode):
        if self.is_ipu_mode(exec_mode):
            self.feed_fp32['x'] = self.feed_fp32['x'].astype(np.int32)
        self.run_op_test(exec_mode)

    def test(self):
        for m in IPUOpTest.ExecutionMode:
            if not self.skip_mode(m):
                self.build_model()
                self.run_model(m)
        self.check()


class TestTrainCase1(TestBase):

    def set_atol(self):
        self.atol = 1e-7
        self.rtol = 1e-6
        self.atol_fp16 = 1e-3
        self.rtol_fp16 = 1e-3

    def set_training(self):
        self.is_training = True
        self.epoch = 10


if __name__ == "__main__":
    unittest.main()
