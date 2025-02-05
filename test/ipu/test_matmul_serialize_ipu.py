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
from op_test_ipu import IPUOpTest

import paddle
import paddle.static


def set_serialize_factor(serialize_factor):
    main_prog = paddle.static.default_main_program()
    op = main_prog.current_block().ops[-1]
    op._set_attr('serialize_factor', serialize_factor)


class TestBase(IPUOpTest):
    def setUp(self):
        self.set_atol()
        self.set_training()
        self.set_data_feed()
        self.set_feed_attr()
        self.set_op_attrs()

    def set_data_feed(self):
        self.feed = {
            "x": np.random.uniform(size=[16, 32]).astype('float32'),
            "y": np.random.uniform(size=[32, 16]).astype('float32'),
        }

    def set_feed_attr(self):
        self.feed_shape = [x.shape for x in self.feed.values()]
        self.feed_list = list(self.feed.keys())
        self.feed_dtype = [x.dtype for x in self.feed.values()]

    def set_op_attrs(self):
        self.attrs = {"transpose_x": False, "transpose_y": False}

    @IPUOpTest.static_graph
    def build_model(self):
        x = paddle.static.data(
            name=self.feed_list[0],
            shape=self.feed_shape[0],
            dtype=self.feed_dtype[0],
        )
        y = paddle.static.data(
            name=self.feed_list[1],
            shape=self.feed_shape[1],
            dtype=self.feed_dtype[1],
        )
        # decorator maybe the best choice, but need to modify api
        out = paddle.matmul(x, y, **self.attrs)
        set_serialize_factor(4)
        self.fetch_list = [out.name]

    def run_model(self, run_ipu):
        self.build_model()
        if run_ipu:
            place = paddle.IPUPlace()
        else:
            place = paddle.CPUPlace()
        exe = paddle.static.Executor(place)
        exe.run(self.startup_prog)
        if run_ipu:
            feed_list = self.feed_list
            ipu_strategy = paddle.static.IpuStrategy()
            ipu_strategy.set_graph_config(is_training=self.is_training)
            program = paddle.static.IpuCompiledProgram(
                self.main_prog, ipu_strategy=ipu_strategy
            ).compile(feed_list, self.fetch_list)
        else:
            program = self.main_prog
        result = exe.run(program, feed=self.feed, fetch_list=self.fetch_list)
        return result[0]

    def test_base(self):
        res0 = self.run_model(False)
        res1 = self.run_model(True)
        np.testing.assert_allclose(
            res0.flatten(), res1.flatten(), rtol=1e-05, atol=self.atol
        )
        self.assertTrue(res0.shape == res1.shape)


if __name__ == "__main__":
    unittest.main()
