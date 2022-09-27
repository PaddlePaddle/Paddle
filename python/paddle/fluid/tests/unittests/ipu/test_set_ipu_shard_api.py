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
import paddle.nn as nn
from paddle.static import set_ipu_shard

paddle.enable_static()


class SimpleNet(paddle.nn.Layer):

    def __init__(self, input_size, output_size):
        super(SimpleNet, self).__init__()
        self.linear1 = nn.Linear(input_size, output_size)
        self.relu1 = nn.ReLU()
        self.linear2 = nn.Linear(input_size, output_size)
        self.relu2 = nn.ReLU()
        self.linear3 = nn.Linear(input_size, output_size)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu1(x)
        x = self.linear_relu2(x)
        x = self.linear3(x)
        return x

    def linear_relu2(self, x):
        x = self.linear2(x)
        x = self.relu2(x)
        return x


class TestSetIpuShard(unittest.TestCase):

    def _test(self):
        # build graph
        main_prog = paddle.static.Program()
        with paddle.static.program_guard(main_prog):
            x = paddle.static.data(name='X', shape=[10, 46], dtype='float32')
            label = paddle.static.data(name='Y',
                                       shape=[10, 46],
                                       dtype='float32')
            model = SimpleNet(46, 46)

            set_ipu_shard(model.linear1, index=1)
            set_ipu_shard(model.relu1, index=2)
            model.linear_relu2 = set_ipu_shard(model.linear_relu2, index=3)
            model.linear3 = set_ipu_shard(model.linear3, index=4)
            out = model(x)

        ipu_index_list = []
        for op in main_prog.global_block().ops:
            if op.desc.has_attr("ipu_index"):
                ipu_index_list.append(op.desc.attr("ipu_index"))

        return ipu_index_list

    def test_set_ipu_shard(self):
        ipu_index_list = self._test()
        expected_ipu_index_list = [1, 1, 2, 3, 3, 3, 4, 4]

        np.testing.assert_allclose(ipu_index_list,
                                   expected_ipu_index_list,
                                   rtol=1e-05,
                                   atol=0)


class TestSetIpuPipeline(unittest.TestCase):

    def _test(self):
        # build graph
        main_prog = paddle.static.Program()
        with paddle.static.program_guard(main_prog):
            x = paddle.static.data(name='X', shape=[10, 46], dtype='float32')
            label = paddle.static.data(name='Y',
                                       shape=[10, 46],
                                       dtype='float32')
            model = SimpleNet(46, 46)

            set_ipu_shard(model.linear1, stage=1)
            set_ipu_shard(model.relu1, stage=2)
            model.linear_relu2 = set_ipu_shard(model.linear_relu2, stage=3)
            model.linear3 = set_ipu_shard(model.linear3, stage=4)
            out = model(x)

        ipu_index_list = []
        for op in main_prog.global_block().ops:
            if op.desc.has_attr("ipu_stage"):
                ipu_index_list.append(op.desc.attr("ipu_stage"))

        return ipu_index_list

    def test_set_ipu_shard(self):
        ipu_index_list = self._test()
        expected_ipu_index_list = [1, 1, 2, 3, 3, 3, 4, 4]

        np.testing.assert_allclose(ipu_index_list,
                                   expected_ipu_index_list,
                                   rtol=1e-05,
                                   atol=0)


class TestSetIpuShardAndPipeline(unittest.TestCase):

    def _test(self):
        # build graph
        main_prog = paddle.static.Program()
        with paddle.static.program_guard(main_prog):
            x = paddle.static.data(name='X', shape=[10, 46], dtype='float32')
            label = paddle.static.data(name='Y',
                                       shape=[10, 46],
                                       dtype='float32')
            model = SimpleNet(46, 46)

            set_ipu_shard(model.linear1, index=1, stage=2)
            set_ipu_shard(model.relu1, index=2, stage=3)
            model.linear_relu2 = set_ipu_shard(model.linear_relu2,
                                               index=3,
                                               stage=4)
            model.linear3 = set_ipu_shard(model.linear3, index=4, stage=1)
            out = model(x)

        ipu_index_list = []
        ipu_stage_list = []
        for op in main_prog.global_block().ops:
            if op.desc.has_attr("ipu_index"):
                ipu_index_list.append(op.desc.attr("ipu_index"))
            if op.desc.has_attr("ipu_stage"):
                ipu_stage_list.append(op.desc.attr("ipu_stage"))

        return ipu_index_list + ipu_stage_list

    def test_set_ipu_shard(self):
        ipu_index_list = self._test()
        expected_ipu_index_list = [
            1, 1, 2, 3, 3, 3, 4, 4, 2, 2, 3, 4, 4, 4, 1, 1
        ]

        np.testing.assert_allclose(ipu_index_list,
                                   expected_ipu_index_list,
                                   rtol=1e-05,
                                   atol=0)


class TestSetIpuForModel(unittest.TestCase):

    def _test(self):
        # build graph
        main_prog = paddle.static.Program()
        with paddle.static.program_guard(main_prog):
            x = paddle.static.data(name='X', shape=[10, 46], dtype='float32')
            label = paddle.static.data(name='Y',
                                       shape=[10, 46],
                                       dtype='float32')
            model = SimpleNet(46, 46)

            set_ipu_shard(model, index=1, stage=2)
            out = model(x)

        ipu_index_list = []
        ipu_stage_list = []
        for op in main_prog.global_block().ops:
            if op.desc.has_attr("ipu_index"):
                ipu_index_list.append(op.desc.attr("ipu_index"))
            if op.desc.has_attr("ipu_stage"):
                ipu_stage_list.append(op.desc.attr("ipu_stage"))

        return ipu_index_list + ipu_stage_list

    def test_set_ipu_shard(self):
        ipu_index_list = self._test()
        expected_ipu_index_list = [
            1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2
        ]

        np.testing.assert_allclose(ipu_index_list,
                                   expected_ipu_index_list,
                                   rtol=1e-05,
                                   atol=0)


class TestSetIpuMixedModel(unittest.TestCase):

    def setUp(self):

        def linear_relu2_mixed(self, x):
            with paddle.static.ipu_shard_guard(index=2, stage=3):
                x = self.linear2(x)
            with paddle.static.ipu_shard_guard(index=3, stage=4):
                x = self.relu2(x)
            return x

        self._old_linear = SimpleNet.linear_relu2
        SimpleNet.linear_relu2 = linear_relu2_mixed

    def tearDown(self):
        SimpleNet.linear_relu2 = self._old_linear

    def _test(self):
        # build graph
        main_prog = paddle.static.Program()
        with paddle.static.program_guard(main_prog):
            x = paddle.static.data(name='X', shape=[10, 46], dtype='float32')
            label = paddle.static.data(name='Y',
                                       shape=[10, 46],
                                       dtype='float32')
            model = SimpleNet(46, 46)

            set_ipu_shard(model.linear1, index=1, stage=2)
            set_ipu_shard(model.relu1, index=2, stage=3)
            model.linear3 = set_ipu_shard(model.linear3, index=4, stage=1)
            out = model(x)

        ipu_index_list = []
        ipu_stage_list = []
        for op in main_prog.global_block().ops:
            if op.desc.has_attr("ipu_index"):
                ipu_index_list.append(op.desc.attr("ipu_index"))
            if op.desc.has_attr("ipu_stage"):
                ipu_stage_list.append(op.desc.attr("ipu_stage"))

        return ipu_index_list + ipu_stage_list

    def test_set_ipu_shard(self):
        ipu_index_list = self._test()
        expected_ipu_index_list = [
            1, 1, 2, 2, 2, 3, 4, 4, 2, 2, 3, 3, 3, 4, 1, 1
        ]

        np.testing.assert_allclose(ipu_index_list,
                                   expected_ipu_index_list,
                                   rtol=1e-05,
                                   atol=0)


if __name__ == "__main__":
    unittest.main()
