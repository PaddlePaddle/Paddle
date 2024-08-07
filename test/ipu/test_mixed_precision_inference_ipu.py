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
import paddle.nn.functional as F
import paddle.static


class TestBase(IPUOpTest):
    def setUp(self):
        self.set_atol()
        self.set_data_feed()
        self.set_feed_attr()
        self.set_attrs()

    def set_atol(self):
        self.atol = 1e-6
        self.rtol = 1e-6
        self.atol_fp16 = 1e-3
        self.rtol_fp16 = 1e-3

    def set_data_feed(self):
        data = np.random.uniform(size=[1, 10, 27, 27])
        self.feed_fp32 = {"in_0": data.astype(np.float32)}

    def set_feed_attr(self):
        self.feed_shape = [x.shape for x in self.feed_fp32.values()]
        self.feed_list = list(self.feed_fp32.keys())

    def set_attrs(self):
        self.num_ipus = 1
        self.enable_pipelining = False
        self.enable_manual_shard = False
        self.batches_per_step = 1

    @IPUOpTest.static_graph
    def build_model(self):
        x = paddle.static.data(
            name=self.feed_list[0], shape=self.feed_shape[0], dtype='float32'
        )

        # using fp32
        x = paddle.static.nn.conv2d(input=x, num_filters=3, filter_size=3)
        x = paddle.static.nn.batch_norm(x, act='relu')
        x = F.max_pool2d(x, kernel_size=2, stride=2)

        # using fp16
        with paddle.static.amp.fp16_guard():
            x = paddle.static.nn.conv2d(input=x, num_filters=6, filter_size=3)
            x = paddle.static.nn.batch_norm(x, act='relu')
            x = F.max_pool2d(x, kernel_size=2, stride=2)

        # using fp32
        x = paddle.static.nn.fc(x, size=10)
        loss = paddle.mean(x)
        self.fetch_list = [loss.name]

    def run_model(self, exec_mode):
        # cast model to fp16
        if self.is_fp16_mode(exec_mode):
            amp_list = paddle.static.amp.CustomOpLists()
            amp_list.unsupported_list = {}
            to_fp16_var_names = paddle.static.amp.cast_model_to_fp16(
                self.main_prog, amp_list, use_fp16_guard=True
            )

        if self.is_ipu_mode(exec_mode):
            place = paddle.CPUPlace()
        else:
            place = paddle.IPUPlace()
        exe = paddle.static.Executor(place)
        exe.run(self.startup_prog)

        # cast parameters to fp16
        if exec_mode == IPUOpTest.ExecutionMode.IPU_FP16:
            paddle.static.amp.cast_parameters_to_fp16(
                paddle.CPUPlace(),
                self.main_prog,
                to_fp16_var_names=to_fp16_var_names,
            )

        if self.is_ipu_mode(exec_mode):
            ipu_strategy = paddle.static.IpuStrategy()
            ipu_strategy.set_graph_config(
                is_training=False,
                num_ipus=self.num_ipus,
                enable_manual_shard=self.enable_manual_shard,
            )
            ipu_strategy.set_pipelining_config(
                enable_pipelining=self.enable_pipelining,
                batches_per_step=self.batches_per_step,
            )
            program = paddle.static.IpuCompiledProgram(
                self.main_prog, ipu_strategy=ipu_strategy
            ).compile(self.feed_list, self.fetch_list)
        else:
            program = self.main_prog

        result = exe.run(
            program, feed=self.feed_fp32, fetch_list=self.fetch_list
        )
        self.output_dict[exec_mode] = result[0]

    def test(self):
        for m in IPUOpTest.ExecutionMode:
            self.build_model()
            self.run_model(m)
        self.check()


class TestPipeline(TestBase):
    @IPUOpTest.static_graph
    def build_model(self, exec_mode):
        feed_shape = list(self.feed_shape[0])
        if self.is_ipu_mode(exec_mode):
            feed_shape[0] = 1
        x = paddle.static.data(
            name=self.feed_list[0], shape=feed_shape, dtype='float32'
        )
        with paddle.static.ipu_shard_guard(index=0, stage=0):
            # using fp32
            x = paddle.static.nn.conv2d(input=x, num_filters=3, filter_size=3)
            x = paddle.static.nn.batch_norm(x, act='relu')
            x = F.max_pool2d(x, kernel_size=2, stride=2)

        with paddle.static.ipu_shard_guard(index=1, stage=1):
            # using fp16
            with paddle.static.amp.fp16_guard():
                x = paddle.static.nn.conv2d(
                    input=x, num_filters=6, filter_size=3
                )
                x = paddle.static.nn.batch_norm(x, act='relu')
                x = F.max_pool2d(x, kernel_size=2, stride=2)

        with paddle.static.ipu_shard_guard(index=2, stage=2):
            # using fp32
            x = paddle.static.nn.fc(x, size=10)
            loss = paddle.mean(x)
        self.fetch_list = [loss.name]

    def set_data_feed(self):
        data = np.random.uniform(size=[3, 10, 27, 27])
        self.feed_fp32 = {"in_0": data.astype(np.float32)}

    def set_attrs(self):
        self.num_ipus = 3
        self.enable_pipelining = True
        self.enable_manual_shard = True
        self.batches_per_step = 3

    def test(self):
        for m in IPUOpTest.ExecutionMode:
            self.build_model(m)
            self.run_model(m)
        # skip check results


if __name__ == "__main__":
    unittest.main()
