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


@unittest.skipIf(not paddle.is_compiled_with_ipu(),
                 "core is not compiled with IPU")
class TestBase(IPUOpTest):
    def setUp(self):
        self.set_atol()
        self.set_training()
        self.set_data_feed()
        self.set_feed_attr()
        self.set_op_attrs()

    def set_data_feed(self):
        self.feed = {
            "x": np.random.uniform(size=[1, 3, 3, 3]).astype('float32'),
        }

    def set_feed_attr(self):
        self.feed_shape = [x.shape for x in self.feed.values()]
        self.feed_list = list(self.feed.keys())
        self.feed_dtype = [x.dtype for x in self.feed.values()]

    def set_op_attrs(self):
        self.attrs = {}

    def _test_base(self, run_ipu=True):
        scope = paddle.static.Scope()
        main_prog = paddle.static.Program()
        startup_prog = paddle.static.Program()
        main_prog.random_seed = self.SEED
        startup_prog.random_seed = self.SEED

        with paddle.static.scope_guard(scope):
            with paddle.static.program_guard(main_prog, startup_prog):
                x = paddle.static.data(
                    name=self.feed_list[0],
                    shape=self.feed_shape[0],
                    dtype=self.feed_dtype[0])
                out = paddle.fluid.layers.conv2d(
                    x, num_filters=3, filter_size=3)
                out = paddle.fluid.layers.Print(out, **self.attrs)

                if self.is_training:
                    loss = paddle.mean(out)
                    adam = paddle.optimizer.Adam(learning_rate=1e-2)
                    adam.minimize(loss)
                    fetch_list = [loss.name]
                else:
                    fetch_list = [out.name]

            if run_ipu:
                place = paddle.IPUPlace()
            else:
                place = paddle.CPUPlace()
            exe = paddle.static.Executor(place)
            exe.run(startup_prog)

            if run_ipu:
                feed_list = self.feed_list
                ipu_strategy = paddle.static.IpuStrategy()
                ipu_strategy.set_graph_config(is_training=self.is_training)
                program = paddle.static.IpuCompiledProgram(
                    main_prog,
                    ipu_strategy=ipu_strategy).compile(feed_list, fetch_list)
            else:
                program = main_prog

            if self.is_training:
                result = []
                for _ in range(self.epoch):
                    loss_res = exe.run(program,
                                       feed=self.feed,
                                       fetch_list=fetch_list)
                    result.append(loss_res[0])
                return np.array(result)
            else:
                result = exe.run(program, feed=self.feed, fetch_list=fetch_list)
                return result[0]

    def test(self):
        res0 = self._test_base(False)
        res1 = self._test_base(True)

        self.assertTrue(
            np.allclose(
                res0.flatten(), res1.flatten(), atol=self.atol))

        self.assertTrue(res0.shape == res1.shape)


class TestCase1(TestBase):
    def set_op_attrs(self):
        self.attrs = {"message": "input_data"}


class TestTrainCase1(TestBase):
    def set_op_attrs(self):
        # "forward" : print forward
        # "backward" : print forward and backward
        # "both": print forward and backward
        self.attrs = {"message": "input_data2", "print_phase": "both"}

    def set_training(self):
        self.is_training = True
        self.epoch = 2


@unittest.skip("attrs are not supported")
class TestCase2(TestBase):
    def set_op_attrs(self):
        self.attrs = {
            "first_n": 10,
            "summarize": 10,
            "print_tensor_name": True,
            "print_tensor_type": True,
            "print_tensor_shape": True,
            "print_tensor_layout": True,
            "print_tensor_lod": True
        }


if __name__ == "__main__":
    unittest.main()
