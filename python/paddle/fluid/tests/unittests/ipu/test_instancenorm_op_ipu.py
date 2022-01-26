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
import paddle.fluid as fluid
import paddle.fluid.compiler as compiler
import paddle.optimizer
import paddle.static
from paddle.fluid.tests.unittests.ipu.op_test_ipu import (IPUOpTest,
                                                          np_dtype_to_fluid_str)

paddle.enable_static()


@unittest.skipIf(not paddle.is_compiled_with_ipu(),
                 "core is not compiled with IPU")
class TestBase(IPUOpTest):
    def setUp(self):
        self.set_atol()
        self.set_training()
        self.set_feed()
        self.set_feed_attr()
        self.set_attrs()

    def set_feed(self):
        self.feed = {
            "x": np.random.uniform(size=[1, 3, 10, 10]).astype('float32'),
        }

    def set_feed_attr(self):
        self.feed_shape = [x.shape for x in self.feed.values()]
        self.feed_list = list(self.feed.keys())
        self.feed_dtype = [
            np_dtype_to_fluid_str(x.dtype) for x in self.feed.values()
        ]

    def set_attrs(self):
        self.attrs = {"epsilon": 1e-05}

    def _test_base(self, run_ipu=True):
        scope = fluid.core.Scope()
        main_prog = paddle.static.Program()
        startup_prog = paddle.static.Program()
        SEED = self.SEED
        main_prog.random_seed = SEED
        startup_prog.random_seed = SEED

        with fluid.scope_guard(scope):
            with paddle.static.program_guard(main_prog, startup_prog):
                x = paddle.static.data(
                    name=self.feed_list[0],
                    shape=self.feed_shape[0],
                    dtype=self.feed_dtype[0])

                if self.is_training:
                    ch = self.feed_shape[0][1]
                    conv1 = paddle.static.nn.conv2d(
                        x, num_filters=ch, filter_size=3, bias_attr=False)
                    scale = paddle.ParamAttr(trainable=True)
                    bias = paddle.ParamAttr(trainable=True)
                    out = paddle.fluid.layers.nn.instance_norm(
                        conv1, param_attr=scale, bias_attr=bias, **self.attrs)
                else:
                    scale = True
                    bias = True
                    out = paddle.fluid.layers.nn.instance_norm(
                        x, param_attr=scale, bias_attr=bias, **self.attrs)

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
                ipu_strategy = compiler.get_ipu_strategy()
                ipu_strategy.is_training = self.is_training
                program = compiler.IPUCompiledProgram(
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
                    result.append(loss_res)
                return np.array(result)
            else:
                result = exe.run(program, feed=self.feed, fetch_list=fetch_list)
                return result[0]

    def test_base(self):
        res0 = self._test_base(False)
        res1 = self._test_base(True)

        self.assertTrue(
            np.allclose(
                res0.flatten(), res1.flatten(), atol=self.atol))

        self.assertTrue(res0.shape == res1.shape)


class TestTrainCase1(TestBase):
    def set_training(self):
        self.is_training = True
        self.epoch = 10


# not support `instance_norm(x, param_attr=False, bias_attr=False, **self.attrs)`

if __name__ == "__main__":
    unittest.main()
