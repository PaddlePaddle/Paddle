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
import shutil

import numpy as np
import paddle
import paddle.fluid as fluid
import paddle.fluid.compiler as compiler
import paddle.optimizer
import paddle.static
from paddle.fluid.tests.unittests.ipu.op_test_ipu import IPUOpTest

paddle.enable_static()


@unittest.skipIf(not paddle.is_compiled_with_ipu(),
                 "core is not compiled with IPU")
class TestBase(IPUOpTest):
    def setUp(self):
        self.set_atol()
        self.set_feed()
        self.set_attrs()

    def set_feed(self):
        self.feed_shape = []
        self.feed_shape.append([1, 3, 10, 10])

        self.feed = {}
        self.feed["in_0"] = np.random.uniform(
            size=self.feed_shape[0]).astype(np.float32)

        self.feed_list = list(self.feed.keys())

    def set_attrs(self):
        self.attrs = {}
        self.attrs['steps'] = 100
        self.attrs['save_at_step'] = 20
        self.attrs['is_training'] = True
        self.attrs['opt_type'] = 'sgd'

    def _test_base(self, save_otherwise_load):
        scope = fluid.core.Scope()
        main_prog = paddle.static.Program()
        startup_prog = paddle.static.Program()
        main_prog.random_seed = self.SEED
        startup_prog.random_seed = self.SEED
        generator = fluid.unique_name.UniqueNameGenerator()

        with fluid.unique_name.guard(generator):
            with fluid.scope_guard(scope):
                with paddle.static.program_guard(main_prog, startup_prog):
                    x = paddle.static.data(
                        name=self.feed_list[0],
                        shape=self.feed_shape[0],
                        dtype='float32')
                    conv1 = paddle.static.nn.conv2d(
                        x,
                        num_filters=3,
                        filter_size=3,
                        bias_attr=False,
                        name='conv2d')
                    loss = paddle.mean(conv1)

                    if self.attrs['is_training']:
                        if self.attrs['opt_type'] == 'sgd':
                            sgd = paddle.optimizer.SGD(learning_rate=1e-2)
                            sgd.minimize(loss)
                        elif self.attrs['opt_type'] == 'adam':
                            adam = paddle.optimizer.Adam(learning_rate=1e-2)
                            adam.minimize(loss)
                        elif self.attrs['opt_type'] == 'lamb':
                            lamb = paddle.optimizer.Lamb(learning_rate=1e-2)
                            lamb.minimize(loss)
                    fetch_list = [loss.name]

                place = paddle.IPUPlace()
                exe = paddle.static.Executor(place)
                exe.run(startup_prog)

                if not save_otherwise_load:
                    paddle.static.load(main_prog, "model/model")

                ipu_strategy = paddle.static.IpuStrategy()
                ipu_strategy.SetGraphConfig(
                    is_training=self.attrs['is_training'])
                program = compiler.IPUCompiledProgram(
                    main_prog, ipu_strategy=ipu_strategy).compile(
                        self.feed_list, fetch_list)

                result = []
                run_steps = self.attrs['steps'] if save_otherwise_load \
                    else self.attrs['steps'] - self.attrs['save_at_step']

                for i in range(run_steps):
                    tmp = exe.run(program,
                                  feed=self.feed,
                                  fetch_list=fetch_list)

                    # currently, we update opt state every sess.run,
                    # will optimize
                    if save_otherwise_load and \
                        i == self.attrs['save_at_step'] - 1:
                        paddle.static.save(main_prog, "model/model")

                    if save_otherwise_load and i >= self.attrs['save_at_step']:
                        result.append(tmp)
                    elif not save_otherwise_load:
                        result.append(tmp)

                return np.asarray(result).flatten()

    def test_base(self):
        res0 = self._test_base(True)
        res1 = self._test_base(False)

        self.assertTrue(
            np.allclose(
                res0.flatten(), res1.flatten(), atol=self.atol))
        shutil.rmtree("model", True)


class TestAdam(TestBase):
    def set_attrs(self):
        self.attrs = {}
        self.attrs['steps'] = 100
        self.attrs['save_at_step'] = 20
        self.attrs['is_training'] = True
        self.attrs['opt_type'] = 'adam'


class TestLamb(TestBase):
    def set_attrs(self):
        self.attrs = {}
        self.attrs['steps'] = 100
        self.attrs['save_at_step'] = 20
        self.attrs['is_training'] = True
        self.attrs['opt_type'] = 'lamb'


if __name__ == "__main__":
    unittest.main()
