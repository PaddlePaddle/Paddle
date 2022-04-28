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

import tempfile
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
        self.set_data_feed()
        self.set_feed_attr()
        self.set_op_attrs()

    def set_data_feed(self):
        data = np.random.uniform(size=[1, 3, 10, 10])
        self.feed_fp32 = {"in_0": data.astype(np.float32)}
        self.feed_fp16 = {"in_0": data.astype(np.float16)}

    def set_feed_attr(self):
        self.feed_shape = [x.shape for x in self.feed_fp32.values()]
        self.feed_list = list(self.feed_fp32.keys())

    def set_op_attrs(self):
        self.attrs = {}
        self.attrs['steps'] = 100
        self.attrs['save_at_step'] = 20
        self.attrs['is_training'] = True
        self.attrs['opt_type'] = 'sgd'
        self.attrs['enable_fp16'] = False
        self.attrs['model_path'] = tempfile.TemporaryDirectory()

    def _test_base(self, save_otherwise_load):
        scope = paddle.static.Scope()
        main_prog = paddle.static.Program()
        startup_prog = paddle.static.Program()
        main_prog.random_seed = self.SEED
        startup_prog.random_seed = self.SEED
        generator = paddle.fluid.unique_name.UniqueNameGenerator()

        with paddle.fluid.unique_name.guard(generator):
            with paddle.static.scope_guard(scope):
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
                    paddle.static.load(main_prog, self.attrs['model_path'].name)

                ipu_strategy = paddle.static.IpuStrategy()
                ipu_strategy.set_graph_config(
                    is_training=self.attrs['is_training'])
                ipu_strategy.set_precision_config(
                    enable_fp16=self.attrs['enable_fp16'])
                ipu_program = paddle.static.IpuCompiledProgram(
                    main_prog, ipu_strategy=ipu_strategy)
                program = ipu_program.compile(self.feed_list, fetch_list)

                result = []
                run_steps = self.attrs['steps'] if save_otherwise_load \
                    else self.attrs['steps'] - self.attrs['save_at_step']

                feed = self.feed_fp16 if self.attrs[
                    'enable_fp16'] else self.feed_fp32
                for i in range(run_steps):
                    tmp = exe.run(program, feed=feed, fetch_list=fetch_list)

                    if save_otherwise_load and \
                        i == self.attrs['save_at_step'] - 1:
                        ipu_program._backend.weights_to_host()
                        paddle.static.save(main_prog,
                                           self.attrs['model_path'].name)

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
        self.attrs['model_path'].cleanup()


class TestAdam(TestBase):
    def set_op_attrs(self):
        self.attrs = {}
        self.attrs['steps'] = 100
        self.attrs['save_at_step'] = 20
        self.attrs['is_training'] = True
        self.attrs['opt_type'] = 'adam'
        self.attrs['enable_fp16'] = False
        self.attrs['model_path'] = tempfile.TemporaryDirectory()


class TestLamb(TestBase):
    def set_op_attrs(self):
        self.attrs = {}
        self.attrs['steps'] = 100
        self.attrs['save_at_step'] = 20
        self.attrs['is_training'] = True
        self.attrs['opt_type'] = 'lamb'
        self.attrs['enable_fp16'] = False
        self.attrs['model_path'] = tempfile.TemporaryDirectory()


@unittest.skipIf(IPUOpTest.use_ipumodel(), "skip for ipumodel")
class TestSGDFP16(TestBase):
    def set_op_attrs(self):
        self.attrs = {}
        self.attrs['steps'] = 100
        self.attrs['save_at_step'] = 20
        self.attrs['is_training'] = True
        self.attrs['opt_type'] = 'sgd'
        self.attrs['enable_fp16'] = True
        self.attrs['model_path'] = tempfile.TemporaryDirectory()


@unittest.skipIf(IPUOpTest.use_ipumodel(), "skip for ipumodel")
class TestAdamFP16(TestBase):
    def set_op_attrs(self):
        self.attrs = {}
        self.attrs['steps'] = 100
        self.attrs['save_at_step'] = 20
        self.attrs['is_training'] = True
        self.attrs['opt_type'] = 'adam'
        self.attrs['enable_fp16'] = True
        self.attrs['model_path'] = tempfile.TemporaryDirectory()


@unittest.skipIf(IPUOpTest.use_ipumodel(), "skip for ipumodel")
class TestLambFP16(TestBase):
    def set_op_attrs(self):
        self.attrs = {}
        self.attrs['steps'] = 100
        self.attrs['save_at_step'] = 20
        self.attrs['is_training'] = True
        self.attrs['opt_type'] = 'lamb'
        self.attrs['enable_fp16'] = True
        self.attrs['model_path'] = tempfile.TemporaryDirectory()


if __name__ == "__main__":
    unittest.main()
