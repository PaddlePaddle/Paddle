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
import shutil

import numpy as np
import paddle
import paddle.fluid as fluid
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

    def set_atol(self):
        self.atol = 1e-6
        self.rtol = 1e-5
        self.atol_fp16 = 1e-2
        self.rtol_fp16 = 1e-3

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
        self.attrs['path'] = 'model'
        self.attrs['model_name'] = 'test'

    def _test_save(self):
        scope = paddle.static.Scope()
        main_prog = paddle.static.Program()
        startup_prog = paddle.static.Program()
        main_prog.random_seed = self.SEED
        startup_prog.random_seed = self.SEED
        generator = paddle.fluid.unique_name.UniqueNameGenerator()
        self.full_name = '/'.join(
            [self.attrs['path'], self.attrs['model_name']])

        with paddle.fluid.unique_name.guard(generator):
            with paddle.static.scope_guard(scope):
                with paddle.static.program_guard(main_prog, startup_prog):
                    x = paddle.static.data(
                        name=self.feed_list[0],
                        shape=self.feed_shape[0],
                        dtype='float32')

                    scale = paddle.fluid.layers.scale(
                        x, scale=1.0, bias=0.0, bias_after_scale=True)
                    conv = paddle.static.nn.conv2d(
                        scale,
                        num_filters=3,
                        filter_size=3,
                        bias_attr=False,
                        name='conv2d')
                    loss = paddle.mean(conv)

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

                ipu_strategy = paddle.static.IpuStrategy()
                ipu_strategy.set_graph_config(is_training=True)
                ipu_strategy.set_precision_config(enable_fp16=True)
                program = paddle.static.IpuCompiledProgram(
                    main_prog, ipu_strategy=ipu_strategy).compile(
                        self.feed_list, fetch_list)

                for _ in range(self.attrs['steps']):
                    exe.run(program, feed=self.feed_fp16, fetch_list=fetch_list)

                paddle.static.save_inference_model(
                    self.full_name, x, loss, exe, program=program.org_program)

    def _test_load(self, run_ipu):
        if run_ipu:
            place = paddle.IPUPlace()
        else:
            place = paddle.CPUPlace()
        exe = paddle.static.Executor(place)

        [inference_program, feed_target_names, fetch_targets] = (
            paddle.static.load_inference_model(self.full_name, exe))

        if run_ipu:
            feed_list = feed_target_names
            fetch_list = [fetch_targets[0].name]
            ipu_strategy = paddle.static.IpuStrategy()
            ipu_strategy.set_graph_config(is_training=False)
            ipu_strategy.set_precision_config(enable_fp16=True)
            program = paddle.static.IpuCompiledProgram(
                inference_program,
                ipu_strategy=ipu_strategy).compile(feed_list, fetch_list)
        else:
            program = inference_program

        feed = self.feed_fp16 if run_ipu else self.feed_fp32
        result = []
        for i in range(10):
            feed["in_0"] += np.array([1.1 * i]).astype(feed["in_0"].dtype)
            out = exe.run(program, feed=feed, fetch_list=[fetch_targets])
            result.append(out)

        return np.array(result)

    def test_base(self):
        self._test_save()
        cpu_res = self._test_load(False)
        ipu_res = self._test_load(True).astype(np.float32)

        self.assertTrue(
            np.allclose(
                cpu_res, ipu_res, rtol=self.rtol_fp16, atol=self.atol_fp16))

        shutil.rmtree(self.attrs['path'], True)


if __name__ == "__main__":
    unittest.main()
