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
<<<<<<< HEAD

=======
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
import paddle
import paddle.static
from paddle.fluid.tests.unittests.ipu.op_test_ipu import IPUOpTest


class TestBase(IPUOpTest):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
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
        self.feed = {"in_0": data.astype(np.float32)}

    def set_feed_attr(self):
        self.feed_shape = [x.shape for x in self.feed.values()]
        self.feed_list = list(self.feed.keys())

    def set_op_attrs(self):
        self.attrs = {}
        self.attrs['steps'] = 100
        self.attrs['save_at_step'] = 20
        self.attrs['is_training'] = True
        self.attrs['opt_type'] = 'sgd'
        self.attrs['path'] = tempfile.TemporaryDirectory()
        self.attrs['model_name'] = 'test'

    def _test_save(self):
        scope = paddle.static.Scope()
        main_prog = paddle.static.Program()
        startup_prog = paddle.static.Program()
        main_prog.random_seed = self.SEED
        startup_prog.random_seed = self.SEED
        generator = paddle.fluid.unique_name.UniqueNameGenerator()
        self.full_name = '/'.join(
<<<<<<< HEAD
            [self.attrs['path'].name, self.attrs['model_name']]
        )
=======
            [self.attrs['path'].name, self.attrs['model_name']])
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

        with paddle.fluid.unique_name.guard(generator):
            with paddle.static.scope_guard(scope):
                with paddle.static.program_guard(main_prog, startup_prog):
<<<<<<< HEAD
                    x = paddle.static.data(
                        name=self.feed_list[0],
                        shape=self.feed_shape[0],
                        dtype='float32',
                    )
                    conv1 = paddle.static.nn.conv2d(
                        x,
                        num_filters=3,
                        filter_size=3,
                        bias_attr=False,
                        name='conv2d',
                    )
=======
                    x = paddle.static.data(name=self.feed_list[0],
                                           shape=self.feed_shape[0],
                                           dtype='float32')
                    conv1 = paddle.static.nn.conv2d(x,
                                                    num_filters=3,
                                                    filter_size=3,
                                                    bias_attr=False,
                                                    name='conv2d')
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
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

                ipu_strategy = paddle.static.IpuStrategy()
                ipu_strategy.set_graph_config(
<<<<<<< HEAD
                    is_training=self.attrs['is_training']
                )
                program = paddle.static.IpuCompiledProgram(
                    main_prog, ipu_strategy=ipu_strategy
                ).compile(self.feed_list, fetch_list)

                result = []
                for i in range(self.attrs['steps']):
                    tmp = exe.run(
                        program, feed=self.feed, fetch_list=fetch_list
                    )
                    result.append(tmp)

                paddle.static.save_inference_model(
                    self.full_name, x, loss, exe, program=program.org_program
                )
=======
                    is_training=self.attrs['is_training'])
                program = paddle.static.IpuCompiledProgram(
                    main_prog,
                    ipu_strategy=ipu_strategy).compile(self.feed_list,
                                                       fetch_list)

                result = []
                for i in range(self.attrs['steps']):
                    tmp = exe.run(program,
                                  feed=self.feed,
                                  fetch_list=fetch_list)
                    result.append(tmp)

                paddle.static.save_inference_model(self.full_name,
                                                   x,
                                                   loss,
                                                   exe,
                                                   program=program.org_program)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

    def _test_load(self, run_ipu):
        if run_ipu:
            place = paddle.IPUPlace()
        else:
            place = paddle.CPUPlace()
        exe = paddle.static.Executor(place)

<<<<<<< HEAD
        [
            inference_program,
            feed_target_names,
            fetch_targets,
        ] = paddle.static.load_inference_model(self.full_name, exe)
=======
        [inference_program, feed_target_names, fetch_targets
         ] = (paddle.static.load_inference_model(self.full_name, exe))
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

        if run_ipu:
            feed_list = feed_target_names
            fetch_list = [fetch_targets[0].name]
            ipu_strategy = paddle.static.IpuStrategy()
            ipu_strategy.set_graph_config(is_training=False)
            program = paddle.static.IpuCompiledProgram(
<<<<<<< HEAD
                inference_program, ipu_strategy=ipu_strategy
            ).compile(feed_list, fetch_list)
=======
                inference_program,
                ipu_strategy=ipu_strategy).compile(feed_list, fetch_list)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        else:
            program = inference_program

        tmp = exe.run(program, feed=self.feed, fetch_list=[fetch_targets])

        return np.array(tmp)

    def test_base(self):
        self._test_save()
        cpu_res = self._test_load(False)
        ipu_res = self._test_load(True)

        np.testing.assert_allclose(cpu_res, ipu_res, rtol=1e-05, atol=self.atol)
        self.attrs['path'].cleanup()


class TestAdam(TestBase):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def set_op_attrs(self):
        self.attrs = {}
        self.attrs['steps'] = 100
        self.attrs['is_training'] = True
        self.attrs['opt_type'] = 'adam'
        self.attrs['path'] = tempfile.TemporaryDirectory()
        self.attrs['model_name'] = 'test'


class TestLamb(TestBase):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def set_op_attrs(self):
        self.attrs = {}
        self.attrs['steps'] = 100
        self.attrs['is_training'] = True
        self.attrs['opt_type'] = 'lamb'
        self.attrs['path'] = tempfile.TemporaryDirectory()
        self.attrs['model_name'] = 'test'


if __name__ == "__main__":
    unittest.main()
