#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import print_function

import unittest
import sys
sys.path.append("..")
import paddle
import paddle.fluid as fluid
import paddle.fluid.core as core
from paddle.fluid.dygraph.nn import Embedding
import paddle.fluid.framework as framework
from paddle.fluid.optimizer import Adam
from paddle.fluid.dygraph.base import to_variable
from test_imperative_base import new_program_scope
from paddle.fluid.executor import global_scope
import numpy as np
import six
import pickle
import os
import errno
from ..unittests.test_static_save_load import SimpleLSTMRNN, PtbModel

paddle.enable_static()


@unittest.skipIf(not paddle.is_compiled_with_npu(),
                 "core is not compiled with NPU")
class TestSaveLoadBase(unittest.TestCase):
    def test_ptb_rnn_cpu_float32(self):
        seed = 90
        hidden_size = 10
        vocab_size = 1000
        num_layers = 1
        num_steps = 3
        init_scale = 0.1
        batch_size = 4
        batch_num = 200

        with new_program_scope():
            fluid.default_startup_program().random_seed = seed
            fluid.default_main_program().random_seed = seed
            ptb_model = PtbModel(
                "ptb_model",
                hidden_size=hidden_size,
                vocab_size=vocab_size,
                num_layers=num_layers,
                num_steps=num_steps,
                init_scale=init_scale)

            place = fluid.CPUPlace() if not core.is_compiled_with_npu(
            ) else paddle.NPUPlace(0)
            exe = fluid.Executor(place)
            sgd = Adam(learning_rate=1e-3)
            x = fluid.layers.data(
                name="x", shape=[-1, num_steps], dtype='int64')
            y = fluid.layers.data(name="y", shape=[-1, 1], dtype='float32')
            init_hidden = fluid.layers.data(
                name="init_hidden", shape=[1], dtype='float32')
            init_cell = fluid.layers.data(
                name="init_cell", shape=[1], dtype='float32')

            static_loss, static_last_hidden, static_last_cell = ptb_model(
                x, y, init_hidden, init_cell)
            sgd.minimize(static_loss)
            static_param_updated = dict()
            static_param_init = dict()

            out = exe.run(framework.default_startup_program())

            static_loss_value = None
            static_last_cell_value = None
            static_last_hidden_value = None
            for i in range(batch_num):
                x_data = np.arange(12).reshape(4, 3).astype('int64')
                y_data = np.arange(1, 13).reshape(4, 3).astype('int64')
                x_data = x_data.reshape((-1, num_steps, 1))
                y_data = y_data.reshape((-1, 1))
                init_hidden_data = np.zeros(
                    (num_layers, batch_size, hidden_size), dtype='float32')
                init_cell_data = np.zeros(
                    (num_layers, batch_size, hidden_size), dtype='float32')
                fetch_list = [static_loss, static_last_hidden, static_last_cell]
                out = exe.run(fluid.default_main_program(),
                              feed={
                                  "x": x_data,
                                  "y": y_data,
                                  "init_hidden": init_hidden_data,
                                  "init_cell": init_cell_data
                              },
                              fetch_list=fetch_list)
                static_loss_value = out[0]
                static_last_hidden_value = out[1]
                static_last_cell_value = out[2]

            # get value before save
            main_program = framework.default_main_program()
            base_map = {}
            for var in main_program.list_vars():
                if isinstance(var, framework.Parameter) or var.persistable:
                    t = np.array(fluid.global_scope().find_var(var.name)
                                 .get_tensor())
                    # make sure all the paramerter or optimizer var have been update
                    self.assertTrue(np.sum(np.abs(t)) != 0)
                    base_map[var.name] = t

            fluid.save(main_program, "./test_1")

            # set var to zero
            for var in main_program.list_vars():
                if isinstance(var, framework.Parameter) or var.persistable:
                    ten = fluid.global_scope().find_var(var.name).get_tensor()
                    ten.set(np.zeros_like(np.array(ten)), place)

                    new_t = np.array(fluid.global_scope().find_var(var.name)
                                     .get_tensor())
                    # make sure all the paramerter or optimizer var have been set to zero
                    self.assertTrue(np.sum(np.abs(new_t)) == 0)

            fluid.load(main_program, "./test_1.pdparams", exe)

            for var in main_program.list_vars():
                if isinstance(var, framework.Parameter) or var.persistable:
                    new_t = np.array(fluid.global_scope().find_var(var.name)
                                     .get_tensor())
                    base_t = base_map[var.name]
                    self.assertTrue(np.array_equal(new_t, base_t))


@unittest.skipIf(not paddle.is_compiled_with_npu(),
                 "core is not compiled with NPU")
class TestSaveLoadPartial(unittest.TestCase):
    def test_ptb_rnn_cpu_float32(self):
        seed = 90
        hidden_size = 10
        vocab_size = 1000
        num_layers = 1
        num_steps = 3
        init_scale = 0.1
        batch_size = 4
        batch_num = 200

        with new_program_scope():
            fluid.default_startup_program().random_seed = seed
            fluid.default_main_program().random_seed = seed
            ptb_model = PtbModel(
                "ptb_model",
                hidden_size=hidden_size,
                vocab_size=vocab_size,
                num_layers=num_layers,
                num_steps=num_steps,
                init_scale=init_scale)

            place = fluid.CPUPlace() if not core.is_compiled_with_npu(
            ) else paddle.NPUPlace(0)
            exe = fluid.Executor(place)
            sgd = Adam(learning_rate=1e-3)
            x = fluid.layers.data(
                name="x", shape=[-1, num_steps], dtype='int64')
            y = fluid.layers.data(name="y", shape=[-1, 1], dtype='float32')
            init_hidden = fluid.layers.data(
                name="init_hidden", shape=[1], dtype='float32')
            init_cell = fluid.layers.data(
                name="init_cell", shape=[1], dtype='float32')

            static_loss, static_last_hidden, static_last_cell = ptb_model(
                x, y, init_hidden, init_cell)

            test_program = fluid.default_main_program().clone(for_test=True)

            add_1 = fluid.layers.fc(static_last_hidden,
                                    size=hidden_size,
                                    num_flatten_dims=2,
                                    bias_attr=False)

            sgd.minimize(static_loss)
            static_param_updated = dict()
            static_param_init = dict()

            out = exe.run(framework.default_startup_program())

            static_loss_value = None
            static_last_cell_value = None
            static_last_hidden_value = None
            for i in range(batch_num):
                x_data = np.arange(12).reshape(4, 3).astype('int64')
                y_data = np.arange(1, 13).reshape(4, 3).astype('int64')
                x_data = x_data.reshape((-1, num_steps, 1))
                y_data = y_data.reshape((-1, 1))
                init_hidden_data = np.zeros(
                    (num_layers, batch_size, hidden_size), dtype='float32')
                init_cell_data = np.zeros(
                    (num_layers, batch_size, hidden_size), dtype='float32')
                fetch_list = [static_loss, static_last_hidden, static_last_cell]
                out = exe.run(fluid.default_main_program(),
                              feed={
                                  "x": x_data,
                                  "y": y_data,
                                  "init_hidden": init_hidden_data,
                                  "init_cell": init_cell_data
                              },
                              fetch_list=fetch_list)
                static_loss_value = out[0]
                static_last_hidden_value = out[1]
                static_last_cell_value = out[2]

            # get value before save
            main_program = framework.default_main_program()
            base_map = {}
            for var in main_program.list_vars():
                if isinstance(var, framework.Parameter) or var.persistable:
                    t = np.array(fluid.global_scope().find_var(var.name)
                                 .get_tensor())
                    # make sure all the paramerter or optimizer var have been update
                    self.assertTrue(np.sum(np.abs(t)) != 0)
                    base_map[var.name] = t

            fluid.save(main_program, "./test_1")

            # set var to zero
            for var in main_program.list_vars():
                if isinstance(var, framework.Parameter) or var.persistable:
                    ten = fluid.global_scope().find_var(var.name).get_tensor()
                    ten.set(np.zeros_like(np.array(ten)), place)

                    new_t = np.array(fluid.global_scope().find_var(var.name)
                                     .get_tensor())
                    # make sure all the paramerter or optimizer var have been set to zero
                    self.assertTrue(np.sum(np.abs(new_t)) == 0)

            fluid.load(test_program, "./test_1.pdopt", None)

            for var in test_program.list_vars():
                if isinstance(var, framework.Parameter) or var.persistable:
                    new_t = np.array(fluid.global_scope().find_var(var.name)
                                     .get_tensor())
                    base_t = base_map[var.name]
                    self.assertTrue(np.array_equal(new_t, base_t))
            fluid.load(test_program, "./test_1.pdmodel", None)


@unittest.skipIf(not paddle.is_compiled_with_npu(),
                 "core is not compiled with NPU")
class TestSaveLoadSetStateDict(unittest.TestCase):
    def test_ptb_rnn_cpu_float32(self):
        seed = 90
        hidden_size = 10
        vocab_size = 1000
        num_layers = 1
        num_steps = 3
        init_scale = 0.1
        batch_size = 4
        batch_num = 200

        with new_program_scope():
            fluid.default_startup_program().random_seed = seed
            fluid.default_main_program().random_seed = seed
            ptb_model = PtbModel(
                "ptb_model",
                hidden_size=hidden_size,
                vocab_size=vocab_size,
                num_layers=num_layers,
                num_steps=num_steps,
                init_scale=init_scale)

            place = fluid.CPUPlace() if not core.is_compiled_with_npu(
            ) else paddle.NPUPlace(0)
            exe = fluid.Executor(place)
            sgd = Adam(learning_rate=1e-3)
            x = fluid.layers.data(
                name="x", shape=[-1, num_steps], dtype='int64')
            y = fluid.layers.data(name="y", shape=[-1, 1], dtype='float32')
            init_hidden = fluid.layers.data(
                name="init_hidden", shape=[1], dtype='float32')
            init_cell = fluid.layers.data(
                name="init_cell", shape=[1], dtype='float32')

            static_loss, static_last_hidden, static_last_cell = ptb_model(
                x, y, init_hidden, init_cell)
            sgd.minimize(static_loss)
            static_param_updated = dict()
            static_param_init = dict()

            out = exe.run(framework.default_startup_program())

            static_loss_value = None
            static_last_cell_value = None
            static_last_hidden_value = None
            for i in range(batch_num):
                x_data = np.arange(12).reshape(4, 3).astype('int64')
                y_data = np.arange(1, 13).reshape(4, 3).astype('int64')
                x_data = x_data.reshape((-1, num_steps, 1))
                y_data = y_data.reshape((-1, 1))
                init_hidden_data = np.zeros(
                    (num_layers, batch_size, hidden_size), dtype='float32')
                init_cell_data = np.zeros(
                    (num_layers, batch_size, hidden_size), dtype='float32')
                fetch_list = [static_loss, static_last_hidden, static_last_cell]
                out = exe.run(fluid.default_main_program(),
                              feed={
                                  "x": x_data,
                                  "y": y_data,
                                  "init_hidden": init_hidden_data,
                                  "init_cell": init_cell_data
                              },
                              fetch_list=fetch_list)
                static_loss_value = out[0]
                static_last_hidden_value = out[1]
                static_last_cell_value = out[2]

            # get value before save
            main_program = framework.default_main_program()
            base_map = {}
            for var in main_program.list_vars():
                if isinstance(var, framework.Parameter) or var.persistable:
                    t = np.array(fluid.global_scope().find_var(var.name)
                                 .get_tensor())
                    # make sure all the paramerter or optimizer var have been update
                    self.assertTrue(np.sum(np.abs(t)) != 0)
                    base_map[var.name] = t

            fluid.save(main_program, "./test_1")

            # set var to zero
            for var in main_program.list_vars():
                if isinstance(var, framework.Parameter) or var.persistable:
                    ten = fluid.global_scope().find_var(var.name).get_tensor()
                    ten.set(np.zeros_like(np.array(ten)), place)

                    new_t = np.array(fluid.global_scope().find_var(var.name)
                                     .get_tensor())
                    # make sure all the paramerter or optimizer var have been set to zero
                    self.assertTrue(np.sum(np.abs(new_t)) == 0)

            fluid.load(main_program, "./test_1", exe)

            for var in main_program.list_vars():
                if isinstance(var, framework.Parameter) or var.persistable:
                    new_t = np.array(fluid.global_scope().find_var(var.name)
                                     .get_tensor())
                    base_t = base_map[var.name]
                    self.assertTrue(np.array_equal(new_t, base_t))


@unittest.skipIf(not paddle.is_compiled_with_npu(),
                 "core is not compiled with NPU")
class TestProgramStatePartial(unittest.TestCase):
    def test_ptb_rnn_cpu_float32(self):
        seed = 90
        hidden_size = 10
        vocab_size = 1000
        num_layers = 1
        num_steps = 3
        init_scale = 0.1
        batch_size = 4
        batch_num = 200

        with new_program_scope():
            fluid.default_startup_program().random_seed = seed
            fluid.default_main_program().random_seed = seed
            ptb_model = PtbModel(
                "ptb_model",
                hidden_size=hidden_size,
                vocab_size=vocab_size,
                num_layers=num_layers,
                num_steps=num_steps,
                init_scale=init_scale)

            place = fluid.CPUPlace() if not core.is_compiled_with_npu(
            ) else paddle.NPUPlace(0)
            exe = fluid.Executor(place)
            sgd = Adam(learning_rate=1e-3)
            x = fluid.layers.data(
                name="x", shape=[-1, num_steps], dtype='int64')
            y = fluid.layers.data(name="y", shape=[-1, 1], dtype='float32')
            init_hidden = fluid.layers.data(
                name="init_hidden", shape=[1], dtype='float32')
            init_cell = fluid.layers.data(
                name="init_cell", shape=[1], dtype='float32')

            static_loss, static_last_hidden, static_last_cell = ptb_model(
                x, y, init_hidden, init_cell)

            test_program = fluid.default_main_program().clone(for_test=True)

            add_1 = fluid.layers.fc(static_last_hidden,
                                    size=hidden_size,
                                    num_flatten_dims=2,
                                    bias_attr=False)

            sgd.minimize(static_loss)
            static_param_updated = dict()
            static_param_init = dict()

            out = exe.run(framework.default_startup_program())

            static_loss_value = None
            static_last_cell_value = None
            static_last_hidden_value = None
            for i in range(batch_num):
                x_data = np.arange(12).reshape(4, 3).astype('int64')
                y_data = np.arange(1, 13).reshape(4, 3).astype('int64')
                x_data = x_data.reshape((-1, num_steps, 1))
                y_data = y_data.reshape((-1, 1))
                init_hidden_data = np.zeros(
                    (num_layers, batch_size, hidden_size), dtype='float32')
                init_cell_data = np.zeros(
                    (num_layers, batch_size, hidden_size), dtype='float32')
                fetch_list = [static_loss, static_last_hidden, static_last_cell]
                out = exe.run(fluid.default_main_program(),
                              feed={
                                  "x": x_data,
                                  "y": y_data,
                                  "init_hidden": init_hidden_data,
                                  "init_cell": init_cell_data
                              },
                              fetch_list=fetch_list)
                static_loss_value = out[0]
                static_last_hidden_value = out[1]
                static_last_cell_value = out[2]

            # get value before save
            main_program = framework.default_main_program()
            base_map = {}
            for var in main_program.list_vars():
                if isinstance(var, framework.Parameter) or var.persistable:
                    t = np.array(fluid.global_scope().find_var(var.name)
                                 .get_tensor())
                    # make sure all the paramerter or optimizer var have been update
                    self.assertTrue(np.sum(np.abs(t)) != 0)
                    base_map[var.name] = t

            fluid.save(main_program, os.path.join('some_dir', 'test_1'))

            # set var to zero
            for var in main_program.list_vars():
                if isinstance(var, framework.Parameter) or var.persistable:
                    ten = fluid.global_scope().find_var(var.name).get_tensor()
                    ten.set(np.zeros_like(np.array(ten)), place)

                    new_t = np.array(fluid.global_scope().find_var(var.name)
                                     .get_tensor())
                    # make sure all the paramerter or optimizer var have been set to zero
                    self.assertTrue(np.sum(np.abs(new_t)) == 0)

            #fluid.load(test_program, "./test_1", None )
            program_state = fluid.load_program_state(
                os.path.join('some_dir', 'test_1'))

            program_state_1 = fluid.load_program_state(
                os.path.join('some_dir', 'test_1.pdparams'))

            program_state_2 = fluid.load_program_state(
                os.path.join('some_dir', 'test_1.pdopt'))

            program_state_3 = fluid.load_program_state(
                os.path.join('some_dir', 'test_1.pdmodel'))

            fluid.set_program_state(test_program, program_state)

            for var in test_program.list_vars():
                if isinstance(var, framework.Parameter) or var.persistable:
                    new_t = np.array(fluid.global_scope().find_var(var.name)
                                     .get_tensor())
                    base_t = base_map[var.name]
                    self.assertTrue(np.array_equal(new_t, base_t))

            # check 1
            for var in main_program.list_vars():
                if isinstance(var, framework.Parameter) or var.persistable:
                    ten = fluid.global_scope().find_var(var.name).get_tensor()
                    ten.set(np.zeros_like(np.array(ten)), place)

                    new_t = np.array(fluid.global_scope().find_var(var.name)
                                     .get_tensor())
                    # make sure all the paramerter or optimizer var have been set to zero
                    self.assertTrue(np.sum(np.abs(new_t)) == 0)

            fluid.set_program_state(test_program, program_state_1)

            for var in test_program.list_vars():
                if isinstance(var, framework.Parameter) or var.persistable:
                    new_t = np.array(fluid.global_scope().find_var(var.name)
                                     .get_tensor())
                    base_t = base_map[var.name]
                    self.assertTrue(np.array_equal(new_t, base_t))

            # check 2
            for var in main_program.list_vars():
                if isinstance(var, framework.Parameter) or var.persistable:
                    ten = fluid.global_scope().find_var(var.name).get_tensor()
                    ten.set(np.zeros_like(np.array(ten)), place)

                    new_t = np.array(fluid.global_scope().find_var(var.name)
                                     .get_tensor())
                    # make sure all the paramerter or optimizer var have been set to zero
                    self.assertTrue(np.sum(np.abs(new_t)) == 0)

            fluid.set_program_state(test_program, program_state_2)

            for var in test_program.list_vars():
                if isinstance(var, framework.Parameter) or var.persistable:
                    new_t = np.array(fluid.global_scope().find_var(var.name)
                                     .get_tensor())
                    base_t = base_map[var.name]
                    self.assertTrue(np.array_equal(new_t, base_t))

            # check 3
            for var in main_program.list_vars():
                if isinstance(var, framework.Parameter) or var.persistable:
                    ten = fluid.global_scope().find_var(var.name).get_tensor()
                    ten.set(np.zeros_like(np.array(ten)), place)

                    new_t = np.array(fluid.global_scope().find_var(var.name)
                                     .get_tensor())
                    # make sure all the paramerter or optimizer var have been set to zero
                    self.assertTrue(np.sum(np.abs(new_t)) == 0)

            fluid.set_program_state(test_program, program_state_3)

            for var in test_program.list_vars():
                if isinstance(var, framework.Parameter) or var.persistable:
                    new_t = np.array(fluid.global_scope().find_var(var.name)
                                     .get_tensor())
                    base_t = base_map[var.name]
                    self.assertTrue(np.array_equal(new_t, base_t))


@unittest.skipIf(not paddle.is_compiled_with_npu(),
                 "core is not compiled with NPU")
class TestVariableInit(unittest.TestCase):
    def test_variable_init(self):

        x = fluid.data(name="x", shape=[10, 10], dtype='float32')
        y = fluid.layers.fc(x, 10)
        z = fluid.layers.fc(y, 10)

        place = fluid.CPUPlace() if not core.is_compiled_with_npu(
        ) else paddle.NPUPlace(0)
        exe = fluid.Executor(place)
        exe.run(fluid.default_startup_program())

        fluid.save(fluid.default_main_program(), "./test_path")

        def set_var(var, ndarray):
            t = var.get_tensor()
            p = t._place()
            if p.is_cpu_place():
                place = paddle.fluid.CPUPlace()
            else:
                place = paddle.NPUPlace(0)

            t.set(ndarray, place)

        program = fluid.default_main_program()
        new_scope = fluid.core.Scope()

        place = fluid.CPUPlace() if not core.is_compiled_with_npu(
        ) else paddle.NPUPlace(0)
        exe = fluid.Executor(place)
        parameter_list = list(
            filter(fluid.io.is_parameter, program.list_vars()))

        fluid.core._create_loaded_parameter(parameter_list, new_scope,
                                            exe._default_executor)
        parameter_file_name = "./test_path.pdparams"
        with open(parameter_file_name, 'rb') as f:
            load_dict = pickle.load(f)

        for v in parameter_list:
            assert v.name in load_dict, \
                "Can not find [{}] in model file [{}]".format(
                    v.name, parameter_file_name)
            new_v = new_scope.find_var(v.name)
            set_var(new_v, load_dict[v.name])

        opt_list = list(
            filter(fluid.io.is_belong_to_optimizer, program.list_vars()))

        fluid.core._create_loaded_parameter(opt_list, new_scope,
                                            exe._default_executor)
        opt_file_name = "./test_path.pdopt"
        with open(opt_file_name, 'rb') as f:
            load_dict = pickle.load(f)

        for v in opt_list:
            assert v.name in load_dict, \
                "Can not find [{}] in model file [{}]".format(
                    v.name, opt_file_name)

            new_v = new_scope.find_var(v.name)
            set_var(new_v, load_dict[v.name])

        base_map = {}
        for var in program.list_vars():
            if isinstance(var, framework.Parameter) or var.persistable:
                t = np.array(fluid.global_scope().find_var(var.name)
                             .get_tensor())
                # make sure all the paramerter or optimizer var have been update
                base_map[var.name] = t

        for var in program.list_vars():
            if isinstance(var, framework.Parameter) or var.persistable:
                new_t = np.array(new_scope.find_var(var.name).get_tensor())
                base_t = base_map[var.name]

                self.assertTrue(np.array_equal(new_t, base_t))


@unittest.skipIf(not paddle.is_compiled_with_npu(),
                 "core is not compiled with NPU")
class TestLoadFromOldInterface(unittest.TestCase):
    def setUp(self):
        if os.path.exists("test_path.pdparams"):
            os.remove("test_path.pdparams")

        if os.path.exists("test_static_load_var_list.pdparams"):
            os.remove("test_static_load_var_list.pdparams")

    def test_load_from_old_interface(self):
        seed = 90
        hidden_size = 10
        vocab_size = 1000
        num_layers = 1
        num_steps = 3
        init_scale = 0.1
        batch_size = 4
        batch_num = 200

        with new_program_scope():
            fluid.default_startup_program().random_seed = seed
            fluid.default_main_program().random_seed = seed
            ptb_model = PtbModel(
                "ptb_model",
                hidden_size=hidden_size,
                vocab_size=vocab_size,
                num_layers=num_layers,
                num_steps=num_steps,
                init_scale=init_scale)

            place = fluid.CPUPlace() if not core.is_compiled_with_npu(
            ) else paddle.NPUPlace(0)
            exe = fluid.Executor(place)
            sgd = Adam(learning_rate=1e-3)
            x = fluid.layers.data(
                name="x", shape=[-1, num_steps], dtype='int64')
            y = fluid.layers.data(name="y", shape=[-1, 1], dtype='float32')
            init_hidden = fluid.layers.data(
                name="init_hidden", shape=[1], dtype='float32')
            init_cell = fluid.layers.data(
                name="init_cell", shape=[1], dtype='float32')

            static_loss, static_last_hidden, static_last_cell = ptb_model(
                x, y, init_hidden, init_cell)

            test_clone_program = fluid.default_main_program().clone()
            sgd.minimize(static_loss)
            static_param_updated = dict()
            static_param_init = dict()

            out = exe.run(framework.default_startup_program())

            static_loss_value = None
            static_last_cell_value = None
            static_last_hidden_value = None
            for i in range(batch_num):
                x_data = np.arange(12).reshape(4, 3).astype('int64')
                y_data = np.arange(1, 13).reshape(4, 3).astype('int64')
                x_data = x_data.reshape((-1, num_steps, 1))
                y_data = y_data.reshape((-1, 1))
                init_hidden_data = np.zeros(
                    (num_layers, batch_size, hidden_size), dtype='float32')
                init_cell_data = np.zeros(
                    (num_layers, batch_size, hidden_size), dtype='float32')
                fetch_list = [static_loss, static_last_hidden, static_last_cell]
                out = exe.run(fluid.default_main_program(),
                              feed={
                                  "x": x_data,
                                  "y": y_data,
                                  "init_hidden": init_hidden_data,
                                  "init_cell": init_cell_data
                              },
                              fetch_list=fetch_list)
                static_loss_value = out[0]
                static_last_hidden_value = out[1]
                static_last_cell_value = out[2]

            # get value before save
            main_program = framework.default_main_program()
            base_map = {}
            for var in main_program.list_vars():
                if isinstance(var, framework.Parameter) or var.persistable:
                    t = np.array(fluid.global_scope().find_var(var.name)
                                 .get_tensor())
                    # make sure all the paramerter or optimizer var have been update
                    self.assertTrue(np.sum(np.abs(t)) != 0)
                    base_map[var.name] = t

            #fluid.save(main_program, "./test_1")
            fluid.io.save_persistables(exe, "test_path", main_program)

            # set var to zero
            for var in main_program.list_vars():
                if isinstance(var, framework.Parameter) or var.persistable:
                    ten = fluid.global_scope().find_var(var.name).get_tensor()
                    ten.set(np.zeros_like(np.array(ten)), place)

                    new_t = np.array(fluid.global_scope().find_var(var.name)
                                     .get_tensor())
                    # make sure all the paramerter or optimizer var have been set to zero
                    self.assertTrue(np.sum(np.abs(new_t)) == 0)

            fluid.load(main_program, "test_path", exe)

            for var in main_program.list_vars():
                if isinstance(var, framework.Parameter) or var.persistable:
                    new_t = np.array(fluid.global_scope().find_var(var.name)
                                     .get_tensor())
                    base_t = base_map[var.name]
                    self.assertTrue(np.array_equal(new_t, base_t))

            for var in main_program.list_vars():
                if isinstance(var, framework.Parameter) or var.persistable:
                    ten = fluid.global_scope().find_var(var.name).get_tensor()
                    old_shape = np.array(ten).shape
                    new_shape = [e + 10 for e in old_shape]

                    var.desc.set_shape(new_shape)
            with self.assertRaises(RuntimeError):
                fluid.load(main_program, "test_path", exe)

            # check unused parameter

            fluid.load(test_clone_program, "test_path", exe)

    def test_load_from_old_interface_var_list(self):
        seed = 90
        hidden_size = 10
        vocab_size = 1000
        num_layers = 1
        num_steps = 3
        init_scale = 0.1
        batch_size = 4
        batch_num = 200

        with new_program_scope():
            fluid.default_startup_program().random_seed = seed
            fluid.default_main_program().random_seed = seed
            ptb_model = PtbModel(
                "ptb_model",
                hidden_size=hidden_size,
                vocab_size=vocab_size,
                num_layers=num_layers,
                num_steps=num_steps,
                init_scale=init_scale)

            place = fluid.CPUPlace() if not core.is_compiled_with_npu(
            ) else paddle.NPUPlace(0)
            exe = fluid.Executor(place)
            sgd = Adam(learning_rate=1e-3)
            x = fluid.layers.data(
                name="x", shape=[-1, num_steps], dtype='int64')
            y = fluid.layers.data(name="y", shape=[-1, 1], dtype='float32')
            init_hidden = fluid.layers.data(
                name="init_hidden", shape=[1], dtype='float32')
            init_cell = fluid.layers.data(
                name="init_cell", shape=[1], dtype='float32')

            static_loss, static_last_hidden, static_last_cell = ptb_model(
                x, y, init_hidden, init_cell)

            test_clone_program = fluid.default_main_program().clone()
            sgd.minimize(static_loss)
            static_param_updated = dict()
            static_param_init = dict()

            out = exe.run(framework.default_startup_program())

            static_loss_value = None
            static_last_cell_value = None
            static_last_hidden_value = None
            for i in range(batch_num):
                x_data = np.arange(12).reshape(4, 3).astype('int64')
                y_data = np.arange(1, 13).reshape(4, 3).astype('int64')
                x_data = x_data.reshape((-1, num_steps, 1))
                y_data = y_data.reshape((-1, 1))
                init_hidden_data = np.zeros(
                    (num_layers, batch_size, hidden_size), dtype='float32')
                init_cell_data = np.zeros(
                    (num_layers, batch_size, hidden_size), dtype='float32')
                fetch_list = [static_loss, static_last_hidden, static_last_cell]
                out = exe.run(fluid.default_main_program(),
                              feed={
                                  "x": x_data,
                                  "y": y_data,
                                  "init_hidden": init_hidden_data,
                                  "init_cell": init_cell_data
                              },
                              fetch_list=fetch_list)
                static_loss_value = out[0]
                static_last_hidden_value = out[1]
                static_last_cell_value = out[2]

            # get value before save
            main_program = framework.default_main_program()
            base_map = {}
            for var in main_program.list_vars():
                if isinstance(var, framework.Parameter) or var.persistable:
                    t = np.array(fluid.global_scope().find_var(var.name)
                                 .get_tensor())
                    # make sure all the paramerter or optimizer var have been update
                    self.assertTrue(np.sum(np.abs(t)) != 0)
                    base_map[var.name] = t

            #fluid.save(main_program, "./test_1")
            fluid.io.save_persistables(exe, "test_static_load_var_list",
                                       main_program)

            # set var to zero            
            var_list = []
            for i, var in enumerate(main_program.list_vars()):
                if isinstance(var, framework.Parameter) or var.persistable:
                    if i % 2 == 0:
                        var_list.append(var)
                    ten = fluid.global_scope().find_var(var.name).get_tensor()
                    ten.set(np.zeros_like(np.array(ten)), place)

                    new_t = np.array(fluid.global_scope().find_var(var.name)
                                     .get_tensor())
                    # make sure all the paramerter or optimizer var have been set to zero
                    self.assertTrue(np.sum(np.abs(new_t)) == 0)

            fluid.load(main_program, "test_static_load_var_list", exe, var_list)
            var_list_names = [var.name for var in var_list]
            for var in main_program.list_vars():
                if isinstance(var, framework.Parameter) or var.persistable:
                    new_t = np.array(fluid.global_scope().find_var(var.name)
                                     .get_tensor())
                    if var.name in var_list_names:
                        # loaded vars
                        base_t = base_map[var.name]
                        self.assertTrue(np.array_equal(new_t, base_t))
                    else:
                        #not loaded vars
                        self.assertTrue(np.sum(np.abs(new_t)) == 0)


@unittest.skipIf(not paddle.is_compiled_with_npu(),
                 "core is not compiled with NPU")
class TestLoadFromOldInterfaceSingleFile(unittest.TestCase):
    def test_load_from_old_interface(self):
        seed = 90
        hidden_size = 10
        vocab_size = 1000
        num_layers = 1
        num_steps = 3
        init_scale = 0.1
        batch_size = 4
        batch_num = 200

        with new_program_scope():
            fluid.default_startup_program().random_seed = seed
            fluid.default_main_program().random_seed = seed
            ptb_model = PtbModel(
                "ptb_model",
                hidden_size=hidden_size,
                vocab_size=vocab_size,
                num_layers=num_layers,
                num_steps=num_steps,
                init_scale=init_scale)

            place = fluid.CPUPlace() if not core.is_compiled_with_npu(
            ) else paddle.NPUPlace(0)
            exe = fluid.Executor(place)
            sgd = Adam(learning_rate=1e-3)
            x = fluid.layers.data(
                name="x", shape=[-1, num_steps], dtype='int64')
            y = fluid.layers.data(name="y", shape=[-1, 1], dtype='float32')
            init_hidden = fluid.layers.data(
                name="init_hidden", shape=[1], dtype='float32')
            init_cell = fluid.layers.data(
                name="init_cell", shape=[1], dtype='float32')

            static_loss, static_last_hidden, static_last_cell = ptb_model(
                x, y, init_hidden, init_cell)
            sgd.minimize(static_loss)
            static_param_updated = dict()
            static_param_init = dict()

            out = exe.run(framework.default_startup_program())

            static_loss_value = None
            static_last_cell_value = None
            static_last_hidden_value = None
            for i in range(batch_num):
                x_data = np.arange(12).reshape(4, 3).astype('int64')
                y_data = np.arange(1, 13).reshape(4, 3).astype('int64')
                x_data = x_data.reshape((-1, num_steps, 1))
                y_data = y_data.reshape((-1, 1))
                init_hidden_data = np.zeros(
                    (num_layers, batch_size, hidden_size), dtype='float32')
                init_cell_data = np.zeros(
                    (num_layers, batch_size, hidden_size), dtype='float32')
                fetch_list = [static_loss, static_last_hidden, static_last_cell]
                out = exe.run(fluid.default_main_program(),
                              feed={
                                  "x": x_data,
                                  "y": y_data,
                                  "init_hidden": init_hidden_data,
                                  "init_cell": init_cell_data
                              },
                              fetch_list=fetch_list)
                static_loss_value = out[0]
                static_last_hidden_value = out[1]
                static_last_cell_value = out[2]

            # get value before save
            main_program = framework.default_main_program()
            base_map = {}
            for var in main_program.list_vars():
                if isinstance(var, framework.Parameter) or var.persistable:
                    t = np.array(fluid.global_scope().find_var(var.name)
                                 .get_tensor())
                    # make sure all the paramerter or optimizer var have been update
                    self.assertTrue(np.sum(np.abs(t)) != 0)
                    base_map[var.name] = t

            #fluid.save(main_program, "./test_1")
            fluid.io.save_persistables(
                exe, "test_path", main_program, filename="model_single")

            # set var to zero
            for var in main_program.list_vars():
                if isinstance(var, framework.Parameter) or var.persistable:
                    ten = fluid.global_scope().find_var(var.name).get_tensor()
                    ten.set(np.zeros_like(np.array(ten)), place)

                    new_t = np.array(fluid.global_scope().find_var(var.name)
                                     .get_tensor())
                    # make sure all the paramerter or optimizer var have been set to zero
                    self.assertTrue(np.sum(np.abs(new_t)) == 0)

            file_model_path = os.path.join("test_path", "model_single")
            fluid.load(main_program, file_model_path, exe,
                       fluid.io.get_program_persistable_vars(main_program))

            for var in main_program.list_vars():
                if isinstance(var, framework.Parameter) or var.persistable:
                    new_t = np.array(fluid.global_scope().find_var(var.name)
                                     .get_tensor())
                    base_t = base_map[var.name]
                    self.assertTrue(np.array_equal(new_t, base_t))

            # test exception
            # change shape
            for var in main_program.list_vars():
                if isinstance(var, framework.Parameter) or var.persistable:
                    ten = fluid.global_scope().find_var(var.name).get_tensor()
                    old_shape = np.array(ten).shape
                    new_shape = [e + 10 for e in old_shape]

                    var.desc.set_shape(new_shape)

            with self.assertRaises(RuntimeError):
                fluid.load(main_program, file_model_path, exe,
                           fluid.io.get_program_persistable_vars(main_program))

            fluid.io.save_params(
                exe, "test_path", main_program, filename="model_single")
            with self.assertRaises(RuntimeError):
                fluid.load(main_program, file_model_path, exe,
                           fluid.io.get_program_persistable_vars(main_program))

            # check when executor is None
            with self.assertRaises(ValueError):
                fluid.load(main_program, file_model_path, None,
                           fluid.io.get_program_persistable_vars(main_program))

            # check when var list is None
            with self.assertRaises(ValueError):
                fluid.load(main_program, file_model_path, exe, None)

            # check save params, load var_list = get_program_persistable_vars
            with self.assertRaises(RuntimeError):
                temp_var = framework.Variable(
                    main_program.global_block(),
                    shape=[1],
                    name="test_temp_var")
                all_var_list = list(main_program.list_vars())
                fluid.load(main_program, file_model_path, exe,
                           all_var_list + [temp_var])


@unittest.skipIf(not paddle.is_compiled_with_npu(),
                 "core is not compiled with NPU")
class TestProgramStateOldSave(unittest.TestCase):
    def test_ptb_rnn_cpu_float32(self):
        seed = 90
        hidden_size = 10
        vocab_size = 1000
        num_layers = 1
        num_steps = 3
        init_scale = 0.1
        batch_size = 4
        batch_num = 200

        with new_program_scope():
            fluid.default_startup_program().random_seed = seed
            fluid.default_main_program().random_seed = seed
            ptb_model = PtbModel(
                "ptb_model",
                hidden_size=hidden_size,
                vocab_size=vocab_size,
                num_layers=num_layers,
                num_steps=num_steps,
                init_scale=init_scale)

            place = fluid.CPUPlace() if not core.is_compiled_with_npu(
            ) else paddle.NPUPlace(0)
            exe = fluid.Executor(place)
            sgd = Adam(learning_rate=1e-3)
            x = fluid.layers.data(
                name="x", shape=[-1, num_steps], dtype='int64')
            y = fluid.layers.data(name="y", shape=[-1, 1], dtype='float32')
            init_hidden = fluid.layers.data(
                name="init_hidden", shape=[1], dtype='float32')
            init_cell = fluid.layers.data(
                name="init_cell", shape=[1], dtype='float32')

            static_loss, static_last_hidden, static_last_cell = ptb_model(
                x, y, init_hidden, init_cell)

            test_program = fluid.default_main_program().clone(for_test=True)

            add_1 = fluid.layers.fc(static_last_hidden,
                                    size=hidden_size,
                                    num_flatten_dims=2,
                                    bias_attr=False)

            sgd.minimize(static_loss)
            static_param_updated = dict()
            static_param_init = dict()

            out = exe.run(framework.default_startup_program())

            static_loss_value = None
            static_last_cell_value = None
            static_last_hidden_value = None
            for i in range(batch_num):
                x_data = np.arange(12).reshape(4, 3).astype('int64')
                y_data = np.arange(1, 13).reshape(4, 3).astype('int64')
                x_data = x_data.reshape((-1, num_steps, 1))
                y_data = y_data.reshape((-1, 1))
                init_hidden_data = np.zeros(
                    (num_layers, batch_size, hidden_size), dtype='float32')
                init_cell_data = np.zeros(
                    (num_layers, batch_size, hidden_size), dtype='float32')
                fetch_list = [static_loss, static_last_hidden, static_last_cell]
                out = exe.run(fluid.default_main_program(),
                              feed={
                                  "x": x_data,
                                  "y": y_data,
                                  "init_hidden": init_hidden_data,
                                  "init_cell": init_cell_data
                              },
                              fetch_list=fetch_list)
                static_loss_value = out[0]
                static_last_hidden_value = out[1]
                static_last_cell_value = out[2]

            # get value before save
            main_program = framework.default_main_program()
            base_map = {}
            for var in main_program.list_vars():
                if isinstance(var, framework.Parameter) or var.persistable:
                    t = np.array(fluid.global_scope().find_var(var.name)
                                 .get_tensor())
                    # make sure all the paramerter or optimizer var have been update
                    self.assertTrue(np.sum(np.abs(t)) != 0)
                    base_map[var.name] = t

            fluid.io.save_persistables(exe, "test_program_1", main_program)

            # set var to zero
            for var in main_program.list_vars():
                if isinstance(var, framework.Parameter) or var.persistable:
                    ten = fluid.global_scope().find_var(var.name).get_tensor()
                    ten.set(np.zeros_like(np.array(ten)), place)

                    new_t = np.array(fluid.global_scope().find_var(var.name)
                                     .get_tensor())
                    # make sure all the paramerter or optimizer var have been set to zero
                    self.assertTrue(np.sum(np.abs(new_t)) == 0)

            # case 1: load basic
            program_state = fluid.load_program_state("test_program_1")
            fluid.set_program_state(main_program, program_state)
            self.check_in_static(main_program, base_map)

            # case 2: load with no need file
            def symlink_force(target, link_name):
                try:
                    os.symlink(target, link_name)
                except OSError as e:
                    if e.errno == errno.EEXIST:
                        os.remove(link_name)
                        os.symlink(target, link_name)
                    else:
                        raise e

            orig_filepath = './test_program_1/fc_0.w_0'
            symlink_filepath = './test_program_1/link_fc_0.w_0'
            # create a needless link file for coverage
            symlink_force(orig_filepath, symlink_filepath)
            program_state = fluid.load_program_state("test_program_1")
            fluid.set_program_state(main_program, program_state)
            self.check_in_static(main_program, base_map)

            # case 3: load with var_list
            program_state = fluid.load_program_state(
                "test_program_1", main_program.all_parameters())
            fluid.set_program_state(main_program, program_state)
            self.check_in_static(main_program, base_map)

    def check_in_static(self, main_program, base_map):
        for var in main_program.list_vars():
            if isinstance(var, framework.Parameter) or var.persistable:
                new_t = np.array(fluid.global_scope().find_var(var.name)
                                 .get_tensor())
                base_t = base_map[var.name]
                self.assertTrue(np.array_equal(new_t, base_t))


@unittest.skipIf(not paddle.is_compiled_with_npu(),
                 "core is not compiled with NPU")
class TestStaticSaveLoadLargeParameters(unittest.TestCase):
    def test_large_parameters_static_save(self):
        # enable static mode
        paddle.enable_static()
        LARGE_PARAM = 2**26
        with new_program_scope():
            # create network
            x = paddle.static.data(
                name="static_save_load_large_x",
                shape=[None, 10],
                dtype='float32')
            z = paddle.static.nn.fc(x, LARGE_PARAM)
            place = paddle.CPUPlace()
            exe = paddle.static.Executor(place)
            exe.run(paddle.static.default_startup_program())
            prog = paddle.static.default_main_program()

            inputs = np.random.randn(1, 10).astype("float32")
            result_z = exe.run(program=prog,
                               feed={"static_save_load_large_x": inputs},
                               fetch_list=[z.name])
            path = "test_static_save_load_large_param/static_save"
            paddle.fluid.save(prog, path)

            paddle.fluid.load(prog, path)
            result_load = exe.run(program=prog,
                                  feed={"static_save_load_large_x": inputs},
                                  fetch_list=[z.name])
            # compare results before and after saving
            self.assertTrue(
                np.sum(np.abs(result_z[0] - result_load[0])) < 1e-15)


@unittest.skipIf(not paddle.is_compiled_with_npu(),
                 "core is not compiled with NPU")
class TestProgramStateOldSaveSingleModel(unittest.TestCase):
    def test_ptb_rnn_cpu_float32(self):
        seed = 90
        hidden_size = 10
        vocab_size = 1000
        num_layers = 1
        num_steps = 3
        init_scale = 0.1
        batch_size = 4
        batch_num = 200

        with new_program_scope():
            fluid.default_startup_program().random_seed = seed
            fluid.default_main_program().random_seed = seed
            ptb_model = PtbModel(
                "ptb_model",
                hidden_size=hidden_size,
                vocab_size=vocab_size,
                num_layers=num_layers,
                num_steps=num_steps,
                init_scale=init_scale)

            place = fluid.CPUPlace() if not core.is_compiled_with_npu(
            ) else paddle.NPUPlace(0)
            exe = fluid.Executor(place)
            sgd = Adam(learning_rate=1e-3)
            x = fluid.layers.data(
                name="x", shape=[-1, num_steps], dtype='int64')
            y = fluid.layers.data(name="y", shape=[-1, 1], dtype='float32')
            init_hidden = fluid.layers.data(
                name="init_hidden", shape=[1], dtype='float32')
            init_cell = fluid.layers.data(
                name="init_cell", shape=[1], dtype='float32')

            static_loss, static_last_hidden, static_last_cell = ptb_model(
                x, y, init_hidden, init_cell)

            test_program = fluid.default_main_program().clone(for_test=True)

            add_1 = fluid.layers.fc(static_last_hidden,
                                    size=hidden_size,
                                    num_flatten_dims=2,
                                    bias_attr=False)

            sgd.minimize(static_loss)
            static_param_updated = dict()
            static_param_init = dict()

            out = exe.run(framework.default_startup_program())

            static_loss_value = None
            static_last_cell_value = None
            static_last_hidden_value = None
            for i in range(batch_num):
                x_data = np.arange(12).reshape(4, 3).astype('int64')
                y_data = np.arange(1, 13).reshape(4, 3).astype('int64')
                x_data = x_data.reshape((-1, num_steps, 1))
                y_data = y_data.reshape((-1, 1))
                init_hidden_data = np.zeros(
                    (num_layers, batch_size, hidden_size), dtype='float32')
                init_cell_data = np.zeros(
                    (num_layers, batch_size, hidden_size), dtype='float32')
                fetch_list = [static_loss, static_last_hidden, static_last_cell]
                out = exe.run(fluid.default_main_program(),
                              feed={
                                  "x": x_data,
                                  "y": y_data,
                                  "init_hidden": init_hidden_data,
                                  "init_cell": init_cell_data
                              },
                              fetch_list=fetch_list)
                static_loss_value = out[0]
                static_last_hidden_value = out[1]
                static_last_cell_value = out[2]

            # get value before save
            main_program = framework.default_main_program()
            base_map = {}
            for var in main_program.list_vars():
                if isinstance(var, framework.Parameter) or var.persistable:
                    t = np.array(fluid.global_scope().find_var(var.name)
                                 .get_tensor())
                    # make sure all the paramerter or optimizer var have been update
                    self.assertTrue(np.sum(np.abs(t)) != 0)
                    base_map[var.name] = t

            fluid.io.save_persistables(
                exe, "test_program_2", main_program, filename="model_1")

            # set var to zero
            for var in main_program.list_vars():
                if isinstance(var, framework.Parameter) or var.persistable:
                    ten = fluid.global_scope().find_var(var.name).get_tensor()
                    ten.set(np.zeros_like(np.array(ten)), place)

                    new_t = np.array(fluid.global_scope().find_var(var.name)
                                     .get_tensor())
                    # make sure all the paramerter or optimizer var have been set to zero
                    self.assertTrue(np.sum(np.abs(new_t)) == 0)

            #fluid.load(test_program, "./test_1", None )
            program_state = fluid.load_program_state(
                os.path.join("test_program_2", "model_1"),
                var_list=fluid.io.get_program_persistable_vars(main_program))
            fluid.set_program_state(main_program, program_state)

            for var in main_program.list_vars():
                if isinstance(var, framework.Parameter) or var.persistable:
                    new_t = np.array(fluid.global_scope().find_var(var.name)
                                     .get_tensor())
                    base_t = base_map[var.name]
                    self.assertTrue(np.array_equal(new_t, base_t))

            with self.assertRaises(ValueError):
                fluid.load_program_state(
                    os.path.join("test_program_2", "model_1"))

            with self.assertRaises(TypeError):
                fluid.load_program_state(
                    os.path.join("test_program_2", "model_1"),
                    var_list=["str"])

            with self.assertRaises(RuntimeError):
                fluid.load_program_state(
                    os.path.join("test_program_2", "model_1"),
                    var_list=[
                        main_program.global_block().create_var(
                            name="fake_var_name", persistable=True)
                    ])


if __name__ == '__main__':
    paddle.enable_static()
    unittest.main()
