# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

import unittest
import numpy as np
from io import BytesIO
import os
import platform
import tempfile

import paddle
import paddle.fluid as fluid
import paddle.fluid.framework as framework
from test_imperative_base import new_program_scope

IMAGE_SIZE = 784


class TestSaveLoadBinaryFormat(unittest.TestCase):

    def setUp(self):
        # enable static graph mode
        paddle.enable_static()
        self.temp_dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.temp_dir.cleanup()

    def set_zero(self, prog, place, scope=None):
        if scope is None:
            scope = fluid.global_scope()
        for var in prog.list_vars():
            if isinstance(var, framework.Parameter) or var.persistable:
                ten = scope.find_var(var.name).get_tensor()
                if ten is not None:
                    ten.set(np.zeros_like(np.array(ten)), place)
                    new_t = np.array(scope.find_var(var.name).get_tensor())
                    self.assertTrue(np.sum(np.abs(new_t)) == 0)

    def replace_save_vars(self, program, dirname):

        def predicate(var):
            return var.persistable

        vars = filter(predicate, program.list_vars())
        for var in vars:
<<<<<<< HEAD
            paddle.save(
                var.get_value(),
                os.path.join(dirname, var.name),
                use_binary_format=True,
            )
=======
            paddle.save(var.get_value(),
                        os.path.join(dirname, var.name),
                        use_binary_format=True)
>>>>>>> 5b0760feb220cd8f9e8a247c638a0f0d6df64baf

    def replace_load_vars(self, program, dirname):

        def predicate(var):
            return var.persistable

        var_list = list(filter(predicate, program.list_vars()))
        for var in var_list:
            var_load = paddle.load(os.path.join(dirname, var.name))
            # set var_load to scope
            var.set_value(var_load)

    def test_replace_save_load_vars(self):
        paddle.enable_static()
        with new_program_scope():
            # create network
<<<<<<< HEAD
            x = paddle.static.data(
                name="x", shape=[None, IMAGE_SIZE], dtype='float32'
            )
=======
            x = paddle.static.data(name="x",
                                   shape=[None, IMAGE_SIZE],
                                   dtype='float32')
>>>>>>> 5b0760feb220cd8f9e8a247c638a0f0d6df64baf
            z = paddle.static.nn.fc(x, 10, bias_attr=False)
            z = paddle.static.nn.fc(z, 128, bias_attr=False)
            loss = fluid.layers.reduce_mean(z)
            place = (
                fluid.CPUPlace()
                if not paddle.fluid.core.is_compiled_with_cuda()
                else fluid.CUDAPlace(0)
            )
            exe = paddle.static.Executor(place)
            exe.run(paddle.static.default_startup_program())
            prog = paddle.static.default_main_program()
            base_map = {}
            for var in prog.list_vars():
                if isinstance(var, framework.Parameter) or var.persistable:
<<<<<<< HEAD
                    t = np.array(
                        fluid.global_scope().find_var(var.name).get_tensor()
                    )
=======
                    t = np.array(fluid.global_scope().find_var(
                        var.name).get_tensor())
>>>>>>> 5b0760feb220cd8f9e8a247c638a0f0d6df64baf
                    # make sure all the paramerter or optimizer var have been update
                    self.assertTrue(np.sum(np.abs(t)) != 0)
                    base_map[var.name] = t
            # test for replace_save_vars/io.load_vars
            path_vars1 = os.path.join(
<<<<<<< HEAD
                self.temp_dir.name, 'test_replace_save_load_vars_binary1/model'
            )
=======
                self.temp_dir.name, 'test_replace_save_load_vars_binary1/model')
>>>>>>> 5b0760feb220cd8f9e8a247c638a0f0d6df64baf
            self.replace_save_vars(prog, path_vars1)
            # set var to zero
            self.set_zero(prog, place)
            var_list = list(
<<<<<<< HEAD
                filter(lambda var: var.persistable, prog.list_vars())
            )
            fluid.io.load_vars(
                exe, path_vars1, main_program=prog, vars=var_list
            )

            for var in prog.list_vars():
                if var.persistable:
                    new_t = np.array(
                        fluid.global_scope().find_var(var.name).get_tensor()
                    )
=======
                filter(lambda var: var.persistable, prog.list_vars()))
            fluid.io.load_vars(exe,
                               path_vars1,
                               main_program=prog,
                               vars=var_list)

            for var in prog.list_vars():
                if var.persistable:
                    new_t = np.array(fluid.global_scope().find_var(
                        var.name).get_tensor())
>>>>>>> 5b0760feb220cd8f9e8a247c638a0f0d6df64baf
                    base_t = base_map[var.name]

                    np.testing.assert_array_equal(new_t, base_t)
            # test for io.save_vars/replace_load_vars
            path_vars2 = os.path.join(
<<<<<<< HEAD
                self.temp_dir.name, 'test_replace_save_load_vars_binary2/model/'
            )
            fluid.io.save_vars(
                exe, path_vars2, main_program=prog, vars=var_list
            )
=======
                self.temp_dir.name,
                'test_replace_save_load_vars_binary2/model/')
            fluid.io.save_vars(exe,
                               path_vars2,
                               main_program=prog,
                               vars=var_list)
>>>>>>> 5b0760feb220cd8f9e8a247c638a0f0d6df64baf
            self.set_zero(prog, place)
            self.replace_load_vars(prog, path_vars2)
            for var in prog.list_vars():
                if var.persistable:
<<<<<<< HEAD
                    new_t = np.array(
                        fluid.global_scope().find_var(var.name).get_tensor()
                    )
=======
                    new_t = np.array(fluid.global_scope().find_var(
                        var.name).get_tensor())
>>>>>>> 5b0760feb220cd8f9e8a247c638a0f0d6df64baf
                    base_t = base_map[var.name]

                    np.testing.assert_array_equal(new_t, base_t)

    def test_save_load_lod_tensor(self):
        paddle.enable_static()
        OUTPUT_NUM = 32
        with new_program_scope():
            x = fluid.data(name="x", shape=[None, IMAGE_SIZE], dtype='float32')
            y = fluid.layers.fc(
                x,
                OUTPUT_NUM,
                name='fc_vars',
            )
            prog = fluid.default_main_program()
            place = (
                fluid.CPUPlace()
                if not paddle.fluid.core.is_compiled_with_cuda()
                else fluid.CUDAPlace(0)
            )
            exe = fluid.Executor(place)
            prog = paddle.static.default_main_program()
            exe.run(fluid.default_startup_program())

<<<<<<< HEAD
            dirname = os.path.join(
                self.temp_dir.name, 'test_save_load_lod_tensor1/tensor_'
            )
=======
            dirname = os.path.join(self.temp_dir.name,
                                   'test_save_load_lod_tensor1/tensor_')
>>>>>>> 5b0760feb220cd8f9e8a247c638a0f0d6df64baf
            for var in prog.list_vars():
                if var.persistable and list(var.shape) == [
                    IMAGE_SIZE,
                    OUTPUT_NUM,
                ]:
                    tensor = var.get_value()
<<<<<<< HEAD
                    paddle.save(
                        tensor, dirname + 'fc_vars.w_0', use_binary_format=True
                    )
=======
                    paddle.save(tensor,
                                dirname + 'fc_vars.w_0',
                                use_binary_format=True)
>>>>>>> 5b0760feb220cd8f9e8a247c638a0f0d6df64baf
                    break

            origin = np.array(var.get_value())
            var.set_value(np.zeros_like(origin))
            is_zeros = np.array(var.get_value())

            loaded_tensor = paddle.load(dirname + 'fc_vars.w_0')
            self.assertTrue(isinstance(loaded_tensor, fluid.core.LoDTensor))
            self.assertTrue(
                list(loaded_tensor.shape()) == [IMAGE_SIZE, OUTPUT_NUM]
            )
            to_array = np.array(loaded_tensor)
            np.testing.assert_array_equal(origin, to_array)

        with self.assertRaises(NotImplementedError):
            path = os.path.join(self.temp_dir.name, 'test_save_load_error/temp')
            paddle.save({}, path, use_binary_format=True)
        # On the Windows platform, when parsing a string that can't be parsed as a `Program`, `desc_.ParseFromString` has a timeout risk.
        if 'Windows' != platform.system():
            with self.assertRaises(ValueError):
<<<<<<< HEAD
                path = os.path.join(
                    self.temp_dir.name, 'test_save_load_error/temp'
                )
=======
                path = os.path.join(self.temp_dir.name,
                                    'test_save_load_error/temp')
>>>>>>> 5b0760feb220cd8f9e8a247c638a0f0d6df64baf
                with open(path, "w") as f:
                    f.write('\0')
                paddle.load(path)

        with self.assertRaises(ValueError):
            temp_lod = fluid.core.LoDTensor()
            paddle.save(temp_lod, path, use_binary_format=True)

        with self.assertRaises(RuntimeError):
            fluid.core.save_lod_tensor(
                temp_lod,
                os.path.join(
                    self.temp_dir.name,
<<<<<<< HEAD
                    'test_save_load_error_not_exist_file/not_exist_file',
                ),
            )
=======
                    'test_save_load_error_not_exist_file/not_exist_file'))
>>>>>>> 5b0760feb220cd8f9e8a247c638a0f0d6df64baf

        with self.assertRaises(RuntimeError):
            fluid.core.load_lod_tensor(
                temp_lod,
                os.path.join(
                    self.temp_dir.name,
<<<<<<< HEAD
                    'test_save_load_error_not_exist_file/not_exist_file',
                ),
            )
=======
                    'test_save_load_error_not_exist_file/not_exist_file'))
>>>>>>> 5b0760feb220cd8f9e8a247c638a0f0d6df64baf

        # save to memory
        byio = BytesIO()
        paddle.save(tensor, byio, use_binary_format=True)
        byio.seek(0)
        # load from memory
        loaded_tensor_mem = paddle.load(byio)
        to_array_mem = np.array(loaded_tensor_mem)
        np.testing.assert_array_equal(np.array(tensor), to_array_mem)

        with self.assertRaises(NotImplementedError):
            paddle.framework.io._save_lod_tensor(tensor, 1)
        with self.assertRaises(NotImplementedError):
            paddle.framework.io._load_lod_tensor(1)

    def test_save_load_selected_rows(self):
        paddle.enable_static()
        place = (
            fluid.CPUPlace()
            if not paddle.fluid.core.is_compiled_with_cuda()
            else fluid.CUDAPlace(0)
        )
        height = 10
        rows = [0, 4, 7]
        row_numel = 12
        selected_rows = fluid.core.SelectedRows(rows, height)
<<<<<<< HEAD
        path = os.path.join(
            self.temp_dir.name, 'test_paddle_save_load_selected_rows/sr.pdsr'
        )
=======
        path = os.path.join(self.temp_dir.name,
                            'test_paddle_save_load_selected_rows/sr.pdsr')
>>>>>>> 5b0760feb220cd8f9e8a247c638a0f0d6df64baf

        with self.assertRaises(ValueError):
            paddle.save(selected_rows, path, use_binary_format=True)

        np_array = np.random.randn(len(rows), row_numel).astype("float32")
        tensor = selected_rows.get_tensor()
        tensor.set(np_array, place)

        paddle.save(selected_rows, path, use_binary_format=True)
        load_sr = paddle.load(path)

        self.assertTrue(isinstance(load_sr, fluid.core.SelectedRows))
        self.assertTrue(list(load_sr.rows()) == rows)
        self.assertTrue(load_sr.height() == height)
<<<<<<< HEAD
        np.testing.assert_array_equal(np.array(load_sr.get_tensor()), np_array)
=======
        self.assertTrue(np.array_equal(np.array(load_sr.get_tensor()),
                                       np_array))
>>>>>>> 5b0760feb220cd8f9e8a247c638a0f0d6df64baf

        with self.assertRaises(RuntimeError):
            fluid.core.save_selected_rows(
                selected_rows,
                os.path.join(
                    self.temp_dir.name,
<<<<<<< HEAD
                    'test_paddle_save_load_selected_rows_not_exist_file/temp',
                ),
            )
=======
                    'test_paddle_save_load_selected_rows_not_exist_file/temp'))
>>>>>>> 5b0760feb220cd8f9e8a247c638a0f0d6df64baf
        with self.assertRaises(RuntimeError):
            fluid.core.load_selected_rows(
                selected_rows,
                os.path.join(
                    self.temp_dir.name,
<<<<<<< HEAD
                    'test_paddle_save_load_selected_rows_not_exist_file/temp',
                ),
            )
=======
                    'test_paddle_save_load_selected_rows_not_exist_file/temp'))
>>>>>>> 5b0760feb220cd8f9e8a247c638a0f0d6df64baf

        # save to memory
        byio = BytesIO()
        paddle.save(selected_rows, byio, use_binary_format=True)
        byio.seek(0)
        # load from memory
        selected_rows_mem = paddle.load(byio)
        to_array_mem = np.array(selected_rows_mem)
        self.assertTrue(isinstance(selected_rows_mem, fluid.core.SelectedRows))
        self.assertTrue(list(selected_rows_mem.rows()) == rows)
        self.assertTrue(selected_rows_mem.height() == height)
        np.testing.assert_array_equal(
            np.array(selected_rows_mem.get_tensor()), np_array
        )

        with self.assertRaises(NotImplementedError):
            paddle.framework.io._save_selected_rows(selected_rows, 1)
        with self.assertRaises(NotImplementedError):
            paddle.framework.io._load_selected_rows(1)
