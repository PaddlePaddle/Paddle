# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
# Copyright (c) 2022 NVIDIA Corporation.  All rights reserved.
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

<<<<<<< HEAD
import unittest

import numpy as np

import paddle
import paddle.fluid as fluid
import paddle.fluid.core as core
from paddle.incubate.asp import ASPHelper
=======
from __future__ import print_function

import unittest
import threading, time
import paddle
import paddle.fluid as fluid
import paddle.fluid.core as core
from paddle.fluid.contrib.sparsity.asp import ASPHelper
import numpy as np
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

paddle.enable_static()


class TestASPStaticOptimize(unittest.TestCase):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def setUp(self):
        self.main_program = fluid.Program()
        self.startup_program = fluid.Program()

        def build_model():
<<<<<<< HEAD
            img = fluid.data(
                name='img', shape=[None, 3, 24, 24], dtype='float32'
            )
            label = fluid.data(name='label', shape=[None, 1], dtype='int64')
            hidden = paddle.static.nn.conv2d(
                input=img, num_filters=4, filter_size=3, padding=2, act="relu"
            )
            hidden = paddle.static.nn.fc(x=hidden, size=32, activation='relu')
            prediction = paddle.static.nn.fc(
                x=hidden, size=10, activation='softmax'
            )
=======
            img = fluid.data(name='img',
                             shape=[None, 3, 24, 24],
                             dtype='float32')
            label = fluid.data(name='label', shape=[None, 1], dtype='int64')
            hidden = fluid.layers.conv2d(input=img,
                                         num_filters=4,
                                         filter_size=3,
                                         padding=2,
                                         act="relu")
            hidden = fluid.layers.fc(input=hidden, size=32, act='relu')
            prediction = fluid.layers.fc(input=hidden, size=10, act='softmax')
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
            return img, label, prediction

        with fluid.program_guard(self.main_program, self.startup_program):
            self.img, self.label, predict = build_model()
            self.loss = paddle.mean(
<<<<<<< HEAD
                paddle.nn.functional.cross_entropy(
                    input=predict,
                    label=self.label,
                    reduction='none',
                    use_softmax=False,
                )
            )
            self.optimizer = fluid.optimizer.SGD(learning_rate=0.01)

    def test_get_not_ASP_relevant_vars(self):
=======
                fluid.layers.cross_entropy(input=predict, label=self.label))
            self.optimizer = fluid.optimizer.SGD(learning_rate=0.01)

    def test_get_not_ASP_relevant_vars(self):

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        def check_params(params, params_from_asp):
            if len(params_from_asp) != len(params):
                return False

            for i, p in enumerate(params_from_asp):
                if p.name != params[i].name:
                    return False
            return True

        params = self.main_program.global_block().all_parameters()
        params_from_asp = ASPHelper._get_not_ASP_relevant_vars(
<<<<<<< HEAD
            self.main_program
        )
        self.assertTrue(check_params(params, params_from_asp))

        with fluid.program_guard(self.main_program, self.startup_program):
            ASPHelper._minimize(
                self.optimizer,
                self.loss,
                self.main_program,
                self.startup_program,
            )
        params_from_asp_after_opt = ASPHelper._get_not_ASP_relevant_vars(
            self.main_program
        )
=======
            self.main_program)
        self.assertTrue(check_params(params, params_from_asp))

        with fluid.program_guard(self.main_program, self.startup_program):
            ASPHelper._minimize(self.optimizer, self.loss, self.main_program,
                                self.startup_program)
        params_from_asp_after_opt = ASPHelper._get_not_ASP_relevant_vars(
            self.main_program)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        self.assertTrue(check_params(params, params_from_asp_after_opt))

    def test_is_supported_layers(self):
        program = paddle.static.default_main_program()

        names = [
<<<<<<< HEAD
            'embedding_0.w_0',
            'fack_layer_0.w_0',
            'conv2d_0.w_0',
            'conv2d_0.b_0',
            'conv2d_1.w_0',
            'conv2d_1.b_0',
            'fc_0.w_0',
            'fc_0.b_0',
            'fc_1.w_0',
            'fc_1.b_0',
            'linear_2.w_0',
            'linear_2.b_0',
        ]
        ref = [
            False,
            False,
            True,
            False,
            True,
            False,
            True,
            False,
            True,
            False,
            True,
            False,
        ]
        for i, name in enumerate(names):
            self.assertTrue(
                ref[i] == ASPHelper._is_supported_layer(program, name)
            )

        paddle.incubate.asp.set_excluded_layers(['fc_1', 'conv2d_0'], program)
        ref = [
            False,
            False,
            False,
            False,
            True,
            False,
            True,
            False,
            False,
            False,
            True,
            False,
        ]
        for i, name in enumerate(names):
            self.assertTrue(
                ref[i] == ASPHelper._is_supported_layer(program, name)
            )

        paddle.incubate.asp.reset_excluded_layers(program)
        ref = [
            False,
            False,
            True,
            False,
            True,
            False,
            True,
            False,
            True,
            False,
            True,
            False,
        ]
        for i, name in enumerate(names):
            self.assertTrue(
                ref[i] == ASPHelper._is_supported_layer(program, name)
            )

    def test_decorate(self):
        param_names = self.__get_param_names(
            self.main_program.global_block().all_parameters()
        )
=======
            'embedding_0.w_0', 'fack_layer_0.w_0', 'conv2d_0.w_0',
            'conv2d_0.b_0', 'conv2d_1.w_0', 'conv2d_1.b_0', 'fc_0.w_0',
            'fc_0.b_0', 'fc_1.w_0', 'fc_1.b_0', 'linear_2.w_0', 'linear_2.b_0'
        ]
        ref = [
            False, False, True, False, True, False, True, False, True, False,
            True, False
        ]
        for i, name in enumerate(names):
            self.assertTrue(
                ref[i] == ASPHelper._is_supported_layer(program, name))

        paddle.incubate.asp.set_excluded_layers(['fc_1', 'conv2d_0'], program)
        ref = [
            False, False, False, False, True, False, True, False, False, False,
            True, False
        ]
        for i, name in enumerate(names):
            self.assertTrue(
                ref[i] == ASPHelper._is_supported_layer(program, name))

        paddle.incubate.asp.reset_excluded_layers(program)
        ref = [
            False, False, True, False, True, False, True, False, True, False,
            True, False
        ]
        for i, name in enumerate(names):
            self.assertTrue(
                ref[i] == ASPHelper._is_supported_layer(program, name))

    def test_decorate(self):
        param_names = self.__get_param_names(
            self.main_program.global_block().all_parameters())
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        with fluid.program_guard(self.main_program, self.startup_program):
            self.optimizer = paddle.incubate.asp.decorate(self.optimizer)
            self.optimizer.minimize(self.loss, self.startup_program)
        param_names_after_minimize = self.__get_param_names(
<<<<<<< HEAD
            self.main_program.global_block().all_parameters()
        )

        self.__check_mask_variables_and_ops(
            param_names, param_names_after_minimize
        )
=======
            self.main_program.global_block().all_parameters())

        self.__check_mask_variables_and_ops(param_names,
                                            param_names_after_minimize)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

    def test_asp_training(self):
        with fluid.program_guard(self.main_program, self.startup_program):
            self.optimizer = paddle.incubate.asp.decorate(self.optimizer)
            self.optimizer.minimize(self.loss, self.startup_program)

        place = paddle.CPUPlace()
        if core.is_compiled_with_cuda():
            place = paddle.CUDAPlace(0)
        exe = fluid.Executor(place)
        feeder = fluid.DataFeeder(feed_list=[self.img, self.label], place=place)

        exe.run(self.startup_program)
        paddle.incubate.asp.prune_model(self.main_program)

<<<<<<< HEAD
        data = (
            np.random.randn(32, 3, 24, 24),
            np.random.randint(10, size=(32, 1)),
        )
=======
        data = (np.random.randn(32, 3, 24,
                                24), np.random.randint(10, size=(32, 1)))
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        exe.run(self.main_program, feed=feeder.feed([data]))

        for param in self.main_program.global_block().all_parameters():
            if ASPHelper._is_supported_layer(self.main_program, param.name):
<<<<<<< HEAD
                mat = np.array(
                    fluid.global_scope().find_var(param.name).get_tensor()
                )
                if (len(param.shape) == 4 and param.shape[1] < 4) or (
                    len(param.shape) == 2 and param.shape[0] < 4
                ):
                    self.assertFalse(
                        paddle.incubate.asp.check_sparsity(mat.T, n=2, m=4)
                    )
                else:
                    self.assertTrue(
                        paddle.incubate.asp.check_sparsity(mat.T, n=2, m=4)
                    )
=======
                mat = np.array(fluid.global_scope().find_var(
                    param.name).get_tensor())
                if (len(param.shape) == 4
                        and param.shape[1] < 4) or (len(param.shape) == 2
                                                    and param.shape[0] < 4):
                    self.assertFalse(
                        paddle.fluid.contrib.sparsity.check_sparsity(mat.T,
                                                                     n=2,
                                                                     m=4))
                else:
                    self.assertTrue(
                        paddle.fluid.contrib.sparsity.check_sparsity(mat.T,
                                                                     n=2,
                                                                     m=4))
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

    def test_asp_training_with_amp(self):
        if core.is_compiled_with_cuda():
            place = paddle.CUDAPlace(0)
            with fluid.program_guard(self.main_program, self.startup_program):
<<<<<<< HEAD
                self.optimizer = paddle.static.amp.decorate(self.optimizer)
=======
                self.optimizer = fluid.contrib.mixed_precision.decorator.decorate(
                    self.optimizer)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
                self.optimizer = paddle.incubate.asp.decorate(self.optimizer)
                self.optimizer.minimize(self.loss, self.startup_program)

            exe = fluid.Executor(place)
<<<<<<< HEAD
            feeder = fluid.DataFeeder(
                feed_list=[self.img, self.label], place=place
            )
=======
            feeder = fluid.DataFeeder(feed_list=[self.img, self.label],
                                      place=place)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

            exe.run(self.startup_program)
            paddle.incubate.asp.prune_model(self.main_program)

<<<<<<< HEAD
            data = (
                np.random.randn(32, 3, 24, 24),
                np.random.randint(10, size=(32, 1)),
            )
=======
            data = (np.random.randn(32, 3, 24,
                                    24), np.random.randint(10, size=(32, 1)))
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
            exe.run(self.main_program, feed=feeder.feed([data]))

            for param in self.main_program.global_block().all_parameters():
                if ASPHelper._is_supported_layer(self.main_program, param.name):
<<<<<<< HEAD
                    mat = np.array(
                        fluid.global_scope().find_var(param.name).get_tensor()
                    )
                    if (len(param.shape) == 4 and param.shape[1] < 4) or (
                        len(param.shape) == 2 and param.shape[0] < 4
                    ):
                        self.assertFalse(
                            paddle.incubate.asp.check_sparsity(mat.T, n=2, m=4)
                        )
                    else:
                        self.assertTrue(
                            paddle.incubate.asp.check_sparsity(mat.T, n=2, m=4)
                        )
=======
                    mat = np.array(fluid.global_scope().find_var(
                        param.name).get_tensor())
                    if (len(param.shape) == 4
                            and param.shape[1] < 4) or (len(param.shape) == 2
                                                        and param.shape[0] < 4):
                        self.assertFalse(
                            paddle.fluid.contrib.sparsity.check_sparsity(mat.T,
                                                                         n=2,
                                                                         m=4))
                    else:
                        self.assertTrue(
                            paddle.fluid.contrib.sparsity.check_sparsity(mat.T,
                                                                         n=2,
                                                                         m=4))
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

    def __get_param_names(self, params):
        param_names = []
        for p in params:
            param_names.append(p.name)
        return param_names

<<<<<<< HEAD
    def __check_mask_variables_and_ops(
        self, param_names, param_names_after_minimize
    ):
        for n in param_names:
            self.assertFalse(
                ASPHelper._is_supported_layer(self.main_program, n)
                and ASPHelper._get_mask_name(n)
                not in param_names_after_minimize
            )
=======
    def __check_mask_variables_and_ops(self, param_names,
                                       param_names_after_minimize):
        for n in param_names:
            self.assertFalse(ASPHelper._is_supported_layer(self.main_program, n) and \
               ASPHelper._get_mask_name(n) not in param_names_after_minimize)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

        mask_names = []
        for n in param_names:
            if ASPHelper._is_supported_layer(self.main_program, n):
                mask_names.append(ASPHelper._get_mask_name(n))

        masking_ops = []
        for op in self.main_program.global_block().ops:
<<<<<<< HEAD
            if op.type == 'elementwise_mul' and op.input('Y')[0] in mask_names:
=======
            if op.type == 'elementwise_mul' and \
               op.input('Y')[0] in mask_names:
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
                masking_ops.append(op.input('Y')[0])

        self.assertTrue(len(masking_ops) == len(mask_names))
        for n in masking_ops:
            self.assertTrue(n in mask_names)

        for n in mask_names:
            self.assertTrue(n in masking_ops)


if __name__ == '__main__':
    unittest.main()
