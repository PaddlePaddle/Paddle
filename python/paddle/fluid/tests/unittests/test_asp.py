# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
# Copyright (c) 2021 NVIDIA Corporation.  All rights reserved.
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
import threading, time
import paddle
import paddle.fluid as fluid
import paddle.fluid.core as core
from paddle.fluid.contrib import sparsity
import numpy as np


class TestASPUtils(unittest.TestCase):
    def test_get_check_method(self):
        self.assertEqual(
            sparsity.CheckMethod.get_checking_method(sparsity.MaskAlgo.MASK_1D),
            sparsity.CheckMethod.CHECK_1D)
        self.assertEqual(
            sparsity.CheckMethod.get_checking_method(
                sparsity.MaskAlgo.MASK_2D_GREEDY),
            sparsity.CheckMethod.CHECK_2D)
        self.assertEqual(
            sparsity.CheckMethod.get_checking_method(
                sparsity.MaskAlgo.MASK_2D_BEST), sparsity.CheckMethod.CHECK_2D)

    def test_density(self):
        x = np.array([[1.0, 1.0, 1.0, 0.0, 1.0], [1.0, 1.0, 0.0, 0.0, 1.0],
                      [1.0, 0.0, 0.0, 0.0, 1.0], [1.0, 1.0, 0.0, 0.0, 1.0],
                      [0.0, 1.0, 0.0, 0.0, 1.0]])
        self.assertEqual(sparsity.density(x), 0.56)
        x[:, 0] = 0.0
        self.assertEqual(sparsity.density(x), 0.4)

    def test_check_mask_1d(self):
        x = np.array([[1.0, 0.0, 0.0, 1.0, 1.0], [1.0, 1.0, 0.0, 0.0, 1.0],
                      [1.0, 1.0, 0.0, 0.0, 1.0], [1.0, 1.0, 0.0, 0.0, 1.0],
                      [0.0, 1.0, 0.0, 0.0, 1.0]])
        self.assertTrue(sparsity.check_mask_1d(x, 2, 4))
        self.assertFalse(sparsity.check_mask_1d(x, 3, 4))
        self.assertTrue(sparsity.check_mask_1d(x, 2, 5))
        self.assertFalse(sparsity.check_mask_1d(x, 3, 5))
        self.assertTrue(sparsity.check_mask_1d(x, 3, 6))
        self.assertFalse(sparsity.check_mask_1d(x, 4, 6))

    def test_get_mask_1d(self):
        for _ in range(10):
            x = np.random.randint(10, size=(5, 5))
            x = sparsity.get_mask_1d(x, 2, 4)
            self.assertTrue(sparsity.check_mask_1d(x, 2, 4))

            x = np.random.randn(5, 4)
            x = sparsity.get_mask_1d(x, 2, 4)
            self.assertTrue(sparsity.check_mask_1d(x, 2, 4))

    def test_check_mask_2d(self):
        x = np.array([[1.0, 0.0, 0.0, 1.0, 1.0], [0.0, 1.0, 0.0, 0.0, 0.0],
                      [0.0, 0.0, 1.0, 0.0, 1.0], [1.0, 1.0, 0.0, 0.0, 0.0],
                      [0.0, 1.0, 0.0, 0.0, 1.0]])
        self.assertTrue(sparsity.check_mask_2d(x, 2, 4))
        self.assertFalse(sparsity.check_mask_2d(x, 3, 4))
        self.assertTrue(sparsity.check_mask_2d(x, 2, 5))
        self.assertFalse(sparsity.check_mask_2d(x, 3, 5))
        self.assertTrue(sparsity.check_mask_2d(x, 3, 6))
        self.assertFalse(sparsity.check_mask_2d(x, 4, 6))

    def test_get_mask_2d_greedy(self):
        for _ in range(10):
            x = np.random.randint(10, size=(5, 5))
            x = sparsity.get_mask_2d_greedy(x, 2, 4)
            self.assertTrue(sparsity.check_mask_2d(x, 2, 4))

            x = np.random.randn(5, 4)
            x = sparsity.get_mask_2d_greedy(x, 2, 4)
            self.assertTrue(sparsity.check_mask_2d(x, 2, 4))

    def test_get_mask_2d_best(self):
        for _ in range(10):
            x = np.random.randint(10, size=(5, 5))
            x = sparsity.get_mask_2d_best(x, 2, 4)
            self.assertTrue(sparsity.check_mask_2d(x, 2, 4))

            x = np.random.randn(5, 4)
            x = sparsity.get_mask_2d_best(x, 2, 4)
            self.assertTrue(sparsity.check_mask_2d(x, 2, 4))

    def test_threadsafe_valid_2d_patterns(self):
        def get_reference(m=4, n=2):
            from itertools import permutations

            patterns = np.zeros(m)
            patterns[:n] = 1
            patterns = list(set(permutations(patterns.tolist())))
            patterns = patterns + patterns
            patterns = np.asarray(list(set(permutations(patterns, m))))

            valid = ((patterns.sum(axis=1) <= n).sum(axis=1) == m
                     ).nonzero()[0].reshape(-1)
            valid_patterns = np.empty((valid.shape[0], m, m))
            valid_patterns[:] = patterns[valid[:]]
            return valid_patterns

        for _ in range(4):
            computing_thread = threading.Thread(
                target=paddle.fluid.contrib.sparsity.utils.
                compute_valid_2d_patterns,
                args=(2, 4))
            computing_thread.start()
        time.sleep(3)
        patterns_map = paddle.fluid.contrib.sparsity.utils.valid_2d_patterns
        reference_patterns = get_reference()
        reference_key = '4_2'

        self.assertTrue(reference_key in patterns_map)
        self.assertTrue(len(patterns_map) == 1)
        self.assertTrue((reference_patterns == patterns_map[reference_key]).all(
        ))

    def test_check_sparsity(self):
        for _ in range(10):
            x = np.random.randint(10, size=(5))
            x_2d = x.reshape(1, x.shape[0])
            self.__test_1D_2D_sparsity_checking_methods(x_2d)

            x = np.random.randint(10, size=(5, 5))
            x_2d = x
            self.__test_1D_2D_sparsity_checking_methods(x_2d)

            x = np.random.randint(10, size=(5, 5, 5))
            x_2d = x.reshape(x.shape[0] * x.shape[1], x.shape[2])
            self.__test_1D_2D_sparsity_checking_methods(x_2d)

            x = np.random.randint(10, size=(5, 5, 5, 5))
            x_2d = x.reshape(x.shape[0], x.shape[1] * x.shape[2] * x.shape[3])
            self.__test_1D_2D_sparsity_checking_methods(x_2d)

    def test_create_mask(self):
        for _ in range(10):
            x = np.random.randint(10, size=(5))
            self.__test_1D_2D_sparse_mask_generation_methods(x)

            x = np.random.randint(10, size=(5, 5))
            self.__test_1D_2D_sparse_mask_generation_methods(x)

            x = np.random.randint(10, size=(5, 5, 5))
            self.__test_1D_2D_sparse_mask_generation_methods(x)

            x = np.random.randint(10, size=(5, 5, 5, 5))
            self.__test_1D_2D_sparse_mask_generation_methods(x)

    def __test_1D_2D_sparsity_checking_methods(self, x_2d):
        mask = sparsity.get_mask_1d(x_2d, 2, 4)
        self.assertEqual(
            sparsity.check_sparsity(
                mask, func_name=sparsity.CheckMethod.CHECK_1D, n=2, m=4),
            sparsity.check_mask_1d(mask, 2, 4))
        mask = sparsity.get_mask_2d_best(x_2d, 2, 4)
        self.assertEqual(
            sparsity.check_sparsity(
                mask, func_name=sparsity.CheckMethod.CHECK_2D, n=2, m=4),
            sparsity.check_mask_2d(mask, 2, 4))

    def __test_1D_2D_sparse_mask_generation_methods(self, x):
        mask = sparsity.create_mask(
            x, func_name=sparsity.MaskAlgo.MASK_1D, n=2, m=4)
        self.assertTrue(
            sparsity.check_sparsity(
                mask, func_name=sparsity.CheckMethod.CHECK_1D, n=2, m=4))
        mask = sparsity.create_mask(
            x, func_name=sparsity.MaskAlgo.MASK_2D_GREEDY, n=2, m=4)
        self.assertTrue(
            sparsity.check_sparsity(
                mask, func_name=sparsity.CheckMethod.CHECK_2D, n=2, m=4))
        mask = sparsity.create_mask(
            x, func_name=sparsity.MaskAlgo.MASK_2D_BEST, n=2, m=4)
        self.assertTrue(
            sparsity.check_sparsity(
                mask, func_name=sparsity.CheckMethod.CHECK_2D, n=2, m=4))


class TestASPHelper(unittest.TestCase):
    def setUp(self):
        paddle.enable_static()
        self.main_program = fluid.Program()
        self.startup_program = fluid.Program()

        def build_model():
            img = fluid.data(
                name='img', shape=[None, 3, 32, 32], dtype='float32')
            label = fluid.data(name='label', shape=[None, 1], dtype='int64')
            hidden = fluid.layers.conv2d(
                input=img, num_filters=32, filter_size=3, padding=2, act="relu")
            hidden = fluid.layers.conv2d(
                input=hidden,
                num_filters=32,
                filter_size=3,
                padding=0,
                act="relu")
            hidden = fluid.layers.pool2d(
                input=hidden,
                pool_size=2,
                pool_type="max",
                pool_stride=1,
                global_pooling=False)
            hidden = fluid.layers.dropout(hidden, dropout_prob=0.25)

            hidden = fluid.layers.fc(input=hidden, size=512, act='relu')
            hidden = fluid.layers.fc(input=hidden, size=128, act='relu')
            hidden = fluid.layers.dropout(hidden, dropout_prob=0.5)
            prediction = fluid.layers.fc(input=hidden, size=10, act='softmax')

            return img, label, prediction

        with fluid.program_guard(self.main_program, self.startup_program):
            self.img, self.label, predict = build_model()
            self.loss = fluid.layers.mean(
                fluid.layers.cross_entropy(
                    input=predict, label=self.label))
            self.optimizer = fluid.optimizer.SGD(learning_rate=0.01)

    def test_get_vars(self):
        def check_params(params, params_from_asp):
            if len(params_from_asp) != len(params):
                return False

            for i, p in enumerate(params_from_asp):
                if p.name != params[i].name:
                    return False
            return True

        params = self.main_program.global_block().all_parameters()
        params_from_asp = sparsity.ASPHelper.get_vars(self.main_program)
        self.assertTrue(check_params(params, params_from_asp))

        with fluid.program_guard(self.main_program, self.startup_program):
            sparsity.ASPHelper.minimize(self.optimizer, self.loss,
                                        self.main_program, self.startup_program)
        params_from_asp_after_opt = sparsity.ASPHelper.get_vars(
            self.main_program)
        self.assertTrue(check_params(params, params_from_asp_after_opt))

    def test_is_supported_layers(self):
        program = paddle.static.default_main_program()

        names = [
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
                ref[i] == sparsity.ASPHelper.is_supported_layer(program, name))

        sparsity.ASPHelper.set_excluded_layers(program, ['fc_1', 'conv2d_0'])
        ref = [
            False, False, False, False, True, False, True, False, False, False,
            True, False
        ]
        for i, name in enumerate(names):
            self.assertTrue(
                ref[i] == sparsity.ASPHelper.is_supported_layer(program, name))

        sparsity.ASPHelper.reset_excluded_layers(program)
        ref = [
            False, False, True, False, True, False, True, False, True, False,
            True, False
        ]
        for i, name in enumerate(names):
            self.assertTrue(
                ref[i] == sparsity.ASPHelper.is_supported_layer(program, name))

    def test_decorate(self):
        param_names = self.__get_param_names(self.main_program.global_block()
                                             .all_parameters())
        with fluid.program_guard(self.main_program, self.startup_program):
            self.optimizer = sparsity.ASPHelper.decorate(self.optimizer)
            self.optimizer.minimize(self.loss, self.startup_program)
        param_names_after_minimize = self.__get_param_names(
            self.main_program.global_block().all_parameters())

        self.__check_mask_variables_and_ops(param_names,
                                            param_names_after_minimize)

    def test_minimize(self):
        param_names = self.__get_param_names(self.main_program.global_block()
                                             .all_parameters())
        with fluid.program_guard(self.main_program, self.startup_program):
            sparsity.ASPHelper.minimize(self.optimizer, self.loss,
                                        self.main_program, self.startup_program)
        param_names_after_minimize = self.__get_param_names(
            self.main_program.global_block().all_parameters())

        self.__check_mask_variables_and_ops(param_names,
                                            param_names_after_minimize)

    def test_inference_pruning(self):
        place = paddle.CPUPlace()
        if core.is_compiled_with_cuda():
            place = paddle.CUDAPlace(0)
        exe = fluid.Executor(place)

        self.__pruning_and_checking(exe, place, sparsity.MaskAlgo.MASK_1D,
                                    sparsity.CheckMethod.CHECK_1D, False)
        self.__pruning_and_checking(exe, place,
                                    sparsity.MaskAlgo.MASK_2D_GREEDY,
                                    sparsity.CheckMethod.CHECK_2D, False)
        self.__pruning_and_checking(exe, place, sparsity.MaskAlgo.MASK_2D_BEST,
                                    sparsity.CheckMethod.CHECK_2D, False)

    def test_training_pruning(self):
        with fluid.program_guard(self.main_program, self.startup_program):
            sparsity.ASPHelper.minimize(self.optimizer, self.loss,
                                        self.main_program, self.startup_program)

        place = paddle.CPUPlace()
        if core.is_compiled_with_cuda():
            place = paddle.CUDAPlace(0)
        exe = fluid.Executor(place)

        self.__pruning_and_checking(exe, place, sparsity.MaskAlgo.MASK_1D,
                                    sparsity.CheckMethod.CHECK_1D, True)
        self.__pruning_and_checking(exe, place,
                                    sparsity.MaskAlgo.MASK_2D_GREEDY,
                                    sparsity.CheckMethod.CHECK_2D, True)
        self.__pruning_and_checking(exe, place, sparsity.MaskAlgo.MASK_2D_BEST,
                                    sparsity.CheckMethod.CHECK_2D, True)

    def test_asp_training(self):
        with fluid.program_guard(self.main_program, self.startup_program):
            sparsity.ASPHelper.minimize(self.optimizer, self.loss,
                                        self.main_program, self.startup_program)

        place = paddle.CPUPlace()
        if core.is_compiled_with_cuda():
            place = paddle.CUDAPlace(0)
        exe = fluid.Executor(place)
        feeder = fluid.DataFeeder(feed_list=[self.img, self.label], place=place)

        exe.run(self.startup_program)
        sparsity.ASPHelper.prune_model(place, self.main_program)

        for _ in range(10):
            data = (np.random.randn(64, 3, 32, 32), np.random.randint(
                10, size=(64, 1)))
            exe.run(self.main_program, feed=feeder.feed([data]))

        for param in self.main_program.global_block().all_parameters():
            if sparsity.ASPHelper.is_supported_layer(self.main_program,
                                                     param.name):
                mat = np.array(fluid.global_scope().find_var(param.name)
                               .get_tensor())
                self.assertTrue(sparsity.check_sparsity(mat.T, n=2, m=4))

    def test_asp_training_with_amp(self):
        if core.is_compiled_with_cuda():
            place = paddle.CUDAPlace(0)
            with fluid.program_guard(self.main_program, self.startup_program):
                self.optimizer = fluid.contrib.mixed_precision.decorator.decorate(
                    self.optimizer)
                self.optimizer = sparsity.ASPHelper.decorate(self.optimizer)
                self.optimizer.minimize(self.loss, self.startup_program)

            exe = fluid.Executor(place)
            feeder = fluid.DataFeeder(
                feed_list=[self.img, self.label], place=place)

            exe.run(self.startup_program)
            sparsity.ASPHelper.prune_model(place, self.main_program)

            for _ in range(10):
                data = (np.random.randn(64, 3, 32, 32), np.random.randint(
                    10, size=(64, 1)))
                exe.run(self.main_program, feed=feeder.feed([data]))

            for param in self.main_program.global_block().all_parameters():
                if sparsity.ASPHelper.is_supported_layer(self.main_program,
                                                         param.name):
                    mat = np.array(fluid.global_scope().find_var(param.name)
                                   .get_tensor())
                    self.assertTrue(sparsity.check_sparsity(mat.T, n=2, m=4))

    def __get_param_names(self, params):
        param_names = []
        for p in params:
            param_names.append(p.name)
        return param_names

    def __check_mask_variables_and_ops(self, param_names,
                                       param_names_after_minimize):
        for n in param_names:
            self.assertFalse(sparsity.ASPHelper.is_supported_layer(self.main_program, n) and \
               sparsity.ASPHelper.get_mask_name(n) not in param_names_after_minimize)

        mask_names = []
        for n in param_names:
            if sparsity.ASPHelper.is_supported_layer(self.main_program, n):
                mask_names.append(sparsity.ASPHelper.get_mask_name(n))

        masking_ops = []
        for op in self.main_program.global_block().ops:
            if op.type == 'elementwise_mul' and \
               op.input('Y')[0] in mask_names:
                masking_ops.append(op.input('Y')[0])

        self.assertTrue(len(masking_ops) == len(mask_names))
        for n in masking_ops:
            self.assertTrue(n in mask_names)

        for n in mask_names:
            self.assertTrue(n in masking_ops)

    def __pruning_and_checking(self, exe, place, mask_func_name,
                               check_func_name, with_mask):
        exe.run(self.startup_program)
        sparsity.ASPHelper.prune_model(
            place,
            self.main_program,
            func_name=mask_func_name,
            with_mask=with_mask)
        for param in self.main_program.global_block().all_parameters():
            if sparsity.ASPHelper.is_supported_layer(self.main_program,
                                                     param.name):
                mat = np.array(fluid.global_scope().find_var(param.name)
                               .get_tensor())
                self.assertTrue(
                    sparsity.check_sparsity(
                        mat.T, func_name=check_func_name, n=2, m=4))


if __name__ == '__main__':
    unittest.main()
