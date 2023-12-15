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

import unittest

import numpy as np

import paddle
from paddle import base
from paddle.base import core
from paddle.incubate.asp import ASPHelper

paddle.enable_static()


class TestASPStaticPruningBase(unittest.TestCase):
    def setUp(self):
        self.main_program = base.Program()
        self.startup_program = base.Program()

        def build_model():
            img = paddle.static.data(
                name='img', shape=[None, 3, 24, 24], dtype='float32'
            )
            label = paddle.static.data(
                name='label', shape=[None, 1], dtype='int64'
            )
            hidden = paddle.static.nn.conv2d(
                input=img, num_filters=2, filter_size=3, padding=2, act="relu"
            )
            hidden = paddle.static.nn.fc(
                x=hidden, size=32, activation='softmax'
            )
            hidden = paddle.static.nn.fc(x=hidden, size=3, activation='softmax')
            prediction = paddle.static.nn.fc(
                x=hidden, size=3, activation='softmax'
            )
            return img, label, prediction

        with base.program_guard(self.main_program, self.startup_program):
            self.img, self.label, self.predict = build_model()

        self.set_config()

    def set_config(self):
        self.mask_gen_func = 'mask_1d'
        self.mask_check_func = paddle.incubate.asp.CheckMethod.CHECK_1D

    def test_inference_pruning(self):
        place = paddle.CPUPlace()
        if core.is_compiled_with_cuda():
            place = paddle.CUDAPlace(0)
        exe = base.Executor(place)

        self.__pruning_and_checking(exe, place, False)

    def test_training_pruning(self):
        with base.program_guard(self.main_program, self.startup_program):
            loss = paddle.mean(
                paddle.nn.functional.cross_entropy(
                    input=self.predict,
                    label=self.label,
                    reduction='none',
                    use_softmax=False,
                )
            )
            optimizer = paddle.incubate.asp.decorate(
                paddle.optimizer.SGD(learning_rate=0.01)
            )
            optimizer.minimize(loss, self.startup_program)

        place = paddle.CPUPlace()
        if core.is_compiled_with_cuda():
            place = paddle.CUDAPlace(0)
        exe = base.Executor(place)

        self.__pruning_and_checking(exe, place, True)

    def __pruning_and_checking(self, exe, place, with_mask):
        exe.run(self.startup_program)
        paddle.incubate.asp.prune_model(
            self.main_program, mask_algo=self.mask_gen_func, with_mask=with_mask
        )
        for param in self.main_program.global_block().all_parameters():
            if ASPHelper._is_supported_layer(self.main_program, param.name):
                mat = np.array(
                    base.global_scope().find_var(param.name).get_tensor()
                )
                if (len(param.shape) == 4 and param.shape[1] < 4) or (
                    len(param.shape) == 2 and param.shape[0] < 4
                ):
                    self.assertFalse(
                        paddle.incubate.asp.check_sparsity(mat.T, n=2, m=4)
                    )
                else:
                    self.assertTrue(
                        paddle.incubate.asp.check_sparsity(
                            mat.T, func_name=self.mask_check_func, n=2, m=4
                        )
                    )


class TestASPStaticPruning1D(TestASPStaticPruningBase):
    def set_config(self):
        self.mask_gen_func = 'mask_1d'
        self.mask_check_func = paddle.incubate.asp.CheckMethod.CHECK_1D


class TestASPStaticPruning2DBest(TestASPStaticPruningBase):
    def set_config(self):
        self.mask_gen_func = 'mask_2d_best'
        self.mask_check_func = paddle.incubate.asp.CheckMethod.CHECK_2D


class TestASPStaticPruning2DGreedy(TestASPStaticPruningBase):
    def set_config(self):
        self.mask_gen_func = 'mask_2d_greedy'
        self.mask_check_func = paddle.incubate.asp.CheckMethod.CHECK_2D


if __name__ == '__main__':
    unittest.main()
