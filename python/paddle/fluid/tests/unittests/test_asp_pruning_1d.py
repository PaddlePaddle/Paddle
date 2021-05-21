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

paddle.enable_static()


class TestASPHelper(unittest.TestCase):
    def setUp(self):
        self.main_program = fluid.Program()
        self.startup_program = fluid.Program()

        def build_model():
            img = fluid.data(
                name='img', shape=[None, 3, 32, 32], dtype='float32')
            label = fluid.data(name='label', shape=[None, 1], dtype='int64')
            hidden = fluid.layers.conv2d(
                input=img, num_filters=8, filter_size=3, padding=2, act="relu")
            hidden = fluid.layers.fc(input=hidden, size=64, act='relu')
            prediction = fluid.layers.fc(input=hidden, size=10, act='softmax')
            return img, label, prediction

        with fluid.program_guard(self.main_program, self.startup_program):
            self.img, self.label, self.predict = build_model()

    def test_inference_pruning(self):
        place = paddle.CPUPlace()
        if core.is_compiled_with_cuda():
            place = paddle.CUDAPlace(0)
        exe = fluid.Executor(place)

        self.__pruning_and_checking(exe, place, sparsity.MaskAlgo.MASK_1D,
                                    sparsity.CheckMethod.CHECK_1D, False)

    def test_training_pruning(self):
        with fluid.program_guard(self.main_program, self.startup_program):
            loss = fluid.layers.mean(
                fluid.layers.cross_entropy(
                    input=self.predict, label=self.label))
            optimizer = fluid.optimizer.SGD(learning_rate=0.01)
            sparsity.ASPHelper.minimize(optimizer, loss, self.main_program,
                                        self.startup_program)

        place = paddle.CPUPlace()
        if core.is_compiled_with_cuda():
            place = paddle.CUDAPlace(0)
        exe = fluid.Executor(place)

        self.__pruning_and_checking(exe, place, sparsity.MaskAlgo.MASK_1D,
                                    sparsity.CheckMethod.CHECK_1D, True)

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
