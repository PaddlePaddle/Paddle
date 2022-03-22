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

from __future__ import print_function

import unittest
import numpy as np

import paddle
from paddle.fluid import core
from paddle.static import sparsity
from paddle.fluid.contrib.sparsity.asp import ASPHelper


class MyLayer(paddle.nn.Layer):
    def __init__(self):
        super(MyLayer, self).__init__()
        self.conv1 = paddle.nn.Conv2D(
            in_channels=3, out_channels=4, kernel_size=3, padding=2)
        self.linear1 = paddle.nn.Linear(32 * 32 * 4, 32)
        self.linear2 = paddle.nn.Linear(32, 32)
        self.linear3 = paddle.nn.Linear(32, 10)

    def forward(self, img):
        hidden = self.conv1(img)
        hidden = paddle.flatten(hidden, start_axis=1)
        hidden = self.linear1(hidden)
        hidden = self.linear2(hidden)
        prediction = self.linear3(hidden)
        return prediction


class TestASPDynamicPruningBase(unittest.TestCase):
    def setUp(self):
        self.layer = MyLayer()

        place = paddle.CPUPlace()
        if core.is_compiled_with_cuda():
            place = paddle.CUDAPlace(0)

        self.img = paddle.to_tensor(
            np.random.uniform(
                low=-0.5, high=0.5, size=(64, 3, 32, 32)),
            dtype=np.float32,
            place=place,
            stop_gradient=False)

    def run_inference_pruning_test(self, get_mask_gen_func,
                                   get_mask_check_func):
        self.__pruning_and_checking(get_mask_gen_func, get_mask_check_func,
                                    False)

    def run_training_pruning_test(self, get_mask_gen_func, get_mask_check_func):

        optimizer = paddle.optimizer.SGD(learning_rate=0.01,
                                         parameters=self.layer.parameters())
        optimizer = sparsity.decorate(optimizer)

        self.__pruning_and_checking(get_mask_gen_func, get_mask_check_func,
                                    True)

    def __pruning_and_checking(self, mask_func_name, check_func_name,
                               with_mask):

        sparsity.prune_model(
            self.layer, mask_algo=mask_func_name, with_mask=with_mask)

        for param in self.layer.parameters():
            if ASPHelper._is_supported_layer(
                    paddle.static.default_main_program(), param.name):
                mat = param.numpy()
                self.assertTrue(
                    paddle.fluid.contrib.sparsity.check_sparsity(
                        mat.T, func_name=check_func_name, n=2, m=4))


class TestASPDynamicPruning1D(TestASPDynamicPruningBase):
    def test_1D_inference_pruning(self):
        self.run_inference_pruning_test(
            'mask_1d', paddle.fluid.contrib.sparsity.CheckMethod.CHECK_1D)

    def test_1D_training_pruning(self):
        self.run_training_pruning_test(
            'mask_1d', paddle.fluid.contrib.sparsity.CheckMethod.CHECK_1D)


class TestASPDynamicPruning2DBest(TestASPDynamicPruningBase):
    def test_1D_inference_pruning(self):
        self.run_inference_pruning_test(
            'mask_2d_best', paddle.fluid.contrib.sparsity.CheckMethod.CHECK_2D)

    def test_1D_training_pruning(self):
        self.run_training_pruning_test(
            'mask_2d_best', paddle.fluid.contrib.sparsity.CheckMethod.CHECK_2D)


class TestASPDynamicPruning2DGreedy(TestASPDynamicPruningBase):
    def test_1D_inference_pruning(self):
        self.run_inference_pruning_test(
            'mask_2d_greedy',
            paddle.fluid.contrib.sparsity.CheckMethod.CHECK_2D)

    def test_1D_training_pruning(self):
        self.run_training_pruning_test(
            'mask_2d_greedy',
            paddle.fluid.contrib.sparsity.CheckMethod.CHECK_2D)


if __name__ == '__main__':
    unittest.main()
