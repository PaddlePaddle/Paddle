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
import paddle.fluid as fluid
import paddle.fluid.core as core
from paddle.fluid.contrib import sparsity
from paddle.fluid.contrib.sparsity.supported_layer_list import supported_layers_and_prune_func_map
from paddle.fluid.dygraph.layers import Layer, _convert_camel_to_snake


class MyOwnLayer(Layer):

    def __init__(self):
        super(MyOwnLayer, self).__init__()

    def forward(self, x):
        return x


static_tensor = None
static_tensor_mask = None


def my_own_pruning(tensor, m, n, mask_algo, param_name):
    global static_tensor
    global static_tensor_mask
    if static_tensor is None:
        static_tensor = np.random.rand(*tensor.shape).astype(np.float32)
    if static_tensor_mask is None:
        static_tensor_mask = np.random.rand(*tensor.shape).astype(np.float32)
    return static_tensor, static_tensor_mask


class TestASPAddSupportedLayer(unittest.TestCase):

    def test_add_supported_layer_via_name(self):
        sparsity.add_supported_layer("test_supported_1")
        sparsity.add_supported_layer("test_supported_2", my_own_pruning)
        sparsity.add_supported_layer(MyOwnLayer)
        my_own_layer_name = _convert_camel_to_snake(MyOwnLayer.__name__)

        self.assertTrue(
            "test_supported_1" in supported_layers_and_prune_func_map)
        self.assertTrue(
            "test_supported_2" in supported_layers_and_prune_func_map)
        self.assertTrue(
            "test_supported_2" in supported_layers_and_prune_func_map)
        self.assertTrue(supported_layers_and_prune_func_map["test_supported_2"]
                        == my_own_pruning)
        self.assertTrue(
            my_own_layer_name in supported_layers_and_prune_func_map)


class TestASPDynamicCustomerizedPruneFunc(unittest.TestCase):

    def setUp(self):
        paddle.disable_static()

        class CustomerLayer(paddle.nn.Layer):

            def __init__(self):
                super(CustomerLayer, self).__init__()

                self.weight = self.create_parameter(shape=[32, 32],
                                                    attr=None,
                                                    dtype='float32',
                                                    is_bias=False)
                self.linear1 = paddle.nn.Linear(32, 32)
                self.linear2 = paddle.nn.Linear(32, 10)

            def forward(self, input_):
                hidden = paddle.nn.functional.linear(x=input_,
                                                     weight=self.weight)
                hidden = self.linear1(hidden)
                out = self.linear2(hidden)
                return out

        sparsity.add_supported_layer(CustomerLayer, my_own_pruning)

        self.layer = CustomerLayer()
        self.customer_prefix = paddle.fluid.dygraph.layers._convert_camel_to_snake(
            CustomerLayer.__name__)
        self.supported_layer_count_ref = 3

    def test_inference_pruning(self):

        sparsity.prune_model(self.layer, mask_algo="mask_1d", with_mask=False)

        supported_layer_count = 0
        for param in self.layer.parameters():
            mat = param.numpy()

            if sparsity.asp.ASPHelper._is_supported_layer(
                    paddle.static.default_main_program(), param.name):
                supported_layer_count += 1
                if (self.customer_prefix in param.name):
                    self.assertLessEqual(
                        np.sum(mat.flatten() - static_tensor.flatten()), 1e-4)
                else:
                    self.assertTrue(
                        sparsity.check_sparsity(
                            mat.T,
                            func_name=sparsity.CheckMethod.CHECK_1D,
                            n=2,
                            m=4))
        self.assertEqual(supported_layer_count, self.supported_layer_count_ref)

    def test_training_pruning(self):
        optimizer = paddle.optimizer.SGD(learning_rate=0.01,
                                         parameters=self.layer.parameters())
        optimizer = sparsity.decorate(optimizer)

        sparsity.prune_model(self.layer, mask_algo="mask_1d", with_mask=True)

        supported_layer_count = 0
        for param in self.layer.parameters():
            mat = param.numpy()

            if sparsity.asp.ASPHelper._is_supported_layer(
                    paddle.static.default_main_program(), param.name):

                mat_mask = sparsity.asp.ASPHelper._get_program_asp_info(
                    paddle.static.default_main_program()).mask_vars[
                        param.name].numpy()

                supported_layer_count += 1
                if (self.customer_prefix in param.name):
                    self.assertLessEqual(
                        np.sum(mat.flatten() - static_tensor.flatten()), 1e-4)
                    self.assertLessEqual(
                        np.sum(mat_mask.flatten() -
                               static_tensor_mask.flatten()), 1e-4)
                else:
                    self.assertTrue(
                        sparsity.check_sparsity(
                            mat.T,
                            func_name=sparsity.CheckMethod.CHECK_1D,
                            n=2,
                            m=4))
                    self.assertTrue(
                        sparsity.check_sparsity(
                            mat_mask.T,
                            func_name=sparsity.CheckMethod.CHECK_1D,
                            n=2,
                            m=4))
        self.assertEqual(supported_layer_count, self.supported_layer_count_ref)


class TestASPStaticCustomerizedPruneFunc(unittest.TestCase):

    def setUp(self):
        paddle.enable_static()

        self.main_program = fluid.Program()
        self.startup_program = fluid.Program()

        self.customer_prefix = "customer_layer"

        def build_model():
            img = fluid.data(name='img',
                             shape=[None, 3, 32, 32],
                             dtype='float32')
            label = fluid.data(name='label', shape=[None, 1], dtype='int64')
            hidden = fluid.layers.conv2d(input=img,
                                         num_filters=4,
                                         filter_size=3,
                                         padding=2,
                                         act="relu")
            hidden = fluid.layers.fc(input=hidden,
                                     size=32,
                                     act='relu',
                                     name=self.customer_prefix)
            hidden = fluid.layers.fc(input=hidden,
                                     size=32,
                                     act='relu',
                                     name=self.customer_prefix)
            hidden = fluid.layers.fc(input=hidden, size=32, act='relu')
            prediction = fluid.layers.fc(input=hidden, size=10, act='softmax')
            return img, label, prediction

        with fluid.program_guard(self.main_program, self.startup_program):
            self.img, self.label, self.predict = build_model()
            self.supported_layer_count_ref = 5

        self.place = paddle.CPUPlace()
        if core.is_compiled_with_cuda():
            self.place = paddle.CUDAPlace(0)
        self.exe = fluid.Executor(self.place)

        sparsity.add_supported_layer(self.customer_prefix, my_own_pruning)

    def test_inference_pruning(self):
        self.exe.run(self.startup_program)

        sparsity.prune_model(self.main_program,
                             mask_algo="mask_1d",
                             with_mask=False)

        supported_layer_count = 0
        for param in self.main_program.global_block().all_parameters():
            mat = np.array(fluid.global_scope().find_var(
                param.name).get_tensor())
            if sparsity.asp.ASPHelper._is_supported_layer(
                    self.main_program, param.name):
                supported_layer_count += 1
                if (self.customer_prefix in param.name):
                    self.assertLessEqual(
                        np.sum(mat.flatten() - static_tensor.flatten()), 1e-4)
                else:
                    if (len(param.shape) == 4
                            and param.shape[1] < 4) or (len(param.shape) == 2
                                                        and param.shape[0] < 4):
                        self.assertFalse(
                            paddle.fluid.contrib.sparsity.check_sparsity(mat.T,
                                                                         n=2,
                                                                         m=4))
                    else:
                        self.assertTrue(
                            sparsity.check_sparsity(
                                mat.T,
                                func_name=sparsity.CheckMethod.CHECK_1D,
                                n=2,
                                m=4))
        self.assertEqual(supported_layer_count, self.supported_layer_count_ref)

    def test_training_pruning(self):
        with fluid.program_guard(self.main_program, self.startup_program):
            loss = paddle.mean(
                fluid.layers.cross_entropy(input=self.predict,
                                           label=self.label))
            optimizer = sparsity.decorate(
                fluid.optimizer.SGD(learning_rate=0.01))
            optimizer.minimize(loss, self.startup_program)

        self.exe.run(self.startup_program)

        sparsity.prune_model(self.main_program,
                             mask_algo="mask_1d",
                             with_mask=True)

        supported_layer_count = 0
        for param in self.main_program.global_block().all_parameters():
            mat = np.array(fluid.global_scope().find_var(
                param.name).get_tensor())
            if sparsity.asp.ASPHelper._is_supported_layer(
                    self.main_program, param.name):
                mat_mask = np.array(fluid.global_scope().find_var(
                    sparsity.asp.ASPHelper._get_mask_name(
                        param.name)).get_tensor())
                supported_layer_count += 1
                if (self.customer_prefix in param.name):
                    self.assertLessEqual(
                        np.sum(mat.flatten() - static_tensor.flatten()), 1e-4)
                    self.assertLessEqual(
                        np.sum(mat_mask.flatten() -
                               static_tensor_mask.flatten()), 1e-4)
                else:
                    if (len(param.shape) == 4
                            and param.shape[1] < 4) or (len(param.shape) == 2
                                                        and param.shape[0] < 4):
                        self.assertFalse(
                            sparsity.check_sparsity(mat.T, n=2, m=4))
                        self.assertFalse(
                            sparsity.check_sparsity(mat_mask.T, n=2, m=4))
                    else:
                        self.assertTrue(
                            sparsity.check_sparsity(
                                mat.T,
                                func_name=sparsity.CheckMethod.CHECK_1D,
                                n=2,
                                m=4))
                        self.assertTrue(
                            sparsity.check_sparsity(
                                mat_mask.T,
                                func_name=sparsity.CheckMethod.CHECK_1D,
                                n=2,
                                m=4))
        self.assertEqual(supported_layer_count, self.supported_layer_count_ref)


if __name__ == '__main__':
    unittest.main()
