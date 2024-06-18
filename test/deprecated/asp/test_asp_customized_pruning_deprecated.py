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
from paddle.incubate import asp as sparsity
from paddle.nn.layer.layers import Layer


class MyOwnLayer(Layer):
    def __init__(self):
        super().__init__()

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


class TestASPStaticCustomizedPruneFunc(unittest.TestCase):
    def setUp(self):
        paddle.enable_static()

        self.main_program = base.Program()
        self.startup_program = base.Program()

        self.customer_prefix = "customer_layer"

        def build_model():
            img = paddle.static.data(
                name='img', shape=[None, 3, 32, 32], dtype='float32'
            )
            label = paddle.static.data(
                name='label', shape=[None, 1], dtype='int64'
            )
            hidden = paddle.static.nn.conv2d(
                input=img, num_filters=4, filter_size=3, padding=2, act="relu"
            )
            hidden = paddle.static.nn.fc(
                x=hidden, size=32, activation='relu', name=self.customer_prefix
            )
            hidden = paddle.static.nn.fc(
                x=hidden, size=32, activation='relu', name=self.customer_prefix
            )
            hidden = paddle.static.nn.fc(x=hidden, size=32, activation='relu')
            prediction = paddle.static.nn.fc(
                x=hidden, size=10, activation='softmax'
            )
            return img, label, prediction

        with base.program_guard(self.main_program, self.startup_program):
            self.img, self.label, self.predict = build_model()
            self.supported_layer_count_ref = 5

        self.place = paddle.CPUPlace()
        if core.is_compiled_with_cuda():
            self.place = paddle.CUDAPlace(0)
        self.exe = base.Executor(self.place)

        sparsity.add_supported_layer(self.customer_prefix, my_own_pruning)

    def test_inference_pruning(self):
        self.exe.run(self.startup_program)

        sparsity.prune_model(
            self.main_program, mask_algo="mask_1d", with_mask=False
        )

        supported_layer_count = 0
        for param in self.main_program.global_block().all_parameters():
            mat = np.array(
                base.global_scope().find_var(param.name).get_tensor()
            )
            if sparsity.asp.ASPHelper._is_supported_layer(
                self.main_program, param.name
            ):
                supported_layer_count += 1
                if self.customer_prefix in param.name:
                    self.assertLessEqual(
                        np.sum(mat.flatten() - static_tensor.flatten()), 1e-4
                    )
                else:
                    if (len(param.shape) == 4 and param.shape[1] < 4) or (
                        len(param.shape) == 2 and param.shape[0] < 4
                    ):
                        self.assertFalse(
                            paddle.incubate.asp.check_sparsity(mat.T, n=2, m=4)
                        )
                    else:
                        self.assertTrue(
                            sparsity.check_sparsity(
                                mat.T,
                                func_name=sparsity.CheckMethod.CHECK_1D,
                                n=2,
                                m=4,
                            )
                        )
        self.assertEqual(supported_layer_count, self.supported_layer_count_ref)

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
            optimizer = sparsity.decorate(
                paddle.optimizer.SGD(learning_rate=0.01)
            )
            optimizer.minimize(loss, self.startup_program)

        self.exe.run(self.startup_program)

        sparsity.prune_model(
            self.main_program, mask_algo="mask_1d", with_mask=True
        )

        supported_layer_count = 0
        for param in self.main_program.global_block().all_parameters():
            mat = np.array(
                base.global_scope().find_var(param.name).get_tensor()
            )
            if sparsity.asp.ASPHelper._is_supported_layer(
                self.main_program, param.name
            ):
                mat_mask = np.array(
                    base.global_scope()
                    .find_var(sparsity.asp.ASPHelper._get_mask_name(param.name))
                    .get_tensor()
                )
                supported_layer_count += 1
                if self.customer_prefix in param.name:
                    self.assertLessEqual(
                        np.sum(mat.flatten() - static_tensor.flatten()), 1e-4
                    )
                    self.assertLessEqual(
                        np.sum(
                            mat_mask.flatten() - static_tensor_mask.flatten()
                        ),
                        1e-4,
                    )
                else:
                    if (len(param.shape) == 4 and param.shape[1] < 4) or (
                        len(param.shape) == 2 and param.shape[0] < 4
                    ):
                        self.assertFalse(
                            sparsity.check_sparsity(mat.T, n=2, m=4)
                        )
                        self.assertFalse(
                            sparsity.check_sparsity(mat_mask.T, n=2, m=4)
                        )
                    else:
                        self.assertTrue(
                            sparsity.check_sparsity(
                                mat.T,
                                func_name=sparsity.CheckMethod.CHECK_1D,
                                n=2,
                                m=4,
                            )
                        )
                        self.assertTrue(
                            sparsity.check_sparsity(
                                mat_mask.T,
                                func_name=sparsity.CheckMethod.CHECK_1D,
                                n=2,
                                m=4,
                            )
                        )
        self.assertEqual(supported_layer_count, self.supported_layer_count_ref)


if __name__ == '__main__':
    unittest.main()
