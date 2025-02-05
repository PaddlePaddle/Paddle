# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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
from paddle.base.framework import paddle_type_to_proto_type


class TestEagerTensorLegacy(unittest.TestCase):
    def setUp(self):
        self.shape = [512, 1234]
        self.dtype = np.float32
        self.array = np.random.uniform(0.1, 1, self.shape).astype(self.dtype)

    def test_block(self):
        var = paddle.to_tensor(self.array)
        self.assertEqual(var.block, base.default_main_program().global_block())

    def test_to_static_var(self):
        with base.dygraph.guard():
            # Convert Tensor into Variable or Parameter
            tensor = paddle.to_tensor(self.array)
            static_var = tensor._to_static_var()
            self._assert_to_static(tensor, static_var)

            tensor = paddle.to_tensor(self.array)
            static_param = tensor._to_static_var(to_parameter=True)
            self._assert_to_static(tensor, static_param, True)

            # Convert EagerParamBase into Parameter
            fc = paddle.nn.Linear(
                10,
                20,
                weight_attr=paddle.ParamAttr(
                    learning_rate=0.001,
                    do_model_average=True,
                    regularizer=paddle.regularizer.L1Decay(),
                ),
            )
            weight = fc.parameters()[0]
            static_param = weight._to_static_var()
            self._assert_to_static(weight, static_param, True)

    def _assert_to_static(self, tensor, static_var, is_param=False):
        if is_param:
            self.assertTrue(isinstance(static_var, base.framework.Parameter))
            self.assertTrue(static_var.persistable, True)
            if isinstance(tensor, base.framework.EagerParamBase):
                for attr in ["trainable", "is_distributed", "do_model_average"]:
                    self.assertEqual(
                        getattr(tensor, attr), getattr(static_var, attr)
                    )

                self.assertEqual(
                    static_var.optimize_attr["learning_rate"], 0.001
                )
                self.assertTrue(
                    isinstance(
                        static_var.regularizer, paddle.regularizer.L1Decay
                    )
                )
        else:
            self.assertTrue(isinstance(static_var, base.framework.Variable))

        attr_keys = ["block", "dtype", "type", "name"]
        for attr in attr_keys:
            if isinstance(getattr(tensor, attr), core.DataType):
                self.assertEqual(
                    paddle_type_to_proto_type[getattr(tensor, attr)],
                    getattr(static_var, attr),
                )
            else:
                self.assertEqual(
                    getattr(tensor, attr),
                    getattr(static_var, attr),
                )

        self.assertListEqual(list(tensor.shape), list(static_var.shape))


if __name__ == "__main__":
    unittest.main()
