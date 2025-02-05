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

import copy
import unittest

import numpy as np

import paddle
from paddle.base.dygraph import guard
from paddle.base.executor import Executor
from paddle.base.framework import Variable, default_main_program

paddle.enable_static()
main_program = default_main_program()


class ParameterChecks(unittest.TestCase):
    def test_parameter(self):
        with paddle.pir_utils.OldIrGuard():
            shape = [784, 100]
            val = 1.0625
            b = main_program.global_block()
            param = b.create_parameter(
                name='fc.w',
                shape=shape,
                dtype='float32',
                initializer=paddle.nn.initializer.Constant(val),
            )
            self.assertIsNotNone(param)
            self.assertEqual('fc.w', param.name)
            self.assertEqual((784, 100), param.shape)
            self.assertEqual(paddle.float32, param.dtype)
            self.assertEqual(0, param.block.idx)
            exe = Executor(paddle.CPUPlace())
            p = exe.run(main_program, fetch_list=[param])[0]
            np.testing.assert_array_equal(p, np.ones(shape) * val)

            zero_dim_param = b.create_parameter(
                name='x', shape=[], dtype='float32'
            )
            self.assertEqual(zero_dim_param.shape, ())

    def test_parambase(self):
        with guard():
            linear = paddle.nn.Linear(10, 10)
            param = linear.weight

            memo = {}
            param_copy = copy.deepcopy(param, memo)
            self.assertEqual(param_copy.shape, param.shape)
            self.assertEqual(param_copy.type, param.type)
            self.assertEqual(param_copy.dtype, param.dtype)
            self.assertEqual(str(param_copy.place), str(param.place))
            np.testing.assert_array_equal(param_copy.numpy(), param.numpy())
            self.assertEqual(param_copy.optimize_attr, param.optimize_attr)
            self.assertEqual(param_copy.regularizer, param.regularizer)
            self.assertEqual(
                param_copy.do_model_average, param.do_model_average
            )
            self.assertEqual(param_copy.need_clip, param.need_clip)
            self.assertEqual(param_copy.is_distributed, param.is_distributed)

            pram_copy2 = copy.deepcopy(param, memo)
            self.assertEqual(id(param_copy), id(pram_copy2))

    def test_create_0_size_param(self):
        with guard():
            shape = [0, 4]
            for dtype in [
                paddle.float32,
                paddle.float64,
            ]:
                zero_size_param = paddle.create_parameter(
                    shape,
                    dtype,
                )
                self.assertEqual(zero_size_param.shape, shape)
                self.assertEqual(zero_size_param.data_ptr(), 0)
                # strides will be same with shape for 0-size tensor in paddle
                self.assertEqual(zero_size_param.strides, shape)

    def func_exception(self):
        b = main_program.global_block()
        with self.assertRaises(ValueError):
            b.create_parameter(
                name='test', shape=None, dtype='float32', initializer=None
            )
        with self.assertRaises(ValueError):
            b.create_parameter(
                name='test', shape=[1], dtype=None, initializer=None
            )
        with self.assertRaises(ValueError):
            b.create_parameter(
                name='test', shape=[], dtype='float32', initializer=None
            )
        with self.assertRaises(ValueError):
            b.create_parameter(
                name='test', shape=[-1], dtype='float32', initializer=None
            )

    def test_parambase_to_vector(self):
        with guard():
            initializer = paddle.ParamAttr(
                initializer=paddle.nn.initializer.Constant(3.0)
            )
            linear1 = paddle.nn.Linear(10, 15, initializer)

            vec = paddle.nn.utils.parameters_to_vector(linear1.parameters())
            self.assertEqual(linear1.weight.shape, [10, 15])
            self.assertEqual(linear1.bias.shape, [15])
            self.assertTrue(isinstance(vec, Variable))
            self.assertTrue(vec.shape, [165])

            linear2 = paddle.nn.Linear(10, 15)
            paddle.nn.utils.vector_to_parameters(vec, linear2.parameters())
            self.assertEqual(linear2.weight.shape, [10, 15])
            self.assertEqual(linear2.bias.shape, [15])
            np.testing.assert_array_equal(
                linear1.weight.numpy(), linear2.weight.numpy()
            )
            np.testing.assert_array_equal(
                linear1.bias.numpy(), linear2.bias.numpy()
            )
            self.assertTrue(linear2.weight.is_leaf, True)
            self.assertTrue(linear2.bias.is_leaf, True)


if __name__ == '__main__':
    unittest.main()
