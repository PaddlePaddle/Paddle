#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

import sys

sys.path.append("..")
import unittest
import numpy as np
import paddle.fluid.core as core
from paddle import _legacy_C_ops
from op_test import OpTest, skip_check_grad_ci
import paddle
import paddle.fluid as fluid
from paddle.fluid import Program, program_guard
from op_test_xpu import XPUOpTest

paddle.enable_static()

from xpu.get_test_cover_info import create_test_class, get_xpu_op_support_types, XPUOpTestWrapper


class XPUTestDropoutOp(XPUOpTestWrapper):

    def __init__(self):
        self.op_name = 'dropout'
        self.use_dynamic_create_class = False

    class TestDropoutOp(XPUOpTest):

        def setUp(self):
            self.init_inputs_shape()
            self.init_attrs()
            self.dtype = self.in_type
            self.op_type = 'dropout'
            self.inputs = {'X': np.random.random(self.shape).astype(self.dtype)}
            self.attrs = {
                'dropout_prob': self.dropout_prob,
                'fix_seed': self.fix_seed,
                'is_test': self.is_test,
                'dropout_implementation': self.dropout_implementation
            }

            out = self.inputs['X'] * (1.0 - self.dropout_prob)
            if self.is_test == False:
                mask = None
                if self.dropout_prob == 0.0:
                    mask = np.ones(self.shape).astype(self.dtype)
                elif self.dropout_prob == 1.0:
                    mask = np.zeros(self.shape).astype(self.dtype)
                self.outputs = {'Out': out, 'Mask': mask}
            else:
                self.outputs = {'Out': out}

        def init_inputs_shape(self):
            self.shape = [32, 64]

        def init_attrs(self):
            self.__class__.no_need_check_grad = False
            self.dropout_prob = 0.0
            self.fix_seed = True
            self.is_test = False
            self.dropout_implementation = "upscale_in_train"

        def test_check_output(self):
            self.check_output()

        def test_check_grad_normal(self):
            if hasattr(self.__class__, "no_need_check_grad"
                       ) and self.__class__.no_need_check_grad == True:
                return

            self.check_grad(['X'], 'Out')

    class TestDropoutOpInput1d(TestDropoutOp):

        def init_inputs_shape(self):
            self.shape = [2000]

    class TestDropoutOp2(TestDropoutOp):

        def init_inputs_shape(self):
            self.shape = [32, 64]

        def init_attrs(self):
            self.dropout_prob = 1.0
            self.fix_seed = True
            self.is_test = False
            self.dropout_implementation = "upscale_in_train"

    class TestDropoutOp3(TestDropoutOp):

        def init_inputs_shape(self):
            self.shape = [32, 64, 2]

    class TestDropoutOp4(TestDropoutOp):

        def init_attrs(self):
            self.__class__.no_need_check_grad = True
            self.dropout_prob = 0.35
            self.fix_seed = True
            self.is_test = True
            self.dropout_implementation = "downgrade_in_infer"

    class TestDropoutOp5(TestDropoutOp):

        def init_inputs_shape(self):
            self.shape = [32, 64, 3]

        def init_attrs(self):
            self.__class__.no_need_check_grad = True
            self.dropout_prob = 0.75
            self.fix_seed = True
            self.is_test = True
            self.dropout_implementation = "downgrade_in_infer"

    class TestDropoutOpError(unittest.TestCase):

        def test_errors(self):
            with program_guard(Program(), Program()):

                def test_Variable():
                    # the input of dropout must be Variable.
                    x1 = fluid.create_lod_tensor(np.array([-1, 3, 5,
                                                           5]), [[1, 1, 1, 1]],
                                                 fluid.CPUPlace())
                    fluid.layers.dropout(x1, dropout_prob=0.5)

                self.assertRaises(TypeError, test_Variable)

                def test_dtype():
                    # the input dtype of dropout must be float16 or float32 or float64
                    # float16 only can be set on GPU place
                    x2 = fluid.layers.data(name='x2',
                                           shape=[3, 4, 5, 6],
                                           dtype="int32")
                    fluid.layers.dropout(x2, dropout_prob=0.5)

                self.assertRaises(TypeError, test_dtype)

    class TestDropoutCAPI(unittest.TestCase):

        def setUp(self):
            np.random.seed(123)
            self.places = [fluid.CPUPlace()]
            self.places.append(fluid.XPUPlace(0))

        def test_dygraph(self):
            for place in self.places:
                with fluid.dygraph.guard(place):
                    input_np = np.random.random([40, 40]).astype(self.in_type)
                    result_np = input_np
                    input = fluid.dygraph.to_variable(input_np)
                    m = paddle.nn.Dropout(p=0.)
                    m.eval()
                    result = m(input)
                    np.testing.assert_allclose(result.numpy(), result_np)

    class TestDropoutBackward(unittest.TestCase):

        def setUp(self):
            np.random.seed(123)
            self.places = [fluid.CPUPlace()]
            self.places.append(fluid.XPUPlace(0))

        def cal_grad_upscale_train(self, mask, prob):
            return mask.astype(self.in_type) / (1 - prob)

        def cal_grad_downscale_in_infer(self, mask):
            return mask.astype(self.in_type)

        def test_backward_downscale_in_infer(self):
            for place in self.places:
                with fluid.dygraph.guard(place):

                    input = paddle.uniform([40, 40], dtype=self.in_type)
                    input.stop_gradient = False
                    out, mask = _legacy_C_ops.dropout(input, 'dropout_prob',
                                                      0.5)
                    out.backward()

                    np.testing.assert_allclose(
                        input.gradient(),
                        self.cal_grad_downscale_in_infer(mask.numpy()))

        def test_backward_upscale_train(self):
            for place in self.places:
                with fluid.dygraph.guard(place):

                    prob = 0.5
                    input = paddle.uniform([40, 40], dtype=self.in_type)
                    input.stop_gradient = False
                    out, mask = _legacy_C_ops.dropout(input, 'dropout_prob',
                                                      prob,
                                                      "dropout_implementation",
                                                      "upscale_in_train")
                    out.backward()

                    np.testing.assert_allclose(
                        input.gradient(),
                        self.cal_grad_upscale_train(mask.numpy(), prob))

        def test_backward_upscale_train_2(self):
            for place in self.places:
                with fluid.dygraph.guard(place):

                    prob = 0.3
                    input = paddle.uniform([40, 40], dtype=self.in_type)
                    input.stop_gradient = False
                    out, mask = _legacy_C_ops.dropout(input, 'dropout_prob',
                                                      prob,
                                                      "dropout_implementation",
                                                      "upscale_in_train")
                    out.backward()

                    np.testing.assert_allclose(
                        input.gradient(),
                        self.cal_grad_upscale_train(mask.numpy(), prob))


support_types = get_xpu_op_support_types('dropout')
for stype in support_types:
    create_test_class(globals(), XPUTestDropoutOp, stype)

if __name__ == '__main__':
    unittest.main()
