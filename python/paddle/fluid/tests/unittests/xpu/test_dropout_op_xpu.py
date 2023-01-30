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

<<<<<<< HEAD
=======
from __future__ import print_function
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
import sys

sys.path.append("..")
import unittest
<<<<<<< HEAD

import numpy as np
from op_test_xpu import XPUOpTest

import paddle
import paddle.fluid as fluid
from paddle import _legacy_C_ops
from paddle.fluid import Program, program_guard

paddle.enable_static()

from xpu.get_test_cover_info import (
    XPUOpTestWrapper,
    create_test_class,
    get_xpu_op_support_types,
)


class XPUTestDropoutOp(XPUOpTestWrapper):
=======
import numpy as np
import paddle.fluid.core as core
from op_test import OpTest, skip_check_grad_ci
import paddle
import paddle.fluid as fluid
from paddle.fluid import Program, program_guard
from op_test_xpu import XPUOpTest

paddle.enable_static()

from xpu.get_test_cover_info import create_test_class, get_xpu_op_support_types, XPUOpTestWrapper


class XPUTestDropoutOp(XPUOpTestWrapper):

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def __init__(self):
        self.op_name = 'dropout'
        self.use_dynamic_create_class = False

    class TestDropoutOp(XPUOpTest):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
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
<<<<<<< HEAD
                'dropout_implementation': self.dropout_implementation,
            }

            out = self.inputs['X'] * (1.0 - self.dropout_prob)
            if not self.is_test:
=======
                'dropout_implementation': self.dropout_implementation
            }

            out = self.inputs['X'] * (1.0 - self.dropout_prob)
            if self.is_test == False:
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
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
<<<<<<< HEAD
            if (
                hasattr(self.__class__, "no_need_check_grad")
                and self.__class__.no_need_check_grad
            ):
=======
            if hasattr(self.__class__, "no_need_check_grad"
                       ) and self.__class__.no_need_check_grad == True:
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
                return

            self.check_grad(['X'], 'Out')

    class TestDropoutOpInput1d(TestDropoutOp):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        def init_inputs_shape(self):
            self.shape = [2000]

    class TestDropoutOp2(TestDropoutOp):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        def init_inputs_shape(self):
            self.shape = [32, 64]

        def init_attrs(self):
            self.dropout_prob = 1.0
            self.fix_seed = True
            self.is_test = False
            self.dropout_implementation = "upscale_in_train"

    class TestDropoutOp3(TestDropoutOp):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        def init_inputs_shape(self):
            self.shape = [32, 64, 2]

    class TestDropoutOp4(TestDropoutOp):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        def init_attrs(self):
            self.__class__.no_need_check_grad = True
            self.dropout_prob = 0.35
            self.fix_seed = True
            self.is_test = True
            self.dropout_implementation = "downgrade_in_infer"

    class TestDropoutOp5(TestDropoutOp):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        def init_inputs_shape(self):
            self.shape = [32, 64, 3]

        def init_attrs(self):
            self.__class__.no_need_check_grad = True
            self.dropout_prob = 0.75
            self.fix_seed = True
            self.is_test = True
            self.dropout_implementation = "downgrade_in_infer"

    class TestDropoutOpError(unittest.TestCase):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        def test_errors(self):
            with program_guard(Program(), Program()):

                def test_Variable():
                    # the input of dropout must be Variable.
<<<<<<< HEAD
                    x1 = fluid.create_lod_tensor(
                        np.array([-1, 3, 5, 5]),
                        [[1, 1, 1, 1]],
                        fluid.CPUPlace(),
                    )
                    paddle.nn.functional.dropout(x1, p=0.5)
=======
                    x1 = fluid.create_lod_tensor(np.array([-1, 3, 5,
                                                           5]), [[1, 1, 1, 1]],
                                                 fluid.CPUPlace())
                    fluid.layers.dropout(x1, dropout_prob=0.5)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

                self.assertRaises(TypeError, test_Variable)

                def test_dtype():
                    # the input dtype of dropout must be float16 or float32 or float64
                    # float16 only can be set on GPU place
<<<<<<< HEAD
                    x2 = paddle.static.data(
                        name='x2', shape=[-1, 3, 4, 5, 6], dtype="int32"
                    )
                    paddle.nn.functional.dropout(x2, p=0.5)
=======
                    x2 = fluid.layers.data(name='x2',
                                           shape=[3, 4, 5, 6],
                                           dtype="int32")
                    fluid.layers.dropout(x2, dropout_prob=0.5)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

                self.assertRaises(TypeError, test_dtype)

    class TestDropoutCAPI(unittest.TestCase):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
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
<<<<<<< HEAD
                    m = paddle.nn.Dropout(p=0.0)
=======
                    m = paddle.nn.Dropout(p=0.)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
                    m.eval()
                    result = m(input)
                    np.testing.assert_allclose(result.numpy(), result_np)

    class TestDropoutBackward(unittest.TestCase):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
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
<<<<<<< HEAD
                    out, mask = _legacy_C_ops.dropout(
                        input, 'dropout_prob', 0.5
                    )
=======
                    out, mask = core.ops.dropout(input, 'dropout_prob', 0.5)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
                    out.backward()

                    np.testing.assert_allclose(
                        input.gradient(),
<<<<<<< HEAD
                        self.cal_grad_downscale_in_infer(mask.numpy()),
                    )
=======
                        self.cal_grad_downscale_in_infer(mask.numpy()))
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

        def test_backward_upscale_train(self):
            for place in self.places:
                with fluid.dygraph.guard(place):

                    prob = 0.5
                    input = paddle.uniform([40, 40], dtype=self.in_type)
                    input.stop_gradient = False
<<<<<<< HEAD
                    out, mask = _legacy_C_ops.dropout(
                        input,
                        'dropout_prob',
                        prob,
                        "dropout_implementation",
                        "upscale_in_train",
                    )
=======
                    out, mask = core.ops.dropout(input, 'dropout_prob', prob,
                                                 "dropout_implementation",
                                                 "upscale_in_train")
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
                    out.backward()

                    np.testing.assert_allclose(
                        input.gradient(),
<<<<<<< HEAD
                        self.cal_grad_upscale_train(mask.numpy(), prob),
                    )
=======
                        self.cal_grad_upscale_train(mask.numpy(), prob))
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

        def test_backward_upscale_train_2(self):
            for place in self.places:
                with fluid.dygraph.guard(place):

                    prob = 0.3
                    input = paddle.uniform([40, 40], dtype=self.in_type)
                    input.stop_gradient = False
<<<<<<< HEAD
                    out, mask = _legacy_C_ops.dropout(
                        input,
                        'dropout_prob',
                        prob,
                        "dropout_implementation",
                        "upscale_in_train",
                    )
=======
                    out, mask = core.ops.dropout(input, 'dropout_prob', prob,
                                                 "dropout_implementation",
                                                 "upscale_in_train")
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
                    out.backward()

                    np.testing.assert_allclose(
                        input.gradient(),
<<<<<<< HEAD
                        self.cal_grad_upscale_train(mask.numpy(), prob),
                    )
=======
                        self.cal_grad_upscale_train(mask.numpy(), prob))
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81


support_types = get_xpu_op_support_types('dropout')
for stype in support_types:
    create_test_class(globals(), XPUTestDropoutOp, stype)

if __name__ == '__main__':
    unittest.main()
