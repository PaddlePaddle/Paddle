# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
import unittest

import numpy as np

sys.path.append("..")

from op_test_xpu import XPUOpTest
from xpu.get_test_cover_info import (
    XPUOpTestWrapper,
    create_test_class,
    get_xpu_op_support_types,
)

import paddle
import paddle.fluid as fluid

paddle.enable_static()


class XPUTestInverseOp(XPUOpTestWrapper):
    def __init__(self):
        self.op_name = 'inverse'
        self.use_dynamic_create_class = False

    class TestInverseOp(XPUOpTest):
        def setUp(self):
            self.op_type = "inverse"
            self.dtype = self.in_type
            self.place = paddle.XPUPlace(0)
            self.python_api = paddle.tensor.math.inverse
            self.init_test_case()

            np.random.seed(123)
            mat = np.random.random(self.matrix_shape).astype(self.dtype)
            inverse = np.linalg.inv(mat)

            self.inputs = {'Input': mat}
            self.outputs = {'Output': inverse}

        def init_test_case(self):
            self.matrix_shape = [10, 10]

        def test_check_output(self):
            self.check_output(atol=0.01, check_eager=True)

        # def test_grad(self):
        #     self.check_grad(['Input'], 'Output', check_eager=True)

    class TestInverseOpBatched(TestInverseOp):
        def init_test_case(self):
            self.matrix_shape = [8, 4, 4]

    class TestInverseOpLarge(TestInverseOp):
        def init_test_case(self):
            self.matrix_shape = [32, 32]

    class TestInverseAPI(unittest.TestCase):
        def setUp(self):
            np.random.seed(123)
            self.places = [fluid.XPUPlace(0)]
            self.dtype = self.in_type

        def check_static_result(self, place):
            with fluid.program_guard(fluid.Program(), fluid.Program()):
                input = fluid.data(name="input", shape=[4, 4], dtype=self.dtype)
                result = paddle.inverse(x=input)
                input_np = np.random.random([4, 4]).astype(self.dtype)
                result_np = np.linalg.inv(input_np)

                exe = fluid.Executor(place)
                fetches = exe.run(
                    fluid.default_main_program(),
                    feed={"input": input_np},
                    fetch_list=[result],
                )
                np.testing.assert_allclose(
                    fetches[0], np.linalg.inv(input_np), rtol=1e-05
                )

        def test_static(self):
            for place in self.places:
                self.check_static_result(place=place)

        def test_dygraph(self):
            for place in self.places:
                with fluid.dygraph.guard(place):
                    input_np = np.random.random([4, 4]).astype(self.dtype)
                    input = fluid.dygraph.to_variable(input_np)
                    result = paddle.inverse(input)
                    np.testing.assert_allclose(
                        result.numpy(), np.linalg.inv(input_np), rtol=1e-05
                    )

    class TestInverseAPIError(unittest.TestCase):
        def test_errors(self):
            input_np = np.random.random([4, 4]).astype("float64")

            # input must be Variable.
            self.assertRaises(TypeError, paddle.inverse, input_np)

            # The data type of input must be float32 or float64.
            for dtype in ["bool", "int32", "int64", "float16"]:
                input = fluid.data(
                    name='input_' + dtype, shape=[4, 4], dtype=dtype
                )
                self.assertRaises(TypeError, paddle.inverse, input)

            # When out is set, the data type must be the same as input.
            input = fluid.data(name='input_1', shape=[4, 4], dtype="float32")
            out = fluid.data(name='output', shape=[4, 4], dtype="float64")
            self.assertRaises(TypeError, paddle.inverse, input, out)

            # The number of dimensions of input must be >= 2.
            input = fluid.data(name='input_2', shape=[4], dtype="float32")
            self.assertRaises(ValueError, paddle.inverse, input)

    class TestInverseSingularAPI(unittest.TestCase):
        def setUp(self):
            self.places = [fluid.XPUPlace(0)]
            self.dtype = self.in_type

        def check_static_result(self, place):
            with fluid.program_guard(fluid.Program(), fluid.Program()):
                input = fluid.data(name="input", shape=[4, 4], dtype=self.dtype)
                result = paddle.inverse(x=input)

                input_np = np.zeros([4, 4]).astype(self.dtype)

                exe = fluid.Executor(place)
                try:
                    fetches = exe.run(
                        fluid.default_main_program(),
                        feed={"input": input_np},
                        fetch_list=[result],
                    )
                except RuntimeError as ex:
                    print("The mat is singular")
                except ValueError as ex:
                    print("The mat is singular")

        def test_static(self):
            for place in self.places:
                self.check_static_result(place=place)

        def test_dygraph(self):
            for place in self.places:
                with fluid.dygraph.guard(place):
                    input_np = np.ones([4, 4]).astype(self.dtype)
                    input = fluid.dygraph.to_variable(input_np)
                    try:
                        result = paddle.inverse(input)
                    except RuntimeError as ex:
                        print("The mat is singular")
                    except ValueError as ex:
                        print("The mat is singular")
                    except OSError as ex:
                        print("The mat is singular")


support_types = get_xpu_op_support_types('inverse')
for stype in support_types:
    create_test_class(globals(), XPUTestInverseOp, stype)


if __name__ == "__main__":
    paddle.enable_static()
    unittest.main()
