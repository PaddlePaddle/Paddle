#   Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
import unittest
import sys

sys.path.append('..')
import numpy as np
from op_test import OpTest
import paddle
import paddle.fluid as fluid
import paddle.tensor as tensor
from paddle.fluid.framework import Program, program_guard

paddle.enable_static()


class TrilTriuOpDefaultTest(OpTest):
<<<<<<< HEAD
    """the base class of other op testcases"""
=======
    """ the base class of other op testcases
    """
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

    def setUp(self):
        self.initTestCase()
        self.__class__.use_mlu = True
        self.place = paddle.device.MLUPlace(0)
<<<<<<< HEAD
        self.python_api = (
            paddle.tril if self.real_op_type == 'tril' else paddle.triu
        )
=======
        self.python_api = paddle.tril if self.real_op_type == 'tril' else paddle.triu
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        self.real_np_op = getattr(np, self.real_op_type)

        self.op_type = "tril_triu"
        self.inputs = {'X': self.X}
        self.attrs = {
            'diagonal': self.diagonal,
            'lower': True if self.real_op_type == 'tril' else False,
        }
        self.outputs = {
<<<<<<< HEAD
            'Out': self.real_np_op(self.X, self.diagonal)
            if self.diagonal
            else self.real_np_op(self.X)
=======
            'Out':
            self.real_np_op(self.X, self.diagonal)
            if self.diagonal else self.real_np_op(self.X)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        }

    def test_check_output(self):
        self.check_output_with_place(self.place)

    def initTestCase(self):
        self.real_op_type = np.random.choice(['triu', 'tril'])
        self.diagonal = None
        self.X = np.arange(1, 101, dtype="float32").reshape([10, -1])


def case_generator(op_type, Xshape, diagonal, expected):
    """
    Generate testcases with the params shape of X, diagonal and op_type.
    If arg`expercted` is 'success', it will register an Optest case and expect to pass.
    Otherwise, it will register an API case and check the expect failure.
    """
<<<<<<< HEAD
    cls_name = "{0}_{1}_shape_{2}_diag_{3}".format(
        expected, op_type, Xshape, diagonal
    )
    errmsg = {
        "diagonal: TypeError": "diagonal in {} must be a python Int".format(
            op_type
        ),
        "input: ValueError": "x shape in {} must be at least 2-D".format(
            op_type
        ),
    }

    class FailureCase(unittest.TestCase):
=======
    cls_name = "{0}_{1}_shape_{2}_diag_{3}".format(expected, op_type, Xshape,
                                                   diagonal)
    errmsg = {
        "diagonal: TypeError":
        "diagonal in {} must be a python Int".format(op_type),
        "input: ValueError":
        "x shape in {} must be at least 2-D".format(op_type),
    }

    class FailureCase(unittest.TestCase):

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        def test_failure(self):
            paddle.enable_static()

            data = fluid.data(shape=Xshape, dtype='float64', name=cls_name)
<<<<<<< HEAD
            with self.assertRaisesRegex(
                eval(expected.split(':')[-1]), errmsg[expected]
            ):
                getattr(tensor, op_type)(x=data, diagonal=diagonal)

    class SuccessCase(TrilTriuOpDefaultTest):
=======
            with self.assertRaisesRegexp(eval(expected.split(':')[-1]),
                                         errmsg[expected]):
                getattr(tensor, op_type)(x=data, diagonal=diagonal)

    class SuccessCase(TrilTriuOpDefaultTest):

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        def initTestCase(self):
            paddle.enable_static()

            self.real_op_type = op_type
            self.diagonal = diagonal
            self.X = np.random.random(Xshape).astype("float32")

    CLASS = locals()['SuccessCase' if expected == "success" else 'FailureCase']
    CLASS.__name__ = cls_name
    globals()[cls_name] = CLASS


## NOTE: meaningful diagonal is [1 - min(H, W), max(H, W) -1]
## test the diagonal just at the border, upper/lower the border,
##     negative/positive integer within range and a zero
cases = {
    'success': {
        (2, 2, 3, 4, 5): [-100, -3, -1, 0, 2, 4, 100],  # normal shape
        (10, 10, 1, 1): [-100, -1, 0, 1, 100],  # small size of matrix
    },
    'diagonal: TypeError': {
        (20, 20): [
            '2020',
            [20],
<<<<<<< HEAD
            {20: 20},
=======
            {
                20: 20
            },
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
            (20, 20),
            20.20,
        ],  # str, list, dict, tuple, float
    },
    'input: ValueError': {
<<<<<<< HEAD
        (2020,): [None],
=======
        (2020, ): [None],
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    },
}
for _op_type in ['tril', 'triu']:
    for _expected, _params in cases.items():
        for _Xshape, _diaglist in _params.items():
            list(
                map(
                    lambda _diagonal: case_generator(
<<<<<<< HEAD
                        _op_type, _Xshape, _diagonal, _expected
                    ),
                    _diaglist,
                )
            )


class TestTrilTriuOpAPI(unittest.TestCase):
    """test case by using API and has -1 dimension"""
=======
                        _op_type, _Xshape, _diagonal, _expected), _diaglist))


class TestTrilTriuOpAPI(unittest.TestCase):
    """ test case by using API and has -1 dimension 
    """
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

    def test_api(self):
        paddle.enable_static()

        dtypes = ['float16', 'float32', 'int32']
        for dtype in dtypes:
            prog = Program()
            startup_prog = Program()
            with program_guard(prog, startup_prog):
                data = np.random.random([1, 9, 9, 4]).astype(dtype)
                x = fluid.data(shape=[1, 9, -1, 4], dtype=dtype, name='x')
                tril_out, triu_out = tensor.tril(x), tensor.triu(x)

                place = fluid.MLUPlace(0)
                exe = fluid.Executor(place)
                tril_out, triu_out = exe.run(
                    fluid.default_main_program(),
                    feed={"x": data},
                    fetch_list=[tril_out, triu_out],
                )
                np.testing.assert_allclose(tril_out, np.tril(data))
                np.testing.assert_allclose(triu_out, np.triu(data))

    def test_api_with_dygraph(self):
        paddle.disable_static()

        dtypes = ['float16', 'float32', 'int32']
        for dtype in dtypes:
            with fluid.dygraph.guard():
                data = np.random.random([1, 9, 9, 4]).astype(dtype)
                x = fluid.dygraph.to_variable(data)
<<<<<<< HEAD
                tril_out, triu_out = (
                    tensor.tril(x).numpy(),
                    tensor.triu(x).numpy(),
                )
=======
                tril_out, triu_out = tensor.tril(x).numpy(), tensor.triu(
                    x).numpy()
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
                np.testing.assert_allclose(tril_out, np.tril(data))
                np.testing.assert_allclose(triu_out, np.triu(data))

    def test_fluid_api(self):
        paddle.enable_static()

        dtypes = ['float16', 'float32', 'int32']
        for dtype in dtypes:
            prog = Program()
            startup_prog = Program()
            with program_guard(prog, startup_prog):
                data = np.random.random([1, 9, 9, 4]).astype(dtype)
                x = fluid.data(shape=[1, 9, -1, 4], dtype=dtype, name='x')
<<<<<<< HEAD
                triu_out = paddle.triu(x)

                place = fluid.MLUPlace(0)
                exe = fluid.Executor(place)
                triu_out = exe.run(
                    fluid.default_main_program(),
                    feed={"x": data},
                    fetch_list=[triu_out],
                )
=======
                triu_out = fluid.layers.triu(x)

                place = fluid.MLUPlace(0)
                exe = fluid.Executor(place)
                triu_out = exe.run(fluid.default_main_program(),
                                   feed={"x": data},
                                   fetch_list=[triu_out])
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81


if __name__ == '__main__':
    unittest.main()
