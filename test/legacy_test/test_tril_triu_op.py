#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at #
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import unittest

import numpy as np
from op_test import OpTest, convert_float_to_uint16

import paddle
from paddle import base, tensor
from paddle.base import core


class TrilTriuOpDefaultTest(OpTest):
    """the base class of other op testcases"""

    def setUp(self):
        self.initTestCase()
        self.python_api = (
            paddle.tril if self.real_op_type == 'tril' else paddle.triu
        )
        self.real_np_op = getattr(np, self.real_op_type)

        self.op_type = "tril_triu"
        self.inputs = {'X': self.X}
        self.attrs = {
            'diagonal': self.diagonal,
            'lower': True if self.real_op_type == 'tril' else False,
        }
        self.outputs = {
            'Out': (
                self.real_np_op(self.X, self.diagonal)
                if self.diagonal
                else self.real_np_op(self.X)
            )
        }

    def test_check_output(self):
        self.check_output(check_pir=True, check_symbol_infer=False)

    def test_check_grad_normal(self):
        self.check_grad(['X'], 'Out', check_pir=True)

    def init_dtype(self):
        self.dtype = np.float64

    def initTestCase(self):
        self.init_dtype()
        self.real_op_type = np.random.choice(['triu', 'tril'])
        self.diagonal = None
        self.X = np.arange(1, 101, dtype=self.dtype).reshape([10, -1])
        if self.dtype == np.complex64 or self.dtype == np.complex128:
            self.X = (
                np.random.uniform(-1, 1, [10, 10])
                + 1j * np.random.uniform(-1, 1, [10, 10])
            ).astype(self.dtype)


class TrilTriuOpDefaultTestFP16(TrilTriuOpDefaultTest):
    def init_dtype(self):
        self.dtype = np.float16


class TrilTriuOpDefaultTestComplex_64(TrilTriuOpDefaultTest):
    def init_dtype(self):
        self.dtype = np.complex64


class TrilTriuOpDefaultTestComplex_128(TrilTriuOpDefaultTest):
    def init_dtype(self):
        self.dtype = np.complex128


@unittest.skipIf(
    not core.is_compiled_with_cuda()
    or not core.is_bfloat16_supported(core.CUDAPlace(0)),
    'not supported bf16',
)
class TrilTriuOpDefaultTestBF16(TrilTriuOpDefaultTest):
    def init_dtype(self):
        self.dtype = np.uint16

    def setUp(self):
        super().setUp()
        self.outputs["Out"] = convert_float_to_uint16(self.outputs["Out"])
        self.inputs['X'] = convert_float_to_uint16(self.inputs['X'])

    def initTestCase(self):
        self.init_dtype()
        self.real_op_type = np.random.choice(['triu', 'tril'])
        self.diagonal = None
        self.X = np.arange(1, 101, dtype="float32").reshape([10, -1])

    def test_check_output(self):
        self.check_output_with_place(core.CUDAPlace(0), check_pir=True)

    def test_check_grad_normal(self):
        self.check_grad_with_place(
            core.CUDAPlace(0),
            ['X'],
            'Out',
            numeric_grad_delta=0.05,
            check_pir=True,
        )


def case_generator(op_type, Xshape, diagonal, expected, dtype):
    """
    Generate testcases with the params shape of X, diagonal and op_type.
    If arg `expected` is 'success', it will register an OpTest case and expect to pass.
    Otherwise, it will register an API case and check the expect failure.
    """
    cls_name = (
        f"{expected}_{op_type}_shape_{Xshape}_diag_{diagonal}_dtype_{dtype}"
    )
    errmsg = {
        "diagonal: TypeError": f"diagonal in {op_type} must be a python Int",
        "input: ValueError": f"x shape in {op_type} must be at least 2-D",
    }

    class FailureCase(unittest.TestCase):
        def test_failure(self):
            paddle.enable_static()

            data = paddle.static.data(
                shape=Xshape, dtype='float64', name=cls_name
            )
            with self.assertRaisesRegex(
                eval(expected.split(':')[-1]), errmsg[expected]
            ):
                getattr(tensor, op_type)(x=data, diagonal=diagonal)

    class SuccessCase(TrilTriuOpDefaultTest):
        def initTestCase(self):
            paddle.enable_static()

            self.real_op_type = op_type
            self.diagonal = diagonal
            self.X = np.random.random(Xshape).astype("float64")

    class SuccessCaseFP16(TrilTriuOpDefaultTestFP16):
        def initTestCase(self):
            self.init_dtype()
            self.real_op_type = op_type
            self.diagonal = diagonal
            self.X = np.random.random(Xshape).astype("float16")

    class SuccessCaseBF16(TrilTriuOpDefaultTestBF16):
        def initTestCase(self):
            self.init_dtype()
            self.real_op_type = op_type
            self.diagonal = diagonal
            self.X = np.random.random(Xshape).astype("float32")

    class SuccessCaseComplex64(TrilTriuOpDefaultTestComplex_64):
        def initTestCase(self):
            self.init_dtype()
            self.real_op_type = op_type
            self.diagonal = diagonal
            self.X = (
                np.random.random(Xshape) + 1j * np.random.random(Xshape)
            ).astype("complex64")

    class SuccessCaseComplex128(TrilTriuOpDefaultTestComplex_128):
        def initTestCase(self):
            self.init_dtype()
            self.real_op_type = op_type
            self.diagonal = diagonal
            self.X = (
                np.random.random(Xshape) + 1j * np.random.random(Xshape)
            ).astype("complex128")

    if dtype == "float64":
        CLASS = locals()[
            'SuccessCase' if expected == "success" else 'FailureCase'
        ]
    elif dtype == "float16":
        CLASS = locals()[
            'SuccessCaseFP16' if expected == "success" else 'FailureCase'
        ]
    elif dtype == "bfloat16":
        CLASS = locals()[
            'SuccessCaseBF16' if expected == "success" else 'FailureCase'
        ]
    elif dtype == "complex64":
        CLASS = locals()[
            'SuccessCaseComplex64' if expected == "success" else 'FailureCase'
        ]
    elif dtype == "complex128":
        CLASS = locals()[
            'SuccessCaseComplex128' if expected == "success" else 'FailureCase'
        ]
    else:
        raise ValueError(f"Not supported dtype {dtype}")
    CLASS.__name__ = cls_name
    globals()[cls_name] = CLASS


# NOTE: meaningful diagonal is [1 - min(H, W), max(H, W) -1]
# test the diagonal just at the border, upper/lower the border,
#     negative/positive integer within range and a zero
cases = {
    'success': {
        (2, 2, 3, 4, 5): [-100, -3, -1, 0, 2, 4, 100],  # normal shape
        (10, 10, 1, 1): [-100, -1, 0, 1, 100],  # small size of matrix
    },
    'diagonal: TypeError': {
        (20, 20): [
            '2020',
            [20],
            {20: 20},
            (20, 20),
            20.20,
        ],  # str, list, dict, tuple, float
    },
    'input: ValueError': {
        (2020,): [None],
    },
}
for dtype in ["float64", "float16", "bfloat16", "complex64", "complex128"]:
    for _op_type in ['tril', 'triu']:
        for _expected, _params in cases.items():
            for _Xshape, _diaglist in _params.items():
                [
                    case_generator(
                        _op_type, _Xshape, _diagonal, _expected, dtype
                    )
                    for _diagonal in _diaglist
                ]


class TestTrilTriuOpAPI(unittest.TestCase):
    """test case by using API and has -1 dimension"""

    def test_api(self):
        paddle.enable_static()

        dtypes = ['float16', 'float32', 'complex64', 'complex128']
        for dtype in dtypes:
            prog = paddle.static.Program()
            startup_prog = paddle.static.Program()
            with paddle.static.program_guard(prog, startup_prog):
                data = np.random.random([1, 9, 9, 4]).astype(dtype)
                x = paddle.static.data(
                    shape=[1, 9, -1, 4], dtype=dtype, name='x'
                )
                if dtype == 'complex64' or dtype == 'complex128':
                    data = (
                        np.random.uniform(-1, 1, [1, 9, 9, 4])
                        + 1j * np.random.uniform(-1, 1, [1, 9, 9, 4])
                    ).astype(dtype)
                tril_out, triu_out = tensor.tril(x), tensor.triu(x)

                place = (
                    base.CUDAPlace(0)
                    if base.core.is_compiled_with_cuda()
                    else base.CPUPlace()
                )
                exe = base.Executor(place)
                tril_out, triu_out = exe.run(
                    prog,
                    feed={"x": data},
                    fetch_list=[tril_out, triu_out],
                )
                np.testing.assert_allclose(tril_out, np.tril(data), rtol=1e-05)
                np.testing.assert_allclose(triu_out, np.triu(data), rtol=1e-05)

    def test_api_with_dygraph(self):
        paddle.disable_static()

        dtypes = ['float16', 'float32', 'complex64', 'complex128']
        for dtype in dtypes:
            with base.dygraph.guard():
                data = np.random.random([1, 9, 9, 4]).astype(dtype)
                if dtype == 'complex64' or dtype == 'complex128':
                    data = (
                        np.random.uniform(-1, 1, [1, 9, 9, 4])
                        + 1j * np.random.uniform(-1, 1, [1, 9, 9, 4])
                    ).astype(dtype)
                x = paddle.to_tensor(data)
                tril_out, triu_out = (
                    tensor.tril(x).numpy(),
                    tensor.triu(x).numpy(),
                )
                np.testing.assert_allclose(tril_out, np.tril(data), rtol=1e-05)
                np.testing.assert_allclose(triu_out, np.triu(data), rtol=1e-05)

    def test_base_api(self):
        paddle.enable_static()

        dtypes = ['float16', 'float32', 'complex64', 'complex128']
        for dtype in dtypes:
            prog = paddle.static.Program()
            startup_prog = paddle.static.Program()
            with paddle.static.program_guard(prog, startup_prog):
                data = np.random.random([1, 9, 9, 4]).astype(dtype)
                x = paddle.static.data(
                    shape=[1, 9, -1, 4], dtype=dtype, name='x'
                )
                if dtype == 'complex64' or dtype == 'complex128':
                    data = (
                        np.random.uniform(-1, 1, [1, 9, 9, 4])
                        + 1j * np.random.uniform(-1, 1, [1, 9, 9, 4])
                    ).astype(dtype)
                triu_out = paddle.triu(x)

                place = (
                    base.CUDAPlace(0)
                    if base.core.is_compiled_with_cuda()
                    else base.CPUPlace()
                )
                exe = base.Executor(place)
                triu_out = exe.run(
                    prog,
                    feed={"x": data},
                    fetch_list=[triu_out],
                )


if __name__ == '__main__':
    unittest.main()
