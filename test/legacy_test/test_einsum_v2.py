#   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

import os
import unittest

import numpy as np

import paddle
from paddle.base import core

os.environ['FLAGS_new_einsum'] = "1"


def error_trans(func, *args, **kargs):
    """
    transport C++ exception into Python exception.
    because einsum_v2 raise different exception with einsum_v1.
    """
    try:
        out = func(*args, **kargs)
    except ValueError as e:
        if "Same label have different shapes" in str(e):
            raise AssertionError(
                "Invalid operands: label i "
                "corresponds to non-broadcastable dimensions."
            )


class TestErrors(unittest.TestCase):
    def setUp(self):
        pass

    def test_param_errors(self):
        a = np.arange(4 * 3 * 4 * 4).reshape(4, 3, 4, 4).astype('float')
        a = paddle.to_tensor(a)
        with self.assertRaisesRegex(
            AssertionError,
            ("Required at least one operand in Einsum API, but received 0"),
        ):
            paddle.einsum('ijk')
        with self.assertRaisesRegex(
            AssertionError, ('Invalid equation: multiple `->` were found.')
        ):
            paddle.einsum('i -> j -> k', a)
        with self.assertRaisesRegex(
            AssertionError,
            (
                "Invalid equation: the number of operands is 2, "
                "but found 3 segments in the label equation."
            ),
        ):
            paddle.einsum('i,j,k', a, a)
        with self.assertRaisesRegex(
            AssertionError,
            (
                "Invalid equation: the number of operands is 2, "
                "but found 1 segments in the label equation."
            ),
        ):
            paddle.einsum('ij -> k', a, a)
        with self.assertRaisesRegex(
            AssertionError,
            (
                "Invalid equation: the number of operands is 1, "
                "but found 2 segments in the label equation."
            ),
        ):
            paddle.einsum('i, -> k', a)
        with self.assertRaisesRegex(
            AssertionError,
            ("Invalid equation: the label string '' misses dimensions."),
        ):
            paddle.einsum('->', a)
        with self.assertRaisesRegex(
            AssertionError,
            ("Invalid equation: the label string 'i' misses dimensions."),
        ):
            paddle.einsum('i', a)
        with self.assertRaisesRegex(
            AssertionError,
            (
                "Invalid equation: _ is not a valid label, "
                "which should be letters."
            ),
        ):
            paddle.einsum('i_', a)
        with self.assertRaisesRegex(
            AssertionError,
            ("Invalid equation: `.` is found outside of an ellipsis."),
        ):
            paddle.einsum('i..j', a)
        with self.assertRaisesRegex(
            AssertionError,
            ("Invalid equation: `.` is found outside of an ellipsis."),
        ):
            paddle.einsum('...k...', a)
        with self.assertRaisesRegex(
            AssertionError,
            ("Invalid equation: missing ellipsis in output labels."),
        ):
            paddle.einsum('i...->i', a)
        with self.assertRaisesRegex(
            AssertionError,
            (
                "Invalid operands: label i "
                "corresponds to non-broadcastable dimensions."
            ),
        ):
            error_trans(paddle.einsum, 'ij...,ji...', a, a)


class TestEinsum(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        np.random.seed(12345)

        cls.TEST_SAMPLES = {
            "a": np.random.rand(1, 1),
            "b": np.random.rand(1),
            "x": np.random.rand(5),
            "y": np.random.rand(7),
            "A": np.random.rand(4, 5),
            "B": np.random.rand(2, 5),
            "C": np.random.rand(3, 7),
            "D": np.random.rand(3, 4, 5),
            "E": np.random.rand(3, 5, 2),
            "F": np.random.rand(2, 4, 5, 3),
            "G": np.random.rand(4, 2, 5),
            "H": np.random.rand(3, 2, 4),
            "I": np.random.rand(2, 2),
            "J": np.random.rand(1, 3, 5),
            "K": np.random.rand(1, 2, 3, 4),
            "X": np.random.rand(5, 5),
            "L": np.random.rand(5, 10, 5),
            "M": np.random.rand(5, 3, 2, 1, 4, 5),
            "N": np.random.rand(5, 5, 5),
            "O": np.random.rand(3, 5, 7, 3),
            "P": np.random.rand(5, 7, 5, 7),
            "S": np.random.rand(4, 3, 4, 4),
        }

    def _get_place(self, force_to_use_cpu=False):
        if force_to_use_cpu:
            return core.CPUPlace()
        else:
            if core.is_compiled_with_cuda():
                return core.CUDAPlace(0)
            return core.CPUPlace()

    def check_output_equal(self, actual, expect, rtol=1.0e-5, atol=1.0e-8):
        error_msg = 'Output has diff at place:{}. \nExpect: {} \nBut Got: {} in class {}'
        np.testing.assert_allclose(
            actual,
            expect,
            rtol=rtol,
            atol=atol,
            err_msg=error_msg.format(
                paddle.get_device(), expect, actual, self.__class__.__name__
            ),
        )

    def setUp(self):
        self.sample = {"paradigm": "i->", "data": ["x"]}

    def test_forward(self):
        operands = [
            TestEinsum.TEST_SAMPLES[operand] for operand in self.sample["data"]
        ]
        expected_result = np.einsum(self.sample["paradigm"], *operands)
        equation = self.sample["paradigm"]

        with paddle.base.dygraph.guard(self._get_place(force_to_use_cpu=False)):
            pd_operands = [paddle.to_tensor(operand) for operand in operands]
            result = paddle.einsum(equation, *pd_operands)
            self.check_output_equal(result.numpy(), expected_result)

        with paddle.base.dygraph.guard(self._get_place(force_to_use_cpu=True)):
            pd_operands = [paddle.to_tensor(operand) for operand in operands]
            result = paddle.einsum(equation, *pd_operands)
            self.check_output_equal(result.numpy(), expected_result)


class TestEinsumTraceDiag1(TestEinsum):
    def setUp(self):
        self.sample = {"paradigm": "ii->", "data": ["X"]}


class TestEinsumTraceDiag2(TestEinsum):
    def setUp(self):
        self.sample = {"paradigm": "iji->j", "data": ["L"]}


class TestEinsumTraceDiag3(TestEinsum):
    def setUp(self):
        self.sample = {"paradigm": "a...a->...", "data": ["M"]}


class TestEinsumTraceDiag4(TestEinsum):
    def setUp(self):
        self.sample = {"paradigm": "a...a->a...", "data": ["M"]}


class TestEinsumTraceDiag5(TestEinsum):
    def setUp(self):
        self.sample = {"paradigm": "aaa->a", "data": ["N"]}


# Numpy don't support i->ii, but paddle.einsum support.
# class TestEinsumTraceDiag6(TestEinsum):
# def setUp(self):
# self.sample = {"paradigm": "i->iii", "data": ["x"]}

# class TestEinsumTraceDiag7(TestEinsum):
# def setUp(self):
# self.sample = {"paradigm": "i...->i...i", "data": ["S"]}


class TestEinsumTraceDiag2Ops(TestEinsum):
    def setUp(self):
        self.sample = {"paradigm": "ijki,jkjk->ik", "data": ["O", "P"]}


class TestEinsumIdentity(TestEinsum):
    def setUp(self):
        self.sample = {"paradigm": "...->...", "data": ["N"]}


class TestEinsumElementwiseProduct(TestEinsum):
    def setUp(self):
        self.sample = {"paradigm": "...,...->...", "data": ["N", "N"]}


class TestEinsumVectorOuter(TestEinsum):
    def setUp(self):
        self.sample = {"paradigm": "i,j->ij", "data": ["x", "y"]}


class TestEinsumMatrixTranspose(TestEinsum):
    def setUp(self):
        self.sample = {"paradigm": "ij->ji", "data": ["A"]}


class TestEinsumMatrixRowSum(TestEinsum):
    def setUp(self):
        self.sample = {"paradigm": "ij->j", "data": ["A"]}


class TestEinsumMatrixColSum(TestEinsum):
    def setUp(self):
        self.sample = {"paradigm": "ij->i", "data": ["A"]}


class TestEinsumMatrixEleMul(TestEinsum):
    def setUp(self):
        self.sample = {"paradigm": "ij,ij->ij", "data": ["A", "A"]}


class TestEinsumDegenerateMatrixVecMul(TestEinsum):
    def setUp(self):
        self.sample = {"paradigm": "ij,j", "data": ["a", "b"]}


class TestEinsumMatrixVecMul(TestEinsum):
    def setUp(self):
        self.sample = {"paradigm": "ij,j->i", "data": ["A", "x"]}


class TestEinsumMatrixMul(TestEinsum):
    def setUp(self):
        self.sample = {"paradigm": "ij,kj->ik", "data": ["A", "B"]}


class TestEinsumMatrixOuter(TestEinsum):
    def setUp(self):
        self.sample = {"paradigm": "ij,kl->ijkl", "data": ["A", "C"]}


class TestEinsumTensorBMM(TestEinsum):
    def setUp(self):
        self.sample = {"paradigm": "bij,bjk->bik", "data": ["D", "E"]}


class TestEinsumTensorContract1(TestEinsum):
    def setUp(self):
        self.sample = {"paradigm": "ijk,jk->i", "data": ["D", "A"]}


class TestEinsumTensorContract2(TestEinsum):
    def setUp(self):
        self.sample = {"paradigm": "ijk,lk->ijl", "data": ["D", "B"]}


class TestEinsumTensorContract3(TestEinsum):
    def setUp(self):
        self.sample = {"paradigm": "abcd,dfg->abcfg", "data": ["F", "D"]}


class TestEinsumTensorContract4(TestEinsum):
    def setUp(self):
        self.sample = {"paradigm": "ijk,jk->ik", "data": ["D", "A"]}


class TestEinsumTensorContract5(TestEinsum):
    def setUp(self):
        self.sample = {"paradigm": "ijk,jk->ij", "data": ["D", "A"]}


class TestEinsumTensorContract6(TestEinsum):
    def setUp(self):
        self.sample = {"paradigm": "ik, ijk->j", "data": ["A", "G"]}


class TestEinsumTensorContract7(TestEinsum):
    def setUp(self):
        self.sample = {"paradigm": "ijk, ik->jk", "data": ["G", "A"]}


class TestEinsumEllipsis1(TestEinsum):
    def setUp(self):
        self.sample = {"paradigm": "i...->...", "data": ["G"]}


class TestEinsumEllipsis2(TestEinsum):
    def setUp(self):
        self.sample = {"paradigm": "ij,...i->j...", "data": ["A", "H"]}


class TestEinsumEllipsis3(TestEinsum):
    def setUp(self):
        self.sample = {"paradigm": "k...,jk", "data": ["F", "I"]}


class TestEinsumTestEinsumBilinear(TestEinsum):
    def setUp(self):
        self.sample = {"paradigm": "bn,anm,bm->ba", "data": ["B", "E", "I"]}


class TestEinsumTestEinsumOthers1(TestEinsum):
    def setUp(self):
        self.sample = {"paradigm": "ijkl, lmn->kmn", "data": ["F", "H"]}


class TestEinsumTestEinsumOthers2(TestEinsum):
    def setUp(self):
        self.sample = {"paradigm": "ijkl, lmn->ijn", "data": ["F", "H"]}


class TestEinsumBatch1(TestEinsum):
    def setUp(self):
        self.sample = {"paradigm": "blq,bhlk->bhlqk", "data": ["J", "K"]}


class TestNumpyTests(unittest.TestCase):
    def setUp(self):
        pass

    def _get_place(self, force_to_use_cpu=False):
        if force_to_use_cpu:
            return core.CPUPlace()
        else:
            if core.is_compiled_with_cuda():
                return core.CUDAPlace(0)
            return core.CPUPlace()

    def check_output_equal(self, actual, expect, rtol=1.0e-5, atol=1.0e-8):
        error_msg = 'Output has diff at place:{}. \nExpect: {} \nBut Got: {} in class {}'
        np.testing.assert_allclose(
            actual,
            expect,
            rtol=rtol,
            atol=atol,
            err_msg=error_msg.format(
                self._get_place(False), expect, actual, self.__class__.__name__
            ),
        )

    def check_output(self, eqn, *ops):
        expect = np.einsum(eqn, *ops)
        with paddle.base.dygraph.guard(self._get_place(force_to_use_cpu=False)):
            pd_operands = [paddle.to_tensor(op) for op in ops]
            actual = paddle.einsum(eqn, *pd_operands)
            self.check_output_equal(actual.numpy(), expect)

    def test_sums(self):
        for n in range(1, 17):
            a = np.arange(n).astype('float')
            self.check_output("i->", a)

        for n in range(1, 17):
            a = np.arange(2 * 3 * n).reshape(2, 3, n).astype('float')
            self.check_output("...i->...", a)

        for n in range(1, 17):
            a = np.arange(2 * n).reshape(2, n).astype('float')
            self.check_output("i...->...", a)

        for n in range(1, 17):
            a = np.arange(2 * 3 * n).reshape(2, 3, n).astype('float')
            self.check_output("i...->...", a)

        for n in range(1, 17):
            a = np.arange(3 * n).reshape(3, n).astype('float')
            b = np.arange(2 * 3 * n).reshape(2, 3, n).astype('float')
            self.check_output("..., ...", a, b)

        for n in range(1, 17):
            a = np.arange(2 * 3 * n).reshape(2, 3, n).astype('float')
            b = np.arange(n).astype('float')
            self.check_output("...i, ...i", a, b)

        for n in range(1, 11):
            a = np.arange(n * 3 * 2).reshape(n, 3, 2).astype('float')
            b = np.arange(n).astype('float')
            self.check_output("i..., i...", a, b)

        for n in range(1, 17):
            a = (np.arange(3) + 1).astype('float')
            b = (np.arange(n) + 1).astype('float')
            self.check_output("i,j", a, b)

        for n in range(1, 17):
            a = np.arange(4 * n).reshape(4, n).astype('float')
            b = np.arange(n).astype('float')
            self.check_output("ij, j", a, b)

        for n in range(1, 17):
            a = np.arange(4 * n).reshape(4, n).astype('float')
            b = np.arange(n).astype('float')
            self.check_output("ji,j", a.T, b.T)

        for n in range(1, 17):
            a = np.arange(4 * n).reshape(4, n).astype('float')
            b = np.arange(n * 6).reshape(n, 6).astype('float')
            self.check_output("ij,jk", a, b)

        a = np.arange(12).reshape(3, 4).astype('float')
        b = np.arange(20).reshape(4, 5).astype('float')
        c = np.arange(30).reshape(5, 6).astype('float')
        self.check_output("ij,jk,kl", a, b, c)

        a = np.arange(60).reshape(3, 4, 5).astype('float')
        b = np.arange(24).reshape(4, 3, 2).astype('float')
        self.check_output("ijk, jil -> kl", a, b)

        for n in range(1, 25):
            a = np.arange(n).astype('float')
            self.check_output("...,...", a, a)
            self.check_output("i,i", a, a)

        x = np.eye(2).astype('float')
        y = np.ones(2).astype('float')
        self.check_output("ji,i->", x, y)
        self.check_output("i,ij->", y, x)
        self.check_output("ij,i->", x, y)

    def test_static_graph(self):
        paddle.enable_static()
        base = paddle.base
        if base.core.is_compiled_with_cuda():
            self.place = base.CUDAPlace(0)
        else:
            self.place = base.CPUPlace()
        main = base.Program()
        startup = base.Program()
        with base.program_guard(main, startup):
            a = paddle.static.data(
                name='a', shape=[3, None, None, None], dtype='float'
            )
            b = paddle.static.data(
                name='b', shape=[2, None, None, None], dtype='float'
            )
            c = paddle.static.data(
                name='c', shape=[None, None, 2, None], dtype='float'
            )
            d = paddle.static.data(
                name='d', shape=[None, None, 5], dtype='float'
            )
            e = paddle.static.data(
                name='e', shape=[None, 2, None], dtype='float'
            )

            outs = []
            outs.append(paddle.einsum("ibnd,jbnd->bnij", a, b))
            outs.append(paddle.einsum('...ik, ...j', c, d))
            outs.append(paddle.einsum('...kj, ...ik', d, e))
            outs.append(paddle.einsum('ijk..., ikj', c, e))
            outs.append(paddle.einsum('ijk..., ikj->...ij', c, e))
        exe = base.Executor(self.place)
        exe.run(startup)
        a = np.arange(72).reshape(3, 2, 3, 4).astype('float')
        b = np.arange(48).reshape(2, 2, 3, 4).astype('float')
        c = np.arange(48).reshape(2, 3, 2, 4).astype('float')
        d = np.arange(30).reshape(2, 3, 5).astype('float')
        e = np.arange(12).reshape(2, 2, 3).astype('float')
        feeds = {'a': a, 'b': b, 'c': c, 'd': d, 'e': e}
        actual = exe.run(main, feed=feeds, fetch_list=[outs])
        expect = []
        expect.append(np.einsum("ibnd,jbnd->bnij", a, b))
        expect.append(np.einsum('...ik, ...j', c, d))
        expect.append(np.einsum('...kj, ...ik', d, e))
        expect.append(np.einsum('ijk..., ikj', c, e))
        expect.append(np.einsum('ijk..., ikj->...ij', c, e))
        for a, e in zip(actual, expect):
            self.check_output_equal(a, e)


class TestStaticGraphShape(unittest.TestCase):
    def setUp(self):
        paddle.enable_static()

    def tearDown(self):
        paddle.disable_static()

    def test_shape(self):
        A = paddle.static.data(name='x', shape=[-1])
        B = paddle.static.data(name='y', shape=[384])
        C = paddle.einsum('i,d->id', A, B)
        self.assertEqual(tuple(C.shape), (-1, 384))


@unittest.skipIf(
    not core.is_compiled_with_cuda()
    or not core.is_bfloat16_supported(core.CUDAPlace(0)),
    "core is not compiled with CUDA or not support the bfloat16",
)
class TestBF16(unittest.TestCase):
    """
    EinsumOp support bfloat16 type, add unittest here for the correctness.
    """

    def test_shape(self):
        cuda_major = paddle.version.cuda().split('.')[0].strip()
        if int(cuda_major) >= 11:
            """MatmulKernel support bfloat16 only if cuda_major > 11.0."""
            A = paddle.to_tensor(np.array([1.0, 2.0])).astype(paddle.bfloat16)
            A = A.cuda()
            B = paddle.to_tensor(np.array([2.0, 3.0])).astype(paddle.bfloat16)
            B = B.cuda()
            C = paddle.einsum('i,i->', A, B)
            D = paddle.to_tensor([8.0]).astype(paddle.bfloat16)
            self.assertEqual(C.item(), D.item())


class TestComplex(unittest.TestCase):
    """
    EinsumOp support Complex type
    """

    def test_shape(self):
        a = paddle.rand([4, 4])
        b = paddle.rand([4, 4])
        c = paddle.einsum('xy,yz->xz', a, b)
        a = paddle.cast(a, 'complex64')
        b = paddle.cast(b, 'complex64')
        c = paddle.einsum('xy,yz->xz', a, b)


class TestSimpleUndiagonal(unittest.TestCase):
    """
    EinsumOp support undiagonalize.
    """

    def test_shape(self):
        paddle.disable_static()
        A = paddle.to_tensor(np.array([1.0, 2.0]))
        A_expect = paddle.to_tensor([[1.0, 0.0], [0.0, 2.0]])
        A_actual = paddle.einsum('i->ii', A)
        np.testing.assert_array_equal(A_expect.numpy(), A_actual.numpy())


class TestSimpleUndiagonal2(unittest.TestCase):
    """
    EinsumOp support undiagonalize.
    """

    def test_shape(self):
        paddle.disable_static()
        A = paddle.to_tensor(np.array([1.0, 2.0]))
        B = paddle.to_tensor(np.array([1.0, 1.0]))
        A_expect = paddle.to_tensor([[2.0, 0.0], [0.0, 4.0]])
        A_actual = paddle.einsum('i,j->ii', A, B)
        np.testing.assert_array_equal(A_expect.numpy(), A_actual.numpy())


class TestSimpleComplexGrad(unittest.TestCase):
    """
    EinsumOp support complex grad. but op_test don't support numeric grad for complex dtype.
    """

    def test_shape(self):
        paddle.disable_static()
        A = paddle.to_tensor(
            [
                [
                    [-1.08644637 + 1.30794563j],
                    [-0.89606513 + 1.84546043j],
                    [-0.30629937 + 0.82911495j],
                ],
                [
                    [-1.33993366 - 0.02329881j],
                    [-1.20658558 - 0.20856395j],
                    [-0.64172681 - 0.91661975j],
                ],
            ]
        )

        B = paddle.to_tensor(
            [
                [[-1.07474258 + 0.39477287j], [-0.08614349 - 0.38770082j]],
                [[1.17583854 + 0.58840176j], [-1.63509173 - 1.43329882j]],
                [[1.228194 - 0.32357468j], [1.07638625 + 1.25298469j]],
            ]
        )

        dOut = paddle.to_tensor(
            [
                [[-0.73074259 - 0.1632133j], [1.42848507 - 0.96410727j]],
                [[0.94465389 - 0.34264733j], [-0.26400278 + 0.04890404j]],
            ]
        )

        d_expect = paddle.to_tensor(
            [
                [
                    [0.971658 + 1.100766j],
                    [-1.909121 + 3.861908j],
                    [-0.515092 - 3.264529j],
                ],
                [
                    [-1.146746 - 0.111233j],
                    [1.270721 - 1.417091j],
                    [1.048197 + 0.268260j],
                ],
            ]
        )

        A.stop_gradient = False
        B.stop_gradient = False
        Out = paddle.einsum('iox,ojx->ijx', A, B)
        dA = paddle.grad(Out, A, dOut)[0]
        np.testing.assert_allclose(
            dA.numpy(), d_expect.numpy(), rtol=1e-6, atol=0
        )


if __name__ == "__main__":
    unittest.main()
