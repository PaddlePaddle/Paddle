# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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
from op_test import OpTest

import paddle
from paddle.base import core
from paddle.base.framework import _dygraph_guard

HAS_CUDA = (
    paddle.is_compiled_with_cuda() and paddle.device.cuda.device_count() > 0
)
NAMES = ('input', 'tensor1', 'tensor2', 'upstream')


def sum_to_shape(array, shape):
    """Undo NumPy broadcasting, including dimensions broadcast from 1 to 0."""
    while array.ndim > len(shape):
        array = array.sum(axis=0)
    axes = tuple(
        i for i, size in enumerate(shape) if size == 1 and array.shape[i] != 1
    )
    if axes:
        array = array.sum(axis=axes, keepdims=True)
    return array.reshape(shape)


class TestAddcmulOp(OpTest):
    def init_shapes(self):
        return (5, 7, 3), (5, 7, 3), (5, 7, 3)

    def setUp(self):
        self.__class__.op_type = self.op_type = 'addcmul'
        self.python_api = paddle.addcmul
        self.public_python_api = paddle.addcmul
        self.dtype = np.float64
        rng = np.random.default_rng(2026)
        self.inputs = dict(
            zip(
                ('input', 'tensor1', 'tensor2'),
                [rng.uniform(0.2, 0.8, s) for s in self.init_shapes()],
            )
        )
        self.attrs = {'value': 0.75}
        x, a, b = self.inputs.values()
        self.outputs = {'out': x + 0.75 * a * b}

    def _get_places(self):
        places = [paddle.CPUPlace()]
        if HAS_CUDA:
            places.append(paddle.CUDAPlace(0))
        return places

    def test_check_output(self):
        self.check_output(check_pir=True)

    def test_check_grad(self):
        self.check_grad(['input', 'tensor1', 'tensor2'], 'out', check_pir=True)


class TestAddcmulBroadcastOp(TestAddcmulOp):
    def init_shapes(self):
        # OpTest requires at least 100 elements in each gradient-check input.
        return (10, 1, 10), (1, 10, 10), (10, 10)


class AddcmulTestBase(unittest.TestCase):
    use_cuda = False

    def setUp(self):
        self.place = paddle.CUDAPlace(0) if self.use_cuda else paddle.CPUPlace()
        guard = paddle.base.dygraph.guard(self.place)
        guard.__enter__()
        self.addCleanup(guard.__exit__, None, None, None)
        self.rng = np.random.default_rng(2026)

    def run_static(self, feed, build):
        """Runs a PIR program that applies build to the paddle.static.data of
        the arrays in feed, and fetches the values build returns."""
        # Leave dygraph mode through _dygraph_guard instead of IrGuard, whose
        # paddle.disable_static resets the place of the later dygraph code.
        with _dygraph_guard(None), paddle.pir_utils.IrGuard():
            main = paddle.static.Program()
            with paddle.static.program_guard(main, paddle.static.Program()):
                data = [
                    paddle.static.data(name, array.shape, array.dtype.name)
                    for name, array in feed.items()
                ]
                fetch_list = build(*data)
            return paddle.static.Executor(self.place).run(
                main, feed=feed, fetch_list=fetch_list
            )


class TestAddcmulAPI(AddcmulTestBase):
    def run_addcmul(self, arrays, value=None, static=False, inplace=False):
        api = paddle.addcmul_ if inplace else paddle.addcmul
        kwargs = {} if value is None else {'value': value}

        def build(*inputs):
            return [api(*inputs, **kwargs)]

        if static:
            return self.run_static(dict(zip(NAMES, arrays)), build)[0]
        inputs = [paddle.to_tensor(array) for array in arrays]
        if not inplace:
            return build(*inputs)[0].numpy()
        ptr = inputs[0].data_ptr()
        self.assertIs(build(*inputs)[0], inputs[0])
        self.assertEqual(inputs[0].data_ptr(), ptr)
        return inputs[0].numpy()

    def check(self, arrays, expected, value=None, inplace=False, exact=False):
        """Checks the results of both dygraph and static graph."""
        for static in (False, True):
            with self.subTest(static=static):
                actual = self.run_addcmul(arrays, value, static, inplace)
                self.assertEqual(actual.shape, expected.shape)
                self.assertEqual(actual.dtype, expected.dtype)
                if not exact and np.issubdtype(expected.dtype, np.inexact):
                    np.testing.assert_allclose(
                        actual, expected, rtol=1e-6, atol=1e-7
                    )
                else:
                    np.testing.assert_array_equal(actual, expected)

    def test_example_and_default_value(self):
        x = np.array([1, 2, 3], 'float32')
        b = np.array([[1], [2]], 'float32')
        expected = np.array([[1.5, 3, 4.5], [2, 4, 6]], 'float32')
        self.check((x, x, b), expected, value=0.5)
        self.check((x, x, b), np.array([[2, 4, 6], [3, 6, 9]], 'float32'))
        out = paddle.to_tensor(x).addcmul(
            paddle.to_tensor(x), paddle.to_tensor(b), value=0.5
        )
        self.assertEqual(out.dtype, paddle.float32)
        np.testing.assert_array_equal(out.numpy(), expected)

    def test_dtypes(self):
        for dtype in (
            'uint8',
            'int8',
            'int16',
            'int32',
            'int64',
            'float16',
            'float32',
            'float64',
            'complex64',
            'complex128',
        ):
            with self.subTest(dtype=dtype):
                x = np.array([1, 2, 3], dtype=dtype)
                a = np.array([2, 3, 4], dtype=dtype)
                b = np.array([3, 2, 1], dtype=dtype)
                if np.issubdtype(x.dtype, np.complexfloating):
                    a += 1j
                self.check((x, a, b), x + 2 * a * b, value=2)

    def test_broadcast_scalar_and_empty(self):
        for shapes in (
            ((), (), ()),
            ((2, 1, 3), (1, 4, 1), (3,)),
            ((2, 1, 3), (3,), (2, 1)),
            ((), (2, 3), (3,)),
            ((), (2, 1), (1, 3)),
            ((1, 3), (), (2, 1)),
            ((2, 1), (1, 3), ()),
            ((0, 3), (1, 3), ()),
            ((0,), (0,), (1,)),
            ((2, 0, 3), (1, 0, 1), (3,)),
        ):
            with self.subTest(shapes=shapes):
                arrays = [
                    np.full(s, i + 1, 'float64') for i, s in enumerate(shapes)
                ]
                expected = arrays[0] - 0.5 * arrays[1] * arrays[2]
                self.check(arrays, expected, value=-0.5)

    def test_noncontiguous_inputs(self):
        arrays = [
            np.arange(6, dtype='float64').reshape(2, 3) + i for i in range(3)
        ]
        tensors = [paddle.to_tensor(a).transpose([1, 0]) for a in arrays]
        self.assertFalse(tensors[0].is_contiguous())
        out = paddle.addcmul(*tensors, value=0.5)
        expected = arrays[0].T + 0.5 * arrays[1].T * arrays[2].T
        np.testing.assert_array_equal(out.numpy(), expected)

    def test_type_promotion(self):
        # Same as PyTorch, a 0-D operand does not widen the N-D operands of the
        # same category, e.g. a float64 0-D tensor with float32 vectors gives
        # float32.
        vector, scalar = (2,), ()
        for dtypes, shapes, expected_dtype in (
            (('int32', 'int64', 'int64'), (vector,) * 3, 'int64'),
            (('float32', 'int64', 'float32'), (vector,) * 3, 'float32'),
            (('float32', 'bool', 'float32'), (vector,) * 3, 'float32'),
            (('float32', 'float64', 'float32'), (vector,) * 3, 'float64'),
            (('float32', 'complex64', 'float32'), (vector,) * 3, 'complex64'),
            (
                ('float64', 'float32', 'float32'),
                (scalar, vector, vector),
                'float32',
            ),
            (
                ('float32', 'float64', 'float32'),
                (vector, scalar, vector),
                'float32',
            ),
            (
                ('float32', 'float32', 'float64'),
                (vector, vector, scalar),
                'float32',
            ),
            (('int64', 'int32', 'int32'), (scalar, vector, vector), 'int32'),
            (
                ('float32', 'complex128', 'float32'),
                (vector, scalar, vector),
                'complex64',
            ),
            (('float32', 'float64', 'float32'), (scalar,) * 3, 'float64'),
        ):
            with self.subTest(dtypes=dtypes, shapes=shapes):
                arrays = [np.ones(s, d) for s, d in zip(shapes, dtypes)]
                shape = np.broadcast_shapes(*shapes)
                self.check(arrays, np.full(shape, 2, expected_dtype))

    def test_value_conversion(self):
        # An integer value is exact for int64 inputs, and a floating-point one
        # is truncated for integer inputs.
        zero, one = np.zeros(1, 'int64'), np.ones(1, 'int64')
        for value in (2**24 + 1, 2**53 + 1, 2**63 - 1, -(2**63)):
            with self.subTest(value=value):
                expected = np.array([value], 'int64')
                self.check((zero, one, one), expected, value=value)
        x, a = np.array([1, 2], 'int64'), np.array([2, 3], 'int64')
        for value, expected in ((0.75, [1, 2]), (2.75, [9, 20])):
            with self.subTest(value=value):
                expected = np.array(expected, 'int64')
                self.check((x, a, a), expected, value=value)

    def test_low_precision_computation_type(self):
        # value is converted to float32 instead of overflowing float16 first.
        dtypes = ['float16']
        if not self.use_cuda or core.is_bfloat16_supported(self.place):
            dtypes.append('bfloat16')
        for dtype in dtypes:
            with self.subTest(dtype=dtype):
                x = paddle.zeros([1], dtype=dtype)
                a = paddle.full([1], 0.5, dtype=dtype)
                out = paddle.addcmul(x, a, a, value=70000)
                expected = paddle.to_tensor([17500.0]).astype(dtype)
                self.assertEqual(out.dtype, expected.dtype)
                np.testing.assert_array_equal(
                    out.astype('float32').numpy(),
                    expected.astype('float32').numpy(),
                )

    def test_complex_value(self):
        x = np.array([1 + 2j, 3 - 1j], 'complex128')
        a = np.array([2 - 1j, -1 + 2j], 'complex128')
        b = np.array([1 + 3j, 2 + 1j], 'complex128')
        self.check((x, a, b), x + (1 + 2j) * a * b, value=1 + 2j)
        real = np.ones(2, 'float32')
        self.check((real,) * 3, np.full(2, 2, 'float32'), value=1 + 0j)

    def test_value_overflow(self):
        # Same as PyTorch, value is checked even when the output is empty.
        for static in (False, True):
            for shape in ((1,), (0,)):
                for dtype, value in (
                    ('int8', 128),
                    ('float32', 1e100),
                    ('float32', 1 + 2j),
                ):
                    arrays = [np.ones(shape, dtype)] * 3
                    with (
                        self.subTest(
                            static=static, shape=shape, dtype=dtype, value=value
                        ),
                        self.assertRaisesRegex(
                            (ValueError, RuntimeError, OverflowError),
                            'overflow|convert|complex',
                        ),
                    ):
                        self.run_addcmul(arrays, value, static)

    def test_nonfinite_values(self):
        one, inf = np.ones(1), float('inf')
        for value in (inf, -inf, float('nan')):
            with self.subTest(value=value):
                self.check((one,) * 3, np.array([value]), value=value)
        self.check((one, np.array([inf]), one), np.array([np.nan]), value=0)

    def test_out(self):
        # out keeps its dtype and identity, rejects an unsafe cast, and can be
        # input itself.
        x = paddle.to_tensor([1.0, 2.0])
        for dtype in ('float32', 'float64'):
            with self.subTest(dtype=dtype):
                target = paddle.empty([2], dtype=dtype)
                ptr = target.data_ptr()
                self.assertIs(paddle.addcmul(x, x, x, out=target), target)
                self.assertEqual(target.data_ptr(), ptr)
                self.assertEqual(target.dtype, getattr(paddle, dtype))
                np.testing.assert_array_equal(target.numpy(), [2, 6])
        with self.assertRaises((ValueError, RuntimeError)):
            paddle.addcmul(x, x, x, out=paddle.empty([2], dtype='int64'))
        a = paddle.to_tensor([2.0, 3.0])
        self.assertIs(paddle.addcmul(x, a, a, out=x), x)
        np.testing.assert_array_equal(x.numpy(), [5, 11])

    def test_out_rejects_autograd(self):
        # Same as PyTorch, out= does not support autograd and keeps out
        # unchanged, but it can still be used under no_grad.
        for dtype in ('float32', 'float64'):
            for i, name in enumerate(('input', 'tensor1', 'tensor2', 'out')):
                with self.subTest(dtype=dtype, requires_grad=name):
                    args = [paddle.to_tensor([1.0]) for _ in range(3)]
                    target = paddle.full([1], -99, dtype=dtype)
                    (*args, target)[i].stop_gradient = False
                    original_shape = tuple(target.shape)
                    original_dtype = target.dtype
                    original_ptr = target.data_ptr()
                    with self.assertRaisesRegex(
                        RuntimeError, r'out=\.\.\. arguments don\'t support'
                    ):
                        paddle.addcmul(*args, out=target)
                    self.assertEqual(tuple(target.shape), original_shape)
                    self.assertEqual(target.dtype, original_dtype)
                    self.assertEqual(target.data_ptr(), original_ptr)
                    np.testing.assert_array_equal(target.numpy(), [-99])
                    with paddle.no_grad():
                        self.assertIs(paddle.addcmul(*args, out=target), target)
                    self.assertEqual(tuple(target.shape), original_shape)
                    self.assertEqual(target.dtype, original_dtype)
                    self.assertEqual(target.data_ptr(), original_ptr)
                    np.testing.assert_array_equal(target.numpy(), [2])

    def test_inplace(self):
        # The multipliers are broadcast to input, which keeps its dtype while
        # the result is computed in the promoted dtype.
        a, b = np.array([[2.0], [4.0]]), np.array([1.0, 2.0, 3.0])
        expected = np.array([[2.0, 3, 4], [3, 5, 7]])
        self.check((np.ones((2, 3)), a, b), expected, value=0.5, inplace=True)
        # 2**-24 is below the default atol, so compare exactly.
        arrays = (np.array([-1], 'float32'), np.array([1 + 2**-24]), np.ones(1))
        self.check(
            arrays, np.array([2**-24], 'float32'), inplace=True, exact=True
        )
        x = paddle.ones([2, 3], dtype='float64')
        out = x.addcmul_(paddle.to_tensor(a), paddle.to_tensor(b), value=0.5)
        self.assertIs(out, x)
        np.testing.assert_array_equal(x.numpy(), expected)

    def test_inplace_errors(self):
        # input can neither be broadcast nor be cast unsafely.
        for static in (False, True):
            for dtype in ('float32', 'float64'):
                arrays = (
                    np.ones((1, 3), 'float32'),
                    np.ones((2, 1), dtype),
                    np.ones(3, dtype),
                )
                with (
                    self.subTest(static=static, multiplier_dtype=dtype),
                    self.assertRaisesRegex(
                        (ValueError, RuntimeError),
                        '(?i)broadcast|shape|dimension',
                    ),
                ):
                    self.run_addcmul(arrays, static=static, inplace=True)
            arrays = (np.ones(2, 'int64'), *[np.ones(2, 'float32')] * 2)
            with (
                self.subTest(static=static, input_dtype='int64'),
                self.assertRaises((ValueError, RuntimeError)),
            ):
                self.run_addcmul(arrays, static=static, inplace=True)

    def test_inplace_grad_leaf(self):
        # A leaf that requires grad can only be updated in place under no_grad.
        for dtype in ('float32', 'float64'):
            with self.subTest(multiplier_dtype=dtype):
                x = paddle.to_tensor([1.0], stop_gradient=False)
                a = paddle.full([1], 2, dtype=dtype)
                b = paddle.full([1], 3, dtype=dtype)
                with self.assertRaisesRegex(
                    (ValueError, RuntimeError), '[Ll]eaf|inplace|in-place'
                ):
                    x.addcmul_(a, b)
                np.testing.assert_array_equal(x.numpy(), [1])
                with paddle.no_grad():
                    self.assertIs(x.addcmul_(a, b), x)
                np.testing.assert_array_equal(x.numpy(), [7])

    def test_invalid_broadcast(self):
        for static in (False, True):
            with (
                self.subTest(static=static),
                self.assertRaises((ValueError, RuntimeError)),
            ):
                arrays = (np.ones(2), np.ones(3), np.ones(2))
                self.run_addcmul(arrays, static=static)


class TestAddcmulGrad(AddcmulTestBase):
    def array(self, shape, dtype):
        data = self.rng.normal(size=shape)
        if np.issubdtype(np.dtype(dtype), np.complexfloating):
            data = data + 1j * self.rng.normal(size=shape)
        return np.asarray(data, dtype=dtype)

    def assert_gradient(self, actual, expected):
        self.assertEqual(actual.shape, expected.shape)
        self.assertEqual(actual.dtype, expected.dtype)
        tolerance = (
            1e-5 if actual.dtype in (np.float32, np.complex64) else 1e-12
        )
        np.testing.assert_allclose(
            actual, expected, rtol=tolerance, atol=tolerance
        )

    def first_grads(self, data, upstream, value, mask, static):
        """Returns the grads of the inputs selected by the bits of mask."""
        selected = [i for i in range(3) if mask & (1 << i)]

        def build(*tensors):
            for i in selected:
                tensors[i].stop_gradient = False
            out = paddle.addcmul(*tensors[:3], value=value)
            inputs = [tensors[i] for i in selected]
            if static:
                return paddle.static.gradients([out], inputs, [tensors[3]])
            return paddle.grad(out, inputs, grad_outputs=tensors[3])

        arrays = (*data, upstream)
        if static:
            return self.run_static(dict(zip(NAMES, arrays)), build)
        tensors = [paddle.to_tensor(array) for array in arrays]
        return [grad.numpy() for grad in build(*tensors)]

    def second_grads(self, data, upstream, seeds, value, wrt, static):
        """Returns the grads of the first order grads with the seeds as their
        upstream grads, with respect to the tensors of NAMES at wrt. A None
        seed leaves that first order grad unused."""
        used = [i for i, seed in enumerate(seeds) if seed is not None]

        def build(*tensors):
            for tensor in tensors[:4]:
                tensor.stop_gradient = False
            out = paddle.addcmul(*tensors[:3], value=value)
            targets = [tensors[i] for i in wrt]
            seed_tensors = list(tensors[4:])
            if static:
                first = paddle.static.gradients(
                    [out], tensors[:3], [tensors[3]]
                )
                outputs = [first[i] for i in used]
                return paddle.static.gradients(outputs, targets, seed_tensors)
            first = paddle.grad(
                out, tensors[:3], grad_outputs=tensors[3], create_graph=True
            )
            outputs = [first[i] for i in used]
            return paddle.grad(outputs, targets, grad_outputs=seed_tensors)

        arrays = (*data, upstream, *(seeds[i] for i in used))
        names = NAMES + tuple(f'seed{i}' for i in used)
        if static:
            return self.run_static(dict(zip(names, arrays)), build)
        tensors = [paddle.to_tensor(array) for array in arrays]
        return [grad.numpy() for grad in build(*tensors)]

    def check_first_gradient(self, shapes, dtype, value, mask=7):
        data = [self.array(shape, dtype) for shape in shapes]
        upstream = self.array(np.broadcast_shapes(*shapes), dtype)
        # Complex gradients use the conjugate Jacobian convention.
        expected = (
            upstream,
            upstream * np.conj(value * data[2]),
            upstream * np.conj(value * data[1]),
        )
        expected = [
            sum_to_shape(expected[i], shapes[i])
            for i in range(3)
            if mask & (1 << i)
        ]
        for static in (False, True):
            with self.subTest(static=static):
                actual = self.first_grads(data, upstream, value, mask, static)
                self.assertEqual(len(actual), len(expected))
                for grad, reference in zip(actual, expected):
                    self.assert_gradient(grad, reference)

    def test_first_gradient(self):
        for dtype in ('float32', 'float64', 'complex64', 'complex128'):
            value = 1 + 2j if dtype.startswith('complex') else -0.75
            for shapes in (
                ((2, 3), (2, 3), (2, 3)),
                ((2, 1), (1, 3), ()),
                ((), (2, 1, 3), (1, 4, 1)),
                ((), (), ()),
                ((0,), (0,), (1,)),
                ((1, 0, 3), (2, 1, 3), ()),
                ((), (0, 3), (1, 3)),
            ):
                with self.subTest(dtype=dtype, shapes=shapes):
                    self.check_first_gradient(shapes, dtype, value)

    def test_first_gradient_selected_inputs(self):
        for dtype, value in (('float64', -0.75), ('complex128', 1 + 2j)):
            for mask in range(1, 8):
                with self.subTest(dtype=dtype, mask=mask):
                    self.check_first_gradient(
                        ((2, 1), (1, 3), ()), dtype, value, mask
                    )

    def test_first_gradient_input_dtypes(self):
        # All products, partial sums, and expected gradients are exactly
        # representable in both float16 and bfloat16.
        cases = [((paddle.float16,) * 3, paddle.float16)]
        if not self.use_cuda or core.is_bfloat16_supported(self.place):
            cases.append(((paddle.bfloat16,) * 3, paddle.bfloat16))
        cases.append(
            ((paddle.float32, paddle.float64, paddle.float32), paddle.float64)
        )
        arrays = (
            np.ones((2, 1), dtype='float32'),
            np.array([[0.5, 1, 2]], dtype='float32'),
            np.array(2, dtype='float32'),
        )
        expected = (
            np.array([[2], [3]], dtype='float32'),
            np.array([[5, -2, 2]], dtype='float32'),
            np.array(2.25, dtype='float32'),
        )
        for dtypes, out_dtype in cases:
            with self.subTest(dtypes=dtypes):
                inputs = [
                    paddle.to_tensor(array, dtype=dtype, stop_gradient=False)
                    for array, dtype in zip(arrays, dtypes)
                ]
                upstream = paddle.to_tensor(
                    [[1, -2, 3], [4, 0, -1]], dtype=out_dtype
                )
                output = paddle.addcmul(*inputs, value=0.5)
                self.assertEqual(output.dtype, out_dtype)
                grads = paddle.grad(output, inputs, grad_outputs=upstream)
                # The grads keep the dtypes of the inputs.
                for grad, reference, dtype in zip(grads, expected, dtypes):
                    self.assertEqual(grad.dtype, dtype)
                    self.assertEqual(tuple(grad.shape), reference.shape)
                    np.testing.assert_array_equal(
                        grad.astype('float32').numpy(), reference
                    )

    def test_first_gradient_noncontiguous(self):
        for dtype, value in (('float64', -0.75), ('complex128', 1 + 2j)):
            with self.subTest(dtype=dtype):
                data = [self.array((3, 2), dtype).T for _ in range(3)]
                inputs = [
                    paddle.to_tensor(array.T, stop_gradient=False).transpose(
                        [1, 0]
                    )
                    for array in data
                ]
                self.assertFalse(any(t.is_contiguous() for t in inputs))
                upstream = self.array((3, 2), dtype).T
                grads = paddle.grad(
                    paddle.addcmul(*inputs, value=value),
                    inputs,
                    grad_outputs=paddle.to_tensor(upstream.T).transpose([1, 0]),
                )
                expected = (
                    upstream,
                    upstream * np.conj(value * data[2]),
                    upstream * np.conj(value * data[1]),
                )
                for grad, reference in zip(grads, expected):
                    self.assert_gradient(grad.numpy(), reference)

    def test_first_gradient_finite_difference(self):
        shapes = ((2, 1), (1, 3), ())
        for dtype, value in (('float64', -0.75), ('complex128', 1 + 2j)):
            with self.subTest(dtype=dtype):
                data = [self.array(shape, dtype) for shape in shapes]
                upstream = self.array((2, 3), dtype)
                grads = self.first_grads(data, upstream, value, 7, False)

                def loss():
                    output = data[0] + value * data[1] * data[2]
                    return np.vdot(upstream, output).real

                epsilon = 1e-5
                directions = (1, 1j) if dtype.startswith('complex') else (1,)
                for array, grad in zip(data, grads):
                    numerical = np.zeros_like(array)
                    for index in np.ndindex(array.shape):
                        original = array[index].copy()
                        for direction in directions:
                            array[index] = original + epsilon * direction
                            positive = loss()
                            array[index] = original - epsilon * direction
                            negative = loss()
                            numerical[index] += (
                                direction
                                * (positive - negative)
                                / (2 * epsilon)
                            )
                        array[index] = original
                    np.testing.assert_allclose(
                        grad, numerical, rtol=1e-8, atol=1e-8
                    )

    def check_second_gradient(self, shapes, dtype, value, mask):
        data = [self.array(shape, dtype) for shape in shapes]
        out_shape = np.broadcast_shapes(*shapes)
        upstream = self.array(out_shape, dtype)
        seeds = [
            self.array(shape, dtype) if mask & (1 << i) else None
            for i, shape in enumerate(shapes)
        ]
        # Only request the connected derivatives.
        wrt, expected = [], []
        if mask & 4:
            wrt.append(1)
            expected.append(
                sum_to_shape(upstream * np.conj(value * seeds[2]), shapes[1])
            )
        if mask & 2:
            wrt.append(2)
            expected.append(
                sum_to_shape(upstream * np.conj(value * seeds[1]), shapes[2])
            )
        wrt.append(3)
        grad_upstream = np.zeros(out_shape, dtype=dtype)
        if mask & 1:
            grad_upstream += seeds[0]
        if mask & 2:
            grad_upstream += value * seeds[1] * data[2]
        if mask & 4:
            grad_upstream += value * seeds[2] * data[1]
        expected.append(grad_upstream)
        for static in (False, True):
            with self.subTest(static=static):
                actual = self.second_grads(
                    data, upstream, seeds, value, wrt, static
                )
                self.assertEqual(len(actual), len(expected))
                for grad, reference in zip(actual, expected):
                    self.assert_gradient(grad, reference)

    def test_second_gradient(self):
        # Omitted first order grads exercise the optional grad_input_grad,
        # grad_tensor1_grad and grad_tensor2_grad of the double grad.
        for dtype, value in (('float64', -0.75), ('complex128', 1 + 2j)):
            for shapes, masks in (
                (((2, 1), (1, 3), ()), range(1, 8)),
                (((), (), ()), range(1, 8)),
                (((2, 1, 3), (3,), (2, 4, 1)), (7,)),
                (((), (2, 1), (1, 3)), (7,)),
                (((0,), (0,), (1,)), (1, 2, 4, 7)),
                (((), (0, 3), (1, 3)), (1, 2, 4, 7)),
            ):
                for mask in masks:
                    with self.subTest(dtype=dtype, shapes=shapes, mask=mask):
                        self.check_second_gradient(shapes, dtype, value, mask)

    def test_second_gradient_float64_value_precision(self):
        ones = np.ones(1)
        for value in (1e-50, 1e100, 1 + 2**-40):
            for static in (False, True):
                with self.subTest(value=value, static=static):
                    (grad,) = self.second_grads(
                        [ones] * 3, ones, [None, ones, None], value, [2], static
                    )
                    np.testing.assert_allclose(
                        grad, [value], rtol=1e-14, atol=0
                    )

    def test_second_gradient_float16(self):
        # Same as PyTorch, float16 is computed in float32, where value and the
        # intermediate results do not overflow.
        x, half, upstream = (np.array([v], 'float16') for v in (0, 0.5, 2**-10))
        seed = np.ones(1, 'float16')
        # PyTorch's double backward gives value * 2**-10 and value * 0.5. 73728
        # overflows float16 but both results are exact in it, so they do not
        # depend on the rounding of the final float32 -> float16 cast, which
        # truncates on ARM CPUs (PADDLE_WITH_ARM).
        for value, expected in ((0.5, (2**-11, 0.25)), (73728, (72, 36864))):
            for static in (False, True):
                with self.subTest(value=value, static=static):
                    grads = self.second_grads(
                        [x, half, half],
                        upstream,
                        [None, seed, None],
                        value,
                        [2, 3],
                        static,
                    )
                    self.assertEqual(len(grads), len(expected))
                    for grad, reference in zip(grads, expected):
                        self.assert_gradient(
                            grad, np.array([reference], 'float16')
                        )

    def test_second_gradient_missing_or_zero_first_grad(self):
        # Same as PyTorch, a missing first order grad is skipped, while a zero
        # one still gives value * 0, which is nan when value is inf or nan.
        data = [np.ones(shape) for shape in ((2, 3), (3,), (2, 1))]
        for value in (0.75, float('inf'), float('nan')):
            for seed_values in (
                (1, None, None),
                (None, 1, None),
                (None, None, 1),
                (1, 1, None),
                (1, None, 1),
                (None, 1, 1),
                (1, 0, None),
            ):
                seeds = [
                    None if seed is None else np.full(array.shape, seed, 'f8')
                    for seed, array in zip(seed_values, data)
                ]
                # All the inputs are ones.
                expected = np.zeros((2, 3))
                for i, seed in enumerate(seed_values):
                    if seed is not None:
                        expected += seed if i == 0 else value * seed
                for static in (False, True):
                    with self.subTest(
                        value=value, seeds=seed_values, static=static
                    ):
                        (grad,) = self.second_grads(
                            data, np.ones((2, 3)), seeds, value, [3], static
                        )
                        np.testing.assert_allclose(grad, expected)

    def test_third_gradient_complex(self):
        # For unit cotangents, d(grad_tensor1)/d(tensor2) = conj(value)
        # and its VJP with respect to grad_out is value, not conj(value).
        inputs = [
            paddle.to_tensor(np.ones(1, 'complex128'), stop_gradient=False)
            for _ in range(4)
        ]
        output = paddle.addcmul(*inputs[:3], value=1 + 2j)
        first = paddle.grad(
            output, inputs[1], grad_outputs=inputs[3], create_graph=True
        )[0]
        second = paddle.grad(
            first,
            inputs[2],
            grad_outputs=paddle.ones_like(first),
            create_graph=True,
        )[0]
        third = paddle.grad(
            second, inputs[3], grad_outputs=paddle.ones_like(second)
        )[0]
        self.assert_gradient(second.numpy(), np.array([1 - 2j]))
        self.assert_gradient(third.numpy(), np.array([1 + 2j]))


@unittest.skipUnless(HAS_CUDA, 'CUDA device is required')
class TestAddcmulAPICUDA(TestAddcmulAPI):
    use_cuda = True


@unittest.skipUnless(HAS_CUDA, 'CUDA device is required')
class TestAddcmulGradCUDA(TestAddcmulGrad):
    use_cuda = True


if __name__ == '__main__':
    unittest.main()
