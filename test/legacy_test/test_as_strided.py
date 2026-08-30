#  Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import unittest

import numpy as np
from op_test import get_device, get_places

import paddle
from paddle import base


class TestAsStrided(unittest.TestCase):
    def setUp(self):
        self.shape = [32, 32]
        self.typelist = ['float32', 'float64', 'int32', 'int64', 'float16']
        self.places = get_places()
        if base.core.is_compiled_with_cuda():
            self.places.append(base.CUDAPinnedPlace())

    def test_as_strided_forward(self):
        for idx, p in enumerate(self.places):
            if idx == 0:
                paddle.set_device('cpu')
            else:
                paddle.set_device(get_device())
            for dtype in self.typelist:
                x_np = np.random.random(self.shape).astype(dtype)
                x = paddle.to_tensor(x_np, place=p)
                a = paddle.as_strided(x, shape=(3, 4), stride=(32, 1))
                np.testing.assert_allclose(a.numpy(), x_np[:3, :4])

    def test_as_strided_backward(self):
        for idx, p in enumerate(self.places):
            if idx == 0:
                paddle.set_device('cpu')
            else:
                paddle.set_device(get_device())
            for dtype in self.typelist:
                x_np = np.random.random(self.shape).astype(dtype)
                x = paddle.to_tensor(x_np, place=p)
                x.stop_gradient = False
                a = paddle.as_strided(x, shape=(3,), stride=(1,))
                b = a * 2
                b.retain_grads()
                loss = b.sum()
                loss.backward()
                self.assertEqual((b.grad.numpy() == 1).all().item(), True)


class TestAsStrided_ZeroSize(unittest.TestCase):
    def setUp(self):
        self.places = get_places()

    def test_as_strided_forward(self):
        for place in self.places:
            with base.dygraph.guard(place):
                a = paddle.to_tensor(
                    np.random.random([0, 32]).astype('float32')
                )
                a.stop_gradient = False
                b = paddle.as_strided(a, shape=(0, 4), stride=(32, 1))
                np.testing.assert_equal(b.shape, [0, 4])
                b.backward(paddle.ones_like(b))
                np.testing.assert_equal(a.grad.shape, [0, 32])

    def test_as_strided_error(self):
        for place in self.places:
            with base.dygraph.guard(place):
                self.assertRaises(
                    ValueError,
                    paddle.as_strided,
                    x=paddle.to_tensor(
                        np.random.random([0, 32]).astype('float32')
                    ),
                    shape=[3, 4],
                    stride=[32, 1],
                )


class TestAsStridedAlias(unittest.TestCase):
    def test_as_strided_alias(self):
        self.shape = [32, 32]
        self.typelist = ['float32', 'float64', 'int32', 'int64', 'float16']
        with base.dygraph.guard():
            for dtype in self.typelist:
                x_np = np.random.random(self.shape).astype(dtype)
                x = paddle.to_tensor(x_np)
                shape = (3, 4)
                stride = (32, 1)
                offset = 0
                # 1. Standard call (Benchmark)
                out_ref = paddle.as_strided(
                    x, shape=shape, stride=stride, offset=offset
                )

                # 2. Test alias: input -> x
                out_input = paddle.as_strided(
                    input=x, shape=shape, stride=stride, offset=offset
                )
                np.testing.assert_array_equal(
                    out_ref.numpy(), out_input.numpy()
                )

                # 3. Test alias: size -> shape
                out_size = paddle.as_strided(
                    x=x, size=shape, stride=stride, offset=offset
                )
                np.testing.assert_array_equal(out_ref.numpy(), out_size.numpy())

                # 4. Test alias: storage_offset -> offset
                out_offset = paddle.as_strided(
                    x=x, shape=shape, stride=stride, storage_offset=offset
                )
                np.testing.assert_array_equal(
                    out_ref.numpy(), out_offset.numpy()
                )

                # 5. Test both aliases: input -> x, shape -> repeat_times
                out_both = paddle.as_strided(
                    input=x, size=shape, stride=stride, storage_offset=offset
                )
                np.testing.assert_array_equal(out_ref.numpy(), out_both.numpy())


class TestAsStridedOverlapBackward(unittest.TestCase):
    """When a view maps several logical positions onto the same element, the
    backward has to sum all the incoming contributions instead of letting the
    writes overwrite each other."""

    def setUp(self):
        self.places = get_places()

    def _check(self, numel, shape, stride, offset_elems=0, dtype='float64'):
        itemsize = np.dtype(dtype).itemsize
        expected = np.zeros([numel], dtype=dtype)
        for idx in np.ndindex(*shape):
            expected[offset_elems + int(np.dot(idx, stride))] += 1
        for place in self.places:
            with base.dygraph.guard(place):
                x = paddle.to_tensor(np.random.random([numel]).astype(dtype))
                x.stop_gradient = False
                y = paddle.as_strided(
                    x,
                    shape=shape,
                    stride=stride,
                    offset=offset_elems * itemsize,
                )
                y.backward(paddle.ones_like(y))
                np.testing.assert_allclose(x.grad.numpy(), expected)

    def test_repeated_stride(self):
        self._check(5, (3, 3), (1, 1))
        self._check(64, (32, 32), (1, 1))

    def test_zero_stride(self):
        self._check(1, (4, 4), (0, 0))
        self._check(3, (3, 4), (1, 0))

    def test_offset(self):
        self._check(8, (3, 3), (1, 1), offset_elems=2)

    def test_float32(self):
        self._check(5, (3, 3), (1, 1), dtype='float32')

    def test_non_overlapping(self):
        self._check(6, (2, 3), (3, 1))
        self._check(6, (3, 2), (1, 3))
        self._check(6, (2, 3), (3, 1), dtype='float32')


@unittest.skipIf(
    not base.core.is_compiled_with_xpu(), "core is not compiled with XPU"
)
class TestAsStridedOverlapBackwardXPU(unittest.TestCase):
    """get_places() reports CPU, CUDA and CustomPlace but never XPU, so without
    this class the serial host fallback of the overlapping backward -- the only
    path an XPU takes, since the device scatter-add needs GPU atomics -- would
    never run. It stages both operands on the host through phi::Copy, so it
    also covers the XPU <-> CPU copies that path depends on."""

    def test_repeated_stride(self):
        with base.dygraph.guard(base.XPUPlace(0)):
            x = paddle.to_tensor(np.random.random([5]).astype('float32'))
            x.stop_gradient = False
            y = paddle.as_strided(x, shape=(3, 3), stride=(1, 1))
            y.backward(paddle.ones_like(y))
            np.testing.assert_allclose(
                x.grad.numpy(), np.array([1, 2, 3, 2, 1], dtype='float32')
            )

    def test_zero_stride(self):
        with base.dygraph.guard(base.XPUPlace(0)):
            x = paddle.to_tensor(np.random.random([1]).astype('float32'))
            x.stop_gradient = False
            y = paddle.as_strided(x, shape=(4, 4), stride=(0, 0))
            y.backward(paddle.ones_like(y))
            np.testing.assert_allclose(
                x.grad.numpy(), np.array([16], dtype='float32')
            )


class TestAsStridedStorageRange(unittest.TestCase):
    """A view must stay inside the allocation of its input, otherwise reads and
    writes through it corrupt unrelated memory."""

    def setUp(self):
        self.places = get_places()

    def test_shape_out_of_range(self):
        for place in self.places:
            with base.dygraph.guard(place):
                x = paddle.zeros([16], dtype='float32')
                self.assertRaises(
                    ValueError,
                    paddle.as_strided,
                    x=x,
                    shape=(64, 64),
                    stride=(64, 1),
                )

    def test_offset_out_of_range(self):
        for place in self.places:
            with base.dygraph.guard(place):
                x = paddle.zeros([16], dtype='float32')
                self.assertRaises(
                    ValueError,
                    paddle.as_strided,
                    x=x,
                    shape=(16,),
                    stride=(1,),
                    offset=4096 * 4,
                )

    def test_misaligned_offset(self):
        for place in self.places:
            with base.dygraph.guard(place):
                x = paddle.zeros([16], dtype='float32')
                self.assertRaises(
                    ValueError,
                    paddle.as_strided,
                    x=x,
                    shape=(2,),
                    stride=(1,),
                    offset=1,
                )

    def test_overflowing_span_is_rejected(self):
        # 4 * 2**62 wraps around to 0 in int64, so unchecked arithmetic would
        # accept this view and let it read and write far outside the allocation.
        for place in self.places:
            with base.dygraph.guard(place):
                x = paddle.zeros([16], dtype='float32')
                self.assertRaises(
                    ValueError,
                    paddle.as_strided,
                    x=x,
                    shape=(5,),
                    stride=(1 << 62,),
                )

    def test_in_range_view_is_allowed(self):
        for place in self.places:
            with base.dygraph.guard(place):
                x_np = np.random.random([16]).astype('float32')
                x = paddle.to_tensor(x_np)
                y = paddle.as_strided(x, shape=(4, 4), stride=(4, 1), offset=0)
                np.testing.assert_allclose(y.numpy(), x_np.reshape(4, 4))


class TestAsStridedNestedViewBackward(unittest.TestCase):
    """`offset` is an absolute byte offset into the shared allocation, while the
    gradient buffer of a view starts at its own element 0. Taking a view of a
    view therefore has to relocate the offset before writing the gradient."""

    def setUp(self):
        self.places = get_places()

    def _check(self, shape, stride, dtype='float32'):
        itemsize = np.dtype(dtype).itemsize
        base_numel = 8
        outer_numel = 6
        outer_offset = 2
        expected = np.zeros([base_numel], dtype=dtype)
        for idx in np.ndindex(*shape):
            expected[outer_offset + int(np.dot(idx, stride))] += 1
        for place in self.places:
            with base.dygraph.guard(place):
                x = paddle.to_tensor(
                    np.random.random([base_numel]).astype(dtype)
                )
                x.stop_gradient = False
                outer = paddle.as_strided(
                    x,
                    shape=(outer_numel,),
                    stride=(1,),
                    offset=outer_offset * itemsize,
                )
                inner = paddle.as_strided(
                    outer,
                    shape=shape,
                    stride=stride,
                    offset=outer_offset * itemsize,
                )
                inner.backward(paddle.ones_like(inner))
                np.testing.assert_allclose(x.grad.numpy(), expected)

    def test_nested_non_overlapping(self):
        self._check((2,), (2,))
        self._check((3,), (2,))

    def test_nested_overlapping(self):
        self._check((3, 3), (1, 1))


class TestAsStridedStorageCoordinateBackward(unittest.TestCase):
    """`dims` / `stride` / `offset` address the shared allocation, while the
    gradient buffer is dense and row-major over the input's own logical
    indices. The two orders coincide only when the input is contiguous and the
    whole window of the view lands inside it, so in every other case the
    gradient has to be routed back through the input's own strides."""

    def setUp(self):
        self.places = get_places()

    def _check(
        self,
        base_numel,
        input_shape,
        input_stride,
        input_offset,
        shape,
        stride,
        offset_elems=0,
        dtype='float64',
    ):
        itemsize = np.dtype(dtype).itemsize
        # Gradient per slot of the shared allocation.
        storage_grad = np.zeros([base_numel], dtype=dtype)
        for idx in np.ndindex(*shape):
            storage_grad[offset_elems + int(np.dot(idx, stride))] += 1
        # Read back through the input geometry: this is what distinguishes a
        # storage index from a row-major one.
        expected_input = np.zeros(input_shape, dtype=dtype)
        expected_base = np.zeros([base_numel], dtype=dtype)
        for idx in np.ndindex(*input_shape):
            slot = input_offset + int(np.dot(idx, input_stride))
            expected_input[idx] = storage_grad[slot]
            expected_base[slot] += storage_grad[slot]
        for place in self.places:
            with base.dygraph.guard(place):
                x = paddle.to_tensor(
                    np.random.random([base_numel]).astype(dtype)
                )
                x.stop_gradient = False
                inp = paddle.as_strided(
                    x,
                    shape=input_shape,
                    stride=input_stride,
                    offset=input_offset * itemsize,
                )
                inp.retain_grads()
                y = paddle.as_strided(
                    inp,
                    shape=shape,
                    stride=stride,
                    offset=offset_elems * itemsize,
                )
                y.backward(paddle.ones_like(y))
                np.testing.assert_allclose(inp.grad.numpy(), expected_input)
                np.testing.assert_allclose(x.grad.numpy(), expected_base)

    def test_transposed_input_overlapping_view(self):
        # 2x2 transpose viewed with a repeated stride: storage slot 1 collects
        # two contributions and belongs to input[1, 0], not to input[0, 1].
        self._check(4, (2, 2), (1, 2), 0, (2, 2), (1, 1))

    def test_transposed_input_non_overlapping_view(self):
        self._check(6, (2, 3), (1, 2), 0, (3,), (2,))
        self._check(6, (2, 3), (1, 2), 0, (2, 2), (1, 2))

    def test_transposed_input_float32(self):
        self._check(4, (2, 2), (1, 2), 0, (2, 2), (1, 1), dtype='float32')

    def test_view_starting_before_input(self):
        # The leading contributions land outside the input and are dropped;
        # they belong to no element of its gradient.
        self._check(8, (4,), (1,), 2, (4,), (1,), offset_elems=0)
        self._check(8, (2, 2), (1, 2), 2, (3,), (1,), offset_elems=1)

    def test_view_ending_after_input(self):
        # A view may legally reach past the end of its input, because the
        # forward validates it against the shared allocation and not against
        # the extent of the input. Only storage slot 5 belongs to the input
        # here, so its gradient is [0, 0, 0, 1].
        self._check(8, (4,), (1,), 2, (3,), (1,), offset_elems=5)

    def test_view_disjoint_from_input(self):
        # No slot is shared, so nothing reaches the input's gradient at all.
        self._check(8, (4,), (1,), 2, (2,), (1,), offset_elems=6)

    def test_view_ending_after_input_overlapping(self):
        self._check(8, (4,), (1,), 2, (2, 2), (1, 1), offset_elems=4)

    def test_overlapping_input_is_rejected(self):
        # The gradient of a view that aliases itself needs a convention for
        # splitting a shared slot, so it is refused rather than guessed.
        for place in self.places:
            with base.dygraph.guard(place):
                x = paddle.to_tensor(np.random.random([4]).astype('float32'))
                x.stop_gradient = False
                inp = paddle.as_strided(x, shape=(2, 2), stride=(1, 1))
                y = paddle.as_strided(inp, shape=(2,), stride=(1,))
                with self.assertRaises(NotImplementedError):
                    y.backward(paddle.ones_like(y))


if __name__ == '__main__':
    unittest.main()
