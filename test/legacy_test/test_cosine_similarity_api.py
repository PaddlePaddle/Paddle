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

import unittest

import numpy as np
from op_test import get_places

import paddle
import paddle.nn.functional as F
from paddle import nn, static
from paddle.base import Executor


def _np_cosine_similarity(x1, x2, axis=1, eps=1e-8):
    """Reference following torch2.12.0: broadcast both inputs first, then divide
    each of them by its own norm along ``axis`` clamped to ``eps``.
    """
    x1, x2 = np.broadcast_arrays(x1, x2)
    n1 = np.maximum(np.sqrt(np.sum(x1 * x1, axis=axis, keepdims=True)), eps)
    n2 = np.maximum(np.sqrt(np.sum(x2 * x2, axis=axis, keepdims=True)), eps)
    return np.sum((x1 / n1) * (x2 / n2), axis=axis)


class TestCosineSimilarityAPI(unittest.TestCase):
    def setUp(self):
        self.places = get_places()

    def _get_numpy_out(self, x1, x2, axis=1, eps=1e-8):
        return _np_cosine_similarity(x1, x2, axis=axis, eps=eps)

    def check_static_result(self, place):
        paddle.enable_static()

        main_program = static.Program()
        startup_program = static.Program()
        with static.program_guard(
            main_program=main_program, startup_program=startup_program
        ):
            shape = [10, 15]
            axis = 1
            eps = 1e-8
            np.random.seed(0)
            np_x1 = np.random.rand(*shape).astype(np.float32)
            np_x2 = np.random.rand(*shape).astype(np.float32)

            x1 = paddle.static.data(name="x1", shape=shape)
            x2 = paddle.static.data(name="x2", shape=shape)
            result = F.cosine_similarity(x1, x2, axis=axis, eps=eps)
            exe = Executor(place)
            fetches = exe.run(
                feed={"x1": np_x1, "x2": np_x2},
                fetch_list=[result],
            )

            np_out = self._get_numpy_out(np_x1, np_x2, axis=axis, eps=eps)
            np.testing.assert_allclose(fetches[0], np_out, rtol=1e-05)

    def test_static(self):
        for place in self.places:
            self.check_static_result(place=place)

    def test_dygraph_1(self):
        paddle.disable_static()

        shape = [10, 15]
        axis = 1
        eps = 1e-8
        np.random.seed(1)
        np_x1 = np.random.rand(*shape).astype(np.float32)
        np_x2 = np.random.rand(*shape).astype(np.float32)
        np_out = self._get_numpy_out(np_x1, np_x2, axis=axis, eps=eps)

        tensor_x1 = paddle.to_tensor(np_x1)
        tensor_x2 = paddle.to_tensor(np_x2)
        y = F.cosine_similarity(tensor_x1, tensor_x2, axis=axis, eps=eps)

        np.testing.assert_allclose(y.numpy(), np_out, rtol=1e-05)

        # test dim alias for axis
        y = F.cosine_similarity(tensor_x1, tensor_x2, dim=axis, eps=eps)
        np.testing.assert_allclose(y.numpy(), np_out, rtol=1e-05)

    def test_dygraph_2(self):
        paddle.disable_static()

        shape = [12, 13]
        axis = 0
        eps = 1e-6
        np.random.seed(1)
        np_x1 = np.random.rand(*shape).astype(np.float32)
        np_x2 = np.random.rand(*shape).astype(np.float32)
        np_out = self._get_numpy_out(np_x1, np_x2, axis=axis, eps=eps)

        tensor_x1 = paddle.to_tensor(np_x1)
        tensor_x2 = paddle.to_tensor(np_x2)
        y = F.cosine_similarity(tensor_x1, tensor_x2, axis=axis, eps=eps)

        np.testing.assert_allclose(y.numpy(), np_out, rtol=1e-05)

    def test_dygraph_3(self):
        paddle.disable_static()

        shape1 = [10, 12, 10]
        shape2 = [10, 1, 10]
        axis = 2
        eps = 1e-6
        np.random.seed(1)
        np_x1 = np.random.rand(*shape1).astype(np.float32)
        np_x2 = np.random.rand(*shape2).astype(np.float32)
        np_out = self._get_numpy_out(np_x1, np_x2, axis=axis, eps=eps)

        tensor_x1 = paddle.to_tensor(np_x1)
        tensor_x2 = paddle.to_tensor(np_x2)
        y = F.cosine_similarity(tensor_x1, tensor_x2, axis=axis, eps=eps)

        np.testing.assert_allclose(y.numpy(), np_out, rtol=1e-05)

    def test_dygraph_4(self):
        paddle.disable_static()

        shape1 = [23, 12, 1]
        shape2 = [23, 1, 10]
        axis = 2
        eps = 1e-6
        np.random.seed(1)
        np_x1 = np.random.rand(*shape1).astype(np.float32)
        np_x2 = np.random.rand(*shape2).astype(np.float32)
        np_out = self._get_numpy_out(np_x1, np_x2, axis=axis, eps=eps)

        cos_sim_func = nn.CosineSimilarity(axis=axis, eps=eps)
        tensor_x1 = paddle.to_tensor(np_x1)
        tensor_x2 = paddle.to_tensor(np_x2)
        y = cos_sim_func(tensor_x1, tensor_x2)

        np.testing.assert_allclose(y.numpy(), np_out, rtol=1e-05)

    def test_dygraph_5(self):
        paddle.disable_static()

        shape1 = [23, 1, 10]
        shape2 = [23, 12, 1]
        axis = 2
        eps = 1e-6
        np.random.seed(1)
        np_x1 = np.random.rand(*shape1).astype(np.float32)
        np_x2 = np.random.rand(*shape2).astype(np.float32)
        np_out = self._get_numpy_out(np_x1, np_x2, axis=axis, eps=eps)

        cos_sim_func = nn.CosineSimilarity(axis=axis, eps=eps)
        tensor_x1 = paddle.to_tensor(np_x1)
        tensor_x2 = paddle.to_tensor(np_x2)
        y = cos_sim_func(tensor_x1, tensor_x2)

        np.testing.assert_allclose(y.numpy(), np_out, rtol=1e-05)

        cos_sim_func = nn.CosineSimilarity(dim=axis + 1, eps=eps)
        cos_sim_func.dim = axis
        y = cos_sim_func(tensor_x1, tensor_x2)
        np.testing.assert_allclose(y.numpy(), np_out, rtol=1e-05)


class TestCosineSimilarityAPI_ZeroSize(unittest.TestCase):
    def setUp(self):
        self.places = get_places()

    def _get_numpy_out(self, x1, x2, axis=1, eps=1e-8):
        return _np_cosine_similarity(x1, x2, axis=axis, eps=eps)

    def test_dygraph_1(self):
        paddle.disable_static()

        shape = [0, 15]
        axis = 1
        eps = 1e-8
        np.random.seed(1)
        np_x1 = np.random.rand(*shape).astype(np.float32)
        np_x2 = np.random.rand(*shape).astype(np.float32)
        np_out = self._get_numpy_out(np_x1, np_x2, axis=axis, eps=eps)

        tensor_x1 = paddle.to_tensor(np_x1)
        tensor_x1.stop_gradient = False
        tensor_x2 = paddle.to_tensor(np_x2)
        y = F.cosine_similarity(tensor_x1, tensor_x2, axis=axis, eps=eps)

        np.testing.assert_allclose(y.numpy(), np_out, rtol=1e-05)
        y.sum().backward()
        np.testing.assert_allclose(tensor_x1.grad.shape, tensor_x1.shape)


class TestCosineSimilarityAPI_RankMismatch(unittest.TestCase):
    """x1 and x2 may have different ranks: they are broadcast first, so
    ``axis`` indexes the common shape rather than each input's own shape.
    """

    def test_dygraph(self):
        paddle.disable_static()
        np.random.seed(1)
        for shape1, shape2, axis in [
            ([2, 3], [1, 2, 3], 1),
            ([2, 3], [1, 2, 3], 2),
            ([2, 3], [1, 2, 3], -1),
            ([2, 3], [1, 2, 3], -2),
            ([4, 5], [3, 4, 5], 0),
            ([5], [3, 4, 5], 2),
            ([1, 7], [2, 3, 7], 1),
            ([3, 4, 5], [4, 5], 1),
        ]:
            np_x1 = np.random.rand(*shape1).astype(np.float32)
            np_x2 = np.random.rand(*shape2).astype(np.float32)
            np_out = _np_cosine_similarity(np_x1, np_x2, axis=axis)

            y = F.cosine_similarity(
                paddle.to_tensor(np_x1), paddle.to_tensor(np_x2), axis=axis
            )
            msg = f"x1={shape1} x2={shape2} axis={axis}"
            self.assertEqual(list(y.shape), list(np_out.shape), msg)
            np.testing.assert_allclose(
                y.numpy(), np_out, rtol=1e-05, err_msg=msg
            )


class TestCosineSimilarityAPI_UnknownReduceDim(unittest.TestCase):
    """A static graph may only know the reduced axis at run time, so the
    ||repeat(x, m)|| == sqrt(m) * ||x|| factor cannot be folded into a python
    scalar there: with shape=[2, -1] fed a [2, 1] input broadcast against
    [2, 5], skipping the factor would report a similarity above 1.
    """

    def _check(self, shape1, shape2, feed1, feed2, axis):
        np_x1 = np.random.rand(*feed1).astype(np.float32)
        np_x2 = np.random.rand(*feed2).astype(np.float32)
        np_out = _np_cosine_similarity(np_x1, np_x2, axis=axis)
        msg = f"x1={shape1}{feed1} x2={shape2}{feed2} axis={axis}"

        main_program = static.Program()
        startup_program = static.Program()
        with static.program_guard(main_program, startup_program):
            x1 = static.data(name="x1", shape=shape1, dtype='float32')
            x2 = static.data(name="x2", shape=shape2, dtype='float32')
            result = F.cosine_similarity(x1, x2, axis=axis)
            for place in get_places():
                fetches = Executor(place).run(
                    main_program,
                    feed={"x1": np_x1, "x2": np_x2},
                    fetch_list=[result],
                )
                self.assertEqual(
                    list(fetches[0].shape), list(np_out.shape), msg
                )
                np.testing.assert_allclose(
                    fetches[0], np_out, rtol=1e-07, err_msg=msg
                )

    def test_static(self):
        paddle.enable_static()
        np.random.seed(1)
        try:
            for shape1, shape2, feed1, feed2, axis in [
                # the unknown axis is broadcast at run time
                ([2, -1], [2, 5], (2, 1), (2, 5), 1),
                # the unknown axis turns out not to be broadcast
                ([2, -1], [2, 5], (2, 5), (2, 5), 1),
                # x2 is the short side, so the factor applies to its norm
                ([2, 5], [2, -1], (2, 5), (2, 1), 1),
                ([-1, -1], [3, 4], (3, 1), (3, 4), 1),
                ([2, -1, 4], [2, 6, 4], (2, 1, 4), (2, 6, 4), 1),
                # a 0-size reduction must not divide by zero
                ([2, -1], [2, 0], (2, 0), (2, 0), 1),
            ]:
                self._check(shape1, shape2, feed1, feed2, axis)
        finally:
            paddle.disable_static()


class TestCosineSimilarityAPI_LargeBroadcast(unittest.TestCase):
    """Large inputs that broadcast along ``axis``: the reduction length is the
    broadcast one, which is what the ||repeat(x, m)|| == sqrt(m) * ||x||
    correction has to reproduce.
    """

    def test_dygraph(self):
        paddle.disable_static()
        np.random.seed(1)
        b, n, d = 4, 512, 256
        for shape1, shape2, axis in [
            # broadcast on the reduced axis, so both norms need the correction
            ([b, 1, d], [b, n, d], 1),
            # broadcast on the reduced axis with mismatched ranks
            ([1, d], [b, n, d], 1),
            # broadcast off the reduced axis, no correction
            ([b, 1, d], [b, n, d], 2),
            ([d], [b, n, d], 2),
        ]:
            np_x1 = np.random.rand(*shape1).astype(np.float32)
            np_x2 = np.random.rand(*shape2).astype(np.float32)
            np_out = _np_cosine_similarity(np_x1, np_x2, axis=axis)

            y = F.cosine_similarity(
                paddle.to_tensor(np_x1), paddle.to_tensor(np_x2), axis=axis
            )
            msg = f"x1={shape1} x2={shape2} axis={axis}"
            self.assertEqual(list(y.shape), list(np_out.shape), msg)
            np.testing.assert_allclose(
                y.numpy(), np_out, rtol=1e-05, err_msg=msg
            )


class TestCosineSimilarityAPI_Numerics(unittest.TestCase):
    def test_small_but_valid_vectors(self):
        # |x1| * |x2| is below eps while both norms are far above it, so
        # clamping the product instead of each norm would attenuate the
        # result to about zero
        paddle.disable_static()
        np_x1 = np.array([[1e-6, 0.0, 0.0]], dtype=np.float32)
        np_x2 = np.array([[2e-6, 0.0, 0.0]], dtype=np.float32)
        y = F.cosine_similarity(
            paddle.to_tensor(np_x1), paddle.to_tensor(np_x2), axis=1
        )
        np.testing.assert_allclose(
            y.numpy(), np.ones([1], dtype=np.float32), rtol=1e-06
        )

    def test_large_vectors_do_not_overflow(self):
        # |x1|^2 * |x2|^2 overflows float32 here while each norm stays finite
        paddle.disable_static()
        np_x1 = np.full([1, 4], 1e10, dtype=np.float32)
        np_x2 = np.full([1, 4], -1e10, dtype=np.float32)
        y = F.cosine_similarity(
            paddle.to_tensor(np_x1), paddle.to_tensor(np_x2), axis=1
        )
        np.testing.assert_allclose(
            y.numpy(), -np.ones([1], dtype=np.float32), rtol=1e-06
        )

    def test_cpu_reduced_dtype_fp32_path(self):
        # CPU has no p_norm/divide/multiply kernels for fp16/bf16, so the
        # inputs are promoted to fp32 for the computation and the result is
        # cast back to the reduced dtype, keeping the output dtype contract.
        paddle.disable_static()
        origin_device = paddle.get_device()
        paddle.set_device('cpu')
        np_x1 = np.random.rand(2, 4).astype(np.float32)
        np_x2 = np.random.rand(2, 4).astype(np.float32)
        np_out = _np_cosine_similarity(np_x1, np_x2, axis=1)
        try:
            for dtype, pd_dtype in [
                ('float16', paddle.float16),
                ('bfloat16', paddle.bfloat16),
            ]:
                x1 = paddle.to_tensor(np_x1, dtype=dtype)
                x2 = paddle.to_tensor(np_x2, dtype=dtype)
                y = F.cosine_similarity(x1, x2, axis=1)
                self.assertEqual(y.dtype, pd_dtype)
                np.testing.assert_allclose(
                    y.astype('float32').numpy(),
                    np_out,
                    rtol=1e-06,
                    err_msg=f"dtype={dtype}",
                )
        finally:
            paddle.set_device(origin_device)


class TestCosineSimilarityAPI_DtypePromotion(unittest.TestCase):
    def test_integral_input_is_promoted(self):
        paddle.disable_static()
        np_x1 = np.array([[1, 2, 3]], dtype=np.int32)
        np_x2 = np.array([[3.0, 2.0, 1.0]], dtype=np.float32)
        y = F.cosine_similarity(
            paddle.to_tensor(np_x1), paddle.to_tensor(np_x2), axis=1
        )
        self.assertEqual(y.dtype, paddle.float32)
        np_out = _np_cosine_similarity(np_x1.astype(np.float32), np_x2)
        np.testing.assert_allclose(y.numpy(), np_out, rtol=1e-06)

    def test_mixed_float_dtypes(self):
        paddle.disable_static()
        np_x1 = np.random.rand(2, 5).astype(np.float32)
        np_x2 = np.random.rand(2, 5).astype(np.float64)
        y = F.cosine_similarity(
            paddle.to_tensor(np_x1), paddle.to_tensor(np_x2), axis=1
        )
        self.assertEqual(y.dtype, paddle.float64)
        np_out = _np_cosine_similarity(np_x1.astype(np.float64), np_x2)
        np.testing.assert_allclose(y.numpy(), np_out, rtol=1e-06)

    def test_non_floating_common_dtype(self):
        paddle.disable_static()
        for dtype in ('int32', 'int64', 'bool', 'complex64'):
            x = paddle.ones([1, 3], dtype=dtype)
            with self.assertRaises(TypeError):
                F.cosine_similarity(x, x, axis=1)

    def test_negative_eps(self):
        paddle.disable_static()
        x = paddle.ones([1, 3])
        with self.assertRaises(ValueError):
            F.cosine_similarity(x, x, axis=1, eps=-1e-8)


if __name__ == '__main__':
    unittest.main()
