#   Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

import paddle
from paddle.testing import assert_close


class TestAssertClose(unittest.TestCase):
    def setUp(self):
        paddle.set_device("cpu")

    def test_scalars_exact_match(self):
        assert_close(1, 1)
        assert_close(1.0, 1.0)
        assert_close(True, True)
        assert_close(None, None)
        assert_close(1 + 2j, 1 + 2j)

    def test_scalars_mismatch(self):
        with self.assertRaisesRegex(AssertionError, "Scalars are not equal!"):
            assert_close(1, 2)
        with self.assertRaisesRegex(AssertionError, "Booleans mismatch"):
            assert_close(True, False)
        with self.assertRaisesRegex(AssertionError, "None mismatch"):
            assert_close(None, 1)

    def test_scalars_tolerances(self):
        assert_close(1.0, 1.0 + 1e-9)

        with self.assertRaises(AssertionError):
            assert_close(1.0, 1.1)

        assert_close(1.0, 1.1, atol=0.2, rtol=0.0)
        with self.assertRaises(AssertionError):
            assert_close(1.0, 1.1, atol=0.05, rtol=0.0)

    def test_numpy_scalars(self):
        if np:
            assert_close(np.float32(1.0), np.float32(1.0))
            assert_close(np.int32(1), np.int32(1))
            assert_close(np.bool_(True), np.bool_(True))

            assert_close(np.float64(1.0), 1.0)

    def test_tensor_exact_match(self):
        t1 = paddle.to_tensor([1.0, 2.0, 3.0])
        t2 = paddle.to_tensor([1.0, 2.0, 3.0])
        assert_close(t1, t2)

    def test_tensor_tolerances(self):
        t1 = paddle.to_tensor([1.0, 2.0, 3.0], dtype='float32')
        t2 = t1 + 1e-6
        assert_close(t1, t2)

        t3 = t1 + 1e-4
        with self.assertRaisesRegex(
            AssertionError, "Tensor-likes are not close"
        ):
            assert_close(t1, t3)

        # assert_close(t1, t3, atol=1e-3, rtol=0)

    def test_tensor_shape_mismatch(self):
        t1 = paddle.zeros([2, 2])
        t2 = paddle.zeros([2, 3])
        with self.assertRaisesRegex(AssertionError, "shape"):
            assert_close(t1, t2)

    def test_tensor_dtype_check(self):
        t_float32 = paddle.to_tensor([1.0], dtype='float32')
        t_float64 = paddle.to_tensor([1.0], dtype='float64')

        with self.assertRaisesRegex(AssertionError, "dtype"):
            assert_close(t_float32, t_float64)

        assert_close(t_float32, t_float64, check_dtype=False)

    def test_tensor_device_check(self):
        t1 = paddle.to_tensor([1.0])
        if paddle.device.is_compiled_with_cuda():
            t_gpu = t1.cuda()
            with self.assertRaisesRegex(AssertionError, "place"):
                assert_close(t1, t_gpu)

            assert_close(t1, t_gpu, check_device=False)
        else:
            assert_close(t1, t1)

    def test_nan_handling(self):
        val_nan = float('nan')
        t_nan = paddle.to_tensor([val_nan])

        with self.assertRaises(AssertionError):
            assert_close(val_nan, val_nan)
        with self.assertRaises(AssertionError):
            assert_close(t_nan, t_nan)

        assert_close(val_nan, val_nan, equal_nan=True)
        assert_close(t_nan, t_nan, equal_nan=True)

    def test_sequences(self):
        l1 = [paddle.to_tensor(1.0), 2.0]
        l2 = [paddle.to_tensor(1.0), 2.0]
        assert_close(l1, l2)

        with self.assertRaisesRegex(
            AssertionError, "length of the sequences mismatch"
        ):
            assert_close([1], [1, 2])

        with self.assertRaisesRegex(AssertionError, "Scalars are not equal!"):
            assert_close([1], [2])

    def test_mappings(self):
        d1 = {"a": 1, "b": paddle.to_tensor(2.0)}
        d2 = {"a": 1, "b": paddle.to_tensor(2.0)}
        assert_close(d1, d2)

        d3 = {"a": 1, "c": 2.0}
        with self.assertRaisesRegex(
            AssertionError, "keys of the mappings do not match"
        ):
            assert_close(d1, d3)

    def test_nested_structure_error_msg(self):
        actual = {"data": [{"val": 10}]}
        expected = {"data": [{"val": 20}]}

        try:
            assert_close(actual, expected)
        except AssertionError as e:
            msg = str(e)
            self.assertIn("data", msg)
            self.assertIn("val", msg)
            # ErrorMeta logic: ''.join(str([item]) for item in self.id) -> "['data'][0]['val']"
            self.assertIn("['data']", msg)

    def test_tensor_mismatch_msg_details(self):
        t1 = paddle.to_tensor([[1.0, 2.0], [3.0, 4.0]])
        t2 = paddle.to_tensor([[1.0, 2.0], [3.0, 5.0]])  # index (1, 1) mismatch

        try:
            assert_close(t1, t2)
        except AssertionError as e:
            msg = str(e)
            self.assertIn("Mismatched elements: 1 / 4", msg)
            self.assertIn("Greatest absolute difference: 1.0", msg)
            self.assertIn("at index (1, 1)", msg)

    def test_msg_override(self):
        with self.assertRaisesRegex(AssertionError, "My custom error"):
            assert_close(1, 2, msg="My custom error")

    def test_unsupported_types(self):
        class A:
            pass

        class B:
            pass

        with self.assertRaises(TypeError):
            assert_close(A(), B())

    def test_complex_numbers(self):
        c1 = 1 + 1j
        c2 = 1 + 1j + 1e-10j  # close
        c3 = 1 + 2j  # not close

        assert_close(c1, c2)
        with self.assertRaises(AssertionError):
            assert_close(c1, c3)

    # def test_int_tensor_promoted_check(self):
    #     t1 = paddle.to_tensor([1, 2], dtype='int32')
    #     t2 = paddle.to_tensor([1, 3], dtype='int32')

    #     with self.assertRaises(AssertionError):
    #         assert_close(t1, t2)

    #     assert_close(t1, t1)


if __name__ == '__main__':
    unittest.main()
