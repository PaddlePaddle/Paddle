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

        assert_close(t1, t3, atol=1e-3, rtol=0.0)

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
            t_gpu = t1.to("gpu")
            with self.assertRaisesRegex(AssertionError, "device"):
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
            self.assertIn("['data']", msg)

    def test_tensor_mismatch_msg_details(self):
        t1 = paddle.to_tensor([[1.0, 2.0], [3.0, 4.0]])
        t2 = paddle.to_tensor([[1.0, 2.0], [3.0, 5.0]])

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
        c2 = 1 + 1j + 1e-10j
        c3 = 1 + 2j

        assert_close(c1, c2)
        with self.assertRaises(AssertionError):
            assert_close(c1, c3)

    def test_tolerance_validation_logic(self):
        with self.assertRaisesRegex(
            ValueError,
            "Both 'rtol' and 'atol' must be either specified or omitted",
        ):
            assert_close(1.0, 1.0, rtol=1e-5)

        with self.assertRaisesRegex(
            ValueError,
            "Both 'rtol' and 'atol' must be either specified or omitted",
        ):
            assert_close(1.0, 1.0, atol=1e-5)

    def test_msg_callable(self):
        def custom_formatter(orig_msg):
            return f"PREFIX -> {orig_msg} <- SUFFIX"

        with self.assertRaisesRegex(
            AssertionError, "PREFIX -> Scalars are not equal!"
        ):
            assert_close(1, 2, msg=custom_formatter)

    def test_zero_dim_tensor_mismatch(self):
        t1 = paddle.to_tensor(1.0)
        t2 = paddle.to_tensor(2.0)

        with self.assertRaisesRegex(AssertionError, "Scalars"):
            assert_close(t1, t2)

        try:
            assert_close(t1, t2)
        except AssertionError as e:
            self.assertNotIn("Tensor-likes", str(e))

    def test_type_promotion_logic(self):
        t_real = paddle.to_tensor([1.0], dtype='float32')
        t_complex = paddle.to_tensor([1.0 + 0j], dtype='complex64')
        assert_close(t_real, t_complex, check_dtype=False)

        t_c64 = paddle.to_tensor([1 + 1j], dtype='complex64')
        t_c128 = paddle.to_tensor([1 + 1j], dtype='complex128')
        assert_close(t_c64, t_c128, check_dtype=False)

    def test_object_pair_broken_eq(self):
        from paddle.testing._comparison import ErrorMeta, ObjectPair

        class BrokenObj:
            def __eq__(self, other):
                raise RuntimeError("Comparison crashed internal error!")

            def __repr__(self):
                return "BrokenObj"

        obj = BrokenObj()
        pair = ObjectPair(obj, obj)

        try:
            pair.compare()
        except ErrorMeta as e:
            actual_error = e.to_error()

            self.assertIsInstance(actual_error, ValueError)
            self.assertIn(
                "failed with:\nComparison crashed internal error!",
                str(actual_error),
            )
        else:
            self.fail("ObjectPair.compare() should have raised an ErrorMeta")

    def test_pair_repr_and_extra_repr(self):
        from paddle.testing._comparison import NumberPair, ObjectPair, Pair

        obj_pair = ObjectPair(actual=10, expected=20, id=("test_id",))
        rep_str = repr(obj_pair)

        self.assertIn("ObjectPair(", rep_str)
        self.assertIn("id=('test_id',),", rep_str)
        self.assertIn("actual=10,", rep_str)
        self.assertIn("expected=20,", rep_str)

        num_pair = NumberPair(1.0, 1.0, rtol=0.5, atol=0.1)
        rep_str_num = repr(num_pair)

        self.assertIn("NumberPair(", rep_str_num)
        self.assertIn("rtol=0.5,", rep_str_num)
        self.assertIn("atol=0.1,", rep_str_num)

        class MockTuplePair(Pair):
            def compare(self):
                pass

            def extra_repr(self):
                return [("custom_key", "custom_value")]

        mock_pair = MockTuplePair("act", "exp")
        rep_str_mock = repr(mock_pair)

        self.assertIn("MockTuplePair(", rep_str_mock)
        self.assertIn("custom_key=custom_value,", rep_str_mock)

    def test_static_graph_variable(self):
        paddle.enable_static()
        try:
            main_prog = paddle.static.Program()
            startup_prog = paddle.static.Program()

            with paddle.static.program_guard(main_prog, startup_prog):
                x = paddle.static.data(name='x', shape=[2, 2], dtype='float32')
                y = paddle.static.data(name='y', shape=[2, 2], dtype='float32')

                assert_close(x, y)
                with self.assertRaisesRegex(
                    AssertionError, "Python types do not match"
                ):
                    assert_close(x, 1)

                z = paddle.static.data(name='z', shape=[2, 2], dtype='int32')
                assert_close(x, z, check_dtype=False)
                with self.assertRaisesRegex(
                    AssertionError,
                    "The values for attribute dtype do not match",
                ):
                    assert_close(x, z)

                w = paddle.static.data(name='w', shape=[2, 3], dtype='float32')
                with self.assertRaisesRegex(
                    AssertionError,
                    "The values for attribute shape do not match",
                ):
                    assert_close(x, w)

                v = paddle.static.data(
                    name='v', shape=[2, 2, 1], dtype='float32'
                )
                with self.assertRaisesRegex(
                    AssertionError,
                    "The values for attribute shape do not match",
                ):
                    assert_close(x, v)

                dynamic_x = paddle.static.data(
                    name='dynamic_x', shape=[-1, 2], dtype='float32'
                )
                assert_close(x, dynamic_x)
                assert_close(dynamic_x, x)

        finally:
            paddle.disable_static()


if __name__ == '__main__':
    unittest.main()
