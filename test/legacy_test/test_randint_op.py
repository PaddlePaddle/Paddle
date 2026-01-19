# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
from op_test import OpTest, get_device, get_device_place, is_custom_device

import paddle

paddle.enable_static()


def output_hist(out):
    hist, _ = np.histogram(out, range=(-10, 10))
    hist = hist.astype("float32")
    hist /= float(out.size)
    prob = 0.1 * np.ones(10)
    return hist, prob


class TestRandintOp(OpTest):
    def setUp(self):
        self.op_type = "randint"
        self.python_api = paddle.randint
        self.inputs = {}
        self.init_attrs()
        self.outputs = {"Out": np.zeros((10000, 784)).astype("float32")}

    def init_attrs(self):
        self.attrs = {"shape": [10000, 784], "low": -10, "high": 10, "seed": 10}
        self.output_hist = output_hist

    def test_check_output(self):
        self.check_output_customized(self.verify_output, check_pir=True)

    def verify_output(self, outs):
        hist, prob = self.output_hist(np.array(outs[0]))
        np.testing.assert_allclose(hist, prob, rtol=0, atol=0.001)


class TestRandintOpError(unittest.TestCase):
    def test_errors(self):
        with paddle.static.program_guard(
            paddle.static.Program(), paddle.static.Program()
        ):
            self.assertRaises(TypeError, paddle.randint, 5, shape=np.array([2]))
            self.assertRaises(TypeError, paddle.randint, 5, dtype='float32')
            self.assertRaises(ValueError, paddle.randint, 5, 5)
            self.assertRaises(ValueError, paddle.randint, -5)
            self.assertRaises(TypeError, paddle.randint, 5, shape=['2'])
            shape_tensor = paddle.static.data('X', [1])
            self.assertRaises(TypeError, paddle.randint, 5, shape=shape_tensor)
            self.assertRaises(
                TypeError, paddle.randint, 5, shape=[shape_tensor]
            )


class TestRandintOp_attr_tensorlist(OpTest):
    def setUp(self):
        self.op_type = "randint"
        self.python_api = paddle.randint
        self.new_shape = (10000, 784)
        shape_tensor = []
        for index, ele in enumerate(self.new_shape):
            shape_tensor.append(
                ("x" + str(index), np.ones(1).astype("int64") * ele)
            )
        self.inputs = {'ShapeTensorList': shape_tensor}
        self.init_attrs()
        self.outputs = {"Out": np.zeros((10000, 784)).astype("int32")}

    def init_attrs(self):
        self.attrs = {"low": -10, "high": 10, "seed": 10}
        self.output_hist = output_hist

    def test_check_output(self):
        self.check_output_customized(self.verify_output, check_pir=True)

    def verify_output(self, outs):
        hist, prob = self.output_hist(np.array(outs[0]))
        np.testing.assert_allclose(hist, prob, rtol=0, atol=0.001)


class TestRandint_attr_tensor(OpTest):
    def setUp(self):
        self.op_type = "randint"
        self.python_api = paddle.randint
        self.inputs = {"ShapeTensor": np.array([10000, 784]).astype("int64")}
        self.init_attrs()
        self.outputs = {"Out": np.zeros((10000, 784)).astype("int64")}

    def init_attrs(self):
        self.attrs = {"low": -10, "high": 10, "seed": 10}
        self.output_hist = output_hist

    def test_check_output(self):
        self.check_output_customized(self.verify_output, check_pir=True)

    def verify_output(self, outs):
        hist, prob = self.output_hist(np.array(outs[0]))
        np.testing.assert_allclose(hist, prob, rtol=0, atol=0.001)


# Test python API
class TestRandintAPI(unittest.TestCase):
    def test_api(self):
        with paddle.static.program_guard(
            paddle.static.Program(), paddle.static.Program()
        ):
            # results are from [0, 5).
            out1 = paddle.randint(5)
            # shape is a list and dtype is 'int32'
            out2 = paddle.randint(
                low=-100, high=100, shape=[64, 64], dtype='int32'
            )
            # shape is a tuple and dtype is 'int64'
            out3 = paddle.randint(
                low=-100, high=100, shape=(32, 32, 3), dtype='int64'
            )
            # shape is a tensorlist and dtype is 'float32'
            dim_1 = paddle.tensor.fill_constant([1], "int64", 32)
            dim_2 = paddle.tensor.fill_constant([1], "int32", 50)
            out4 = paddle.randint(
                low=-100, high=100, shape=[dim_1, 5, dim_2], dtype='int32'
            )
            # shape is a tensor and dtype is 'float64'
            var_shape = paddle.static.data(
                name='var_shape', shape=[2], dtype="int64"
            )
            out5 = paddle.randint(
                low=1, high=1000, shape=var_shape, dtype='int64'
            )

            place = get_device_place()
            exe = paddle.static.Executor(place)
            outs = exe.run(
                feed={'var_shape': np.array([100, 100]).astype('int64')},
                fetch_list=[out1, out2, out3, out4, out5],
            )


class TestRandintImperative(unittest.TestCase):
    def test_case(self):
        paddle.disable_static()
        n = 10
        x1 = paddle.randint(n, shape=[10], dtype="int32")
        x2 = paddle.tensor.randint(n)
        x3 = paddle.tensor.random.randint(n)
        for i in [x1, x2, x3]:
            for j in i.numpy().tolist():
                self.assertTrue(j >= 0 and j < n)
        paddle.enable_static()


class TestRandomValue(unittest.TestCase):
    def test_fixed_random_number(self):
        # Test GPU Fixed random number, which is generated by 'curandStatePhilox4_32_10_t'
        if not (paddle.is_compiled_with_cuda() or is_custom_device()):
            return

        # Different GPU generatte different random value. Only test V100 here.
        if "V100" not in paddle.device.get_device_name():
            return

        print("Test Fixed Random number on GPU------>")
        paddle.disable_static()

        self.run_test_case()

        paddle.enable_static()

    def run_test_case(self):
        paddle.set_device(get_device())
        paddle.seed(100)

        x = paddle.randint(
            -10000, 10000, [32, 3, 1024, 1024], dtype='int32'
        ).numpy()
        self.assertTrue(x.mean(), -0.7517569760481516)
        self.assertTrue(x.std(), 5773.696619107639)
        expect = [2535, 2109, 5916, -5011, -261]
        np.testing.assert_array_equal(x[10, 0, 100, 100:105], expect)
        expect = [3465, 7206, -8660, -9628, -6574]
        np.testing.assert_array_equal(x[20, 1, 600, 600:605], expect)
        expect = [881, 1560, 1100, 9664, 1669]
        np.testing.assert_array_equal(x[30, 2, 1000, 1000:1005], expect)

        x = paddle.randint(
            -10000, 10000, [32, 3, 1024, 1024], dtype='int64'
        ).numpy()
        self.assertTrue(x.mean(), -1.461287518342336)
        self.assertTrue(x.std(), 5773.023477548159)
        expect = [7213, -9597, 754, 8129, -1158]
        np.testing.assert_array_equal(x[10, 0, 100, 100:105], expect)
        expect = [-7159, 8054, 7675, 6980, 8506]
        np.testing.assert_array_equal(x[20, 1, 600, 600:605], expect)
        expect = [3581, 3420, -8027, -5237, -2436]
        np.testing.assert_array_equal(x[30, 2, 1000, 1000:1005], expect)


# Test API shape
class TestRandintAPI_ZeroDim(unittest.TestCase):
    def test_dygraph(self):
        paddle.disable_static()
        x = paddle.randint(0, 2, [])
        self.assertEqual(x.shape, [])
        paddle.enable_static()

    def test_static(self):
        with paddle.static.program_guard(
            paddle.static.Program(), paddle.static.Program()
        ):
            x = paddle.randint(-10, 10, [])

            # Test compile shape
            self.assertEqual(tuple(x.shape), ())

            # Test runtime shape
            exe = paddle.static.Executor()
            result = exe.run(fetch_list=[x])
            self.assertEqual(tuple(result[0].shape), ())

        paddle.enable_static()


class TestRandintAliasAndOut(unittest.TestCase):
    def test_alias_and_out(self):
        paddle.disable_static()

        # Test size alias (param_one_alias decorator: shape -> size)
        result_1 = paddle.randint(5, size=[3, 4])
        result_2 = paddle.randint(5, size=paddle.to_tensor([3, 4]))
        self.assertEqual(result_1.shape, [3, 4])
        self.assertEqual(result_2.shape, [3, 4])

        # Test out parameter with int32 dtype
        result_3 = paddle.randint(high=5, shape=[3, 4], dtype='int32')
        out = paddle.zeros([3, 4], dtype='int32')
        result_4 = paddle.randint(high=5, shape=[3, 4], dtype='int32', out=out)
        self.assertTrue(paddle.equal_all(result_4, out))
        self.assertEqual(result_4.dtype, paddle.int32)

        # Test out parameter with int64 dtype
        out_int64 = paddle.zeros([2, 5], dtype='int64')
        result_5 = paddle.randint(
            high=10, shape=[2, 5], dtype='int64', out=out_int64
        )
        self.assertTrue(paddle.equal_all(result_5, out_int64))
        self.assertEqual(result_5.dtype, paddle.int64)

        # Test WITHOUT out parameter (out=None, triggers 'if out is None' branch)
        result_6 = paddle.randint(high=5, shape=[3, 4], dtype='int32')
        self.assertEqual(result_6.shape, [3, 4])
        self.assertEqual(result_6.dtype, paddle.int32)

        result_7 = paddle.randint(high=5, shape=[2, 3], dtype='int64')
        self.assertEqual(result_7.shape, [2, 3])
        self.assertEqual(result_7.dtype, paddle.int64)

        paddle.enable_static()

    def test_out_static_mode(self):
        paddle.enable_static()
        with paddle.static.program_guard(
            paddle.static.Program(), paddle.static.Program()
        ):
            # In static mode (PIR), out parameter is not supported (as shown by warning)
            # Test creates new tensor (out=None), triggering 'if out is None' branch
            result1 = paddle.randint(high=5, shape=[3, 4], dtype='int32')
            self.assertEqual(result1.shape, (3, 4))

            result2 = paddle.randint(high=10, shape=[2, 5], dtype='int64')
            self.assertEqual(result2.shape, (2, 5))

    def test_size_alias_static_mode(self):
        paddle.enable_static()
        with paddle.static.program_guard(
            paddle.static.Program(), paddle.static.Program()
        ):
            # Test size parameter as an alias for shape in static mode
            result = paddle.randint(high=5, size=[3, 4], dtype='int32')
            self.assertEqual(result.shape, (3, 4))


class TestRandintOldStaticMode(unittest.TestCase):
    """Test randint in old static graph mode (non-PIR mode).

    This test specifically covers the else branch in randint:
        if out is None:
            out = helper.create_variable_for_type_inference(dtype=dtype)

    This branch is only executed when:
    1. Not in dynamic mode (in_dynamic_mode() returns False)
    2. Not in PIR mode (in_pir_mode() returns False)
    """

    def test_out_none_old_static_mode(self):
        """Test that 'if out is None' branch is covered in old static mode."""
        from paddle.pir_utils import OldIrGuard

        with OldIrGuard():
            main_program = paddle.static.Program()
            startup_program = paddle.static.Program()

            with paddle.static.program_guard(main_program, startup_program):
                # This should go through the else branch (old static mode)
                # and trigger 'if out is None: out = helper.create_variable_for_type_inference(dtype=dtype)'
                result1 = paddle.randint(high=5, shape=[3, 4], dtype='int32')
                result2 = paddle.randint(high=10, shape=[2, 5], dtype='int64')

                # Verify shapes are correct
                self.assertEqual(result1.shape, (3, 4))
                self.assertEqual(result2.shape, (2, 5))

            # Execute the program to verify it works
            place = paddle.CPUPlace()
            exe = paddle.static.Executor(place)
            exe.run(startup_program)
            outs = exe.run(main_program, fetch_list=[result1, result2])

            # Verify the outputs
            self.assertEqual(outs[0].shape, (3, 4))
            self.assertEqual(outs[1].shape, (2, 5))
            # Verify values are in expected range
            self.assertTrue(np.all(outs[0] >= 0) and np.all(outs[0] < 5))
            self.assertTrue(np.all(outs[1] >= 0) and np.all(outs[1] < 10))

    def test_size_alias_old_static_mode(self):
        """Test size alias in old static mode."""
        from paddle.pir_utils import OldIrGuard

        with OldIrGuard():
            main_program = paddle.static.Program()
            startup_program = paddle.static.Program()

            with paddle.static.program_guard(main_program, startup_program):
                # Test using 'size' parameter alias
                result = paddle.randint(high=5, size=[4, 5], dtype='int32')
                self.assertEqual(result.shape, (4, 5))

            # Execute the program
            place = paddle.CPUPlace()
            exe = paddle.static.Executor(place)
            exe.run(startup_program)
            outs = exe.run(main_program, fetch_list=[result])

            self.assertEqual(outs[0].shape, (4, 5))
            self.assertTrue(np.all(outs[0] >= 0) and np.all(outs[0] < 5))


if __name__ == "__main__":
    unittest.main()
