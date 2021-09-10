# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

import re
import sys
import unittest

from scipy.fft import hfftn, hfft2
import numpy as np
import paddle

TEST_CASE_NAME = "test_case"


def setUpModule():
    global rtol
    global atol
    rtol = {'float32': 1e-6, 'float64': 1e-6}
    atol = {'float32': 0.0, 'float64': 0.0}


def tearDownModule():
    pass


def rand_x(dims=1, dtype='float32', min_dim_len=1, max_dim_len=10):
    """generate random input"""
    shape = [np.random.randint(min_dim_len, max_dim_len) for i in range(dims)]
    return np.random.randn(*shape).astype(dtype)


def parameterize(attrs, input_values=None):
    """ Parameterizes a test class by setting attributes on the class.
    """
    if isinstance(attrs, str):
        attrs = [attrs]
    input_dicts = (attrs if input_values is None else
                   [dict(zip(attrs, vals)) for vals in input_values])

    def decorator(base_class):
        """class decorator"""
        test_class_module = sys.modules[base_class.__module__].__dict__
        for idx, input_dict in enumerate(input_dicts):
            test_class_dict = dict(base_class.__dict__)
            test_class_dict.update(input_dict)

            name = class_name(base_class, idx, input_dict)

            test_class_module[name] = type(name, (base_class, ),
                                           test_class_dict)

        for method_name in list(base_class.__dict__):
            if method_name.startswith("test"):
                delattr(base_class, method_name)
        return base_class

    return decorator


def class_name(cls, num, params_dict):
    suffix = to_safe_name(
        next((v for v in params_dict.values() if isinstance(v, str)), ""))
    if TEST_CASE_NAME in params_dict:
        suffix = to_safe_name(params_dict["test_case"])
    return "{}_{}{}".format(cls.__name__, num, suffix and "_" + suffix)


def to_safe_name(s):
    return str(re.sub("[^a-zA-Z0-9_]+", "_", s))


@parameterize((TEST_CASE_NAME, 'x', 'n', 'axis', 'norm'), [
    ('test_x_complex128',
     (np.random.randn(4, 4, 4) + 1j * np.random.randn(4, 4, 4)
      ).astype(np.complex128), None, -1, "backward"),
    ('test_n_grater_than_input_length',
     np.random.randn(4, 4, 4) + 1j * np.random.randn(4, 4, 4), 4, -1,
     "backward"),
    ('test_n_smaller_than_input_length',
     np.random.randn(4, 4, 4) + 1j * np.random.randn(4, 4, 4), 2, -1,
     "backward"),
    ('test_axis_not_last',
     np.random.randn(4, 4, 4) + 1j * np.random.randn(4, 4, 4), None, 1,
     "backward"),
    ('test_norm_forward',
     np.random.randn(4, 4, 4) + 1j * np.random.randn(4, 4, 4), None, 1,
     "forward"),
    ('test_norm_ortho',
     np.random.randn(4, 4, 4) + 1j * np.random.randn(4, 4, 4), None, -1,
     "ortho"),
])
class TestHfft(unittest.TestCase):
    """Test hfft with norm condition
    """

    def test_hfft(self):
        np.testing.assert_allclose(
            np.fft.hfft(self.x, self.n, self.axis, self.norm),
            paddle.tensor.fft.hfft(
                paddle.to_tensor(self.x), self.n, self.axis, self.norm),
            rtol=1e-5,
            atol=0)


@parameterize((TEST_CASE_NAME, 'x', 'n', 'axis', 'norm'), [
    ('test_x_complex128',
     (np.random.randn(4, 4, 4) + 1j * np.random.randn(4, 4, 4)
      ).astype(np.complex128), None, -1, "backward"),
    ('test_n_grater_than_input_length',
     np.random.randn(4, 4, 4) + 1j * np.random.randn(4, 4, 4), 4, -1,
     "backward"),
    ('test_n_smaller_than_input_length',
     np.random.randn(4, 4, 4) + 1j * np.random.randn(4, 4, 4), 2, -1,
     "backward"),
    ('test_axis_not_last',
     np.random.randn(4, 4, 4) + 1j * np.random.randn(4, 4, 4), None, -1,
     "backward"),
    ('test_norm_forward',
     np.random.randn(4, 4, 4) + 1j * np.random.randn(4, 4, 4), None, -1,
     "forward"),
    ('test_norm_ortho',
     np.random.randn(4, 4, 4) + 1j * np.random.randn(4, 4, 4), None, -1,
     "ortho"),
])
class TestIrfft(unittest.TestCase):
    """Test irfft with norm condition
    """

    def test_irfft(self):
        np.testing.assert_allclose(
            np.fft.irfft(self.x, self.n, self.axis, self.norm),
            paddle.tensor.fft.irfft(
                paddle.to_tensor(self.x), self.n, self.axis, self.norm),
            rtol=1e-5,
            atol=0)


@parameterize((TEST_CASE_NAME, 'x', 'n', 'axis', 'norm'), [
    ('test_x_complex128',
     (np.random.randn(4, 4, 4) + 1j * np.random.randn(4, 4, 4)
      ).astype(np.complex128), None, None, "backward"),
    ('test_n_grater_than_input_length',
     np.random.randn(4, 4, 4) + 1j * np.random.randn(4, 4, 4), [4], None,
     "backward"),
    ('test_n_smaller_than_input_length',
     np.random.randn(4, 4, 4) + 1j * np.random.randn(4, 4, 4), [2], None,
     "backward"),
    ('test_axis_not_last',
     np.random.randn(4, 4, 4) + 1j * np.random.randn(4, 4, 4), None, None,
     "backward"),
    ('test_norm_forward',
     np.random.randn(4, 4, 4) + 1j * np.random.randn(4, 4, 4), None, None,
     "forward"),
    ('test_norm_ortho',
     np.random.randn(4, 4, 4) + 1j * np.random.randn(4, 4, 4), None, None,
     "ortho"),
])
class Testirfftn(unittest.TestCase):
    """Test irfftn with norm condition
    """

    def test_irfftn(self):
        np.testing.assert_allclose(
            np.fft.irfftn(self.x, self.n, self.axis, self.norm),
            paddle.tensor.fft.irfftn(
                paddle.to_tensor(self.x), self.n, self.axis, self.norm),
            rtol=1e-5,
            atol=0)


@parameterize((TEST_CASE_NAME, 'x', 'n', 'axis', 'norm'), [
    ('test_x_complex128',
     (np.random.randn(4, 4, 4) + 1j * np.random.randn(4, 4, 4)
      ).astype(np.complex128), None, None, "backward"),
    ('test_n_grater_than_input_length',
     np.random.randn(4, 4, 4) + 1j * np.random.randn(4, 4, 4), [4], None,
     "backward"),
    ('test_n_smaller_than_input_length',
     np.random.randn(4, 4, 4) + 1j * np.random.randn(4, 4, 4), [2], None,
     "backward"),
    ('test_axis_not_last',
     np.random.randn(4, 4, 4) + 1j * np.random.randn(4, 4, 4), None, None,
     "backward"),
    ('test_norm_forward',
     np.random.randn(4, 4, 4) + 1j * np.random.randn(4, 4, 4), None, None,
     "forward"),
    ('test_norm_ortho',
     np.random.randn(4, 4, 4) + 1j * np.random.randn(4, 4, 4), None, None,
     "ortho"),
])
class Testhfftn(unittest.TestCase):
    """Test hfftn with norm condition
    """

    def test_hfftn(self):
        np.testing.assert_allclose(
            hfftn(self.x, self.n, self.axis, self.norm),
            paddle.tensor.fft.hfftn(
                paddle.to_tensor(self.x), self.n, self.axis, self.norm),
            rtol=1e-5,
            atol=0)


@parameterize((TEST_CASE_NAME, 'x', 's', 'axis', 'norm'), [
    ('test_x_complex128',
     (np.random.randn(4, 4, 4) + 1j * np.random.randn(4, 4, 4)
      ).astype(np.complex128), None, (-2, -1), "backward"),
    ('test_n_grater_than_input_length',
     np.random.randn(4, 4, 4) + 1j * np.random.randn(4, 4, 4), [1, 2], (-2, -1),
     "backward"),
    ('test_n_smaller_than_input_length',
     np.random.randn(4, 4, 4) + 1j * np.random.randn(4, 4, 4), [2, 1], (-2, -1),
     "backward"),
    ('test_axis_not_last',
     np.random.randn(4, 4, 4) + 1j * np.random.randn(4, 4, 4), None, (-2, -1),
     "backward"),
    ('test_norm_forward',
     np.random.randn(4, 4, 4) + 1j * np.random.randn(4, 4, 4), None, (-2, -1),
     "forward"),
    ('test_norm_ortho',
     np.random.randn(4, 4, 4) + 1j * np.random.randn(4, 4, 4), None, (-2, -1),
     "ortho"),
])
class Testhfft2(unittest.TestCase):
    """Test hfft2 with norm condition
    """

    def test_hfft2(self):
        np.testing.assert_allclose(
            hfft2(self.x, self.s, self.axis, self.norm),
            paddle.tensor.fft.hfft2(
                paddle.to_tensor(self.x), self.s, self.axis, self.norm),
            rtol=1e-5,
            atol=0)


@parameterize((TEST_CASE_NAME, 'x', 's', 'axis', 'norm'), [
    ('test_x_complex128',
     (np.random.randn(4, 4, 4) + 1j * np.random.randn(4, 4, 4)
      ).astype(np.complex128), None, (-2, -1), "backward"),
    ('test_n_equal_input_length',
     np.random.randn(4, 4, 4) + 1j * np.random.randn(4, 4, 4), (2, 1), (-2, -1),
     "backward"),
    ('test_axis_not_last',
     np.random.randn(4, 4, 4) + 1j * np.random.randn(4, 4, 4), None, (-2, -1),
     "backward"),
    ('test_norm_forward',
     np.random.randn(4, 4, 4) + 1j * np.random.randn(4, 4, 4), None, (-2, -1),
     "forward"),
    ('test_norm_ortho',
     np.random.randn(4, 4, 4) + 1j * np.random.randn(4, 4, 4), None, (-2, -1),
     "ortho"),
])
class TestIrfft2(unittest.TestCase):
    """Test irfft2 with norm condition
    """

    def test_irfft2(self):
        np.testing.assert_allclose(
            np.fft.irfft2(self.x, self.s, self.axis, self.norm),
            paddle.tensor.fft.irfft2(
                paddle.to_tensor(self.x), self.s, self.axis, self.norm),
            rtol=1e-5,
            atol=0)


@parameterize(
    (TEST_CASE_NAME, 'x', 'n', 'axis', 'norm', 'expect_exception'),
    [('test_input_dtype', np.random.randn(4, 4, 4), None, -1, 'backward',
      ValueError), ('test_bool_input',
                    (np.random.randn(4, 4, 4) + 1j * np.random.randn(4, 4, 4)
                     ).astype(np.bool8), None, -1, 'backward', ValueError),
     ('test_n_nagative',
      np.random.randn(4, 4, 4) + 1j * np.random.randn(4, 4, 4), -1, -1,
      'backward', ValueError),
     ('test_n_zero', np.random.randn(4, 4) + 1j * np.random.randn(4, 4), 0, -1,
      'backward', ValueError),
     ('test_n_type', np.random.randn(4, 4, 4) + 1j * np.random.randn(4, 4, 4),
      (1, 2, 3), -1, 'backward', ValueError),
     ('test_axis_out_of_range', np.random.randn(4) + 1j * np.random.randn(4),
      None, 10, 'backward', ValueError), (
          'test_axis_with_array', np.random.randn(4) + 1j * np.random.randn(4),
          None, (0, 1), 'backward',
          ValueError), ('test_norm_not_in_enum_value',
                        np.random.randn(4, 4) + 1j * np.random.randn(4, 4),
                        None, -1, 'random', ValueError)])
class TestHfftException(unittest.TestCase):
    '''Test hfft with buoudary condition
    Test case include:
    - non complex input
    - n out of range
    - axis out of range
    - norm out of range
    '''

    def test_hfft(self):
        with self.assertRaises(self.expect_exception):
            paddle.tensor.fft.rfft(self.x, self.n, self.axis, self.norm)


@parameterize(
    (TEST_CASE_NAME, 'x', 'n', 'axis', 'norm', 'expect_exception'),
    [('test_input_dtype', np.random.randn(4, 4, 4), None, -1, 'backward',
      ValueError), ('test_bool_input',
                    (np.random.randn(4, 4, 4) + 1j * np.random.randn(4, 4, 4)
                     ).astype(np.bool8), None, -1, 'backward', ValueError),
     ('test_n_nagative',
      np.random.randn(4, 4, 4) + 1j * np.random.randn(4, 4, 4), -1, -1,
      'backward', ValueError),
     ('test_n_zero', np.random.randn(4, 4) + 1j * np.random.randn(4, 4), 0, -1,
      'backward', ValueError),
     ('test_n_type', np.random.randn(4, 4, 4) + 1j * np.random.randn(4, 4, 4),
      (1, 2), -1, 'backward', ValueError),
     ('test_axis_out_of_range', np.random.randn(4) + 1j * np.random.randn(4),
      None, 10, 'backward', ValueError), (
          'test_axis_with_array', np.random.randn(4) + 1j * np.random.randn(4),
          None, (0, 1), 'backward',
          ValueError), ('test_norm_not_in_enum_value',
                        np.random.randn(4, 4) + 1j * np.random.randn(4, 4),
                        None, None, 'random', ValueError)])
class TestIrfftException(unittest.TestCase):
    '''Test Irfft with buoudary condition
    Test case include:
    - non complex input
    - n out of range
    - axis out of range
    - norm out of range
    - the dimensions of n and axis are different
    '''

    def test_irfft(self):
        with self.assertRaises(self.expect_exception):
            paddle.tensor.fft.irfft(self.x, self.n, self.axis, self.norm)


@parameterize((TEST_CASE_NAME, 'x', 'n', 'axis', 'norm', 'expect_exception'), [
    ('test_input_dtype', np.random.randn(4, 4, 4), None, None, 'backward',
     ValueError), ('test_bool_input',
                   (np.random.randn(4, 4, 4) + 1j * np.random.randn(4, 4, 4)
                    ).astype(np.bool8), None, (-2, -1), 'backward', ValueError),
    ('test_n_nagative',
     np.random.randn(4, 4, 4) + 1j * np.random.randn(4, 4, 4), (-1, -2),
     (-2, -1), 'backward', ValueError),
    ('test_n_zero', np.random.randn(4, 4, 4) + 1j * np.random.randn(4, 4, 4),
     (0, 0), (-2, -1), 'backward', ValueError),
    ('test_n_type', np.random.randn(4, 4, 4) + 1j * np.random.randn(4, 4, 4), 3,
     None, 'backward',
     ValueError), ('test_n_axis_dim',
                   np.random.randn(4, 4, 4) + 1j * np.random.randn(4, 4, 4),
                   (1, 2), (-1), 'backward', ValueError),
    ('test_axis_out_of_range', np.random.randn(4) + 1j * np.random.randn(4),
     None, (1, 2), 'backward', ValueError), (
         'test_axis_type', np.random.randn(4) + 1j * np.random.randn(4), None,
         -1, 'backward',
         ValueError), ('test_norm_not_in_enum_value',
                       np.random.randn(4, 4) + 1j * np.random.randn(4, 4), None,
                       None, 'random', ValueError)
])
class TestHfft2Exception(unittest.TestCase):
    '''Test hfft2 with buoudary condition
    Test case include:
    - non complex input
    - n out of range
    - axis out of range
    - the dimensions of n and axis are different
    - norm out of range
    '''

    def test_hfft2(self):
        with self.assertRaises(self.expect_exception):
            paddle.tensor.fft.hfft2(self.x, self.n, self.axis, self.norm)


@parameterize((TEST_CASE_NAME, 'x', 'n', 'axis', 'norm', 'expect_exception'), [
    ('test_input_dtype', np.random.randn(4, 4, 4), None, None, 'backward',
     ValueError), ('test_bool_input',
                   (np.random.randn(4, 4, 4) + 1j * np.random.randn(4, 4, 4)
                    ).astype(np.bool8), None, (-2, -1), 'backward', ValueError),
    ('test_n_nagative',
     np.random.randn(4, 4, 4) + 1j * np.random.randn(4, 4, 4), (-1, -2),
     (-2, -1), 'backward', ValueError),
    ('test_n_zero', np.random.randn(4, 4, 4) + 1j * np.random.randn(4, 4, 4),
     (0, 0), (-2, -1), 'backward', ValueError),
    ('test_n_type', np.random.randn(4, 4, 4) + 1j * np.random.randn(4, 4, 4), 3,
     -1, 'backward',
     ValueError), ('test_n_axis_dim',
                   np.random.randn(4, 4, 4) + 1j * np.random.randn(4, 4, 4),
                   (1, 2), (-3, -2, -1), 'backward', ValueError),
    ('test_axis_out_of_range', np.random.randn(4) + 1j * np.random.randn(4),
     None, (1, 2), 'backward', ValueError), (
         'test_axis_type', np.random.randn(4) + 1j * np.random.randn(4), None,
         1, 'backward',
         ValueError), ('test_norm_not_in_enum_value',
                       np.random.randn(4, 4) + 1j * np.random.randn(4, 4), None,
                       None, 'random', ValueError)
])
class TestIrfft2Exception(unittest.TestCase):
    '''Test irfft2 with buoudary condition
    Test case include:
    - non complex input
    - n out of range
    - axis out of range
    - norm out of range
    - the dimensions of n and axis are different
    '''

    def test_irfft2(self):
        with self.assertRaises(self.expect_exception):
            paddle.tensor.fft.irfft2(self.x, self.n, self.axis, self.norm)


@parameterize((TEST_CASE_NAME, 'x', 'n', 'axis', 'norm', 'expect_exception'), [
    ('test_input_dtype', np.random.randn(4, 4, 4), None, None, 'backward',
     ValueError), ('test_bool_input',
                   (np.random.randn(4, 4, 4) + 1j * np.random.randn(4, 4, 4)
                    ).astype(np.bool8), None, (-2, -1), 'backward', ValueError),
    ('test_n_nagative',
     np.random.randn(4, 4, 4) + 1j * np.random.randn(4, 4, 4), (-1, -2),
     (-2, -1), 'backward', ValueError),
    ('test_n_zero', np.random.randn(4, 4, 4) + 1j * np.random.randn(4, 4, 4),
     (0, 0), (-2, -1), 'backward', ValueError),
    ('test_n_type', np.random.randn(4, 4, 4) + 1j * np.random.randn(4, 4, 4), 3,
     -1, 'backward',
     ValueError), ('test_n_axis_dim',
                   np.random.randn(4, 4, 4) + 1j * np.random.randn(4, 4, 4),
                   (1, 2), (-3, -2, -1), 'backward', ValueError),
    ('test_axis_out_of_range', np.random.randn(4) + 1j * np.random.randn(4),
     None, (10, 20), 'backward', ValueError), (
         'test_axis_type', np.random.randn(4) + 1j * np.random.randn(4), None,
         1, 'backward',
         ValueError), ('test_norm_not_in_enum_value',
                       np.random.randn(4, 4) + 1j * np.random.randn(4, 4), None,
                       None, 'random', ValueError)
])
class TestHfftnException(unittest.TestCase):
    '''Test hfftn with buoudary condition
    Test case include:
    - non complex input
    - n out of range
    - axis out of range
    - norm out of range
    - the dimensions of n and axis are different
    '''

    def test_hfftn(self):
        with self.assertRaises(self.expect_exception):
            paddle.tensor.fft.hfftn(self.x, self.n, self.axis, self.norm)


@parameterize((TEST_CASE_NAME, 'x', 'n', 'axis', 'norm', 'expect_exception'), [
    ('test_input_dtype', np.random.randn(4, 4, 4), None, None, 'backward',
     ValueError), ('test_bool_input',
                   (np.random.randn(4, 4, 4) + 1j * np.random.randn(4, 4, 4)
                    ).astype(np.bool8), None, (-2, -1), 'backward', ValueError),
    ('test_n_nagative',
     np.random.randn(4, 4, 4) + 1j * np.random.randn(4, 4, 4), (-1, -2),
     (-2, -1), 'backward', ValueError),
    ('test_n_zero', np.random.randn(4, 4, 4) + 1j * np.random.randn(4, 4, 4),
     (0, 0), (-2, -1), 'backward', ValueError),
    ('test_n_type', np.random.randn(4, 4, 4) + 1j * np.random.randn(4, 4, 4), 3,
     -1, 'backward',
     ValueError), ('test_n_axis_dim',
                   np.random.randn(4, 4, 4) + 1j * np.random.randn(4, 4, 4),
                   (1, 2), (-3, -2, -1), 'backward', ValueError),
    ('test_axis_out_of_range', np.random.randn(4) + 1j * np.random.randn(4),
     None, (10, 20), 'backward', ValueError), (
         'test_axis_type', np.random.randn(4) + 1j * np.random.randn(4), None,
         1, 'backward',
         ValueError), ('test_norm_not_in_enum_value',
                       np.random.randn(4, 4) + 1j * np.random.randn(4, 4), None,
                       None, 'random', ValueError)
])
class TestIrfftnException(unittest.TestCase):
    '''Test irfftn with buoudary condition
    Test case include:
    - non complex input
    - n out of range
    - axis out of range
    - norm out of range
    - the dimensions of n and axis are different
    '''

    def test_irfftn(self):
        with self.assertRaises(self.expect_exception):
            paddle.tensor.fft.irfftn(self.x, self.n, self.axis, self.norm)


if __name__ == '__main__':
    unittest.main()
