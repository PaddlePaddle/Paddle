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
import scipy.fft

paddle.set_default_dtype('float64')

TEST_CASE_NAME = 'test_case'


def setUpModule():
    global rtol
    global atol
    # All test case will use float64 for compare percision, refs:
    # https://github.com/PaddlePaddle/Paddle/wiki/Upgrade-OP-Precision-to-Float64
    rtol = {'float32': 1e-06, 'float64': 1e-7}
    atol = {'float32': 0.0, 'float64': 0.0}


def tearDownModule():
    pass


def rand_x(dims=1,
           dtype='float64',
           min_dim_len=1,
           max_dim_len=10,
           complex=False):
    shape = [np.random.randint(min_dim_len, max_dim_len) for i in range(dims)]
    if complex:
        return np.random.randn(*shape).astype(dtype) + 1.j * np.random.randn(
            *shape).astype(dtype)
    else:
        return np.random.randn(*shape).astype(dtype)


def parameterize(attrs, input_values=None):

    if isinstance(attrs, str):
        attrs = [attrs]
    input_dicts = (attrs if input_values is None else
                   [dict(zip(attrs, vals)) for vals in input_values])

    def decorator(base_class):
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


# yapf: disable
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
     np.random.randn(4, 4, 4) + 1j *
     np.random.randn(4, 4, 4), [1, 2], (-2, -1),
     "backward"),
    ('test_n_smaller_than_input_length',
     np.random.randn(4, 4, 4) + 1j *
     np.random.randn(4, 4, 4), [2, 1], (-2, -1),
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
     np.random.randn(4, 4, 4) + 1j *
     np.random.randn(4, 4, 4), (2, 1), (-2, -1),
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
                       np.random.randn(4, 4) + 1j *
                       np.random.randn(4, 4), None,
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
                       np.random.randn(4, 4) + 1j *
                       np.random.randn(4, 4), None,
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
                       np.random.randn(4, 4) + 1j *
                       np.random.randn(4, 4), None,
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
                       np.random.randn(4, 4) + 1j *
                       np.random.randn(4, 4), None,
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


@parameterize(
    (TEST_CASE_NAME, 'x', 'n', 'axis', 'norm'),
    [('test_x_float64', rand_x(5, np.float64), None, -1, 'backward'), (
        'test_n_grater_than_input_length', rand_x(
            5, max_dim_len=5), 11, -1, 'backward'),
     ('test_n_smaller_than_input_length', rand_x(
         5, min_dim_len=5), 3, -1,
      'backward'), ('test_axis_not_last', rand_x(5), None, 3, 'backward'),
     ('test_norm_forward', rand_x(5), None, 3, 'forward'),
     ('test_norm_ortho', rand_x(5), None, 3, 'ortho')])
class TestRfft(unittest.TestCase):
    def test_rfft(self):
        self.assertTrue(
            np.allclose(
                np.fft.rfft(self.x, self.n, self.axis, self.norm),
                paddle.tensor.fft.rfft(
                    paddle.to_tensor(self.x), self.n, self.axis, self.norm),
                rtol=rtol.get(str(self.x.dtype)),
                atol=atol.get(str(self.x.dtype))))


@parameterize(
    (TEST_CASE_NAME, 'x', 'n', 'axis', 'norm', 'expect_exception'),
    [('test_n_nagative', rand_x(2), -1, -1, 'backward', ValueError),
     ('test_n_zero', rand_x(2), 0, -1, 'backward', ValueError),
     ('test_axis_out_of_range', rand_x(1), None, 10, 'backward',
      ValueError), ('test_axis_with_array', rand_x(1), None, (0, 1),
                    'backward', ValueError), ('test_norm_not_in_enum_value',
                                              rand_x(2), None, -1,
                                              'random', ValueError)])
class TestRfftException(unittest.TestCase):
    def test_rfft(self):
        with self.assertRaises(self.expect_exception):
            paddle.tensor.fft.rfft(self.x, self.n, self.axis, self.norm)


@parameterize(
    (TEST_CASE_NAME, 'x', 'n', 'axis', 'norm'), [
        ('test_x_float64', rand_x(5), None, (0, 1), 'backward'),
        ('test_n_grater_input_length', rand_x(
            5, max_dim_len=5), (6, 6), (0, 1), 'backward'),
        ('test_n_smaller_than_input_length', rand_x(
            5, min_dim_len=5), (4, 4), (0, 1), 'backward'),
        ('test_axis_random', rand_x(5), None, (1, 2), 'backward'),
        ('test_axis_none', rand_x(5), None, None, 'backward'),
        ('test_norm_forward', rand_x(5), None, (0, 1), 'forward'),
        ('test_norm_ortho', rand_x(5), None, (0, 1), 'ortho'),
    ])
class TestRfft2(unittest.TestCase):
    def test_rfft2(self):
        self.assertTrue(
            np.allclose(
                np.fft.rfft2(self.x, self.n, self.axis, self.norm),
                paddle.fft.rfft2(
                    paddle.to_tensor(self.x), self.n, self.axis, self.norm),
                rtol=rtol.get(str(self.x.dtype)),
                atol=atol.get(str(self.x.dtype))))


@parameterize((TEST_CASE_NAME, 'x', 'n', 'axis', 'norm', 'expect_exception'), [
    ('test_x_complex_input', rand_x(2,  complex=True), None, (0, 1), 'backward',
     ValueError),
    # ('test_x_not_tensor', [0, 1], None, (0, 1), 'backward', ValueError),
    ('test_x_1dim_tensor', rand_x(1), None, (0, 1), 'backward', ValueError),
    ('test_n_nagative', rand_x(2), -1, (0, 1), 'backward', ValueError),
    ('test_n_zero', rand_x(2), 0, (0, 1), 'backward', ValueError),
    ('test_axis_out_of_range', rand_x(2), None, (0, 1, 2), 'backward',
     ValueError),
    ('test_axis_with_array', rand_x(1), None, (0, 1), 'backward',
     ValueError),
    ('test_axis_not_sequence', rand_x(5), None, -10, 'backward', ValueError),
    ('test_norm_not_enum', rand_x(2), None, -1, 'random', ValueError)])
class TestRfft2Exception(unittest.TestCase):
    def test_rfft(self):
        with self.assertRaises(self.expect_exception):
            paddle.fft.rfft2(self.x, self.n, self.axis, self.norm)


@parameterize((TEST_CASE_NAME, 'x', 'n', 'axis', 'norm'), [
    ('test_x_float64', rand_x(5, np.float64), None, None, 'backward'),
    ('test_n_grater_input_length', rand_x(5, max_dim_len=5), (6, 6), (1, 2),
     'backward'),
    ('test_n_smaller_input_length', rand_x(5, min_dim_len=5), (3, 3), (1, 2),
     'backward'),
    ('test_axis_not_default', rand_x(5), None, (1, 2),
     'backward'), ('test_norm_forward', rand_x(5), None, None, 'forward'),
    ('test_norm_ortho', rand_x(5), None, None, 'ortho')])
class TestRfftn(unittest.TestCase):
    def test_rfftn(self):
        self.assertTrue(
            np.allclose(
                np.fft.rfftn(self.x, self.n, self.axis, self.norm),
                paddle.fft.rfftn(
                    paddle.to_tensor(self.x), self.n, self.axis, self.norm),
                rtol=rtol.get(str(self.x.dtype)),
                atol=atol.get(str(self.x.dtype))))


@parameterize((TEST_CASE_NAME, 'x', 'n', 'axis', 'norm', 'expect_exception'), [
    ('test_x_complex', rand_x(4, complex=True), None, None, 'backward',
     ValueError),
    ('test_n_nagative', rand_x(4), (-1, -1), (1, 2), 'backward',
     ValueError),
    ('test_n_not_sequence', rand_x(4), -1, None, 'backward',
     ValueError),
    ('test_n_zero', rand_x(4), 0, None, 'backward', ValueError),
    ('test_axis_out_of_range', rand_x(1), None, [0, 1], 'backward',
     ValueError),
    ('test_norm_not_in_enum', rand_x(2), None, -1,
     'random', ValueError)])
class TestRfftnException(unittest.TestCase):
    def test_rfft(self):
        with self.assertRaises(self.expect_exception):
            paddle.fft.rfftn(self.x, self.n, self.axis, self.norm)


@parameterize(
    (TEST_CASE_NAME, 'x', 'n', 'axis', 'norm'),
    [('test_x_float64', rand_x(5, np.float64), None, -1, 'backward'), (
        'test_n_grater_than_input_length', rand_x(
            5, max_dim_len=5), 11, -1, 'backward'),
     ('test_n_smaller_than_input_length', rand_x(
         5, min_dim_len=5), 3, -1,
      'backward'), ('test_axis_not_last', rand_x(5), None, 3, 'backward'),
     ('test_norm_forward', rand_x(5), None, 3, 'forward'),
     ('test_norm_ortho', rand_x(5), None, 3, 'ortho')])
class TestIhfft(unittest.TestCase):
    def test_ihfft(self):
        self.assertTrue(
            np.allclose(
                np.fft.ihfft(self.x, self.n, self.axis, self.norm),
                paddle.fft.ihfft(
                    paddle.to_tensor(self.x), self.n, self.axis, self.norm),
                rtol=rtol.get(str(self.x.dtype)),
                atol=atol.get(str(self.x.dtype))))


@parameterize((TEST_CASE_NAME, 'x', 'n', 'axis', 'norm', 'expect_exception'), [
    ('test_n_nagative', rand_x(2), -1, -1, 'backward', ValueError),
    ('test_n_zero', rand_x(2), 0, -1, 'backward', ValueError),
    ('test_axis_out_of_range', rand_x(1), None, 10, 'backward', ValueError),
    ('test_axis_with_array', rand_x(1), None, (0, 1), 'backward', ValueError),
    ('test_norm_not_in_enum_value', rand_x(2), None, -1, 'random', ValueError)])
class TestIhfftException(unittest.TestCase):
    def test_ihfft(self):
        with self.assertRaises(self.expect_exception):
            paddle.fft.ihfft(self.x, self.n, self.axis, self.norm)


@parameterize(
    (TEST_CASE_NAME, 'x', 'n', 'axis', 'norm'), [
        ('test_x_float64', rand_x(5), None, (0, 1), 'backward'),
        ('test_n_grater_input_length', rand_x(
            5,  max_dim_len=5), (11, 11), (0, 1), 'backward'),
        ('test_n_smaller_than_input_length', rand_x(
            5,  min_dim_len=5), (1, 1), (0, 1), 'backward'),
        ('test_axis_random', rand_x(5), None, (1, 2), 'backward'),
        ('test_axis_none', rand_x(5), None, None, 'backward'),
        ('test_norm_forward', rand_x(5), None, (0, 1), 'forward'),
        ('test_norm_ortho', rand_x(5), None, (0, 1), 'ortho'),
    ])
class TestIhfft2(unittest.TestCase):
    def test_ihfft2(self):
        self.assertTrue(
            np.allclose(
                scipy.fft.ihfft2(self.x, self.n, self.axis, self.norm),
                paddle.fft.ihfft2(
                    paddle.to_tensor(self.x), self.n, self.axis, self.norm),
                rtol=rtol.get(str(self.x.dtype)),
                atol=atol.get(str(self.x.dtype))))


@parameterize((TEST_CASE_NAME, 'x', 'n', 'axis', 'norm', 'expect_exception'), [
    ('test_x_complex_input', rand_x(2, complex=True), None, (0, 1), None,
     ValueError),
    # ('test_x_not_tensor', [0, 1], None, (0, 1), None, ValueError),
    ('test_x_1dim_tensor', rand_x(1), None, (0, 1), None, ValueError),
    ('test_n_nagative', rand_x(2), -1, (0, 1), 'backward',
     ValueError),
    ('test_n_len_not_equal_axis', rand_x(5, max_dim_len=5), 11, (0, 1),
     'backward', ValueError),
    ('test_n_zero', rand_x(2), (0, 0), (0, 1), 'backward',
     ValueError),
    ('test_axis_out_of_range', rand_x(2), None,
     (0, 1, 2), 'backward', ValueError),
    ('test_axis_with_array', rand_x(1), None, (0, 1), 'backward',
     ValueError),
    ('test_axis_not_sequence', rand_x(5), None, -10, 'backward', ValueError),
    ('test_norm_not_enum', rand_x(2), None, -1, 'random', ValueError)])
class TestIhfft2Exception(unittest.TestCase):
    def test_rfft(self):
        with self.assertRaises(self.expect_exception):
            paddle.fft.ihfft2(self.x, self.n, self.axis, self.norm)


@parameterize((TEST_CASE_NAME, 'x', 'n', 'axis', 'norm'), [
    ('test_x_float64', rand_x(5, np.float64), None, None, 'backward'),
    ('test_n_grater_input_length', rand_x(5, max_dim_len=5), (11, 11), (0, 1),
     'backward'),
    ('test_n_smaller_input_length', rand_x(5, min_dim_len=5), (1, 1), (0, 1),
     'backward'),
    ('test_axis_not_default', rand_x(5), None, (1, 2),
     'backward'),
    ('test_norm_forward', rand_x(5), None, None, 'forward'),
    ('test_norm_ortho', rand_x(5), None, None, 'ortho')])
class TestIhfftn(unittest.TestCase):
    def test_rfftn(self):
        self.assertTrue(
            np.allclose(
                scipy.fft.ihfftn(self.x, self.n, self.axis, self.norm),
                paddle.fft.ihfftn(
                    paddle.to_tensor(self.x), self.n, self.axis, self.norm),
                rtol=rtol.get(str(self.x.dtype)),
                atol=atol.get(str(self.x.dtype))))


@parameterize(
    (TEST_CASE_NAME, 'x', 'n', 'axis', 'norm', 'expect_exception'),
    [('test_x_complex', rand_x(
        4, complex=True), None, None, 'backward',
      ValueError), ('test_n_nagative', rand_x(4), -1, None,
                    'backward', ValueError),
     ('test_n_zero', rand_x(4), 0, None, 'backward', ValueError),
     ('test_axis_out_of_range', rand_x(1), None, [0, 1], 'backward',
      ValueError), ('test_norm_not_in_enum', rand_x(2), None, -1,
                    'random', ValueError)])
class TestIhfftnException(unittest.TestCase):
    def test_rfft(self):
        with self.assertRaises(self.expect_exception):
            paddle.fft.ihfftn(self.x, self.n, self.axis, self.norm)


if __name__ == '__main__':
    unittest.main()


# yapf: enable
