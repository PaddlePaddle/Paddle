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

import contextlib
import re
import sys
import unittest

import numpy as np
import paddle
import scipy.fft

DEVICES = [paddle.CPUPlace()]
if paddle.is_compiled_with_cuda():
    DEVICES.append(paddle.CUDAPlace(0))

TEST_CASE_NAME = 'suffix'
# All test case will use float64 for compare percision, refs:
# https://github.com/PaddlePaddle/Paddle/wiki/Upgrade-OP-Precision-to-Float64
RTOL = {
    'float32': 1e-03,
    'complex64': 1e-3,
    'float64': 1e-7,
    'complex128': 1e-7
}
ATOL = {'float32': 0.0, 'complex64': 0, 'float64': 0.0, 'complex128': 0}


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


def place(devices, key='place'):
    def decorate(cls):
        module = sys.modules[cls.__module__].__dict__
        raw_classes = {
            k: v
            for k, v in module.items() if k.startswith(cls.__name__)
        }

        for raw_name, raw_cls in raw_classes.items():
            for d in devices:
                test_cls = dict(raw_cls.__dict__)
                test_cls.update({key: d})
                new_name = raw_name + '.' + d.__class__.__name__
                module[new_name] = type(new_name, (raw_cls, ), test_cls)
            del module[raw_name]
        return cls

    return decorate


def parameterize(fields, values=None):

    fields = [fields] if isinstance(fields, str) else fields
    params = [dict(zip(fields, vals)) for vals in values]

    def decorate(cls):
        test_cls_module = sys.modules[cls.__module__].__dict__
        for k, v in enumerate(params):
            test_cls = dict(cls.__dict__)
            test_cls.update(v)
            name = cls.__name__ + str(k)
            name = name + '.' + v.get('suffix') if v.get('suffix') else name

            test_cls_module[name] = type(name, (cls, ), test_cls)

        for m in list(cls.__dict__):
            if m.startswith("test"):
                delattr(cls, m)
        return cls

    return decorate


@place(DEVICES)
@parameterize(
    (TEST_CASE_NAME, 'x', 'n', 'axis', 'norm'),
    [('test_x_float64', rand_x(5, np.float64), None, -1, 'backward'),
     ('test_x_complex', rand_x(
         5, complex=True), None, -1,
      'backward'), ('test_n_grater_input_length', rand_x(
          5, max_dim_len=5), 11, -1,
                    'backward'), ('test_n_smaller_than_input_length', rand_x(
                        5, min_dim_len=5, complex=True), 3, -1, 'backward'),
     ('test_axis_not_last', rand_x(5), None, 3, 'backward'),
     ('test_norm_forward', rand_x(5), None, 3, 'forward'),
     ('test_norm_ortho', rand_x(5), None, 3, 'ortho')])
class TestFft(unittest.TestCase):
    def test_fft(self):
        """Test fft with norm condition
        """
        with paddle.fluid.dygraph.guard(self.place):
            self.assertTrue(
                np.allclose(
                    scipy.fft.fft(self.x, self.n, self.axis, self.norm),
                    paddle.fft.fft(
                        paddle.to_tensor(self.x), self.n, self.axis, self.norm),
                    rtol=RTOL.get(str(self.x.dtype)),
                    atol=ATOL.get(str(self.x.dtype))))


@place(DEVICES)
@parameterize(
    (TEST_CASE_NAME, 'x', 'n', 'axis', 'norm'),
    [('test_x_float64', rand_x(5, np.float64), None, -1, 'backward'),
     ('test_x_complex', rand_x(
         5, complex=True), None, -1,
      'backward'), ('test_n_grater_input_length', rand_x(
          5, max_dim_len=5), 11, -1,
                    'backward'), ('test_n_smaller_than_input_length', rand_x(
                        5, min_dim_len=5, complex=True), 3, -1, 'backward'),
     ('test_axis_not_last', rand_x(5), None, 3, 'backward'),
     ('test_norm_forward', rand_x(5), None, 3, 'forward'),
     ('test_norm_ortho', rand_x(5), None, 3, 'ortho')])
class TestIfft(unittest.TestCase):
    def test_fft(self):
        """Test ifft with norm condition
        """
        with paddle.fluid.dygraph.guard(self.place):
            self.assertTrue(
                np.allclose(
                    scipy.fft.ifft(self.x, self.n, self.axis, self.norm),
                    paddle.fft.ifft(
                        paddle.to_tensor(self.x), self.n, self.axis, self.norm),
                    rtol=RTOL.get(str(self.x.dtype)),
                    atol=ATOL.get(str(self.x.dtype))))


@place(DEVICES)
@parameterize((TEST_CASE_NAME, 'x', 'n', 'axis', 'norm', 'expect_exception'), [
    ('test_n_nagative', rand_x(2), -1, -1, 'backward', ValueError),
    ('test_n_zero', rand_x(2), 0, -1, 'backward', ValueError),
    ('test_axis_out_of_range', rand_x(1), None, 10, 'backward', ValueError),
    ('test_axis_with_array', rand_x(1), None, (0, 1), 'backward', ValueError),
    ('test_norm_not_in_enum_value', rand_x(2), None, -1, 'random', ValueError)
])
class TestFftException(unittest.TestCase):
    def test_fft(self):
        """Test fft with buoudary condition
        Test case include:
        - n out of range
        - axis out of range
        - axis type error
        - norm out of range
        """
        with self.assertRaises(self.expect_exception):
            paddle.fft.fft(
                paddle.to_tensor(self.x), self.n, self.axis, self.norm)


@place(DEVICES)
@parameterize(
    (TEST_CASE_NAME, 'x', 'n', 'axis', 'norm'), [
        ('test_x_float64', rand_x(5), None, (0, 1), 'backward'),
        ('test_x_complex128', rand_x(
            5, complex=True), None, (0, 1), 'backward'),
        ('test_n_grater_input_length', rand_x(
            5, max_dim_len=5), (6, 6), (0, 1), 'backward'),
        ('test_n_smaller_than_input_length', rand_x(
            5, min_dim_len=5, complex=True), (4, 4), (0, 1), 'backward'),
        ('test_axis_random', rand_x(5), None, (1, 2), 'backward'),
        ('test_axis_none', rand_x(5), None, None, 'backward'),
        ('test_norm_forward', rand_x(5), None, (0, 1), 'forward'),
        ('test_norm_ortho', rand_x(5), None, (0, 1), 'ortho'),
    ])
class TestFft2(unittest.TestCase):
    def test_fft2(self):
        """Test fft2 with norm condition
        """
        with paddle.fluid.dygraph.guard(self.place):
            self.assertTrue(
                np.allclose(
                    scipy.fft.fft2(self.x, self.n, self.axis, self.norm),
                    paddle.fft.fft2(
                        paddle.to_tensor(self.x), self.n, self.axis, self.norm),
                    rtol=RTOL.get(str(self.x.dtype)),
                    atol=ATOL.get(str(self.x.dtype))))


@place(DEVICES)
@parameterize(
    (TEST_CASE_NAME, 'x', 'n', 'axis', 'norm', 'expect_exception'),
    [('test_x_complex_input', rand_x(
        2, complex=True), None, (0, 1), None,
      ValueError), ('test_x_1dim_tensor', rand_x(1), None, (0, 1), None,
                    ValueError), ('test_n_nagative', rand_x(2), -1, (0, 1),
                                  'backward', ValueError),
     ('test_n_len_not_equal_axis', rand_x(
         5, max_dim_len=5), 11, (0, 1), 'backward',
      ValueError), ('test_n_zero', rand_x(2), (0, 0), (0, 1), 'backward',
                    ValueError), ('test_axis_out_of_range', rand_x(2), None,
                                  (0, 1, 2), 'backward', ValueError),
     ('test_axis_with_array', rand_x(1), None, (0, 1), 'backward', ValueError),
     ('test_axis_not_sequence', rand_x(5), None, -10, 'backward', ValueError),
     ('test_norm_not_enum', rand_x(2), None, -1, 'random', ValueError)])
class TestFft2Exception(unittest.TestCase):
    def test_fft2(self):
        """Test fft2 with buoudary condition
        Test case include:
        - input type error
        - input dim error
        - n out of range
        - axis out of range
        - axis type error
        - norm out of range
        """
        with paddle.fluid.dygraph.guard(self.place):
            with self.assertRaises(self.expect_exception):
                paddle.fft.fft2(
                    paddle.to_tensor(self.x), self.n, self.axis, self.norm)


@place(DEVICES)
@parameterize(
    (TEST_CASE_NAME, 'x', 'n', 'axis', 'norm'),
    [('test_x_float64', rand_x(5, np.float64), None, None, 'backward'),
     ('test_x_complex128', rand_x(
         5, complex=True), None, None,
      'backward'), ('test_n_grater_input_length', rand_x(
          5, max_dim_len=5), (6, 6), (1, 2), 'backward'), (
              'test_n_smaller_input_length', rand_x(
                  5, min_dim_len=5, complex=True), (3, 3), (1, 2), 'backward'),
     ('test_axis_not_default', rand_x(5), None, (1, 2),
      'backward'), ('test_norm_forward', rand_x(5), None, None, 'forward'),
     ('test_norm_ortho', rand_x(5), None, None, 'ortho')])
class TestFftn(unittest.TestCase):
    def test_fftn(self):
        """Test fftn with norm condition
        """
        with paddle.fluid.dygraph.guard(self.place):
            np.testing.assert_allclose(
                scipy.fft.fftn(self.x, self.n, self.axis, self.norm),
                paddle.fft.fftn(
                    paddle.to_tensor(self.x), self.n, self.axis, self.norm),
                rtol=RTOL.get(str(self.x.dtype)),
                atol=ATOL.get(str(self.x.dtype)))


@place(DEVICES)
@parameterize(
    (TEST_CASE_NAME, 'x', 'n', 'axis', 'norm'),
    [('test_x_float64', rand_x(5, np.float64), None, None, 'backward'),
     ('test_x_complex128', rand_x(
         5, complex=True), None, None,
      'backward'), ('test_n_grater_input_length', rand_x(
          5, max_dim_len=5), (6, 6), (1, 2), 'backward'), (
              'test_n_smaller_input_length', rand_x(
                  5, min_dim_len=5, complex=True), (3, 3), (1, 2), 'backward'),
     ('test_axis_not_default', rand_x(5), None, (1, 2),
      'backward'), ('test_norm_forward', rand_x(5), None, None, 'forward'),
     ('test_norm_ortho', rand_x(5), None, None, 'ortho')])
class TestIFftn(unittest.TestCase):
    def test_ifftn(self):
        """Test ifftn with norm condition
        """
        with paddle.fluid.dygraph.guard(self.place):
            np.testing.assert_allclose(
                scipy.fft.ifftn(self.x, self.n, self.axis, self.norm),
                paddle.fft.ifftn(
                    paddle.to_tensor(self.x), self.n, self.axis, self.norm),
                rtol=RTOL.get(str(self.x.dtype)),
                atol=ATOL.get(str(self.x.dtype)))


@place(DEVICES)
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
    def test_hfft(self):
        """Test hfft with norm condition
        """
        with paddle.fluid.dygraph.guard(self.place):
            np.testing.assert_allclose(
                scipy.fft.hfft(self.x, self.n, self.axis, self.norm),
                paddle.fft.hfft(
                    paddle.to_tensor(self.x), self.n, self.axis, self.norm),
                rtol=1e-5,
                atol=0)


@place(DEVICES)
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
    def test_irfft(self):
        """Test irfft with norm condition
        """
        with paddle.fluid.dygraph.guard(self.place):
            np.testing.assert_allclose(
                scipy.fft.irfft(self.x, self.n, self.axis, self.norm),
                paddle.fft.irfft(
                    paddle.to_tensor(self.x), self.n, self.axis, self.norm),
                rtol=1e-5,
                atol=0)


@place(DEVICES)
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
class TestIrfftn(unittest.TestCase):
    def test_irfftn(self):
        """Test irfftn with norm condition
        """
        with paddle.fluid.dygraph.guard(self.place):
            np.testing.assert_allclose(
                scipy.fft.irfftn(self.x, self.n, self.axis, self.norm),
                paddle.fft.irfftn(
                    paddle.to_tensor(self.x), self.n, self.axis, self.norm),
                rtol=1e-5,
                atol=0)


@place(DEVICES)
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
class TestHfftn(unittest.TestCase):
    def test_hfftn(self):
        """Test hfftn with norm condition
        """
        with paddle.fluid.dygraph.guard(self.place):
            np.testing.assert_allclose(
                scipy.fft.hfftn(self.x, self.n, self.axis, self.norm),
                paddle.fft.hfftn(
                    paddle.to_tensor(self.x), self.n, self.axis, self.norm),
                rtol=1e-5,
                atol=0)


@place(DEVICES)
@parameterize((TEST_CASE_NAME, 'x', 's', 'axis', 'norm'), [
    ('test_x_complex128',
     (np.random.randn(4, 4, 4) + 1j * np.random.randn(4, 4, 4)
      ).astype(np.complex128), None, (-2, -1), "backward"),
    ('test_with_s', np.random.randn(4, 4, 4) + 1j * np.random.randn(4, 4, 4),
     [2, 2], (-2, -1), "backward", ValueError),
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
class TestHfft2(unittest.TestCase):
    def test_hfft2(self):
        """Test hfft2 with norm condition
        """
        with paddle.fluid.dygraph.guard(self.place):
            np.testing.assert_allclose(
                scipy.fft.hfft2(self.x, self.s, self.axis, self.norm),
                paddle.fft.hfft2(
                    paddle.to_tensor(self.x), self.s, self.axis, self.norm),
                rtol=1e-5,
                atol=0)


@place(DEVICES)
@parameterize((TEST_CASE_NAME, 'x', 's', 'axis', 'norm'), [
    ('test_x_complex128',
     (np.random.randn(4, 4, 4) + 1j * np.random.randn(4, 4, 4)
      ).astype(np.complex128), None, (-2, -1), "backward"),
    ('test_n_equal_input_length',
     np.random.randn(4, 4, 4) + 1j * np.random.randn(4, 4, 4), (4, 6), (-2, -1),
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
    def test_irfft2(self):
        """Test irfft2 with norm condition
        """
        with paddle.fluid.dygraph.guard(self.place):
            np.testing.assert_allclose(
                scipy.fft.irfft2(self.x, self.s, self.axis, self.norm),
                paddle.fft.irfft2(
                    paddle.to_tensor(self.x), self.s, self.axis, self.norm),
                rtol=1e-5,
                atol=0)


@place(DEVICES)
@parameterize((TEST_CASE_NAME, 'x', 'n', 'axis', 'norm', 'expect_exception'), [(
    'test_bool_input',
    (np.random.randn(4, 4, 4) + 1j * np.random.randn(4, 4, 4)).astype(np.bool8),
    None, -1, 'backward', NotImplementedError), (
        'test_n_nagative',
        np.random.randn(4, 4, 4) + 1j * np.random.randn(4, 4, 4), -1, -1,
        'backward', ValueError), (
            'test_n_zero', np.random.randn(4, 4) + 1j * np.random.randn(4, 4),
            0, -1, 'backward', ValueError), (
                'test_n_type',
                np.random.randn(4, 4, 4) + 1j * np.random.randn(4, 4, 4),
                (1, 2, 3), -1, 'backward', ValueError), (
                    'test_axis_out_of_range',
                    np.random.randn(4) + 1j * np.random.randn(4), None, 10,
                    'backward', ValueError), (
                        'test_axis_with_array',
                        np.random.randn(4) + 1j * np.random.randn(4), None,
                        (0, 1), 'backward', ValueError), (
                            'test_norm_not_in_enum_value',
                            np.random.randn(4, 4) + 1j * np.random.randn(4, 4),
                            None, -1, 'random', ValueError)])
class TestHfftException(unittest.TestCase):
    def test_hfft(self):
        """Test hfft with buoudary condition
        Test case include:
        Test case include:
        - n out of range
        - n type error
        - axis out of range
        - axis type error
        - norm out of range
        """
        with paddle.fluid.dygraph.guard(self.place):
            with self.assertRaises(self.expect_exception):
                paddle.fft.hfft(
                    paddle.to_tensor(self.x), self.n, self.axis, self.norm)


@place(DEVICES)
@parameterize(
    (TEST_CASE_NAME, 'x', 'n', 'axis', 'norm', 'expect_exception'),
    [('test_n_nagative',
      np.random.randn(4, 4, 4) + 1j * np.random.randn(4, 4, 4), -1, -1,
      'backward', ValueError),
     ('test_n_zero', np.random.randn(4, 4) + 1j * np.random.randn(4, 4), 0, -1,
      'backward', ValueError),
     ('test_n_type', np.random.randn(4, 4, 4) + 1j * np.random.randn(4, 4, 4),
      (1, 2), -1, 'backward', ValueError),
     ('test_axis_out_of_range', np.random.randn(4) + 1j * np.random.randn(4),
      None, 10, 'backward', ValueError),
     ('test_axis_with_array', np.random.randn(4) + 1j * np.random.randn(4),
      None, (0, 1), 'backward',
      ValueError), ('test_norm_not_in_enum_value',
                    np.random.randn(4, 4) + 1j * np.random.randn(4, 4), None,
                    None, 'random', ValueError)])
class TestIrfftException(unittest.TestCase):
    def test_irfft(self):
        """
        Test irfft with buoudary condition
        Test case include:
        - n out of range
        - n type error
        - axis type error
        - axis out of range
        - norm out of range
        """
        with paddle.fluid.dygraph.guard(self.place):
            with self.assertRaises(self.expect_exception):
                paddle.fft.irfft(
                    paddle.to_tensor(self.x), self.n, self.axis, self.norm)


@place(DEVICES)
@parameterize(
    (TEST_CASE_NAME, 'x', 'n', 'axis', 'norm', 'expect_exception'),
    [('test_bool_input',
      (np.random.randn(4, 4, 4) + 1j * np.random.randn(4, 4, 4)
       ).astype(np.bool8), None, (-2, -1), 'backward', NotImplementedError),
     ('test_n_nagative',
      np.random.randn(4, 4, 4) + 1j * np.random.randn(4, 4, 4), (-1, -2),
      (-2, -1), 'backward', ValueError),
     ('test_n_zero', np.random.randn(4, 4, 4) + 1j * np.random.randn(4, 4, 4),
      (0, 0), (-2, -1), 'backward', ValueError),
     ('test_n_type', np.random.randn(4, 4, 4) + 1j * np.random.randn(4, 4, 4),
      3, None, 'backward', ValueError),
     ('test_n_axis_dim',
      np.random.randn(4, 4, 4) + 1j * np.random.randn(4, 4, 4), (1, 2), (-1),
      'backward', ValueError), ('test_axis_out_of_range',
                                np.random.randn(4) + 1j * np.random.randn(4),
                                None, (1, 2), 'backward', ValueError),
     ('test_axis_type', np.random.randn(4) + 1j * np.random.randn(4), None, -1,
      'backward',
      ValueError), ('test_norm_not_in_enum_value',
                    np.random.randn(4, 4) + 1j * np.random.randn(4, 4), None,
                    None, 'random', ValueError)])
class TestHfft2Exception(unittest.TestCase):
    def test_hfft2(self):
        """
        Test hfft2 with buoudary condition
        Test case include:
        - input type error
        - n type error
        - n out of range
        - axis out of range
        - the dimensions of n and axis are different
        - norm out of range
        """
        with paddle.fluid.dygraph.guard(self.place):
            with self.assertRaises(self.expect_exception):
                paddle.fft.hfft2(
                    paddle.to_tensor(self.x), self.n, self.axis, self.norm)


@place(DEVICES)
@parameterize(
    (TEST_CASE_NAME, 'x', 'n', 'axis', 'norm', 'expect_exception'),
    [('test_n_nagative',
      np.random.randn(4, 4, 4) + 1j * np.random.randn(4, 4, 4), (-1, -2),
      (-2, -1), 'backward', ValueError),
     ('test_zero_point',
      np.random.randn(4, 4, 1) + 1j * np.random.randn(4, 4, 1), None, (-2, -1),
      "backward", ValueError),
     ('test_n_zero', np.random.randn(4, 4, 4) + 1j * np.random.randn(4, 4, 4),
      (0, 0), (-2, -1), 'backward', ValueError),
     ('test_n_type', np.random.randn(4, 4, 4) + 1j * np.random.randn(4, 4, 4),
      3, -1, 'backward',
      ValueError), ('test_n_axis_dim',
                    np.random.randn(4, 4, 4) + 1j * np.random.randn(4, 4, 4),
                    (1, 2), (-3, -2, -1), 'backward', ValueError),
     ('test_axis_out_of_range', np.random.randn(4) + 1j * np.random.randn(4),
      None, (1, 2), 'backward', ValueError), (
          'test_axis_type', np.random.randn(4) + 1j * np.random.randn(4), None,
          1, 'backward',
          ValueError), ('test_norm_not_in_enum_value',
                        np.random.randn(4, 4) + 1j * np.random.randn(4, 4),
                        None, None, 'random', ValueError)])
class TestIrfft2Exception(unittest.TestCase):
    def test_irfft2(self):
        """
        Test irfft2 with buoudary condition
        Test case include:
        - input type error
        - n type error
        - n out of range
        - axis out of range
        - the dimensions of n and axis are different
        - norm out of range
        """
        with paddle.fluid.dygraph.guard(self.place):
            with self.assertRaises(self.expect_exception):
                paddle.fft.irfft2(
                    paddle.to_tensor(self.x), self.n, self.axis, self.norm)


@place(DEVICES)
@parameterize(
    (TEST_CASE_NAME, 'x', 'n', 'axis', 'norm', 'expect_exception'),
    [('test_bool_input',
      (np.random.randn(4, 4, 4) + 1j * np.random.randn(4, 4, 4)
       ).astype(np.bool8), None, (-2, -1), 'backward', NotImplementedError),
     ('test_n_nagative',
      np.random.randn(4, 4, 4) + 1j * np.random.randn(4, 4, 4), (-1, -2),
      (-2, -1), 'backward', ValueError),
     ('test_n_zero', np.random.randn(4, 4, 4) + 1j * np.random.randn(4, 4, 4),
      (0, 0), (-2, -1), 'backward', ValueError),
     ('test_n_type', np.random.randn(4, 4, 4) + 1j * np.random.randn(4, 4, 4),
      3, -1, 'backward', ValueError),
     ('test_n_axis_dim',
      np.random.randn(4, 4, 4) + 1j * np.random.randn(4, 4, 4),
      (1, 2), (-3, -2, -1), 'backward',
      ValueError), ('test_axis_out_of_range',
                    np.random.randn(4) + 1j * np.random.randn(4), None,
                    (10, 20), 'backward', ValueError),
     ('test_axis_type', np.random.randn(4) + 1j * np.random.randn(4), None, 1,
      'backward',
      ValueError), ('test_norm_not_in_enum_value',
                    np.random.randn(4, 4) + 1j * np.random.randn(4, 4), None,
                    None, 'random', ValueError)])
class TestHfftnException(unittest.TestCase):
    def test_hfftn(self):
        """Test hfftn with buoudary condition
        Test case include:
        - input type error
        - n type error
        - n out of range
        - axis out of range
        - the dimensions of n and axis are different
        - norm out of range
        """
        with paddle.fluid.dygraph.guard(self.place):
            with self.assertRaises(self.expect_exception):
                paddle.fft.hfftn(
                    paddle.to_tensor(self.x), self.n, self.axis, self.norm)


@place(DEVICES)
@parameterize(
    (TEST_CASE_NAME, 'x', 'n', 'axis', 'norm', 'expect_exception'),
    [('test_n_nagative',
      np.random.randn(4, 4, 4) + 1j * np.random.randn(4, 4, 4), (-1, -2),
      (-2, -1), 'backward', ValueError),
     ('test_n_zero', np.random.randn(4, 4, 4) + 1j * np.random.randn(4, 4, 4),
      (0, 0), (-2, -1), 'backward', ValueError),
     ('test_n_type', np.random.randn(4, 4, 4) + 1j * np.random.randn(4, 4, 4),
      3, -1, 'backward',
      ValueError), ('test_n_axis_dim',
                    np.random.randn(4, 4, 4) + 1j * np.random.randn(4, 4, 4),
                    (1, 2), (-3, -2, -1), 'backward', ValueError),
     ('test_axis_out_of_range', np.random.randn(4) + 1j * np.random.randn(4),
      None, (10, 20), 'backward', ValueError),
     ('test_axis_type', np.random.randn(4) + 1j * np.random.randn(4), None, 1,
      'backward',
      ValueError), ('test_norm_not_in_enum_value',
                    np.random.randn(4, 4) + 1j * np.random.randn(4, 4), None,
                    None, 'random', ValueError)])
class TestIrfftnException(unittest.TestCase):
    def test_irfftn(self):
        """Test irfftn with buoudary condition
        Test case include:
        - n out of range
        - n type error
        - axis out of range
        - norm out of range
        - the dimensions of n and axis are different
        """
        with paddle.fluid.dygraph.guard(self.place):
            with self.assertRaises(self.expect_exception):
                paddle.fft.irfftn(
                    paddle.to_tensor(self.x), self.n, self.axis, self.norm)


@place(DEVICES)
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
        """Test rfft with norm condition
        """
        with paddle.fluid.dygraph.guard(self.place):
            self.assertTrue(
                np.allclose(
                    scipy.fft.rfft(self.x, self.n, self.axis, self.norm),
                    paddle.fft.rfft(
                        paddle.to_tensor(self.x), self.n, self.axis, self.norm),
                    rtol=RTOL.get(str(self.x.dtype)),
                    atol=ATOL.get(str(self.x.dtype))))


@place(DEVICES)
@parameterize((TEST_CASE_NAME, 'x', 'n', 'axis', 'norm', 'expect_exception'), [
    ('test_n_nagative', rand_x(2), -1, -1, 'backward', ValueError),
    ('test_n_zero', rand_x(2), 0, -1, 'backward', ValueError),
    ('test_axis_out_of_range', rand_x(1), None, 10, 'backward', ValueError),
    ('test_axis_with_array', rand_x(1), None, (0, 1), 'backward', ValueError),
    ('test_norm_not_in_enum_value', rand_x(2), None, -1, 'random', ValueError)
])
class TestRfftException(unittest.TestCase):
    def test_rfft(self):
        """Test rfft with buoudary condition
        Test case include:
        - n out of range
        - axis out of range
        - axis type error
        - norm out of range
        - the dimensions of n and axis are different
        """
        with self.assertRaises(self.expect_exception):
            paddle.fft.rfft(
                paddle.to_tensor(self.x), self.n, self.axis, self.norm)


@place(DEVICES)
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
        """Test rfft2 with norm condition
        """
        with paddle.fluid.dygraph.guard(self.place):
            self.assertTrue(
                np.allclose(
                    scipy.fft.rfft2(self.x, self.n, self.axis, self.norm),
                    paddle.fft.rfft2(
                        paddle.to_tensor(self.x), self.n, self.axis, self.norm),
                    rtol=RTOL.get(str(self.x.dtype)),
                    atol=ATOL.get(str(self.x.dtype))))


@place(DEVICES)
@parameterize(
    (TEST_CASE_NAME, 'x', 'n', 'axis', 'norm', 'expect_exception'), [
        ('test_x_complex_input', rand_x(
            2, complex=True), None, (0, 1), 'backward', RuntimeError),
        ('test_x_1dim_tensor', rand_x(1), None, (0, 1), 'backward', ValueError),
        ('test_n_nagative', rand_x(2), -1, (0, 1), 'backward', ValueError),
        ('test_n_zero', rand_x(2), 0, (0, 1), 'backward', ValueError),
        ('test_axis_out_of_range', rand_x(2), None, (0, 1, 2), 'backward',
         ValueError),
        ('test_axis_with_array', rand_x(1), None, (0, 1), 'backward',
         ValueError),
        ('test_axis_not_sequence', rand_x(5), None, -10, 'backward',
         ValueError),
        ('test_norm_not_enum', rand_x(2), None, -1, 'random', ValueError),
    ])
class TestRfft2Exception(unittest.TestCase):
    def test_rfft2(self):
        """Test rfft2 with buoudary condition
        Test case include:
        - input type error
        - input dim error
        - n out of range
        - axis out of range
        - norm out of range
        - the dimensions of n and axis are different
        """
        with paddle.fluid.dygraph.guard(self.place):
            with self.assertRaises(self.expect_exception):
                paddle.fft.rfft2(
                    paddle.to_tensor(self.x), self.n, self.axis, self.norm)


@place(DEVICES)
@parameterize(
    (TEST_CASE_NAME, 'x', 'n', 'axis', 'norm'), [
        ('test_x_float64', rand_x(5, np.float64), None, None, 'backward'),
        ('test_n_grater_input_length', rand_x(
            5, max_dim_len=5), (6, 6), (1, 2), 'backward'),
        ('test_n_smaller_input_length', rand_x(
            5, min_dim_len=5), (3, 3), (1, 2), 'backward'),
        ('test_axis_not_default', rand_x(5), None, (1, 2), 'backward'),
        ('test_norm_forward', rand_x(5), None, None, 'forward'),
        ('test_norm_ortho', rand_x(5), None, None, 'ortho'),
    ])
class TestRfftn(unittest.TestCase):
    def test_rfftn(self):
        """Test rfftn with norm condition
        """
        with paddle.fluid.dygraph.guard(self.place):
            self.assertTrue(
                np.allclose(
                    scipy.fft.rfftn(self.x, self.n, self.axis, self.norm),
                    paddle.fft.rfftn(
                        paddle.to_tensor(self.x), self.n, self.axis, self.norm),
                    rtol=RTOL.get(str(self.x.dtype)),
                    atol=ATOL.get(str(self.x.dtype))))


@place(DEVICES)
@parameterize(
    (TEST_CASE_NAME, 'x', 'n', 'axis', 'norm', 'expect_exception'),
    [('test_x_complex', rand_x(
        4, complex=True), None, None, 'backward',
      RuntimeError), ('test_n_nagative', rand_x(4), (-1, -1), (1, 2),
                      'backward', ValueError),
     ('test_n_not_sequence', rand_x(4), -1, None, 'backward', ValueError),
     ('test_n_zero', rand_x(4), 0, None, 'backward', ValueError), (
         'test_axis_out_of_range', rand_x(1), None, [0, 1], 'backward',
         ValueError),
     ('test_norm_not_in_enum', rand_x(2), None, -1, 'random', ValueError)])
class TestRfftnException(unittest.TestCase):
    def test_rfftn(self):
        """Test rfftn with buoudary condition
        Test case include:
        - n out of range
        - axis out of range
        - norm out of range
        - the dimensions of n and axis are different
        """
        with paddle.fluid.dygraph.guard(self.place):
            with self.assertRaises(self.expect_exception):
                paddle.fft.rfftn(
                    paddle.to_tensor(self.x), self.n, self.axis, self.norm)


@place(DEVICES)
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
        """Test ihfft with norm condition
        """
        with paddle.fluid.dygraph.guard(self.place):
            np.testing.assert_allclose(
                scipy.fft.ihfft(self.x, self.n, self.axis, self.norm),
                paddle.fft.ihfft(
                    paddle.to_tensor(self.x), self.n, self.axis, self.norm),
                rtol=RTOL.get(str(self.x.dtype)),
                atol=ATOL.get(str(self.x.dtype)))


@place(DEVICES)
@parameterize((TEST_CASE_NAME, 'x', 'n', 'axis', 'norm', 'expect_exception'), [
    ('test_n_nagative', rand_x(2), -1, -1, 'backward', ValueError),
    ('test_n_zero', rand_x(2), 0, -1, 'backward', ValueError),
    ('test_axis_out_of_range', rand_x(1), None, 10, 'backward', ValueError),
    ('test_axis_with_array', rand_x(1), None, (0, 1), 'backward', ValueError),
    ('test_norm_not_in_enum_value', rand_x(2), None, -1, 'random', ValueError)
])
class TestIhfftException(unittest.TestCase):
    def test_ihfft(self):
        """Test ihfft with buoudary condition
        Test case include:
        - axis type error
        - axis out of range
        - norm out of range
        """
        with paddle.fluid.dygraph.guard(self.place):
            with self.assertRaises(self.expect_exception):
                paddle.fft.ihfft(
                    paddle.to_tensor(self.x), self.n, self.axis, self.norm)


@place(DEVICES)
@parameterize(
    (TEST_CASE_NAME, 'x', 'n', 'axis', 'norm'), [
        ('test_x_float64', rand_x(5), None, (0, 1), 'backward'),
        ('test_n_grater_input_length', rand_x(
            5, max_dim_len=5), (11, 11), (0, 1), 'backward'),
        ('test_n_smaller_than_input_length', rand_x(
            5, min_dim_len=5), (1, 1), (0, 1), 'backward'),
        ('test_axis_random', rand_x(5), None, (1, 2), 'backward'),
        ('test_axis_none', rand_x(5), None, None, 'backward'),
        ('test_norm_forward', rand_x(5), None, (0, 1), 'forward'),
        ('test_norm_ortho', rand_x(5), None, (0, 1), 'ortho'),
    ])
class TestIhfft2(unittest.TestCase):
    def test_ihfft2(self):
        """Test ihfft2 with norm condition
        """
        with paddle.fluid.dygraph.guard(self.place):
            np.testing.assert_allclose(
                scipy.fft.ihfft2(self.x, self.n, self.axis, self.norm),
                paddle.fft.ihfft2(
                    paddle.to_tensor(self.x), self.n, self.axis, self.norm),
                rtol=RTOL.get(str(self.x.dtype)),
                atol=ATOL.get(str(self.x.dtype)))


@place(DEVICES)
@parameterize(
    (TEST_CASE_NAME, 'x', 'n', 'axis', 'norm', 'expect_exception'),
    [('test_x_complex_input', rand_x(
        2, complex=True), None, (0, 1), None, ValueError),
     ('test_x_1dim_tensor', rand_x(1), None, (0, 1), None,
      ValueError), ('test_n_nagative', rand_x(2), -1, (0, 1), 'backward',
                    ValueError), ('test_n_len_not_equal_axis', rand_x(
                        5, max_dim_len=5), 11, (0, 1), 'backward', ValueError),
     ('test_n_zero', rand_x(2), (0, 0), (0, 1), 'backward', ValueError),
     ('test_axis_out_of_range', rand_x(2), None, (0, 1, 2), 'backward',
      ValueError), ('test_axis_with_array', rand_x(1), None, (0, 1), 'backward',
                    ValueError), ('test_axis_not_sequence', rand_x(5), None,
                                  -10, 'backward', ValueError),
     ('test_norm_not_enum', rand_x(2), None, -1, 'random', ValueError)])
class TestIhfft2Exception(unittest.TestCase):
    def test_ihfft2(self):
        """Test ihfft2 with buoudary condition
        Test case include:
        - input type error
        - input dim error
        - n out of range
        - axis type error
        - axis out of range
        - norm out of range
        """
        with paddle.fluid.dygraph.guard(self.place):
            with self.assertRaises(self.expect_exception):
                paddle.fft.ihfft2(
                    paddle.to_tensor(self.x), self.n, self.axis, self.norm)


@place(DEVICES)
@parameterize(
    (TEST_CASE_NAME, 'x', 'n', 'axis', 'norm'),
    [('test_x_float64', rand_x(5, np.float64), None, None, 'backward'),
     ('test_n_grater_input_length', rand_x(
         5, max_dim_len=5), (11, 11), (0, 1),
      'backward'), ('test_n_smaller_input_length', rand_x(
          5, min_dim_len=5), (1, 1), (0, 1), 'backward'),
     ('test_axis_not_default', rand_x(5), None, (1, 2),
      'backward'), ('test_norm_forward', rand_x(5), None, None, 'forward'),
     ('test_norm_ortho', rand_x(5), None, None, 'ortho')])
class TestIhfftn(unittest.TestCase):
    def test_ihfftn(self):
        """Test ihfftn with norm condition
        """
        with paddle.fluid.dygraph.guard(self.place):
            self.assertTrue(
                np.allclose(
                    scipy.fft.ihfftn(self.x, self.n, self.axis, self.norm),
                    paddle.fft.ihfftn(
                        paddle.to_tensor(self.x), self.n, self.axis, self.norm),
                    rtol=RTOL.get(str(self.x.dtype)),
                    atol=ATOL.get(str(self.x.dtype))))


@place(DEVICES)
@parameterize(
    (TEST_CASE_NAME, 'x', 'n', 'axis', 'norm', 'expect_exception'),
    [('test_x_complex', rand_x(
        4, complex=True), None, None, 'backward', RuntimeError),
     ('test_n_nagative', rand_x(4), -1, None, 'backward', ValueError),
     ('test_n_zero', rand_x(4), 0, None, 'backward', ValueError), (
         'test_axis_out_of_range', rand_x(1), None, [0, 1], 'backward',
         ValueError),
     ('test_norm_not_in_enum', rand_x(2), None, -1, 'random', ValueError)])
class TestIhfftnException(unittest.TestCase):
    def test_ihfftn(self):
        """Test ihfftn with buoudary condition
        Test case include:
        - input type error
        - n out of range
        - axis out of range
        - norm out of range
        """
        with paddle.fluid.dygraph.guard(self.place):
            with self.assertRaises(self.expect_exception):
                paddle.fft.ihfftn(
                    paddle.to_tensor(self.x), self.n, self.axis, self.norm)


@place(DEVICES)
@parameterize((TEST_CASE_NAME, 'n', 'd', 'dtype'), [
    ('test_without_d', 20, 1, 'float32'),
    ('test_with_d', 20, 0.5, 'float32'),
])
class TestFftFreq(unittest.TestCase):
    def test_fftfreq(self):
        """Test fftfreq with norm condition
        """
        with paddle.fluid.dygraph.guard(self.place):
            np.testing.assert_allclose(
                scipy.fft.fftfreq(self.n, self.d).astype(self.dtype),
                paddle.fft.fftfreq(self.n, self.d, self.dtype).numpy(),
                rtol=RTOL.get(str(self.dtype)),
                atol=ATOL.get(str(self.dtype)))


@place(DEVICES)
@parameterize((TEST_CASE_NAME, 'n', 'd', 'dtype'), [
    ('test_without_d', 20, 1, 'float32'),
    ('test_with_d', 20, 0.5, 'float32'),
])
class TestRfftFreq(unittest.TestCase):
    def test_rfftfreq(self):
        """Test rfftfreq with norm condition
        """
        with paddle.fluid.dygraph.guard(self.place):
            np.testing.assert_allclose(
                scipy.fft.rfftfreq(self.n, self.d).astype(self.dtype),
                paddle.fft.rfftfreq(self.n, self.d, self.dtype).numpy(),
                rtol=RTOL.get(str(self.dtype)),
                atol=ATOL.get(str(self.dtype)))


@place(DEVICES)
@parameterize((TEST_CASE_NAME, 'x', 'axes', 'dtype'), [
    ('test_1d', np.random.randn(10), (0, ), 'float64'),
    ('test_2d', np.random.randn(10, 10), (0, 1), 'float64'),
    ('test_2d_with_all_axes', np.random.randn(10, 10), None, 'float64'),
    ('test_2d_odd_with_all_axes',
     np.random.randn(5, 5) + 1j * np.random.randn(5, 5), None, 'complex128'),
])
class TestFftShift(unittest.TestCase):
    def test_fftshift(self):
        """Test fftshift with norm condition
        """
        with paddle.fluid.dygraph.guard(self.place):
            np.testing.assert_allclose(
                scipy.fft.fftshift(self.x, self.axes),
                paddle.fft.fftshift(paddle.to_tensor(self.x),
                                    self.axes).numpy(),
                rtol=RTOL.get(str(self.x.dtype)),
                atol=ATOL.get(str(self.x.dtype)))


@place(DEVICES)
@parameterize(
    (TEST_CASE_NAME, 'x', 'axes'),
    [('test_1d', np.random.randn(10), (0, ),
      'float64'), ('test_2d', np.random.randn(10, 10), (0, 1), 'float64'),
     ('test_2d_with_all_axes', np.random.randn(10, 10), None, 'float64'),
     ('test_2d_odd_with_all_axes',
      np.random.randn(5, 5) + 1j * np.random.randn(5, 5), None, 'complex128')])
class TestIfftShift(unittest.TestCase):
    def test_ifftshift(self):
        """Test ifftshift with norm condition
        """
        with paddle.fluid.dygraph.guard(self.place):
            np.testing.assert_allclose(
                scipy.fft.ifftshift(self.x, self.axes),
                paddle.fft.ifftshift(paddle.to_tensor(self.x),
                                     self.axes).numpy(),
                rtol=RTOL.get(str(self.x.dtype)),
                atol=ATOL.get(str(self.x.dtype)))


if __name__ == '__main__':
    unittest.main()

# yapf: enable
