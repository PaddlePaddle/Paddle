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

from test_fft import (ATOL, DEVICES, RTOL, TEST_CASE_NAME, parameterize, place,
                      rand_x)


@contextlib.contextmanager
def stgraph(func, place, x, n, axes, norm):
    """static graph exec context"""
    paddle.enable_static()
    mp, sp = paddle.static.Program(), paddle.static.Program()
    with paddle.static.program_guard(mp, sp):
        input = paddle.static.data('input', x.shape, dtype=x.dtype)
        output = func(input, n, axes, norm)

    exe = paddle.static.Executor(place)
    exe.run(sp)
    [output] = exe.run(mp, feed={'input': x}, fetch_list=[output])
    yield output
    paddle.disable_static()


@place(DEVICES)
@parameterize((TEST_CASE_NAME, 'x', 'n', 'axis', 'norm'),
              [('test_x_float64', rand_x(5, np.float64), None, -1, 'backward'),
               ('test_x_complex64', rand_x(5, np.float64,
                                           complex=True), None, -1, 'backward'),
               ('test_n_grater_than_input_length', rand_x(
                   5, max_dim_len=5), 11, -1, 'backward'),
               ('test_n_smaller_than_input_length', rand_x(
                   5, min_dim_len=5), 3, -1, 'backward'),
               ('test_axis_not_last', rand_x(5), None, 3, 'backward'),
               ('test_norm_forward', rand_x(5), None, 3, 'forward'),
               ('test_norm_ortho', rand_x(5), None, 3, 'ortho')])
class TestFft(unittest.TestCase):

    def test_static_rfft(self):
        with stgraph(paddle.fft.fft, self.place, self.x, self.n, self.axis,
                     self.norm) as y:
            np.testing.assert_allclose(scipy.fft.fft(self.x, self.n, self.axis,
                                                     self.norm),
                                       y,
                                       rtol=RTOL.get(str(self.x.dtype)),
                                       atol=ATOL.get(str(self.x.dtype)))


@place(DEVICES)
@parameterize(
    (TEST_CASE_NAME, 'x', 'n', 'axis', 'norm', 'expect_exception'),
    [('test_n_nagative', rand_x(2), -1, -1, 'backward', ValueError),
     ('test_n_zero', rand_x(2), 0, -1, 'backward', ValueError),
     ('test_axis_out_of_range', rand_x(1), None, 10, 'backward', ValueError),
     ('test_axis_with_array', rand_x(1), None, (0, 1), 'backward', ValueError),
     ('test_norm_not_in_enum_value', rand_x(2), None, -1, 'random', ValueError)]
)
class TestFftException(unittest.TestCase):

    def test_fft(self):
        with self.assertRaises(self.expect_exception):
            with stgraph(paddle.fft.fft, self.place, self.x, self.n, self.axis,
                         self.norm) as y:
                pass


@place(DEVICES)
@parameterize((TEST_CASE_NAME, 'x', 'n', 'axis', 'norm'), [
    ('test_x_float64', rand_x(5), None, (0, 1), 'backward'),
    ('test_x_complex128', rand_x(5, complex=True), None, (0, 1), 'backward'),
    ('test_n_grater_input_length', rand_x(5, max_dim_len=5), (6, 6),
     (0, 1), 'backward'),
    ('test_n_smaller_than_input_length', rand_x(5, min_dim_len=5), (4, 4),
     (0, 1), 'backward'),
    ('test_axis_random', rand_x(5), None, (1, 2), 'backward'),
    ('test_axis_none', rand_x(5), None, None, 'backward'),
    ('test_norm_forward', rand_x(5), None, (0, 1), 'forward'),
    ('test_norm_ortho', rand_x(5), None, (0, 1), 'ortho'),
])
class TestFft2(unittest.TestCase):

    def test_static_fft2(self):
        with stgraph(paddle.fft.fft2, self.place, self.x, self.n, self.axis,
                     self.norm) as y:
            np.testing.assert_allclose(scipy.fft.fft2(self.x, self.n, self.axis,
                                                      self.norm),
                                       y,
                                       rtol=RTOL.get(str(self.x.dtype)),
                                       atol=ATOL.get(str(self.x.dtype)))


@place(DEVICES)
@parameterize(
    (TEST_CASE_NAME, 'x', 'n', 'axis', 'norm', 'expect_exception'),
    [
        # ('test_x_not_tensor', [0, 1], None, (0, 1), 'backward', ValueError),
        ('test_x_1dim_tensor', rand_x(1), None, (0, 1), 'backward', ValueError),
        ('test_n_nagative', rand_x(2), -1, (0, 1), 'backward', ValueError),
        ('test_n_zero', rand_x(2), 0, (0, 1), 'backward', ValueError),
        ('test_axis_out_of_range', rand_x(2), None,
         (0, 1, 2), 'backward', ValueError),
        ('test_axis_with_array', rand_x(1), None,
         (0, 1), 'backward', ValueError),
        ('test_axis_not_sequence', rand_x(5), None, -10, 'backward',
         ValueError),
        ('test_norm_not_enum', rand_x(2), None, -1, 'random', ValueError)
    ])
class TestFft2Exception(unittest.TestCase):

    def test_static_fft2(self):
        with self.assertRaises(self.expect_exception):
            with stgraph(paddle.fft.fft2, self.place, self.x, self.n, self.axis,
                         self.norm) as y:
                pass


@place(DEVICES)
@parameterize(
    (TEST_CASE_NAME, 'x', 'n', 'axis', 'norm'),
    [('test_x_float64', rand_x(5, np.float64), None, None, 'backward'),
     ('test_x_complex128', rand_x(5, np.float64,
                                  complex=True), None, None, 'backward'),
     ('test_n_grater_input_length', rand_x(5, max_dim_len=5), (6, 6),
      (1, 2), 'backward'),
     ('test_n_smaller_input_length', rand_x(5, min_dim_len=5), (3, 3),
      (1, 2), 'backward'),
     ('test_axis_not_default', rand_x(5), None, (1, 2), 'backward'),
     ('test_norm_forward', rand_x(5), None, None, 'forward'),
     ('test_norm_ortho', rand_x(5), None, None, 'ortho')])
class TestFftn(unittest.TestCase):

    def test_static_fftn(self):
        with stgraph(paddle.fft.fftn, self.place, self.x, self.n, self.axis,
                     self.norm) as y:
            np.testing.assert_allclose(scipy.fft.fftn(self.x, self.n, self.axis,
                                                      self.norm),
                                       y,
                                       rtol=RTOL.get(str(self.x.dtype)),
                                       atol=ATOL.get(str(self.x.dtype)))


@place(DEVICES)
@parameterize((TEST_CASE_NAME, 'x', 'n', 'axis', 'norm', 'expect_exception'), [
    ('test_x_complex', rand_x(4,
                              complex=True), None, None, 'backward', TypeError),
    ('test_n_nagative', rand_x(4), (-1, -1), (1, 2), 'backward', ValueError),
    ('test_n_not_sequence', rand_x(4), -1, None, 'backward', ValueError),
    ('test_n_zero', rand_x(4), 0, None, 'backward', ValueError),
    ('test_axis_out_of_range', rand_x(1), None, [0, 1], 'backward', ValueError),
    ('test_norm_not_in_enum', rand_x(2), None, -1, 'random', ValueError)
])
class TestRfftnException(unittest.TestCase):

    def test_static_rfftn(self):
        with self.assertRaises(self.expect_exception):
            with stgraph(paddle.fft.rfftn, self.place, self.x, self.n,
                         self.axis, self.norm) as y:
                pass


@place(DEVICES)
@parameterize((TEST_CASE_NAME, 'x', 'n', 'axis', 'norm'), [
    ('test_x_complex128',
     (np.random.randn(4, 4, 4) + 1j * np.random.randn(4, 4, 4)).astype(
         np.complex128), None, -1, "backward"),
    ('test_n_grater_than_input_length', np.random.randn(4, 4, 4) +
     1j * np.random.randn(4, 4, 4), 4, -1, "backward"),
    ('test_n_smaller_than_input_length', np.random.randn(4, 4, 4) +
     1j * np.random.randn(4, 4, 4), 2, -1, "backward"),
    ('test_axis_not_last', np.random.randn(4, 4, 4) +
     1j * np.random.randn(4, 4, 4), None, 1, "backward"),
    ('test_norm_forward', np.random.randn(4, 4, 4) +
     1j * np.random.randn(4, 4, 4), None, 1, "forward"),
    ('test_norm_ortho', np.random.randn(4, 4, 4) +
     1j * np.random.randn(4, 4, 4), None, -1, "ortho"),
])
class TestHfft(unittest.TestCase):
    """Test hfft with norm condition
    """

    def test_hfft(self):
        with stgraph(paddle.fft.hfft, self.place, self.x, self.n, self.axis,
                     self.norm) as y:
            np.testing.assert_allclose(scipy.fft.hfft(self.x, self.n, self.axis,
                                                      self.norm),
                                       y,
                                       rtol=1e-5,
                                       atol=0)


@place(DEVICES)
@parameterize((TEST_CASE_NAME, 'x', 'n', 'axis', 'norm'), [
    ('test_x_complex128',
     (np.random.randn(4, 4, 4) + 1j * np.random.randn(4, 4, 4)).astype(
         np.complex128), None, -1, "backward"),
    ('test_n_grater_than_input_length', np.random.randn(4, 4, 4) +
     1j * np.random.randn(4, 4, 4), 4, -1, "backward"),
    ('test_n_smaller_than_input_length', np.random.randn(4, 4, 4) +
     1j * np.random.randn(4, 4, 4), 2, -1, "backward"),
    ('test_axis_not_last', np.random.randn(4, 4, 4) +
     1j * np.random.randn(4, 4, 4), None, -1, "backward"),
    ('test_norm_forward', np.random.randn(4, 4, 4) +
     1j * np.random.randn(4, 4, 4), None, -1, "forward"),
    ('test_norm_ortho', np.random.randn(4, 4, 4) +
     1j * np.random.randn(4, 4, 4), None, -1, "ortho"),
])
class TestIrfft(unittest.TestCase):
    """Test irfft with norm condition
    """

    def test_irfft(self):
        with stgraph(paddle.fft.irfft, self.place, self.x, self.n, self.axis,
                     self.norm) as y:
            np.testing.assert_allclose(scipy.fft.irfft(self.x, self.n,
                                                       self.axis, self.norm),
                                       y,
                                       rtol=1e-5,
                                       atol=0)


@place(DEVICES)
@parameterize((TEST_CASE_NAME, 'x', 'n', 'axis', 'norm'), [
    ('test_x_complex128',
     (np.random.randn(4, 4, 4) + 1j * np.random.randn(4, 4, 4)).astype(
         np.complex128), None, None, "backward"),
    ('test_n_grater_than_input_length', np.random.randn(4, 4, 4) +
     1j * np.random.randn(4, 4, 4), [4], None, "backward"),
    ('test_n_smaller_than_input_length', np.random.randn(4, 4, 4) +
     1j * np.random.randn(4, 4, 4), [2], None, "backward"),
    ('test_axis_not_last', np.random.randn(4, 4, 4) +
     1j * np.random.randn(4, 4, 4), None, None, "backward"),
    ('test_norm_forward', np.random.randn(4, 4, 4) +
     1j * np.random.randn(4, 4, 4), None, None, "forward"),
    ('test_norm_ortho', np.random.randn(4, 4, 4) +
     1j * np.random.randn(4, 4, 4), None, None, "ortho"),
])
class Testirfftn(unittest.TestCase):
    """Test irfftn with norm condition
    """

    def test_static_irfftn(self):
        with stgraph(paddle.fft.irfftn, self.place, self.x, self.n, self.axis,
                     self.norm) as y:
            np.testing.assert_allclose(scipy.fft.irfftn(self.x, self.n,
                                                        self.axis, self.norm),
                                       y,
                                       rtol=1e-5,
                                       atol=0)


@place(DEVICES)
@parameterize((TEST_CASE_NAME, 'x', 'n', 'axis', 'norm'), [
    ('test_x_complex128',
     (np.random.randn(4, 4, 4) + 1j * np.random.randn(4, 4, 4)).astype(
         np.complex128), None, None, "backward"),
    ('test_n_grater_than_input_length', np.random.randn(4, 4, 4) +
     1j * np.random.randn(4, 4, 4), [4], None, "backward"),
    ('test_n_smaller_than_input_length', np.random.randn(4, 4, 4) +
     1j * np.random.randn(4, 4, 4), [2], None, "backward"),
    ('test_axis_not_last', np.random.randn(4, 4, 4) +
     1j * np.random.randn(4, 4, 4), None, None, "backward"),
    ('test_norm_forward', np.random.randn(4, 4, 4) +
     1j * np.random.randn(4, 4, 4), None, None, "forward"),
    ('test_norm_ortho', np.random.randn(4, 4, 4) +
     1j * np.random.randn(4, 4, 4), None, None, "ortho"),
])
class Testhfftn(unittest.TestCase):
    """Test hfftn with norm condition
    """

    def test_static_hfftn(self):
        with stgraph(paddle.fft.hfftn, self.place, self.x, self.n, self.axis,
                     self.norm) as y:
            np.testing.assert_allclose(scipy.fft.hfftn(self.x, self.n,
                                                       self.axis, self.norm),
                                       y,
                                       rtol=1e-5,
                                       atol=0)


@place(DEVICES)
@parameterize((TEST_CASE_NAME, 'x', 's', 'axis', 'norm'), [
    ('test_x_complex128',
     (np.random.randn(4, 4, 4) + 1j * np.random.randn(4, 4, 4)).astype(
         np.complex128), None, (-2, -1), "backward"),
    ('test_n_grater_input_length',
     np.random.randn(4, 4, 4) + 1j * np.random.randn(4, 4, 4), [4, 8],
     (-2, -1), "backward"),
    ('test_n_smaller_input_length',
     np.random.randn(4, 4, 4) + 1j * np.random.randn(4, 4, 4), [2, 4],
     (-2, -1), "backward"),
    ('test_axis_not_last',
     np.random.randn(4, 4, 4) + 1j * np.random.randn(4, 4, 4), None,
     (-2, -1), "backward"),
    ('test_norm_forward',
     np.random.randn(4, 4, 4) + 1j * np.random.randn(4, 4, 4), None,
     (-2, -1), "forward"),
    ('test_norm_ortho',
     np.random.randn(4, 4, 4) + 1j * np.random.randn(4, 4, 4), None,
     (-2, -1), "ortho"),
])
class Testhfft2(unittest.TestCase):
    """Test hfft2 with norm condition
    """

    def test_static_hfft2(self):
        with stgraph(paddle.fft.hfft2, self.place, self.x, self.s, self.axis,
                     self.norm) as y:
            np.testing.assert_allclose(scipy.fft.hfft2(self.x, self.s,
                                                       self.axis, self.norm),
                                       y,
                                       rtol=1e-5,
                                       atol=0)


@place(DEVICES)
@parameterize((TEST_CASE_NAME, 'x', 's', 'axis', 'norm'), [
    ('test_x_complex128',
     (np.random.randn(4, 4, 4) + 1j * np.random.randn(4, 4, 4)).astype(
         np.complex128), None, (-2, -1), "backward"),
    ('test_n_equal_input_length',
     np.random.randn(4, 4, 4) + 1j * np.random.randn(4, 4, 4), (2, 4),
     (-2, -1), "backward"),
    ('test_axis_not_last',
     np.random.randn(4, 4, 4) + 1j * np.random.randn(4, 4, 4), None,
     (-2, -1), "backward"),
    ('test_norm_forward',
     np.random.randn(4, 4, 4) + 1j * np.random.randn(4, 4, 4), None,
     (-2, -1), "forward"),
    ('test_norm_ortho',
     np.random.randn(4, 4, 4) + 1j * np.random.randn(4, 4, 4), None,
     (-2, -1), "ortho"),
])
class TestIrfft2(unittest.TestCase):
    """Test irfft2 with norm condition
    """

    def test_static_irfft2(self):
        with stgraph(paddle.fft.irfft2, self.place, self.x, self.s, self.axis,
                     self.norm) as y:
            np.testing.assert_allclose(scipy.fft.irfft2(self.x, self.s,
                                                        self.axis, self.norm),
                                       y,
                                       rtol=1e-5,
                                       atol=0)


@place(DEVICES)
@parameterize((TEST_CASE_NAME, 'x', 'n', 'axis', 'norm', 'expect_exception'), [
    ('test_input_dtype', np.random.randn(4, 4,
                                         4), None, -1, 'backward', TypeError),
    ('test_bool_input',
     (np.random.randn(4, 4, 4) + 1j * np.random.randn(4, 4, 4)).astype(
         np.bool_), None, -1, 'backward', TypeError),
    ('test_n_nagative', np.random.randn(4, 4, 4) +
     1j * np.random.randn(4, 4, 4), -1, -1, 'backward', ValueError),
    ('test_n_zero', np.random.randn(4, 4) + 1j * np.random.randn(4, 4), 0, -1,
     'backward', ValueError),
    ('test_n_type', np.random.randn(4, 4, 4) + 1j * np.random.randn(4, 4, 4),
     (1, 2, 3), -1, 'backward', ValueError),
    ('test_axis_out_of_range', np.random.randn(4) + 1j * np.random.randn(4),
     None, 10, 'backward', ValueError),
    ('test_axis_with_array', np.random.randn(4) + 1j * np.random.randn(4), None,
     (0, 1), 'backward', ValueError),
    ('test_norm_not_in_enum_value', np.random.randn(4, 4) +
     1j * np.random.randn(4, 4), None, -1, 'random', ValueError)
])
class TestHfftException(unittest.TestCase):
    '''Test hfft with buoudary condition
    Test case include:
    - non complex input
    - n out of range
    - axis out of range
    - norm out of range
    '''

    def test_static_hfft(self):
        with self.assertRaises(self.expect_exception):
            with stgraph(paddle.fft.hfft, self.place, self.x, self.n, self.axis,
                         self.norm) as y:
                pass


@place(DEVICES)
@parameterize((TEST_CASE_NAME, 'x', 'n', 'axis', 'norm', 'expect_exception'), [
    ('test_input_dtype', np.random.randn(4, 4,
                                         4), None, -1, 'backward', TypeError),
    ('test_bool_input',
     (np.random.randn(4, 4, 4) + 1j * np.random.randn(4, 4, 4)).astype(
         np.bool_), None, -1, 'backward', TypeError),
    ('test_n_nagative', np.random.randn(4, 4, 4) +
     1j * np.random.randn(4, 4, 4), -1, -1, 'backward', ValueError),
    ('test_n_zero', np.random.randn(4, 4) + 1j * np.random.randn(4, 4), 0, -1,
     'backward', ValueError),
    ('test_n_type', np.random.randn(4, 4, 4) + 1j * np.random.randn(4, 4, 4),
     (1, 2), -1, 'backward', ValueError),
    ('test_axis_out_of_range', np.random.randn(4) + 1j * np.random.randn(4),
     None, 10, 'backward', ValueError),
    ('test_axis_with_array', np.random.randn(4) + 1j * np.random.randn(4), None,
     (0, 1), 'backward', ValueError),
    ('test_norm_not_in_enum_value', np.random.randn(4, 4) +
     1j * np.random.randn(4, 4), None, None, 'random', ValueError)
])
class TestIrfftException(unittest.TestCase):
    '''Test Irfft with buoudary condition
    Test case include:
    - non complex input
    - n out of range
    - axis out of range
    - norm out of range
    - the dimensions of n and axis are different
    '''

    def test_static_irfft(self):
        with self.assertRaises(self.expect_exception):
            with stgraph(paddle.fft.irfft, self.place, self.x, self.n,
                         self.axis, self.norm) as y:
                pass


@place(DEVICES)
@parameterize(
    (TEST_CASE_NAME, 'x', 'n', 'axis', 'norm', 'expect_exception'),
    [('test_input_dtype', np.random.randn(
        4, 4, 4), None, None, 'backward', TypeError),
     ('test_bool_input',
      (np.random.randn(4, 4, 4) + 1j * np.random.randn(4, 4, 4)).astype(
          np.bool_), None, (-2, -1), 'backward', TypeError),
     ('test_n_nagative',
      np.random.randn(4, 4, 4) + 1j * np.random.randn(4, 4, 4), (-1, -2),
      (-2, -1), 'backward', ValueError),
     ('test_n_zero', np.random.randn(4, 4, 4) + 1j * np.random.randn(4, 4, 4),
      (0, 0), (-2, -1), 'backward', ValueError),
     ('test_n_type', np.random.randn(4, 4, 4) + 1j * np.random.randn(4, 4, 4),
      3, None, 'backward', ValueError),
     ('test_n_axis_dim',
      np.random.randn(4, 4, 4) + 1j * np.random.randn(4, 4, 4), (1, 2),
      (-1), 'backward', ValueError),
     ('test_axis_out_of_range', np.random.randn(4) + 1j * np.random.randn(4),
      None, (1, 2), 'backward', ValueError),
     ('test_axis_type', np.random.randn(4) + 1j * np.random.randn(4), None, -1,
      'backward', ValueError),
     ('test_norm_not_in_enum_value', np.random.randn(4, 4) +
      1j * np.random.randn(4, 4), None, None, 'random', ValueError)])
class TestHfft2Exception(unittest.TestCase):
    '''Test hfft2 with buoudary condition
    Test case include:
    - non complex input
    - n out of range
    - axis out of range
    - the dimensions of n and axis are different
    - norm out of range
    '''

    def test_static_hfft2(self):
        with self.assertRaises(self.expect_exception):
            with stgraph(paddle.fft.hfft2, self.place, self.x, self.n,
                         self.axis, self.norm) as y:
                pass


@place(DEVICES)
@parameterize(
    (TEST_CASE_NAME, 'x', 'n', 'axis', 'norm', 'expect_exception'),
    [('test_input_dtype', np.random.randn(
        4, 4, 4), None, None, 'backward', TypeError),
     ('test_bool_input',
      (np.random.randn(4, 4, 4) + 1j * np.random.randn(4, 4, 4)).astype(
          np.bool_), None, (-2, -1), 'backward', TypeError),
     ('test_n_nagative',
      np.random.randn(4, 4, 4) + 1j * np.random.randn(4, 4, 4), (-1, -2),
      (-2, -1), 'backward', ValueError),
     ('test_n_zero', np.random.randn(4, 4, 4) + 1j * np.random.randn(4, 4, 4),
      (0, 0), (-2, -1), 'backward', ValueError),
     ('test_n_type', np.random.randn(4, 4, 4) + 1j * np.random.randn(4, 4, 4),
      3, -1, 'backward', ValueError),
     ('test_n_axis_dim',
      np.random.randn(4, 4, 4) + 1j * np.random.randn(4, 4, 4), (1, 2),
      (-3, -2, -1), 'backward', ValueError),
     ('test_axis_out_of_range', np.random.randn(4) + 1j * np.random.randn(4),
      None, (1, 2), 'backward', ValueError),
     ('test_axis_type', np.random.randn(4) + 1j * np.random.randn(4), None, 1,
      'backward', ValueError),
     ('test_norm_not_in_enum_value', np.random.randn(4, 4) +
      1j * np.random.randn(4, 4), None, None, 'random', ValueError)])
class TestIrfft2Exception(unittest.TestCase):
    '''Test irfft2 with buoudary condition
    Test case include:
    - non complex input
    - n out of range
    - axis out of range
    - norm out of range
    - the dimensions of n and axis are different
    '''

    def test_static_irfft2(self):
        with self.assertRaises(self.expect_exception):
            with stgraph(paddle.fft.irfft2, self.place, self.x, self.n,
                         self.axis, self.norm) as y:
                pass


@place(DEVICES)
@parameterize(
    (TEST_CASE_NAME, 'x', 'n', 'axis', 'norm', 'expect_exception'),
    [('test_input_dtype', np.random.randn(
        4, 4, 4), None, None, 'backward', TypeError),
     ('test_bool_input',
      (np.random.randn(4, 4, 4) + 1j * np.random.randn(4, 4, 4)).astype(
          np.bool_), None, (-2, -1), 'backward', TypeError),
     ('test_n_nagative',
      np.random.randn(4, 4, 4) + 1j * np.random.randn(4, 4, 4), (-1, -2),
      (-2, -1), 'backward', ValueError),
     ('test_n_zero', np.random.randn(4, 4, 4) + 1j * np.random.randn(4, 4, 4),
      (0, 0), (-2, -1), 'backward', ValueError),
     ('test_n_type', np.random.randn(4, 4, 4) + 1j * np.random.randn(4, 4, 4),
      3, -1, 'backward', ValueError),
     ('test_n_axis_dim',
      np.random.randn(4, 4, 4) + 1j * np.random.randn(4, 4, 4), (1, 2),
      (-3, -2, -1), 'backward', ValueError),
     ('test_axis_out_of_range', np.random.randn(4) + 1j * np.random.randn(4),
      None, (10, 20), 'backward', ValueError),
     ('test_axis_type', np.random.randn(4) + 1j * np.random.randn(4), None, 1,
      'backward', ValueError),
     ('test_norm_not_in_enum_value', np.random.randn(4, 4) +
      1j * np.random.randn(4, 4), None, None, 'random', ValueError)])
class TestHfftnException(unittest.TestCase):
    '''Test hfftn with buoudary condition
    Test case include:
    - non complex input
    - n out of range
    - axis out of range
    - norm out of range
    - the dimensions of n and axis are different
    '''

    def test_static_hfftn(self):
        with self.assertRaises(self.expect_exception):
            with stgraph(paddle.fft.hfftn, self.place, self.x, self.n,
                         self.axis, self.norm) as y:
                pass


@place(DEVICES)
@parameterize(
    (TEST_CASE_NAME, 'x', 'n', 'axis', 'norm', 'expect_exception'),
    [
        ('test_input_dtype', np.random.randn(
            4, 4, 4), None, None, 'backward', TypeError),
        #  ('test_bool_input',
        #                (np.random.randn(4, 4, 4) + 1j * np.random.randn(4, 4, 4)
        #                 ).astype(np.bool_), None, (-2, -1), 'backward', ValueError),
        ('test_n_nagative',
         np.random.randn(4, 4, 4) + 1j * np.random.randn(4, 4, 4), (-1, -2),
         (-2, -1), 'backward', ValueError),
        ('test_n_zero',
         np.random.randn(4, 4, 4) + 1j * np.random.randn(4, 4, 4), (0, 0),
         (-2, -1), 'backward', ValueError),
        ('test_n_type', np.random.randn(4, 4, 4) +
         1j * np.random.randn(4, 4, 4), 3, -1, 'backward', ValueError),
        ('test_n_axis_dim',
         np.random.randn(4, 4, 4) + 1j * np.random.randn(4, 4, 4), (1, 2),
         (-3, -2, -1), 'backward', ValueError),
        ('test_axis_out_of_range', np.random.randn(4) + 1j * np.random.randn(4),
         None, (10, 20), 'backward', ValueError),
        ('test_axis_type', np.random.randn(4) + 1j * np.random.randn(4), None,
         1, 'backward', ValueError),
        ('test_norm_not_in_enum_value', np.random.randn(4, 4) +
         1j * np.random.randn(4, 4), None, None, 'random', ValueError)
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

    def test_static_irfftn(self):
        with self.assertRaises(self.expect_exception):
            with stgraph(paddle.fft.irfftn, self.place, self.x, self.n,
                         self.axis, self.norm) as y:
                pass


@place(DEVICES)
@parameterize((TEST_CASE_NAME, 'x', 'n', 'axis', 'norm'),
              [('test_x_float64', rand_x(5, np.float64), None, -1, 'backward'),
               ('test_n_grater_than_input_length', rand_x(
                   5, max_dim_len=5), 11, -1, 'backward'),
               ('test_n_smaller_than_input_length', rand_x(
                   5, min_dim_len=5), 3, -1, 'backward'),
               ('test_axis_not_last', rand_x(5), None, 3, 'backward'),
               ('test_norm_forward', rand_x(5), None, 3, 'forward'),
               ('test_norm_ortho', rand_x(5), None, 3, 'ortho')])
class TestRfft(unittest.TestCase):

    def test_static_rfft(self):
        with stgraph(paddle.fft.rfft, self.place, self.x, self.n, self.axis,
                     self.norm) as y:
            np.testing.assert_allclose(scipy.fft.rfft(self.x, self.n, self.axis,
                                                      self.norm),
                                       y,
                                       rtol=RTOL.get(str(self.x.dtype)),
                                       atol=ATOL.get(str(self.x.dtype)))


@place(DEVICES)
@parameterize(
    (TEST_CASE_NAME, 'x', 'n', 'axis', 'norm', 'expect_exception'),
    [('test_n_nagative', rand_x(2), -1, -1, 'backward', ValueError),
     ('test_n_zero', rand_x(2), 0, -1, 'backward', ValueError),
     ('test_axis_out_of_range', rand_x(1), None, 10, 'backward', ValueError),
     ('test_axis_with_array', rand_x(1), None, (0, 1), 'backward', ValueError),
     ('test_norm_not_in_enum_value', rand_x(2), None, -1, 'random', ValueError)]
)
class TestRfftException(unittest.TestCase):

    def test_rfft(self):
        with self.assertRaises(self.expect_exception):
            with stgraph(paddle.fft.rfft, self.place, self.x, self.n, self.axis,
                         self.norm) as y:
                pass


@place(DEVICES)
@parameterize((TEST_CASE_NAME, 'x', 'n', 'axis', 'norm'), [
    ('test_x_float64', rand_x(5), None, (0, 1), 'backward'),
    ('test_n_grater_input_length', rand_x(5, max_dim_len=5), (6, 6),
     (0, 1), 'backward'),
    ('test_n_smaller_than_input_length', rand_x(5, min_dim_len=5), (4, 4),
     (0, 1), 'backward'),
    ('test_axis_random', rand_x(5), None, (1, 2), 'backward'),
    ('test_axis_none', rand_x(5), None, None, 'backward'),
    ('test_norm_forward', rand_x(5), None, (0, 1), 'forward'),
    ('test_norm_ortho', rand_x(5), None, (0, 1), 'ortho'),
])
class TestRfft2(unittest.TestCase):

    def test_static_rfft2(self):
        with stgraph(paddle.fft.rfft2, self.place, self.x, self.n, self.axis,
                     self.norm) as y:
            np.testing.assert_allclose(scipy.fft.rfft2(self.x, self.n,
                                                       self.axis, self.norm),
                                       y,
                                       rtol=RTOL.get(str(self.x.dtype)),
                                       atol=ATOL.get(str(self.x.dtype)))


@place(DEVICES)
@parameterize(
    (TEST_CASE_NAME, 'x', 'n', 'axis', 'norm', 'expect_exception'),
    [
        ('test_x_complex_input', rand_x(2, complex=True), None,
         (0, 1), 'backward', TypeError),
        # ('test_x_not_tensor', [0, 1], None, (0, 1), 'backward', ValueError),
        ('test_x_1dim_tensor', rand_x(1), None, (0, 1), 'backward', ValueError),
        ('test_n_nagative', rand_x(2), -1, (0, 1), 'backward', ValueError),
        ('test_n_zero', rand_x(2), 0, (0, 1), 'backward', ValueError),
        ('test_axis_out_of_range', rand_x(2), None,
         (0, 1, 2), 'backward', ValueError),
        ('test_axis_with_array', rand_x(1), None,
         (0, 1), 'backward', ValueError),
        ('test_axis_not_sequence', rand_x(5), None, -10, 'backward',
         ValueError),
        ('test_norm_not_enum', rand_x(2), None, -1, 'random', ValueError)
    ])
class TestRfft2Exception(unittest.TestCase):

    def test_static_rfft(self):
        with self.assertRaises(self.expect_exception):
            with stgraph(paddle.fft.rfft2, self.place, self.x, self.n,
                         self.axis, self.norm) as y:
                pass


@place(DEVICES)
@parameterize(
    (TEST_CASE_NAME, 'x', 'n', 'axis', 'norm'),
    [('test_x_float64', rand_x(5, np.float64), None, None, 'backward'),
     ('test_n_grater_input_length', rand_x(5, max_dim_len=5), (6, 6),
      (1, 2), 'backward'),
     ('test_n_smaller_input_length', rand_x(5, min_dim_len=5), (3, 3),
      (1, 2), 'backward'),
     ('test_axis_not_default', rand_x(5), None, (1, 2), 'backward'),
     ('test_norm_forward', rand_x(5), None, None, 'forward'),
     ('test_norm_ortho', rand_x(5), None, None, 'ortho')])
class TestRfftn(unittest.TestCase):

    def test_static_rfft(self):
        with stgraph(paddle.fft.rfftn, self.place, self.x, self.n, self.axis,
                     self.norm) as y:
            np.testing.assert_allclose(scipy.fft.rfftn(self.x, self.n,
                                                       self.axis, self.norm),
                                       y,
                                       rtol=RTOL.get(str(self.x.dtype)),
                                       atol=ATOL.get(str(self.x.dtype)))


@place(DEVICES)
@parameterize((TEST_CASE_NAME, 'x', 'n', 'axis', 'norm', 'expect_exception'), [
    ('test_x_complex', rand_x(4,
                              complex=True), None, None, 'backward', TypeError),
    ('test_n_nagative', rand_x(4), (-1, -1), (1, 2), 'backward', ValueError),
    ('test_n_not_sequence', rand_x(4), -1, None, 'backward', ValueError),
    ('test_n_zero', rand_x(4), 0, None, 'backward', ValueError),
    ('test_axis_out_of_range', rand_x(1), None, [0, 1], 'backward', ValueError),
    ('test_norm_not_in_enum', rand_x(2), None, -1, 'random', ValueError)
])
class TestRfftnException(unittest.TestCase):

    def test_static_rfftn(self):
        with self.assertRaises(self.expect_exception):
            with stgraph(paddle.fft.rfftn, self.place, self.x, self.n,
                         self.axis, self.norm) as y:
                pass


@place(DEVICES)
@parameterize((TEST_CASE_NAME, 'x', 'n', 'axis', 'norm'),
              [('test_x_float64', rand_x(5, np.float64), None, -1, 'backward'),
               ('test_n_grater_than_input_length', rand_x(
                   5, max_dim_len=5), 11, -1, 'backward'),
               ('test_n_smaller_than_input_length', rand_x(
                   5, min_dim_len=5), 3, -1, 'backward'),
               ('test_axis_not_last', rand_x(5), None, 3, 'backward'),
               ('test_norm_forward', rand_x(5), None, 3, 'forward'),
               ('test_norm_ortho', rand_x(5), None, 3, 'ortho')])
class TestIhfft(unittest.TestCase):

    def test_static_ihfft(self):
        with stgraph(paddle.fft.ihfft, self.place, self.x, self.n, self.axis,
                     self.norm) as y:
            np.testing.assert_allclose(scipy.fft.ihfft(self.x, self.n,
                                                       self.axis, self.norm),
                                       y,
                                       rtol=RTOL.get(str(self.x.dtype)),
                                       atol=ATOL.get(str(self.x.dtype)))


@place(DEVICES)
@parameterize(
    (TEST_CASE_NAME, 'x', 'n', 'axis', 'norm', 'expect_exception'),
    [('test_n_nagative', rand_x(2), -1, -1, 'backward', ValueError),
     ('test_n_zero', rand_x(2), 0, -1, 'backward', ValueError),
     ('test_axis_out_of_range', rand_x(1), None, 10, 'backward', ValueError),
     ('test_axis_with_array', rand_x(1), None, (0, 1), 'backward', ValueError),
     ('test_norm_not_in_enum_value', rand_x(2), None, -1, 'random', ValueError)]
)
class TestIhfftException(unittest.TestCase):

    def test_static_ihfft(self):
        with self.assertRaises(self.expect_exception):
            with stgraph(paddle.fft.ihfft, self.place, self.x, self.n,
                         self.axis, self.norm) as y:
                pass


@place(DEVICES)
@parameterize((TEST_CASE_NAME, 'x', 'n', 'axis', 'norm'), [
    ('test_x_float64', rand_x(5), None, (0, 1), 'backward'),
    ('test_n_grater_input_length', rand_x(5, max_dim_len=5), (11, 11),
     (0, 1), 'backward'),
    ('test_n_smaller_than_input_length', rand_x(5, min_dim_len=5), (1, 1),
     (0, 1), 'backward'),
    ('test_axis_random', rand_x(5), None, (1, 2), 'backward'),
    ('test_axis_none', rand_x(5), None, None, 'backward'),
    ('test_norm_forward', rand_x(5), None, (0, 1), 'forward'),
    ('test_norm_ortho', rand_x(5), None, (0, 1), 'ortho'),
])
class TestIhfft2(unittest.TestCase):

    def test_static_ihfft2(self):
        with stgraph(paddle.fft.ihfft2, self.place, self.x, self.n, self.axis,
                     self.norm) as y:
            np.testing.assert_allclose(scipy.fft.ihfft2(self.x, self.n,
                                                        self.axis, self.norm),
                                       y,
                                       rtol=RTOL.get(str(self.x.dtype)),
                                       atol=ATOL.get(str(self.x.dtype)))


@place(DEVICES)
@parameterize(
    (TEST_CASE_NAME, 'x', 'n', 'axis', 'norm', 'expect_exception'),
    [
        ('test_x_complex_input', rand_x(2, complex=True), None,
         (0, 1), None, ValueError),
        # ('test_x_not_tensor', [0, 1], None, (0, 1), None, ValueError),
        ('test_x_1dim_tensor', rand_x(1), None, (0, 1), None, ValueError),
        ('test_n_nagative', rand_x(2), -1, (0, 1), 'backward', ValueError),
        ('test_n_len_not_equal_axis', rand_x(5, max_dim_len=5), 11,
         (0, 1), 'backward', ValueError),
        ('test_n_zero', rand_x(2), (0, 0), (0, 1), 'backward', ValueError),
        ('test_axis_out_of_range', rand_x(2), None,
         (0, 1, 2), 'backward', ValueError),
        ('test_axis_with_array', rand_x(1), None,
         (0, 1), 'backward', ValueError),
        ('test_axis_not_sequence', rand_x(5), None, -10, 'backward',
         ValueError),
        ('test_norm_not_enum', rand_x(2), None, -1, 'random', ValueError)
    ])
class TestIhfft2Exception(unittest.TestCase):

    def test_static_ihfft2(self):
        with self.assertRaises(self.expect_exception):
            with stgraph(paddle.fft.ihfft2, self.place, self.x, self.n,
                         self.axis, self.norm) as y:
                pass


@place(DEVICES)
@parameterize(
    (TEST_CASE_NAME, 'x', 'n', 'axis', 'norm'),
    [('test_x_float64', rand_x(5, np.float64), None, None, 'backward'),
     ('test_n_grater_input_length', rand_x(5, max_dim_len=5), (11, 11),
      (0, 1), 'backward'),
     ('test_n_smaller_input_length', rand_x(5, min_dim_len=5), (1, 1),
      (0, 1), 'backward'),
     ('test_axis_not_default', rand_x(5), None, (1, 2), 'backward'),
     ('test_norm_forward', rand_x(5), None, None, 'forward'),
     ('test_norm_ortho', rand_x(5), None, None, 'ortho')])
class TestIhfftn(unittest.TestCase):

    def test_static_ihfftn(self):
        with stgraph(paddle.fft.ihfftn, self.place, self.x, self.n, self.axis,
                     self.norm) as y:
            np.testing.assert_allclose(scipy.fft.ihfftn(self.x, self.n,
                                                        self.axis, self.norm),
                                       y,
                                       rtol=RTOL.get(str(self.x.dtype)),
                                       atol=ATOL.get(str(self.x.dtype)))


@place(DEVICES)
@parameterize((TEST_CASE_NAME, 'x', 'n', 'axis', 'norm', 'expect_exception'), [
    ('test_x_complex', rand_x(4,
                              complex=True), None, None, 'backward', TypeError),
    ('test_n_nagative', rand_x(4), -1, None, 'backward', ValueError),
    ('test_n_zero', rand_x(4), 0, None, 'backward', ValueError),
    ('test_axis_out_of_range', rand_x(1), None, [0, 1], 'backward', ValueError),
    ('test_norm_not_in_enum', rand_x(2), None, -1, 'random', ValueError)
])
class TestIhfftnException(unittest.TestCase):

    def test_static_ihfftn(self):
        with self.assertRaises(self.expect_exception):
            with stgraph(paddle.fft.ihfftn, self.place, self.x, self.n,
                         self.axis, self.norm) as y:
                pass


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
        paddle.enable_static()
        mp, sp = paddle.static.Program(), paddle.static.Program()
        with paddle.static.program_guard(mp, sp):
            input = paddle.static.data('input', x.shape, dtype=x.dtype)
            output = paddle.fft.fftshift(input, axes)

        exe = paddle.static.Executor(place)
        exe.run(sp)
        [output] = exe.run(mp, feed={'input': x}, fetch_list=[output])
        yield output
        paddle.disable_static()


@place(DEVICES)
@parameterize(
    (TEST_CASE_NAME, 'x', 'axes'),
    [('test_1d', np.random.randn(10), (0, ), 'float64'),
     ('test_2d', np.random.randn(10, 10), (0, 1), 'float64'),
     ('test_2d_with_all_axes', np.random.randn(10, 10), None, 'float64'),
     ('test_2d_odd_with_all_axes',
      np.random.randn(5, 5) + 1j * np.random.randn(5, 5), None, 'complex128')])
class TestIfftShift(unittest.TestCase):

    def test_ifftshift(self):
        """Test ifftshift with norm condition
        """
        paddle.enable_static()
        mp, sp = paddle.static.Program(), paddle.static.Program()
        with paddle.static.program_guard(mp, sp):
            input = paddle.static.data('input', x.shape, dtype=x.dtype)
            output = paddle.fft.ifftshift(input, axes)

        exe = paddle.static.Executor(place)
        exe.run(sp)
        [output] = exe.run(mp, feed={'input': x}, fetch_list=[output])
        yield output
        paddle.disable_static()


if __name__ == '__main__':
    unittest.main()
