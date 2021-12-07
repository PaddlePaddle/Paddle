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

from __future__ import print_function
import unittest

import numpy as np
import paddle

import re
import sys
from spectral_op_np import fft_c2c, fft_r2c, fft_c2r
import paddle.fluid.core as core
import paddle.fluid.dygraph as dg
import paddle.static as static
from numpy.random import random as rand
from paddle.fluid import Program, program_guard
sys.path.append("../")
from op_test import OpTest

paddle.enable_static()

TEST_CASE_NAME = 'test_case'


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


def to_safe_name(s):
    return str(re.sub("[^a-zA-Z0-9_]+", "_", s))


def class_name(cls, num, params_dict):
    suffix = to_safe_name(
        next((v for v in params_dict.values() if isinstance(v, str)), ""))
    if TEST_CASE_NAME in params_dict:
        suffix = to_safe_name(params_dict["test_case"])
    return "{}_{}{}".format(cls.__name__, num, suffix and "_" + suffix)


@parameterize((TEST_CASE_NAME, 'x', 'axes', 'norm', 'forward'), [
    ('test_axes_is_sqe_type', (np.random.random(
        (12, 14)) + 1j * np.random.random((12, 14))).astype(np.complex128),
     [0, 1], 'forward', True), ('test_axis_not_last', (np.random.random(
         (4, 4, 4)) + 1j * np.random.random((4, 4, 4))).astype(np.complex128),
                                (0, 1), "backward", False),
    ('test_norm_forward', (np.random.random((12, 14)) + 1j * np.random.random(
        (12, 14))).astype(np.complex128), (0, ), "forward",
     False), ('test_norm_backward', (np.random.random(
         (12, 14)) + 1j * np.random.random((12, 14))).astype(np.complex128),
              (0, ), "backward", True), ('test_norm_ortho', (np.random.random(
                  (12, 14)) + 1j * np.random.random(
                      (12, 14))).astype(np.complex128), (1, ), "ortho", True)
])
class TestFFTC2COp(OpTest):
    # Because framwork not support complex numerial gradient, we skip gradient check. 
    no_need_check_grad = True

    def setUp(self):
        self.op_type = "fft_c2c"

        out = fft_c2c(self.x, self.axes, self.norm, self.forward)

        self.inputs = {'X': self.x}
        self.attrs = {
            'axes': self.axes,
            'normalization': self.norm,
            "forward": self.forward
        }
        self.outputs = {'Out': out}

    def test_check_output(self):
        self.check_output()


@parameterize(
    (TEST_CASE_NAME, 'x', 'axes', 'norm', 'forward', 'last_dim_size'),
    [('test_axes_is_sqe_type', (np.random.random(
        (12, 14)) + 1j * np.random.random((12, 14))).astype(np.complex128),
      [0, 1], 'forward', True, 26), ('test_axis_not_last', (np.random.random(
          (4, 4, 4)) + 1j * np.random.random((4, 4, 4))).astype(np.complex128),
                                     (0, 1), "backward", False, None),
     ('test_norm_forward', (np.random.random((12, 14)) + 1j * np.random.random(
         (12, 14))).astype(np.complex128), (0, ), "forward", False, 22),
     ('test_norm_backward', (np.random.random((12, 14)) + 1j * np.random.random(
         (12, 14))).astype(np.complex128), (0, ), "backward", True,
      22), ('test_norm_ortho', (np.random.random(
          (12, 14)) + 1j * np.random.random((12, 14))).astype(np.complex128),
            (1, ), "ortho", True, 26)])
class TestFFTC2ROp(OpTest):
    # Because framwork not support complex numerial gradient, we skip gradient check. 
    no_need_check_grad = True

    def setUp(self):
        self.op_type = "fft_c2r"

        out = fft_c2r(self.x, self.axes, self.norm, self.forward,
                      self.last_dim_size)

        self.inputs = {'X': self.x}
        self.attrs = {
            "axes": self.axes,
            "normalization": self.norm,
            "forward": self.forward,
            "last_dim_size": self.last_dim_size
        }
        self.outputs = {'Out': out}

    def test_check_output(self):
        self.check_output()


@parameterize(
    (TEST_CASE_NAME, 'x', 'axes', 'norm', 'forward', 'onesided'),
    [('test_axes_is_sqe_type', np.random.randn(12, 14).astype(np.float64),
      (0, 1), 'forward', True,
      True), ('test_axis_not_last', np.random.randn(4, 4, 4).astype(np.float64),
              (0, 1), "backward", False, True),
     ('test_norm_forward', np.random.randn(12, 14).astype(np.float64), (0, 1),
      "forward", False, False),
     ('test_norm_backward', np.random.randn(12, 14).astype(np.float64), (0, ),
      "backward", True, False), ('test_norm_ortho',
                                 np.random.randn(12, 14).astype(np.float64),
                                 (1, ), "ortho", True, False)])
class TestFFTR2COp(OpTest):
    # Because framwork not support complex numerial gradient, we skip gradient check. 
    no_need_check_grad = True

    def setUp(self):
        self.op_type = "fft_r2c"

        out = fft_r2c(self.x, self.axes, self.norm, self.forward, self.onesided)

        self.inputs = {'X': self.x}
        self.attrs = {
            'axes': self.axes,
            'normalization': self.norm,
            "forward": self.forward,
            'onesided': self.onesided
        }
        self.outputs = {'Out': out}

    def test_check_output(self):
        self.check_output()
