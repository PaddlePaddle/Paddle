# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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
import unittest
import numpy as np
import six

import paddle
import paddle.fluid as fluid
from paddle.fluid import core

from paddle.fluid.dygraph.base import to_variable

from paddle.fluid.clip import GradientClipByValue, GradientClipByNorm, GradientClipByGlobalNorm


class TestGradClipByGlobalNorm(unittest.TestCase):

    def init_value(self):
        self.max_global_norm = 5.0
        self.init_scale = 1.0

        self.shape = (20, 20)

    def generate_p_g(self):

        self.para_and_grad = []
        for i in range(10):
            self.para_and_grad.append(
                (np.random.uniform(-self.init_scale, self.init_scale,
                                   self.shape).astype('float32'),
                 np.random.uniform(-self.init_scale, self.init_scale,
                                   self.shape).astype('float32')))

    def get_numpy_global_norm_result(self):
        gloabl_norm = 0.0
        for p, g in self.para_and_grad:
            gloabl_norm += np.sum(np.square(g))

        gloabl_norm_np = np.sqrt(gloabl_norm)

        new_np_p_g = []
        scale = 1.0
        if gloabl_norm_np > self.max_global_norm:
            scale = self.max_global_norm / gloabl_norm_np

        for p, g in self.para_and_grad:
            new_np_p_g.append((p, g * scale))

        return new_np_p_g

    def get_dygrap_global_norm_result(self):
        with fluid.dygraph.guard():

            gloabl_norm_clip = GradientClipByGlobalNorm(self.max_global_norm)
            p_g_var = []
            for p, g in self.para_and_grad:
                new_p = to_variable(p)
                new_g = to_variable(g)
                p_g_var.append((new_p, new_g))

            new_p_g_var = gloabl_norm_clip(p_g_var)

            p_g_dy_out = []
            for p, g in new_p_g_var:
                p_g_dy_out.append((p.numpy(), g.numpy()))

            return p_g_dy_out

    def test_clip_by_global_norm(self):
        self.init_value()
        self.generate_p_g()
        np_p_g = self.get_numpy_global_norm_result()
        dy_out_p_g = self.get_dygrap_global_norm_result()

        for (p_np, g_np), (p_dy, g_dy) in zip(np_p_g, dy_out_p_g):
            np.testing.assert_allclose(g_np, g_dy, rtol=1e-06, atol=1e-08)

    def test_clip_by_global_norm_2(self):
        self.init_value()

        self.init_scale = 0.2
        self.max_global_norm = 10
        self.generate_p_g()
        np_p_g = self.get_numpy_global_norm_result()
        dy_out_p_g = self.get_dygrap_global_norm_result()

        for (p_np, g_np), (p_dy, g_dy) in zip(np_p_g, dy_out_p_g):
            np.testing.assert_allclose(g_np, g_dy, rtol=1e-06, atol=1e-08)


class TestGradClipByNorm(unittest.TestCase):

    def init_value(self):
        self.max_norm = 5.0
        self.init_scale = 1.0

        self.shape = (10, 10)

    def generate_p_g(self):

        self.para_and_grad = []
        for i in range(10):
            self.para_and_grad.append(
                (np.random.uniform(-self.init_scale, self.init_scale,
                                   self.shape).astype('float32'),
                 np.random.uniform(-self.init_scale, self.init_scale,
                                   self.shape).astype('float32')))

    def get_numpy_norm_result(self):

        new_p_g = []
        for p, g in self.para_and_grad:
            norm = np.sqrt(np.sum(np.square(g)))

            if norm > self.max_norm:
                new_p_g.append((p, g * self.max_norm / norm))
            else:
                new_p_g.append((p, g))

        return new_p_g

    def get_dygrap_norm_result(self):
        with fluid.dygraph.guard():

            norm_clip = GradientClipByNorm(self.max_norm)
            p_g_var = []
            for p, g in self.para_and_grad:
                new_p = to_variable(p)
                new_g = to_variable(g)
                p_g_var.append((new_p, new_g))

            new_p_g_var = norm_clip(p_g_var)

            p_g_dy_out = []
            for p, g in new_p_g_var:
                p_g_dy_out.append((p.numpy(), g.numpy()))

            return p_g_dy_out

    def test_clip_by_norm(self):
        self.init_value()
        self.generate_p_g()
        np_p_g = self.get_numpy_norm_result()
        dy_out_p_g = self.get_dygrap_norm_result()

        for (p_np, g_np), (p_dy, g_dy) in zip(np_p_g, dy_out_p_g):
            np.testing.assert_allclose(g_np, g_dy, rtol=1e-06, atol=1e-08)

    def test_clip_by_norm_2(self):
        self.init_value()

        self.init_scale = 0.2
        self.max_norm = 10.0
        self.generate_p_g()
        np_p_g = self.get_numpy_norm_result()
        dy_out_p_g = self.get_dygrap_norm_result()

        for (p_np, g_np), (p_dy, g_dy) in zip(np_p_g, dy_out_p_g):
            np.testing.assert_allclose(g_np, g_dy, rtol=1e-06, atol=1e-08)


class TestGradClipByValue(unittest.TestCase):

    def init_value(self):
        self.max_value = 0.8
        self.min_value = -0.1
        self.init_scale = 1.0

        self.shape = (10, 10)

    def generate_p_g(self):

        self.para_and_grad = []
        for i in range(10):
            self.para_and_grad.append(
                (np.random.uniform(-self.init_scale, self.init_scale,
                                   self.shape).astype('float32'),
                 np.random.uniform(-self.init_scale, self.init_scale,
                                   self.shape).astype('float32')))

    def get_numpy_clip_result(self):

        new_p_g = []
        for p, g in self.para_and_grad:
            new_p_g.append((p, np.clip(g, self.min_value, self.max_value)))

        return new_p_g

    def get_dygrap_clip_result(self):
        with fluid.dygraph.guard():
            value_clip = GradientClipByValue(max=self.max_value,
                                             min=self.min_value)
            p_g_var = []
            for p, g in self.para_and_grad:
                new_p = to_variable(p)
                new_g = to_variable(g)
                p_g_var.append((new_p, new_g))

            new_p_g_var = value_clip(p_g_var)

            p_g_dy_out = []
            for p, g in new_p_g_var:
                p_g_dy_out.append((p.numpy(), g.numpy()))

            return p_g_dy_out

    def test_clip_by_value(self):
        self.init_value()
        self.generate_p_g()
        np_p_g = self.get_numpy_clip_result()
        dy_out_p_g = self.get_dygrap_clip_result()

        for (p_np, g_np), (p_dy, g_dy) in zip(np_p_g, dy_out_p_g):
            np.testing.assert_allclose(g_np, g_dy, rtol=1e-06, atol=1e-08)

    def test_clip_by_value_2(self):
        self.init_value()

        self.init_scale = 0.2
        self.generate_p_g()
        np_p_g = self.get_numpy_clip_result()
        dy_out_p_g = self.get_dygrap_clip_result()

        for (p_np, g_np), (p_dy, g_dy) in zip(np_p_g, dy_out_p_g):
            np.testing.assert_allclose(g_np, g_dy, rtol=1e-06, atol=1e-08)

    def test_clip_by_value_3(self):
        self.init_value()

        self.init_scale = 0.5
        self.max_value = 0.6
        self.min_value = None
        self.generate_p_g()
        np_p_g = self.get_numpy_clip_result()
        dy_out_p_g = self.get_dygrap_clip_result()

        for (p_np, g_np), (p_dy, g_dy) in zip(np_p_g, dy_out_p_g):
            np.testing.assert_allclose(g_np, g_dy, rtol=1e-06, atol=1e-08)


if __name__ == '__main__':
    unittest.main()
