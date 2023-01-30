#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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
<<<<<<< HEAD

import numpy as np
from op_test import OpTest

=======
import numpy as np
from op_test import OpTest
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
import paddle


def AffineGrid4D(theta, size, align_corners):
    n = size[0]
    w = size[3]
    h = size[2]
    h_factor = w_factor = 1
    if not align_corners:
        h_factor = (h - 1) / float(h)
        w_factor = (w - 1) / float(w)
<<<<<<< HEAD
    h_idx = (
        np.repeat(np.linspace(-1, 1, h)[np.newaxis, :], w, axis=0).T[
            :, :, np.newaxis
        ]
        * h_factor
    )
    w_idx = (
        np.repeat(np.linspace(-1, 1, w)[np.newaxis, :], h, axis=0)[
            :, :, np.newaxis
        ]
        * w_factor
    )
    grid = np.concatenate(
        [w_idx, h_idx, np.ones([h, w, 1])], axis=2
    )  # h * w * 3
=======
    h_idx = np.repeat(np.linspace(-1, 1, h)[np.newaxis, :], w,
                      axis=0).T[:, :, np.newaxis] * h_factor
    w_idx = np.repeat(np.linspace(-1, 1, w)[np.newaxis, :], h,
                      axis=0)[:, :, np.newaxis] * w_factor
    grid = np.concatenate([w_idx, h_idx, np.ones([h, w, 1])],
                          axis=2)  # h * w * 3
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    grid = np.repeat(grid[np.newaxis, :], size[0], axis=0)  # n * h * w *3

    ret = np.zeros([n, h * w, 2])
    theta = theta.transpose([0, 2, 1])
    for i in range(len(theta)):
        ret[i] = np.dot(grid[i].reshape([h * w, 3]), theta[i])
    return ret.reshape([n, h, w, 2]).astype("float32")


def AffineGrid5D(theta, size, align_corners):
    n = size[0]
    d = size[2]
    h = size[3]
    w = size[4]
    d_factor = h_factor = w_factor = 1
    if not align_corners:
        d_factor = (d - 1) / float(d)
        h_factor = (h - 1) / float(h)
        w_factor = (w - 1) / float(w)
<<<<<<< HEAD
    d_idx = (
        np.repeat(
            np.repeat(
                np.linspace(-1, 1, d)[:, np.newaxis, np.newaxis], h, axis=1
            ),
            w,
            axis=2,
        )[:, :, :, np.newaxis]
        * d_factor
    )
    h_idx = (
        np.repeat(
            np.repeat(
                np.linspace(-1, 1, h)[np.newaxis, :, np.newaxis], w, axis=2
            ),
            d,
            axis=0,
        )[:, :, :, np.newaxis]
        * h_factor
    )
    w_idx = (
        np.repeat(
            np.repeat(
                np.linspace(-1, 1, w)[np.newaxis, np.newaxis, :], h, axis=1
            ),
            d,
            axis=0,
        )[:, :, :, np.newaxis]
        * w_factor
    )
    grid = np.concatenate(
        [w_idx, h_idx, d_idx, np.ones([d, h, w, 1])], axis=3
    )  # d * h * w * 4
=======
    d_idx = np.repeat(np.repeat(
        np.linspace(-1, 1, d)[:, np.newaxis, np.newaxis], h, axis=1),
                      w,
                      axis=2)[:, :, :, np.newaxis] * d_factor
    h_idx = np.repeat(np.repeat(
        np.linspace(-1, 1, h)[np.newaxis, :, np.newaxis], w, axis=2),
                      d,
                      axis=0)[:, :, :, np.newaxis] * h_factor
    w_idx = np.repeat(np.repeat(
        np.linspace(-1, 1, w)[np.newaxis, np.newaxis, :], h, axis=1),
                      d,
                      axis=0)[:, :, :, np.newaxis] * w_factor
    grid = np.concatenate(
        [w_idx, h_idx, d_idx, np.ones([d, h, w, 1])], axis=3)  # d * h * w * 4
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    grid = np.repeat(grid[np.newaxis, :], size[0], axis=0)  # n * d * h * w * 4

    ret = np.zeros([n, d * h * w, 3])
    theta = theta.transpose([0, 2, 1])
    for i in range(len(theta)):
        ret[i] = np.dot(grid[i].reshape([d * h * w, 4]), theta[i])
    return ret.reshape([n, d, h, w, 3]).astype("float32")


class TestAffineGridOp(OpTest):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def setUp(self):
        self.initTestCase()
        self.op_type = "affine_grid"
        self.python_api = paddle.nn.functional.vision.affine_grid
        theta = np.random.randint(1, 3, self.theta_shape).astype("float32")
        self.inputs = {'Theta': theta}
        self.attrs = {
            "use_cudnn": self.use_cudnn,
<<<<<<< HEAD
            "align_corners": self.align_corners,
=======
            "align_corners": self.align_corners
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        }
        if self.dynamic_shape:
            self.inputs['OutputShape'] = self.output_shape
        else:
            self.attrs['output_shape'] = self.output_shape
<<<<<<< HEAD
        if self.theta_shape[1] == 2 and self.theta_shape[2] == 3:
            self.outputs = {
                'Output': AffineGrid4D(
                    theta, self.output_shape, self.align_corners
                )
            }
        else:
            self.outputs = {
                'Output': AffineGrid5D(
                    theta, self.output_shape, self.align_corners
                )
=======
        if (self.theta_shape[1] == 2 and self.theta_shape[2] == 3):
            self.outputs = {
                'Output': AffineGrid4D(theta, self.output_shape,
                                       self.align_corners)
            }
        else:
            self.outputs = {
                'Output': AffineGrid5D(theta, self.output_shape,
                                       self.align_corners)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
            }

    def test_check_output(self):
        self.check_output(check_eager=True)

    def test_check_grad_normal(self):
<<<<<<< HEAD
        self.check_grad(
            ['Theta'], 'Output', no_grad_set=['OutputShape'], check_eager=True
        )
=======
        self.check_grad(['Theta'],
                        'Output',
                        no_grad_set=['OutputShape'],
                        check_eager=True)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

    def initTestCase(self):
        self.theta_shape = (17, 2, 3)
        self.output_shape = np.array([17, 2, 5, 7]).astype("int32")
        self.dynamic_shape = False
        self.use_cudnn = False
        self.align_corners = True


class TestAffineGridOpCase1(TestAffineGridOp):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def initTestCase(self):
        self.theta_shape = (20, 2, 3)
        self.output_shape = np.array([20, 2, 5, 7]).astype("int32")
        self.dynamic_shape = True
        self.use_cudnn = True
        if paddle.fluid.core.is_compiled_with_rocm():
<<<<<<< HEAD
            self.use_cudnn = (
                False  # ROCM platform do not have MIOPEN kernel for affine_grid
            )
=======
            self.use_cudnn = False  # ROCM platform do not have MIOPEN kernel for affine_grid
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        self.align_corners = True


class TestAffineGridOpCase2(TestAffineGridOp):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def initTestCase(self):
        self.theta_shape = (20, 2, 3)
        self.output_shape = np.array([20, 2, 5, 7]).astype("int32")
        self.dynamic_shape = True
        self.use_cudnn = False
        self.align_corners = True


class TestAffineGridOpCase3(TestAffineGridOp):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def initTestCase(self):
        self.theta_shape = (20, 2, 3)
        self.output_shape = np.array([20, 2, 5, 7]).astype("int32")
        self.dynamic_shape = True
        self.use_cudnn = False
        self.align_corners = False


class TestAffineGridOpCase4(TestAffineGridOp):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def initTestCase(self):
        self.theta_shape = (25, 2, 3)
        self.output_shape = np.array([25, 2, 5, 6]).astype("int32")
        self.dynamic_shape = False
        self.use_cudnn = False
        self.align_corners = False


class TestAffineGridOp5DCase1(TestAffineGridOp):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def initTestCase(self):
        self.theta_shape = (20, 3, 4)
        self.output_shape = np.array([20, 1, 2, 5, 7]).astype("int32")
        self.dynamic_shape = True
        self.use_cudnn = False
        self.align_corners = False


class TestAffineGridOp5DCase2(TestAffineGridOp):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def initTestCase(self):
        self.theta_shape = (20, 3, 4)
        self.output_shape = np.array([20, 1, 2, 5, 7]).astype("int32")
        self.dynamic_shape = True
        self.use_cudnn = False
        self.align_corners = True


class TestAffineGridOp5DCase3(TestAffineGridOp):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def initTestCase(self):
        self.theta_shape = (20, 3, 4)
        self.output_shape = np.array([20, 1, 2, 5, 7]).astype("int32")
        self.dynamic_shape = True
        self.use_cudnn = False
        self.align_corners = False


class TestAffineGridOp5DCase4(TestAffineGridOp):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def initTestCase(self):
        self.theta_shape = (25, 3, 4)
        self.output_shape = np.array([25, 1, 2, 5, 6]).astype("int32")
        self.dynamic_shape = False
        self.use_cudnn = False
        self.align_corners = False


if __name__ == '__main__':
    paddle.enable_static()
    unittest.main()
