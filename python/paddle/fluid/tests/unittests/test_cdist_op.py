# # Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
# #
# # Licensed under the Apache License, Version 2.0 (the "License");
# # you may not use this file except in compliance with the License.
# # You may obtain a copy of the License at
# #
# #     http://www.apache.org/licenses/LICENSE-2.0
# #
# # Unless required by applicable law or agreed to in writing, software
# # distributed under the License is distributed on an "AS IS" BASIS,
# # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# # See the License for the specific language governing permissions and
# # limitations under the License.

import unittest

import numpy as np
from eager_op_test import OpTest

import paddle


def _cdist(x, y, p=2.0):
    r1 = x.shape[-2]
    r2 = y.shape[-2]
    if r1 == 0 or r2 == 0:
        return np.empty((r1, r2), x.dtype)
    return np.linalg.norm(x[..., None, :] - y[..., None, :, :], ord=p, axis=-1)


class TestCdistOp(OpTest):
    def setUp(self):
        self.op_type = 'cdist'
        self.python_api = paddle.nn.functional.cdist
        self.init_config()
        self.outputs = {'out': self.target}

    def test_check_output(self):
        self.check_output()

    def init_config(self):
        self.x = np.random.random((50, 3)).astype('float64')
        self.y = np.random.random((51, 3)).astype('float64')
        self.inputs = {'x': self.x, 'y': self.y}
        self.attrs = {
            'p': 2.0,
            'compute_mode': 'use_mm_for_euclid_dist_if_necessary',
        }
        self.target = _cdist(self.x, self.y, p=2.0)

    def test_check_grad(self):
        self.check_grad(['x', 'y'], 'out')


class TestCdistOpNormCase1(TestCdistOp):
    def init_conig(self):
        self.x = np.random.random((40, 3)).astype('float64')
        self.y = np.random.random((50, 3)).astype('float64')
        self.inputs = {'x': self.x, 'y': self.y}
        self.attrs = {
            'p': 0.0,
            'compute_mode': 'use_mm_for_euclid_dist_if_necessary',
        }
        self.target = _cdist(self.x, self.y, p=0.0)


class TestCdistOpNormCase2(TestCdistOp):
    def init_conig(self):
        self.x = np.random.random((40, 2)).astype('float64')
        self.y = np.random.random((50, 2)).astype('float64')
        self.inputs = {'x': self.x, 'y': self.y}
        self.attrs = {
            'p': 1.0,
            'compute_mode': 'use_mm_for_euclid_dist_if_necessary',
        }
        self.target = _cdist(self.x, self.y, p=1.0)


class TestCdistOpNormCase3(TestCdistOp):
    def init_conig(self):
        self.x = np.random.random((40, 10)).astype('float64')
        self.y = np.random.random((50, 10)).astype('float64')
        self.inputs = {'x': self.x, 'y': self.y}
        self.attrs = {
            'p': 2,
            'compute_mode': 'use_mm_for_euclid_dist_if_necessary',
        }
        self.target = _cdist(self.x, self.y, p=2)


class TestCdistOpNormCase4(TestCdistOp):
    def init_conig(self):
        self.x = np.random.random((30, 4)).astype('float64')
        self.y = np.random.random((70, 4)).astype('float64')
        self.inputs = {'x': self.x, 'y': self.y}
        self.attrs = {
            'p': 3.0,
            'compute_mode': 'use_mm_for_euclid_dist_if_necessary',
        }
        self.target = _cdist(self.x, self.y, p=3.0)


class TestCdistOpNormCase5(TestCdistOp):
    def init_conig(self):
        self.x = np.random.random((60, 10)).astype('float64')
        self.y = np.random.random((80, 10)).astype('float64')
        self.inputs = {'x': self.x, 'y': self.y}
        self.attrs = {
            'p': float('inf'),
            'compute_mode': 'use_mm_for_euclid_dist_if_necessary',
        }
        self.target = _cdist(self.x, self.y, p=float('inf'))


class TestCdistOpNormCase6(TestCdistOp):
    def init_conig(self):
        self.x = np.random.random((40, 3)).astype('float64')
        self.y = np.random.random((50, 3)).astype('float64')
        self.inputs = {'x': self.x, 'y': self.y}
        self.attrs = {
            'p': 1.5,
            'compute_mode': 'use_mm_for_euclid_dist_if_necessary',
        }
        self.target = _cdist(self.x, self.y, p=1.5)


class TestCdistOpNormCase7(TestCdistOp):
    def init_conig(self):
        self.x = np.random.random((40, 3)).astype('float64')
        self.y = np.random.random((50, 3)).astype('float64')
        self.inputs = {'x': self.x, 'y': self.y}
        self.attrs = {
            'p': 2.5,
            'compute_mode': 'use_mm_for_euclid_dist_if_necessary',
        }
        self.target = _cdist(self.x, self.y, p=2.5)


class TestCdistOpNormCase8(TestCdistOp):
    def init_conig(self):
        self.x = np.random.random((40, 10)).astype('float64')
        self.y = np.random.random((50, 10)).astype('float64')
        self.inputs = {'x': self.x, 'y': self.y}
        self.attrs = {'p': 2, 'compute_mode': 'use_mm_for_euclid_dist'}
        self.target = _cdist(self.x, self.y, p=2)


class TestCdistOpNormCase9(TestCdistOp):
    def init_conig(self):
        self.x = np.random.random((40, 10)).astype('float64')
        self.y = np.random.random((50, 10)).astype('float64')
        self.inputs = {'x': self.x, 'y': self.y}
        self.attrs = {'p': 2, 'compute_mode': 'donot_use_mm_for_euclid_dist'}
        self.target = _cdist(self.x, self.y, p=2)


class TestCdistOpNormBatchCase1(TestCdistOp):
    def init_conig(self):
        self.x = np.random.random((20, 30, 6, 4, 3)).astype('float64')
        self.y = np.random.random((20, 30, 6, 5, 3)).astype('float64')
        self.inputs = {'x': self.x, 'y': self.y}
        self.attrs = {
            'p': 0.0,
            'compute_mode': 'use_mm_for_euclid_dist_if_necessary',
        }
        self.target = _cdist(self.x, self.y, p=0.0)


class TestCdistOpNormBatchCase2(TestCdistOp):
    def init_conig(self):
        self.x = np.random.random((2, 30, 6, 4, 2)).astype('float64')
        self.y = np.random.random((2, 30, 6, 5, 2)).astype('float64')
        self.inputs = {'x': self.x, 'y': self.y}
        self.attrs = {
            'p': 1.0,
            'compute_mode': 'use_mm_for_euclid_dist_if_necessary',
        }
        self.target = _cdist(self.x, self.y, p=1.0)


class TestCdistOpNormBatchCase3(TestCdistOp):
    def init_conig(self):
        self.x = np.random.random((2, 3, 60, 4, 10)).astype('float64')
        self.y = np.random.random((2, 3, 60, 5, 10)).astype('float64')
        self.inputs = {'x': self.x, 'y': self.y}
        self.attrs = {
            'p': 2,
            'compute_mode': 'use_mm_for_euclid_dist_if_necessary',
        }
        self.target = _cdist(self.x, self.y, p=2)


class TestCdistOpNormBatchCase4(TestCdistOp):
    def init_conig(self):
        self.x = np.random.random((2, 3, 60, 3, 4)).astype('float64')
        self.y = np.random.random((2, 3, 60, 7, 4)).astype('float64')
        self.inputs = {'x': self.x, 'y': self.y}
        self.attrs = {
            'p': 3.0,
            'compute_mode': 'use_mm_for_euclid_dist_if_necessary',
        }
        self.target = _cdist(self.x, self.y, p=3.0)


class TestCdistOpNormBatchCase5(TestCdistOp):
    def init_conig(self):
        self.x = np.random.random((20, 3, 6, 6, 10)).astype('float64')
        self.y = np.random.random((20, 3, 6, 8, 10)).astype('float64')
        self.inputs = {'x': self.x, 'y': self.y}
        self.attrs = {
            'p': float('inf'),
            'compute_mode': 'use_mm_for_euclid_dist_if_necessary',
        }
        self.target = _cdist(self.x, self.y, p=float('inf'))


class TestCdistOpNormBatchCase6(TestCdistOp):
    def init_conig(self):
        self.x = np.random.random((20, 3, 6, 4, 3)).astype('float64')
        self.y = np.random.random((20, 3, 6, 5, 3)).astype('float64')
        self.inputs = {'x': self.x, 'y': self.y}
        self.attrs = {
            'p': 1.5,
            'compute_mode': 'use_mm_for_euclid_dist_if_necessary',
        }
        self.target = _cdist(self.x, self.y, p=1.5)


class TestCdistOpNormBatchCase7(TestCdistOp):
    def init_conig(self):
        self.x = np.random.random((2, 30, 6, 4, 3)).astype('float64')
        self.y = np.random.random((2, 30, 6, 5, 3)).astype('float64')
        self.inputs = {'x': self.x, 'y': self.y}
        self.attrs = {
            'p': 2.5,
            'compute_mode': 'use_mm_for_euclid_dist_if_necessary',
        }
        self.target = _cdist(self.x, self.y, p=2.5)


class TestCdistOpNormBatchCase8(TestCdistOp):
    def init_conig(self):
        self.x = np.random.random((2, 3, 60, 4, 10)).astype('float64')
        self.y = np.random.random((2, 3, 60, 5, 10)).astype('float64')
        self.inputs = {'x': self.x, 'y': self.y}
        self.attrs = {'p': 2, 'compute_mode': 'use_mm_for_euclid_dist'}
        self.target = _cdist(self.x, self.y, p=2)


class TestCdistOpNormBatchCase9(TestCdistOp):
    def init_conig(self):
        self.x = np.random.random((2, 3, 60, 4, 10)).astype('float64')
        self.y = np.random.random((2, 3, 60, 5, 10)).astype('float64')
        self.inputs = {'x': self.x, 'y': self.y}
        self.attrs = {'p': 2, 'compute_mode': 'donot_use_mm_for_euclid_dist'}
        self.target = _cdist(self.x, self.y, p=2)


if __name__ == '__main__':
    paddle.enable_static()
    unittest.main()
