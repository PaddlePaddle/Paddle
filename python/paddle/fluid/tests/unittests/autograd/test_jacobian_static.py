# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
import paddle.fluid as fluid
from utils import _compute_numerical_jacobian, _compute_numerical_batch_jacobian

def approx_jacobian(f, xs, dtype, eps=1e-5, batch=False):
    r"""Computes an approximate Jacobian matrix of a multi-valued function 
    using finite differences.
    
    The function input is required to be an np array or a list of list of np 
    arrays. 
    """
    def flatten(x):
        to = [x.shape[0], -1] if batch else [-1]
        return x.reshape(to)
    
    def flatten_all(xs):
        if isinstance(xs, np.ndarray):
            flattened = flatten(xs)
        else:
            flattened = np.concatenate([flatten(x) for x in xs], axis=-1)
        return flattened

    def x_like(x, orig_x):
        return x.reshape(orig_x.shape)

    def _f(x):
        if multi_inps:
            _xs = np.split(x, splits, axis=-1)
            _xs = [x_like(_x, _o) for _x, _o in zip(_xs, xs)]      
            outs = f(_xs)
        else:
            outs = f(x)
        return flatten_all(outs)

    multi_inps = False if isinstance(xs, np.ndarray) else True
    x = flatten_all(xs)
    xdim = x.shape[-1]
    splits = []

    if multi_inps:
        split = 0
        for inp in xs:
            split += flatten(inp).shape[-1]
            splits.append(split)

    ds = eps * np.eye(xdim, dtype=dtype) 

    fprimes_by_x = [(0.5/eps) * (_f(x + d) - _f(x - d)) for d in ds]
    fprimes_by_y = np.stack(fprimes_by_x, axis=-1)
    return np.transpose(fprimes_by_y, [1, 0, 2]) if batch else fprimes_by_y

class TestJacobian(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        paddle.enable_static()
        self.np_dtype = np.float32
        self.A = np.array([[1., 2.]]).astype('float32')
        self.B = np.array([[1., 2.], [2., 1.]]).astype('float32')
        self.C = np.array([[2., 2.], [2., 1.]]).astype('float32')
        self.D = np.array([[[2., 2.], [2., 1.]], [[1., 2.], [2., 1.]]]).astype('float32')
        self.E = np.array([[[3., 4.], [2., 3.]], [[2., 1.], [1., 3.]]]).astype('float32')
        self.eps = 1e-4
        self.rtol = 1e-3
        self.atol = 1e-3

    def run_test(self, pd_f, np_f, inps, dtype, batch=False):
        def make_tensors(inps):
            if isinstance(inps, list):
                xs = [paddle.static.data(f'x{i}', inp.shape, dtype=inp.dtype) 
                            for i, inp in enumerate(inps)]
            else:
                xs = paddle.static.data(name='x', shape=inps.shape, dtype=inps.dtype)
            return xs

        main = fluid.Program()
        startup = fluid.Program()
        with fluid.program_guard(main, startup):
            xs = make_tensors(inps)
            JJ = paddle.autograd.functional.Jacobian(pd_f, xs, batch=batch)
            nrow, ncol = JJ.shape()
            full_jacobian = JJ[:]
        place = fluid.CUDAPlace(0)
        exe = fluid.Executor(place)
        exe.run(startup)
        if isinstance(inps, list):
            feeds = {f'x{i}': x for i, x in enumerate(inps)}
        else:
            feeds = {'x': inps}
        pd_jacobians = exe.run(main, feed=feeds, fetch_list=[full_jacobian])[0]
        np_jacobians = approx_jacobian(np_f, inps, dtype, self.eps, batch=batch)
        self.assertTrue(np.allclose(pd_jacobians, np_jacobians, self.rtol, self.atol))

    def test_square(self):
        def pd_f(x):
            return paddle.multiply(x, x)
        def np_f(x):
            return np.multiply(x, x)
        self.run_test(pd_f, np_f, self.A, np.dtype('float32'))

    def test_mul(self):
        def pd_f(xs):
            x, y = xs
            return paddle.multiply(x, y)
        def np_f(xs):
            x, y = xs
            return np.multiply(x, y)
        self.run_test(pd_f, np_f, [self.B, self.C], np.dtype('float32'))        

    def test_matmul(self):
        def pd_f(xs):
            x, y = xs
            return paddle.matmul(x, y)
        def np_f(xs):
            x, y = xs
            return np.matmul(x, y)
        self.run_test(pd_f, np_f, [self.B, self.C], np.dtype('float32'))

    def test_batch_matmul(self):
        def pd_f(xs):
            x, y = xs
            return paddle.matmul(x, y)
        def np_f(xs):
            x, y = xs
            return np.matmul(x, y)
        self.run_test(pd_f, np_f, [self.B, self.C], np.dtype('float32'), batch=True)

if __name__ == "__main__":
    unittest.main()