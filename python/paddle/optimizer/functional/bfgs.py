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

from line_search import strong_wolfe
import paddle
from utils import _value_and_gradient
import numpy as np
from paddle.fluid.framework import in_dygraph_mode

def miminize_bfgs(objective_func,
                  initial_position,
                  max_iters=50,
                  tolerance_grad=1e-8,
                  tolerance_change=1e-9,
                  initial_inverse_hessian_estimate=None,
                  line_search_fn='strong_wolfe',
                  max_line_search_iters=50,
                  initial_step_length=1.0,
                  dtype='float32',
                  name=None):
    
    
    I = paddle.eye(initial_position.shape[0], dtype=dtype)
    if initial_inverse_hessian_estimate == None:
        initial_inverse_hessian_estimate = I
    Hk = paddle.assign(initial_inverse_hessian_estimate)
    x1 = paddle.assign(initial_position)
    f1,g1 = _value_and_gradient(objective_func, x1)
    if in_dygraph_mode():
        
        k = 0
        while k < max_iters:
            gnorm = paddle.linalg.norm(g1, p=np.inf)
            if gnorm < tolerance_grad:
                break
            if paddle.any(paddle.isinf(x1)):
                break
            
            pk = -paddle.matmul(Hk, g1)
            alpha, value, _, _ = strong_wolfe(f=objective_func, xk=x1, pk=pk)
            
            x2 = x1 + alpha * pk
            sk = x2 - x1
            _, g2 = _value_and_gradient(objective_func, x2)
            yk = g2 - g1
            
            x1 = x2
            g1 = g2
            
            yk = paddle.unsqueeze(yk,0)
            sk = paddle.unsqueeze(sk,0)
            
            rhok = 1. / paddle.dot(yk, sk)
            
            if paddle.any(paddle.isinf(rhok)):
                rhok = 1000.0
            
            Vk_transpose = I - rhok * sk * yk.t()
            Vk = I - rhok * yk * sk.t()
            Hk = paddle.matmul(paddle.matmul(Vk_transpose, Hk), Vk) + rhok * sk * sk.t()
            k += 1

        return x1,f1,g1,Hk
    else:
        k = paddle.full(shape=[1], fill_value=0, dtype='int64')
        done = paddle.full(shape=[1], fill_value=False, dtype='bool')
        def cond(k, x1, g1, Hk, done):
            gnorm = paddle.linalg.norm(g1, p=np.inf)
            done = done | (gnorm < tolerance_grad) | paddle.any(paddle.isinf(x1))
            return (k < max_iters) & ~done
        
        def body(k, x1, g1, Hk, done):
            pk = -paddle.matmul(Hk, g1)
            alpha, value, _, _ = strong_wolfe(f=objective_func, xk=x1, pk=pk)
            x2 = x1 + alpha * pk
            sk = x2 - x1
            paddle.assign(paddle.linalg.norm(sk, p=np.inf) < tolerance_change, done)
            _, g2 = _value_and_gradient(objective_func, x2)
            yk = g2 - g1
            
            paddle.assign(x2, x1)
            paddle.assign(g2, g1)
            
            yk = paddle.unsqueeze(yk, 0)
            sk = paddle.unsqueeze(sk, 0)
            
            rhok = 1. / paddle.dot(yk, sk)
            def true_fn(rhok):
                paddle.assign(1000.0, rhok)
            paddle.static.nn.cond(paddle.any(paddle.isinf(rhok)), lambda: true_fn(rhok), None)
            static_print = paddle.static.Print(rhok, message="body1 rhok")

            Vk_transpose = I - rhok * sk * yk.t()
            Vk = I - rhok * yk * sk.t()
            Hk = paddle.matmul(paddle.matmul(Vk_transpose, Hk), Vk) + rhok * sk * sk.t()
            paddle.assign(k+1,k)
            return [k, x1, g1, Hk, done]
        paddle.static.nn.while_loop(
            cond=cond,
            body=body,
            loop_vars=[k, x1, g1, Hk, done])
        return x1,f1,g1,Hk
