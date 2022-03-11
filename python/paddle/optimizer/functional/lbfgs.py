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
from paddle.fluid.dygraph.dygraph_to_static.convert_operators import convert_pop,convert_len

def miminize_lbfgs(objective_func,
                  initial_position,
                  history_size=100,
                  max_iters=50,
                  tolerance_grad=1e-8,
                  tolerance_change=1e-6,
                  initial_inverse_hessian_estimate=None,
                  line_search_fn='strong_wolfe',
                  max_line_search_iters=50,
                  initial_step_length=1.0,
                  dtype='float32',
                  name=None):
    if initial_inverse_hessian_estimate == None:
        initial_inverse_hessian_estimate = paddle.eye(initial_position.shape[0], dtype=dtype)

    H0 = paddle.assign(initial_inverse_hessian_estimate)
    x1 = paddle.assign(initial_position)
    f1,g1 = _value_and_gradient(objective_func, x1)
    
    ai_vec = [None] * history_size
    if in_dygraph_mode():
        k = 0
        sk_vec = []
        yk_vec = []
        rhok_vec = []
        while k < max_iters:
            gnorm = paddle.linalg.norm(g1, p=np.inf)
            if gnorm < tolerance_grad:
                break
            if paddle.any(paddle.isinf(x1)):
                break
            
            vec_len = len(sk_vec)
            _, q = _value_and_gradient(objective_func, x1)
            for i in range(vec_len - 1, -1, -1):
                ai_vec[i] = rhok_vec[i] * paddle.dot(sk_vec[i], q)
                q = q - ai_vec[i] * yk_vec[i]
            
            r = paddle.matmul(H0, q)
            
            for i in range(vec_len):
                beta = rhok_vec[i] * paddle.dot(yk_vec[i], r)
                r = r + sk_vec[i] * (ai_vec[i] - beta)
                
            pk = -r
            if paddle.linalg.norm(pk, p=np.inf) < tolerance_change:
                break
            
            alpha, value, _, _ = strong_wolfe(f=objective_func, xk=x1, pk=pk)
            
            x2 = x1 + alpha * pk
            sk = x2 - x1
            _, g2 = _value_and_gradient(objective_func, x2)
            yk = g2 - g1
            rhok = 1. / paddle.dot(yk, sk)
            
            if len(sk_vec) > history_size:
                sk_vec.pop(0)
                yk_vec.pop(0)
                rhok_vec.pop(0)
            sk_vec.append(sk)
            yk_vec.append(yk)
            rhok_vec.append(rhok)
            x1 = x2
            g1 = g2
            
            k += 1

        return x1,f1,g1
    else:
        shape = initial_position.shape[0]
        k = paddle.full(shape=[1], fill_value=0, dtype='int64')
        history_size = paddle.assign(history_size)
        head = paddle.full(shape=[1], fill_value=1, dtype='int64')
        tail = paddle.full(shape=[1], fill_value=0, dtype='int64')
        done = paddle.full(shape=[1], fill_value=False, dtype='bool')
        sk_vec = paddle.zeros((history_size + 1, shape), dtype="float32")
        yk_vec = paddle.zeros((history_size + 1, shape), dtype="float32")
        rhok_vec = paddle.zeros((history_size + 1, 1), dtype="float32")
        ai_vec = paddle.zeros((history_size + 1, 1), dtype="float32")

        def cond(k, x1, g1, done, sk_vec, yk_vec, rhok_vec, head, tail):
            gnorm = paddle.linalg.norm(g1, p=np.inf)
            done = done | (gnorm < tolerance_grad) | paddle.any(paddle.isinf(x1))
            return (k < max_iters) & ~done
        
        def body(k, x1, g1, done, sk_vec, yk_vec, rhok_vec, head, tail):
            q = paddle.assign(g1)
            
            i = paddle.full(shape=[1], fill_value=(head - 1).mod(history_size), dtype='int64')
            
            def cond(i, q):
                return i != tail
            def body(i, q):
                ai_vec[i] = rhok_vec[i] * paddle.dot(sk_vec[i], q)
                q = q - ai_vec[i] * yk_vec[i]
                i = (i - 1).mod(history_size)
                return i,q            
            paddle.static.nn.while_loop(cond=cond, body=body, loop_vars=[i, q])
            
            r = paddle.matmul(H0, q)

            i = paddle.full(shape=[1], fill_value=tail+1, dtype='int64')
            static_print = paddle.static.Print(i, message="i")
            def cond(i, r):
                return i != head
            def body(i, r):
                beta = rhok_vec[i] * paddle.dot(yk_vec[i], r)
                r = r + sk_vec[i] * (ai_vec[i] - beta)
                i = (i + 1).mod(history_size)
                static_print = paddle.static.Print(beta, message="beta")
                return i, r            
            paddle.static.nn.while_loop(cond=cond, body=body, loop_vars=[i, r])
            
            pk = -r
            
            done = paddle.linalg.norm(pk, p=np.inf) < tolerance_change
            
            alpha, _, _, _ = strong_wolfe(f=objective_func, xk=x1, pk=pk)
            static_print = paddle.static.Print(alpha, message="alpha")
            x2 = x1 + alpha * pk
            sk = x2 - x1
            _, g2 = _value_and_gradient(objective_func, x2)
            yk = g2 - g1

            yk = paddle.unsqueeze(yk, 0)
            sk = paddle.unsqueeze(sk, 0)
            
            rhok = 1. / paddle.dot(yk, sk)
            
            sk_vec[head] = sk
            yk_vec[head] = yk
            rhok_vec[head] = rhok
            head = (head + 1) % history_size
            
            def true_fn(tail):
                paddle.assign(tail+1, tail)
            paddle.static.nn.cond(head == tail, lambda: true_fn(tail), None)
            
            paddle.assign(x2, x1)
            paddle.assign(g2, g1)
            
            paddle.assign(k+1,k)
            return [k, x1, g1, done, sk_vec, yk_vec, rhok_vec, head, tail]
        paddle.static.nn.while_loop(
            cond=cond,
            body=body,
            loop_vars=[k, x1, g1, done, sk_vec, yk_vec, rhok_vec, head, tail])
        return x1,f1,g1
