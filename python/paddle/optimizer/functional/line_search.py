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

import paddle
from utils import _value_and_gradient
import numpy as np
from paddle.fluid.framework import in_dygraph_mode
import paddle.fluid as fluid
import math

def strong_wolfe(f,
                xk,
                pk,
                max_iters=50,
                tolerance_grad=1e-8,
                tolerance_change=1e-9,
                initial_step_length = 1.0,
                c1 = 1e-4,
                c2 = 0.9,
                alpha_max=None,
                dtype='float32'):
    
    def phi(alpha):
        return f(xk + alpha * pk)
    
    def derphi(alpha):
        value, f_grad = _value_and_gradient(f, xk + alpha * pk)
        f_grad = f_grad[0]
        phi_grad = paddle.dot(f_grad, pk)
        '''
        if in_dygraph_mode():
            phi_grad = paddle.dot(f_grad, pk)
        else:
            print(f_grad)
            print(np.array([pk]))
            main = fluid.Program()
            startup = fluid.Program()
            with fluid.program_guard(main, startup):
                x = paddle.static.data(name='x', shape=np.array(f_grad).shape)
                y = paddle.static.data(name='y', shape=np.array(f_grad).shape)
                grad = paddle.dot(x, y)
            place = fluid.CUDAPlace(0)
            exe = fluid.Executor(place)
            exe.run(startup)

            feeds = {'x': np.array(f_grad).astype('float32'), 'y': np.array([pk]).astype('float32')}
            phi_grad = exe.run(main, feed=feeds, fetch_list=[grad])
        '''

        return phi_grad

    def zoom(alpha_lo, alpha_hi, phi_0, phi_prime_0):
        max_zoom_iters = 10
        j = 0
        alpha_star = None
        phi_lo = 0
        phi_prime_lo = 0
        while j < max_zoom_iters:
            if abs(alpha_hi - alpha_lo) < tolerance_change:
                break
            alpha_star = None
            alpha_j = 0.5 * (alpha_lo + alpha_hi)
            phi_j, phi_prime_j = phi(alpha_j),derphi(alpha_j)
            phi_lo, phi_prime_lo = phi(alpha_lo),derphi(alpha_lo)
            if (phi_j > phi_0 + c1 * alpha_j * phi_prime_0) or phi_j >= phi_lo:
                
                alpha_hi = alpha_j
                #print("here 1, alpha_lo: {}, alpha_hi: {}".format(alpha_lo, alpha_hi))
            else:
                #print("here 2")
                if paddle.abs(phi_prime_j) <= -c2 * phi_prime_0:
                    #print("here 3")
                    alpha_star = alpha_j
                    phi_star = phi_j
                    phi_prime_star = phi_prime_j
                    break
                if phi_prime_j * (alpha_hi - alpha_lo) >= 0:
                    #print("here 4")
                    alpha_hi = alpha_lo
                alpha_lo = alpha_j
            
            j += 1
        
        if alpha_star == None:
            alpha_star, phi_star, phi_prime_star = alpha_lo, phi(alpha_lo), derphi(alpha_lo)
        return alpha_star, phi_star, phi_prime_star

    
    alpha_0 = 0
    alpha_1 = initial_step_length


    phi_0, phi_prime_0 = phi(alpha_0),derphi(alpha_0)
    num_func_calls = 1
    
    i = 1
    alpha_star = None
    phi_star = 0
    phi_prime_star = 0
    while i < max_iters:
        phi_1,phi_prime_1 = phi(alpha_1),derphi(alpha_1)
        num_func_calls += 1
        if paddle.isinf(phi_1).item():
            break
        if (phi_1 > phi_0 + c1 * alpha_1 * phi_prime_0) or (phi_1 >= phi_0 and i > 1):
            #print('phi_1: {}, phi_0: {}, alpha_1: {}, phi_prime_0: {}'.format(phi_1,phi_0,alpha_1,phi_prime_0))
            #print('alpha_0: {}, alpha_1: {}'.format(alpha_0, alpha_1))
            alpha_star, phi_star, phi_prime_star = zoom(alpha_0, alpha_1, phi_0, phi_prime_0)
            break
        
        if paddle.abs(phi_prime_1) <= -c2 * phi_prime_0:
            #print('phi_prime_1: {}, phi_prime_0: {}'.format(phi_prime_1, phi_prime_0))
            alpha_star = alpha_1
            phi_star = phi_1
            phi_prime_star = phi_prime_1
            break
        
        if phi_prime_1 >= 0:
            #print('phi_prime_1: {}'.format(phi_prime_1))
            alpha_star, phi_star, phi_prime_star = zoom(alpha_1, alpha_0, phi_0, phi_prime_0)
            break

        alpha_2 = 2 * alpha_1
        if alpha_max is not None:
            alpha_2 = min(alpha_2, alpha_max) 

        alpha_0 = alpha_1
        alpha_1 = alpha_2
        phi_0 = phi_1
        phi_prime_0 = phi_prime_1
        i += 1

    if alpha_star == None:
        alpha_star = alpha_1
        phi_star = phi(alpha_star)
        phi_prime_star = derphi(alpha_star)
    return alpha_star, phi_star, phi_prime_star, num_func_calls
