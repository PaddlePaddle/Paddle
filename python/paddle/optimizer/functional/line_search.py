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
                 initial_step_length=1.0,
                 c1=1e-4,
                 c2=0.9,
                 alpha_max=10,
                 dtype='float32'):
    def phi(alpha):
        return f(xk + alpha * pk)

    def derphi(alpha):
        value, f_grad = _value_and_gradient(f, xk + alpha * pk)
        print(f_grad)
        phi_grad = paddle.dot(f_grad, pk)
        return phi_grad

    def zoom(alpha_lo, alpha_hi, phi_lo, derphi_lo, phi_0, derphi_0):
        max_zoom_iters = 10

        if in_dygraph_mode():
            j = 0
            while j < max_zoom_iters:
                if abs(alpha_hi - alpha_lo) < tolerance_change:
                    break

                alpha_j = 0.5 * (alpha_lo + alpha_hi)
                phi_j = phi(alpha_j)
                num_func_calls += 1
                if (phi_j > phi_0 + c1 * alpha_j * derphi_0) or phi_j >= phi_lo:
                    alpha_hi = alpha_j
                else:
                    _, derphi_j = derphi(alpha_j)
                    if paddle.abs(derphi_j) <= -c2 * derphi_0:
                        alpha_lo = alpha_j
                        phi_lo = phi_j
                        derphi_lo = derphi_j
                        break
                    if derphi_j * (alpha_hi - alpha_lo) >= 0:
                        alpha_hi = alpha_lo
                    alpha_lo = alpha_j
                    phi_lo = phi_j
                    derphi_lo = derphie_j

                j += 1

            return alpha_lo, phi_lo, derphi_lo

####################    static mode    ####################
###########################################################
        j = paddle.full(shape=[1], fill_value=0, dtype='int64')

        def cond(j, alpha_lo, alpha_hi, phi_lo, derphi_lo, phi_0, derphi_0):
            return j < max_zoom_iters or abs(alpha_hi - alpha_lo) < tolerance_change \
            or paddle.abs(derphie_j) <= -c2 * derphi_0

        def body(j, alpha_lo, alpha_hi, phi_lo, derphi_lo, phi_0, derphi_0):
            j += 1
            alpha_j = 0.5 * (alpha_lo + alpha_hi)
            phi_j = phi(alpha_j)

            pred = phi_j > phi_0 + c1 * alpha_j * derphi_0 or phi_j >= phi_lo

            def true_fn():
                alpha_hi = alpha_j

            def false_fn(alpha_lo):
                derphi_j = derphi(alpha_j)
                pred = paddle.abs(derphi_j) <= -c2 * derphi_0

                def true_fn():
                    alpha_lo = alpha_j
                    phi_lo = phi_j
                    derphi_lo = derphi_j

                paddle.static.nn.cond(pred, true_fn, None)

                pred = derphi_j * (alpha_hi - alpha_lo) >= 0

                def true_fn():
                    alpha_hi = alpha_lo

                paddle.static.nn.cond(pred, true_fn, None)

                alpha_lo = alpha_j
                phi_lo = phi_j
                derphi_lo = derphi_j

            paddle.static.nn.cond(pred, true_fn, lambda: false_fn(alpha_lo))
            return [j, alpha_lo, alpha_hi, phi_lo, derphi_lo, phi_0, derphi_0]

        paddle.static.nn.while_loop(
            cond=cond,
            body=body,
            loop_vars=[
                j, alpha_lo, alpha_hi, phi_lo, derphi_lo, phi_0, derphi_0
            ])
        return alpha_lo, phi_lo, derphi_lo

    alpha_0 = 0
    alpha_1 = alpha_0
    alpha_2 = initial_step_length
    phi_0, derphi_0 = phi_1, derphi_1 = phi(alpha_0), derphi(alpha_0)
    num_func_calls = 1

    if in_dygraph_mode():
        i = 1
        while i < max_iters:
            phi_2 = phi(alpha_2)
            num_func_calls += 1
            if paddle.isinf(phi_2).item():
                break
            if (phi_2 > phi_0 + c1 * alpha_2 * derphi_0) or (phi_2 >= phi_0 and
                                                             i > 1):
                #print('phi_1: {}, phi_0: {}, alpha_1: {}, phi_prime_0: {}'.format(phi_1,phi_0,alpha_1,phi_prime_0))
                #print('alpha_0: {}, alpha_1: {}'.format(alpha_0, alpha_1))
                alpha_star, phi_star, phi_prime_star = zoom(
                    alpha_1, alpha_2, phi_1, derphi_1, phi_0, derphi_0)
                break

            derphi_2 = derphi(alpha_2)
            if paddle.abs(derphi_2) <= -c2 * derphi_0:
                #print('phi_prime_1: {}, phi_prime_0: {}'.format(phi_prime_1, phi_prime_0))
                alpha_star = alpha_2
                phi_star = phi_2
                derphi_star = derphi_2
                break

            if phi_prime_2 >= 0:
                #print('phi_prime_1: {}'.format(phi_prime_1))
                alpha_star, phi_star, phi_prime_star = zoom(
                    alpha_2, alpha_1, phi_2, derphi_2, phi_0, derphi_0)
                break

            alpha_2 = 2 * alpha_2
            if alpha_max is not None:
                alpha_2 = min(alpha_2, alpha_max)

            alpha_1 = alpha_2
            phi_1 = phi_2
            derphi_1 = derphi_2
            i += 1

        else:
            alpha_star = alpha_2
            phi_star = phi(alpha_star)
            phi_prime_star = derphi(alpha_star)
        return alpha_star, phi_star, phi_prime_star, num_func_calls

    ####################    static mode    ####################
    ###########################################################
    i = paddle.full(shape=[1], fill_value=0, dtype='int64')
    num_func_calls = paddle.full(shape=[1], fill_value=0, dtype='int64')
    phi_2 = phi(alpha_1)
    derphi_2 = derphi(alpha_1)

    def cond(i, num_func_calls, alpha_0, alpha_1, alpha_2, phi_0, phi_1, phi_2,
             derphi_0, derphi_1, derphi_2):
        pred0 = i < max_iters
        pred1 = paddle.isinf(phi_2)
        pred2 = (phi_2 > phi_0 + c1 * alpha_2 * derphi_0) or (phi_2 >= phi_0 and
                                                              i > 1)
        pred3 = paddle.abs(derphi_2) <= -c2 * derphi_0
        pred4 = derphi_2 >= 0
        #return (pred1 or pred2 or pred3 or pred4)
        return i < 10

    def body(i, num_func_calls, alpha_0, alpha_1, alpha_2, phi_0, phi_1, phi_2,
             derphi_0, derphi_1, derphi_2):
        phi_2 = phi(alpha_2)
        num_func_calls += 1

        pred = (phi_2 > phi_0 + c1 * alpha_2 * derphi_0) or (phi_2 >= phi_0 and
                                                             i > 1)

        def true_fn():
            alpha_star, phi_star, phi_prime_star = zoom(
                alpha_1, alpha_2, phi_1, derphi_1, phi_0, derphi_0)

        paddle.static.nn.cond(pred, true_fn, None)

        derphi_2 = derphi(alpha_2)
        pred = paddle.abs(derphi_2) <= -c2 * derphi_0

        def true_fn():
            alpha_star = alpha_2
            phi_star = phi_2
            phi_prime_star = derphi_2

        paddle.static.nn.cond(pred, true_fn, None)

        pred = derphi_2 >= 0

        def true_fn():
            alpha_star, phi_star, derphi_star = zoom(alpha_2, alpha_1, phi_2,
                                                     derphi_2, phi_0, derphi_0)

        paddle.static.nn.cond(pred, true_fn, None)

        alpha_1 = alpha_2
        phi_1 = phi_2
        derphi_1 = derphi_2

        alpha_2 = min(2 * alpha_2, alpha_max)
        i += 1
        return [
            i, num_func_calls, alpha_0, alpha_1, alpha_2, phi_0, phi_1, phi_2,
            derphi_0, derphi_1, derphi_2
        ]

    paddle.static.nn.while_loop(
        cond=cond,
        body=body,
        loop_vars=[
            i, num_func_calls, alpha_0, alpha_1, alpha_2, phi_0, phi_1, phi_2,
            derphi_0, derphi_1, derphi_2
        ])

    #alpha_star = alpha_2
    alpha_star = paddle.full(shape=[1], fill_value=alpha_2, dtype='int64')
    phi_star = phi(alpha_2)
    phi_prime_star = derphi(alpha_2)
    print(alpha_star, phi_star, phi_prime_star, num_func_calls)
    return alpha_star, phi_star, phi_prime_star, num_func_calls
