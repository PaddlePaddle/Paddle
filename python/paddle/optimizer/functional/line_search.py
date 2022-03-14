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
from paddle.fluid.framework import in_dygraph_mode

def strong_wolfe(f,
                 xk,
                 pk,
                 max_iters=20,
                 tolerance_grad=1e-8,
                 tolerance_change=1e-8,
                 initial_step_length=1.0,
                 c1=1e-4,
                 c2=0.9,
                 alpha_max=10,
                 dtype='float32'):
    r"""Implements of line search algorithm that satisfies the strong Wolfe conditions using double zoom.
    
    Reference:
        Jorge Nocedal, Stephen J. Wright, Numerical Optimization,
        Second Edition, 2006.
    Args:
        f: the objective function to minimize. ``f`` accepts
            a multivariate input and returns a scalar.
        xk (Tensor): the starting point of the iterates.
        pk (Tensor): .
        gtol (Scalar): terminates if the gradient norm is smaller than
            this `gtol`. Currently gradient norm uses inf norm.
            The default value is 1e-8.
        xtol (Scalar): terminates if the distance of succesive iterates
            is smaller than this value. The default value is 0.
        iters (Scalar): the maximum number minimization iterations.
            The default value is 50.
        ls_iters (Scalar): the maximum number of line search iterations.
            The default value is 50.
        summary_only (boolean, optional): specifies the result type. If True 
            then returns the final result. Otherwise returns the results of
            all steps.
        dtype ('float' | 'float32' | 'float64' | 'double'): the data
        type to be used.
    
    Returns:
        summary (BfgsResult): The final optimization results if `summary_only`
            is set True.
        results (list[BfgsResult]): the results of all steps if `summary_only`
            is set False.
    """
    def phi(alpha):
        return f(xk + alpha * pk)

    def derphi(alpha):
        #print("xk + alpha * pk: ", xk + alpha * pk)
        value, f_grad = _value_and_gradient(f, xk + alpha * pk)
        phi_grad = paddle.dot(f_grad, pk)
        return phi_grad

    def zoom(alpha_lo, alpha_hi, phi_lo, derphi_lo, phi_0, derphi_0):
        max_zoom_iters = 50

        if in_dygraph_mode():
            j = 0
            while j < max_zoom_iters:
                #print("j, alpha_lo, alpha_hi: ",j,alpha_lo, alpha_hi)

                if abs(alpha_hi - alpha_lo) < tolerance_change:
                    break

                alpha_j = 0.5 * (alpha_lo + alpha_hi)
                phi_j = phi(alpha_j)
                #print("alpha_j: ", alpha_j)
                #print("xk + alpha * pk: ", xk + alpha_j * pk)
                #print("x_diff: ",paddle.abs(1.1+xk + alpha_j * pk))
                #print("x_change: ",alpha_j * pk)
                #print("pk: ",pk)
                #num_func_calls += 1
                if (phi_j > phi_0 + c1 * alpha_j * derphi_0) or phi_j >= phi_lo:
                    #print(phi_j,phi_0 + c1 * alpha_j * derphi_0,phi_lo)
                    alpha_hi = alpha_j
                else:
                    derphi_j = derphi(alpha_j)
                    if paddle.abs(derphi_j) <= -c2 * derphi_0:
                        #print("derphi_j: {} derphi_0: {}".format(derphi_j,derphi_0))
                        alpha_lo = alpha_j
                        phi_lo = phi_j
                        derphi_lo = derphi_j
                        break
                    if derphi_j * (alpha_hi - alpha_lo) >= 0:
                        #print("here 3")
                        alpha_hi = alpha_lo
                    alpha_lo = alpha_j
                    phi_lo = phi_j
                    derphi_lo = derphi_j

                j += 1

            return alpha_lo, phi_lo, derphi_lo

####################    static mode    ####################
###########################################################
        j = paddle.full(shape=[1], fill_value=0, dtype='int64')
        done_zoom = paddle.full(shape=[1], fill_value=False, dtype='bool')

        def cond(j, alpha_lo, alpha_hi, phi_lo, derphi_lo, phi_0, derphi_0,
                 done_zoom):
            pred1 = paddle.abs(alpha_hi - alpha_lo) < tolerance_change
            paddle.assign(done | pred1, done)
            #done_zoom_print = paddle.static.Print(done_zoom, message="done_zoom")
            #j_print = paddle.static.Print(j, message="j")
            return (j < max_zoom_iters) & ~done_zoom

        def body(j, alpha_lo, alpha_hi, phi_lo, derphi_lo, phi_0, derphi_0,
                 done_zoom):
            #alpha_lo_print = paddle.static.Print(alpha_lo, message="alpha_lo")
            paddle.assign(j + 1, j)
            #j_print = paddle.static.Print(j, message="j")
            alpha_j = 0.5 * (alpha_lo + alpha_hi)
            phi_j = phi(alpha_j)
            #alpha_j_print = paddle.static.Print(alpha_j, message="alpha_j")
            pred2 = (phi_j > phi_0 + c1 * alpha_j * derphi_0) | (
                phi_j >= phi_lo)

            def true_fn():
                paddle.assign(alpha_j, alpha_hi)

            def false_fn(alpha_lo, done_zoom):
                derphi_j = derphi(alpha_j)
                pred3 = (paddle.abs(derphi_j) <= -c2 * derphi_0)
                paddle.assign(pred3, done_zoom)

                #pred_print = paddle.static.Print(pred, message="pred")

                def true_fn():
                    paddle.assign(alpha_j, alpha_lo)
                    paddle.assign(phi_j, phi_lo)
                    paddle.assign(derphi_j, derphi_lo)

                paddle.static.nn.cond(pred3, true_fn, None)

                pred4 = ~done_zoom & (derphi_j * (alpha_hi - alpha_lo) >= 0)

                def true_fn():
                    paddle.assign(alpha_hi, alpha_lo)

                paddle.static.nn.cond(pred4, true_fn, None)

                paddle.assign(alpha_j, alpha_lo)
                paddle.assign(phi_j, phi_lo)
                paddle.assign(derphi_j, derphi_lo)

            paddle.static.nn.cond(pred2, true_fn,
                                  lambda: false_fn(alpha_lo, done_zoom))
            return [
                j, alpha_lo, alpha_hi, phi_lo, derphi_lo, phi_0, derphi_0,
                done_zoom
            ]

        paddle.static.nn.while_loop(
            cond=cond,
            body=body,
            loop_vars=[
                j, alpha_lo, alpha_hi, phi_lo, derphi_lo, phi_0, derphi_0,
                done_zoom
            ])
        return alpha_lo, phi_lo, derphi_lo

    alpha_0 = 0
    alpha_1 = alpha_0
    alpha_2 = initial_step_length
    phi_0, derphi_0 = phi_1, derphi_1 = phi(alpha_0), derphi(alpha_0)
    num_func_calls = 1

    alpha_star, phi_star, derphi_star = 0, 0, 0
    if in_dygraph_mode():
        i = 1
        while i < max_iters:
            phi_2 = phi(alpha_2)
            num_func_calls += 1
            if paddle.any(paddle.isinf(xk)):
                break

            #print('phi_2: {} \nphi_0: {} \nalpha_2: {} \nderphi_0: {}'.format(phi_2, phi_0, alpha_2, derphi_0))
            if (phi_2 > phi_0 + c1 * alpha_2 * derphi_0) or (phi_2 >= phi_0 and
                                                             i > 1):
                #print("xk + alpha * pk: ",xk + alpha_2 * pk)
                alpha_star, phi_star, derphi_star = zoom(
                    alpha_1, alpha_2, phi_1, derphi_1, phi_0, derphi_0)
                #print("here1")
                break

            derphi_2 = derphi(alpha_2)
            if paddle.abs(derphi_2) <= -c2 * derphi_0:
                #print('derphi_2: {}, derphi_0: {}'.format(derphi_2, derphi_0))
                alpha_star = alpha_2
                phi_star = phi_2
                derphi_star = derphi_2
                #print("here2")
                break
            #print('derphi_2: {}'.format(derphi_2))
            if derphi_2 >= 0:
                alpha_star, phi_star, derphi_star = zoom(
                    alpha_2, alpha_1, phi_2, derphi_2, phi_0, derphi_0)
                #print("here3")
                break

            alpha_1 = alpha_2
            phi_1 = phi_2
            derphi_1 = derphi_2

            alpha_2 = 2 * alpha_2
            if alpha_max is not None:
                alpha_2 = min(alpha_2, alpha_max)
            i += 1

        #print("i: ", i)
        if i == max_iters:
            alpha_star = alpha_2
            phi_star = phi(alpha_star)
            derphi_star = derphi(alpha_star)

        return alpha_star, phi_star, derphi_star, num_func_calls

    ####################    static mode    ####################
    ###########################################################
    i = paddle.full(shape=[1], fill_value=1, dtype='int64')
    num_func_calls = paddle.full(shape=[1], fill_value=0, dtype='int64')
    alpha_1 = paddle.full(shape=[1], fill_value=alpha_0, dtype='float32')
    alpha_2 = paddle.full(
        shape=[1], fill_value=initial_step_length, dtype='float32')
    #alpha_2 = initial_step_length
    phi_2 = phi(alpha_2)
    derphi_2 = derphi(alpha_2)
    done = paddle.full(shape=[1], fill_value=False, dtype='bool')
    alpha_star = paddle.full(shape=[1], fill_value=0, dtype='float32')
    phi_star = paddle.full(shape=[1], fill_value=0, dtype='float32')
    derphi_star = paddle.full(shape=[1], fill_value=0, dtype='float32')
    alpha_max = paddle.full(shape=[1], fill_value=alpha_max, dtype='float32')

    def cond(i, num_func_calls, alpha_0, alpha_1, alpha_2, phi_0, phi_1, phi_2,
             derphi_0, derphi_1, derphi_2, done):
        #i_print = paddle.static.Print(i, message="i")
        paddle.assign(done | paddle.any(paddle.isinf(xk)), done)
        #done_print = paddle.static.Print(done, message="done")
        return (i < max_iters) & ~done

    def body(i, num_func_calls, alpha_0, alpha_1, alpha_2, phi_0, phi_1, phi_2,
             derphi_0, derphi_1, derphi_2, done):
        phi_2 = phi(alpha_2)
        #num_func_calls += 1

        pred1 = (phi_2 > phi_0 + c1 * alpha_2 * derphi_0) or (phi_2 >= phi_0 and
                                                              i > 1)
        paddle.assign(done | pred1, done)

        #done_print = paddle.static.Print(done, message="done")
        #pred1_print = paddle.static.Print(pred1, message="pred1")

        def true_fn():
            a, b, c = zoom(alpha_1, alpha_2, phi_1, derphi_1, phi_0, derphi_0)
            paddle.assign(a, alpha_star)
            paddle.assign(b, phi_star)
            paddle.assign(c, derphi_star)

        #new_x = xk + alpha_2 * pk
        #new_x_print = paddle.static.Print(new_x, message="new_x")
        #phi_2_print = paddle.static.Print(phi_2, message="phi_2")
        #phi_0_print = paddle.static.Print(phi_0, message="phi_0")
        #print("alpha_2: ", alpha_2)
        #derphi_0_print = paddle.static.Print(derphi_0, message="derphi_0")
        paddle.static.nn.cond(pred1, true_fn, None)

        derphi_2 = derphi(alpha_2)
        pred2 = ~done & (paddle.abs(derphi_2) <= -c2 * derphi_0)
        #pred2_print = paddle.static.Print(pred2, message="pred2")
        paddle.assign(done | pred2, done)

        #done_print = paddle.static.Print(done, message="done")
        def true_fn():
            paddle.assign(alpha_2, alpha_star)
            paddle.assign(phi_2, phi_star)
            paddle.assign(derphi_2, derphi_star)

        paddle.static.nn.cond(pred2, true_fn, None)

        pred3 = ~done & (derphi_2 >= 0)
        #pred3_print = paddle.static.Print(pred3, message="pred3")
        paddle.assign(done | pred3, done)

        #done_print = paddle.static.Print(done, message="done")
        def true_fn():
            a, b, c = zoom(alpha_1, alpha_2, phi_1, derphi_1, phi_0, derphi_0)
            paddle.assign(a, alpha_star)
            paddle.assign(b, phi_star)
            paddle.assign(c, derphi_star)

        paddle.static.nn.cond(pred3, true_fn, None)

        paddle.assign(alpha_2, alpha_1)
        paddle.assign(phi_2, phi_1)
        paddle.assign(derphi_2, derphi_1)
        paddle.assign(paddle.minimum(2 * alpha_2, alpha_max), alpha_2)

        paddle.assign(i + 1, i)
        return [
            i, num_func_calls, alpha_0, alpha_1, alpha_2, phi_0, phi_1, phi_2,
            derphi_0, derphi_1, derphi_2, done
        ]

    paddle.static.nn.while_loop(
        cond=cond,
        body=body,
        loop_vars=[
            i, num_func_calls, alpha_0, alpha_1, alpha_2, phi_0, phi_1, phi_2,
            derphi_0, derphi_1, derphi_2, done
        ])

    def true_fn():
        paddle.assign(alpha_2, alpha_star)
        paddle.assign(phi_2, phi_star)
        paddle.assign(derphi_2, derphi_star)

    paddle.static.nn.cond(i == max_iters, true_fn)
    return alpha_star, phi_star, derphi_star, num_func_calls
