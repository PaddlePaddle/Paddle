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

def cubic_interpolation_(x1, f1, g1, x2, f2, g2):
    # cubic interpolation use two points and their function value and gradient.
    # Usually it has a better covergence than bisect.
    if in_dygraph_mode():
        if x1 <= x2:
            xmin, xmax = (x1, x2)
        else:
            xmin, xmax = (x2, x1)
        
        d1 = g1 + g2 - 3 * (f1 - f2) / (x1 - x2)
        d2_square = d1 ** 2 - g1 * g2
        if d2_square >= 0:
            d2 = d2_square.sqrt()
            if x1 <= x2:
                min_pos = x2 - (x2 - x1) * ((g2 + d2 - d1) / (g2 - g1 + 2 * d2))
            else:
                min_pos = x1 - (x1 - x2) * ((g1 + d2 - d1) / (g1 - g2 + 2 * d2))
            return min(max(min_pos, xmin), xmax)
        else:
            return (xmin + xmax) / 2.
    else:
        xmin, xmax = paddle.static.nn.cond(x1 <= x2, lambda: (x1, x2), lambda: (x2, x1))
        d1 = g1 + g2 - 3 * (f1 - f2) / (x1 - x2)
        d2_square = d1 ** 2 - g1 * g2
        def true_func():
            d2 = d2_square.sqrt()
            def true_fn():
                return x2 - (x2 - x1) * ((g2 + d2 - d1) / (g2 - g1 + 2 * d2))
            def false_fn():
                return x1 - (x1 - x2) * ((g1 + d2 - d1) / (g1 - g2 + 2 * d2))
            pred = paddle.less_equal(x=x1, y=x2)
            min_pos = paddle.static.nn.cond(pred, true_fn, false_fn)
            
            return paddle.minimum(paddle.maximum(min_pos, xmin), xmax)

        def false_func():
            return (xmin + xmax) / 2.

        min_pos = paddle.static.nn.cond(d2_square >= 0., true_func, false_func)
        return min_pos


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
        Jorge Nocedal, Stephen J. Wright, Numerical Optimization, Second Edition, 2006.
    Args:
        f: the objective function to minimize. ``f`` accepts a multivariate input and returns a scalar.
        xk (Tensor): the starting point of the iterates.
        pk (Tensor): search direction.
        max_iters (Scalar): the maximum number of iterations.
        tolerance_grad (Scalar): terminates if the gradient norm is smaller than
            this. Currently gradient norm uses inf norm.
        tolerance_change (Scalar): terminates if the change of function value/position/parameter between 
            two iterations is smaller than this value.
        initial_step_length (Scalar): step length used in first iteration.
        c1 (Scalar): parameter for sufficient decrease condition.
        c2 (Scalar): parameter for curvature condition.
        alpha_max (Scalar): max step length.
        dtype ('float' | 'float32' | 'float64' | 'double'): the datatype to be used.
    
    Returns:
        num_func_calls : number of objective function called in line search process.
        alpha_star : optimal step length, or None if the line search algorithm did not converge.
        phi_star : phi at alpha_star.
        derphi_star : derphi at alpha_star.
    """

    def phi_and_derphi(alpha):
        # phi = f(xk + alpha * pk)
        phi_value, f_grad = _value_and_gradient(f, xk + alpha * pk)
        phi_grad = paddle.dot(f_grad, pk)
        return phi_value, f_grad, phi_grad
    
    def zoom(alpha_lo, phi_lo, derphi_lo, derf_lo, alpha_hi, phi_hi, derphi_hi, phi_0, derphi_0):
        max_zoom_iters = max_iters
        done_zoom = paddle.full(shape=[1], fill_value=False, dtype='bool')
        j = paddle.full(shape=[1], fill_value=0, dtype='int64')
        
        if in_dygraph_mode():
            while j < max_zoom_iters:
                if paddle.abs(alpha_hi - alpha_lo) < tolerance_change:
                    break

                alpha_j = cubic_interpolation_(alpha_lo, phi_lo, derphi_lo, alpha_hi, phi_hi, derphi_hi)
                min_change = 0.1 * paddle.abs(alpha_hi - alpha_lo)
                if paddle.minimum(paddle.abs(alpha_j - alpha_lo), paddle.abs(alpha_j - alpha_hi)) < min_change:
                    alpha_j = 0.5 * (alpha_lo + alpha_hi)

                phi_j, derf_j, derphi_j = phi_and_derphi(alpha_j)

                if (phi_j > phi_0 + c1 * alpha_j * derphi_0) or phi_j >= phi_lo:
                    alpha_hi = alpha_j
                    phi_hi = phi_j
                    derphi_hi = derphi_j
                else:
                    if paddle.abs(derphi_j) <= -c2 * derphi_0:
                        alpha_lo = alpha_j
                        phi_lo = phi_j
                        derphi_lo = derphi_j
                        derf_lo = derf_j
                        break
                    if derphi_j * (alpha_hi - alpha_lo) >= 0:
                        alpha_hi = alpha_lo
                        phi_hi = phi_lo
                        derphi_hi = derphi_lo
                    alpha_lo = alpha_j
                    phi_lo = phi_j
                    derphi_lo = derphi_j
                    derf_lo = derf_j

                j += 1
            return alpha_lo, phi_lo, derf_lo, j
        else:
            done_zoom = paddle.full(shape=[1], fill_value=False, dtype='bool')
            def cond(j, done_zoom, alpha_lo, phi_lo, derphi_lo, derf_lo, alpha_hi, phi_hi, derphi_hi):
                pred = paddle.abs(alpha_hi - alpha_lo) < tolerance_change
                paddle.assign(done | pred, done)
                return (j < max_zoom_iters) & ~done_zoom

            def body(j, done_zoom, alpha_lo, phi_lo, derphi_lo, derf_lo, alpha_hi, phi_hi, derphi_hi):
                alpha_j = cubic_interpolation_(alpha_lo, phi_lo, derphi_lo, alpha_hi, phi_hi, derphi_hi) # 21
                min_change = 0.1 * paddle.abs(alpha_hi - alpha_lo)
                pred = paddle.minimum(paddle.abs(alpha_j - alpha_lo), paddle.abs(alpha_j - alpha_hi)) < min_change
                alpha_j = paddle.static.nn.cond(pred, lambda: 0.5 * (alpha_lo + alpha_hi), lambda: alpha_j)

                phi_j, derf_j, derphi_j = phi_and_derphi(alpha_j)

                def true_fn():
                    paddle.assign(alpha_j, alpha_hi)
                    paddle.assign(phi_j, phi_hi)
                    paddle.assign(derphi_j, derphi_hi)

                def false_fn(alpha_lo, done_zoom):
                    def true_fn():
                        paddle.assign(alpha_j, alpha_lo)
                        paddle.assign(phi_j, phi_lo)
                        paddle.assign(derphi_j, derphi_lo)
                        paddle.assign(derf_j, derf_lo)

                    pred3 = (paddle.abs(derphi_j) <= -c2 * derphi_0)
                    paddle.assign(pred3, done_zoom)
                    paddle.static.nn.cond(pred3, true_fn, None)

                    def true_fn():
                        paddle.assign(alpha_hi, alpha_lo)

                    pred4 = ~done_zoom & (derphi_j * (alpha_hi - alpha_lo) >= 0)
                    paddle.static.nn.cond(pred4, true_fn, None)

                    paddle.assign(alpha_j, alpha_lo)
                    paddle.assign(phi_j, phi_lo)
                    paddle.assign(derphi_j, derphi_lo)
                    paddle.assign(derf_j, derf_lo)

                pred2 = (phi_j > phi_0 + c1 * alpha_j * derphi_0) | (
                    phi_j >= phi_lo)
                paddle.static.nn.cond(pred2, true_fn,
                                    lambda: false_fn(alpha_lo, done_zoom))
                j = paddle.static.nn.cond(done_zoom, lambda: j, lambda: j + 1)
                return [
                    j, done_zoom, alpha_lo, phi_lo, derphi_lo, derf_lo, alpha_hi, phi_hi, derphi_hi
                ]

            paddle.static.nn.while_loop(
                cond=cond,
                body=body,
                loop_vars=[
                    j, done_zoom, alpha_lo, phi_lo, derphi_lo, derf_lo, alpha_hi, phi_hi, derphi_hi
                ])
            
            return alpha_lo, phi_lo, derf_lo, j
    
    # variable that can be used by both mode
    alpha_max = paddle.full(shape=[1], fill_value=alpha_max, dtype=dtype)

    alpha_1 = paddle.full(shape=[1], fill_value=0., dtype=dtype)
    alpha_2 = paddle.full(shape=[1], fill_value=initial_step_length, dtype=dtype)

    phi_1, derf_1, derphi_1  = phi_and_derphi(alpha_1)
    phi_0 = paddle.assign(phi_1)
    derphi_0 = paddle.assign(derphi_1)
    ls_func_calls = paddle.full(shape=[1], fill_value=1, dtype='int64')

    alpha_star = paddle.full(shape=[1], fill_value=0, dtype=dtype)
    phi_star = paddle.assign(phi_1)
    derf_star = paddle.assign(derf_1)

    i = paddle.full(shape=[1], fill_value=0, dtype='int64')
    if in_dygraph_mode():
        while i < max_iters:
            phi_2, derf_2, derphi_2 = phi_and_derphi(alpha_2)
            ls_func_calls += 1
            if paddle.any(paddle.isinf(phi_2)):
                break
            
            if (phi_2 > phi_0 + c1 * alpha_2 * derphi_0) or (phi_2 >= phi_0 and
                                                            i > 1):
                alpha_star, phi_star, derf_star, zoom_func_calls = zoom(
                    alpha_1, phi_1, derphi_1, derf_1, alpha_2, phi_2, derphi_2, phi_0, derphi_0)
                
                ls_func_calls += zoom_func_calls
                break

            if paddle.abs(derphi_2) <= -c2 * derphi_0:
                alpha_star = alpha_2
                phi_star = phi_2
                derf_star = derf_2
                break
            if derphi_2 >= 0:
                alpha_star, phi_star, derf_star, zoom_func_calls = zoom(
                    alpha_2, phi_2, derphi_2, derf_2, alpha_1, phi_1, derphi_1, phi_0, derphi_0)
                
                ls_func_calls += zoom_func_calls
                break

            alpha_1 = alpha_2
            phi_1 = phi_2
            derf_1 = derf_2

            alpha_2 = min(2 * alpha_2, alpha_max)
            i += 1
        return alpha_star, phi_star, derf_star, ls_func_calls
    else:
        done = paddle.full(shape=[1], fill_value=False, dtype='bool')

        def cond(i, ls_func_calls, alpha_1, alpha_2, phi_1, derf_1, done):
            return (i < max_iters) & ~done

        def body(i, ls_func_calls, alpha_1, alpha_2, phi_1, derf_1, done):
            phi_2, derf_2, derphi_2 = phi_and_derphi(alpha_2)
            paddle.assign(ls_func_calls + 1, ls_func_calls)
            paddle.assign(done | paddle.any(paddle.isinf(phi_2)), done)

            def true_fn1():
                a, b, c, d = zoom(alpha_1, phi_1, derphi_1, derf_1, alpha_2, phi_2, derphi_2, phi_0, derphi_0)
                paddle.assign(a, alpha_star)
                paddle.assign(b, phi_star)
                paddle.assign(c, derf_star)
                paddle.assign(ls_func_calls + d, ls_func_calls)

            pred1 = ~done & ((phi_2 > phi_0 + c1 * alpha_2 * derphi_0) | ((phi_2 >= phi_0) & (i > 1)))
            paddle.assign(done | pred1, done)
            paddle.static.nn.cond(pred1, true_fn1, None)

            def true_fn2():
                paddle.assign(alpha_2, alpha_star)
                paddle.assign(phi_2, phi_star)
                paddle.assign(derf_2, derf_star)

            pred2 = ~done & (paddle.abs(derphi_2) <= -c2 * derphi_0)
            paddle.assign(done | pred2, done)
            paddle.static.nn.cond(pred2, true_fn2, None)

            def true_fn3():
                a, b, c, d = zoom(alpha_2, phi_2, derphi_2, derf_2, alpha_1, phi_1, derphi_1, phi_0, derphi_0)
                paddle.assign(a, alpha_star)
                paddle.assign(b, phi_star)
                paddle.assign(c, derf_star)
                paddle.assign(ls_func_calls + d, ls_func_calls)

            pred3 = ~done & (derphi_2 >= 0)
            paddle.assign(done | pred3, done)
            paddle.static.nn.cond(pred3, true_fn3, None)

            def false_fn():
                paddle.assign(alpha_2, alpha_1)
                paddle.assign(phi_2, phi_1)
                paddle.assign(derf_2, derf_1)
                paddle.assign(paddle.minimum(2 * alpha_2, alpha_max), alpha_2)
                paddle.assign(i + 1, i)
            
            paddle.static.nn.cond(done, None, false_fn)
            return [
                i, ls_func_calls, alpha_1, alpha_2, phi_1, derf_1, done
            ]

        paddle.static.nn.while_loop(
            cond=cond,
            body=body,
            loop_vars=[
                i, ls_func_calls, alpha_1, alpha_2, phi_1, derf_1, done
            ])
        
        return alpha_star, phi_star, derf_star, ls_func_calls
