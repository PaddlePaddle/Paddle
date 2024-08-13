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


def _cubic_interpolate(x1, f1, g1, x2, f2, g2, bounds=None):
    r"""Cubic interpolation between (x1, f1, g1) and (x2, f2, g2).
        Use two points and their gradient to determine a cubic function and get the minimum point
        between them in the cubic curve.

    Reference:
        Jorge Nocedal, Stephen J. Wright, Numerical Optimization, Second Edition, 2006.
        pp59: formula 3.59

    Args:
        x1, f1, g1: point1's position, value and gradient.
        x2, f2, g2: point2's position, value and gradient.
        bounds: bounds of interpolation area

    Returns:
        min_pos: the minimum point between the specified points in the cubic curve.
    """
    # Compute bounds of interpolation area
    if bounds is not None:
        xmin_bound, xmax_bound = bounds
    else:
        xmin_bound, xmax_bound = (x1, x2) if x1 <= x2 else (x2, x1)

    d1 = g1 + g2 - 3 * (f1 - f2) / (x1 - x2)
    d2_square = d1**2 - g1 * g2
    if d2_square >= 0:
        d2 = d2_square.sqrt()
        if x1 <= x2:
            min_pos = x2 - (x2 - x1) * ((g2 + d2 - d1) / (g2 - g1 + 2 * d2))
        else:
            min_pos = x1 - (x1 - x2) * ((g1 + d2 - d1) / (g1 - g2 + 2 * d2))
        return min(max(min_pos, xmin_bound), xmax_bound)
    else:
        return (xmin_bound + xmax_bound) / 2.0


def _strong_wolfe(
    obj_func,
    xk,
    alpha,
    d,
    loss,
    grad,
    gtd,
    c1=1e-4,
    c2=0.9,
    tolerance_change=1e-9,
    max_ls=25,
):
    r"""Implements of line search algorithm that satisfies the strong Wolfe conditions using double zoom.

    Reference:
        Jorge Nocedal, Stephen J. Wright, Numerical Optimization, Second Edition, 2006.
        pp60: Algorithm 3.5 (Line Search Algorithm).

    Args:
        obj_func: the objective function to minimize. ```` accepts a multivariate input and returns a scalar.
        xk (Tensor): the starting point of the iterates.
        alpha (Scalar): the initial step size.
        d (Tensor): search direction.
        loss (scalar): the initial loss
        grad (Tensor): the initial grad
        c1 (Scalar): parameter for sufficient decrease condition.
        c2 (Scalar): parameter for curvature condition.
        tolerance_change (Scalar): terminates if the change of function value/position/parameter between
            two iterations is smaller than this value.
        max_ls(int): max iteration of line search.
        alpha_max (float): max step length.

    Returns:
        loss_new (Scaler): loss of obj_func at final alpha.
        grad_new, (Tensor): derivative of obj_func at final alpha.
        alpha(Tensor): optimal step length, or 0. if the line search algorithm did not converge.
        ls_func_evals (Scaler): number of objective function called in line search process.

    Following summarizes the essentials of the strong Wolfe line search algorithm.
    Some notations used in the description:

        - `func` denotes the objective function.
        - `obi_func` is a function of step size alpha, restricting `obj_func` on a line.

            obi_func = func(xk + alpha * d),
            where xk is the position of k'th iterate, d is the line search direction(decent direction),
            and a is the step size.
        - alpha : substitute of alpha
        - a1 is alpha of last iteration, which is alpha_(i-1).
        - a2 is alpha of current iteration, which is alpha_i.
        - a_lo is alpha in left position when calls zoom, which is alpha_low.
        - a_hi is alpha in right position when calls zoom, which is alpha_high.

    Line Search Algorithm:
        repeat
            Compute obi_func(a2) and derphi(a2).
            1. If obi_func(a2) > obi_func(0) + c_1 * a2 * obi_func'(0) or [obi_func(a2) >= obi_func(a1) and i > 1],
                alpha= zoom(a1, a2) and stop;

            2. If |obi_func'(a2)| <= -c_2 * obi_func'(0),
                alpha= a2 and stop;

            3. If obi_func'(a2) >= 0,
                alpha= zoom(a2, a1) and stop;

            a1 = a2
            a2 = min(2 * a2, a2)
            i = i + 1
        end(repeat)

    zoom(a_lo, a_hi) Algorithm:
        repeat
            aj = cubic_interpolation(a_lo, a_hi)
            Compute obi_func(aj) and derphi(aj).
            1. If obi_func(aj) > obi_func(0) + c_1 * aj * obi_func'(0) or obi_func(aj) >= obi_func(a_lo),
                then a_hi <- aj;
            2.
                2.1. If |obi_func'(aj)| <= -c_2 * obi_func'(0), then alpha= a2 and stop;

                2.2. If obi_func'(aj) * (a2 - a1) >= 0, then a_hi = a_lo

                a_lo = aj;
        end(repeat)
    """

    d_norm = d.abs().max()
    grad = grad.clone()
    # evaluate objective and gradient using initial step
    loss_new, grad_new = obj_func(xk, alpha, d)
    ls_func_evals = 1
    gtd_new = paddle.dot(grad_new, d)

    # bracket an interval containing a point satisfying the Wolfe criteria
    t_prev, f_prev, g_prev, gtd_prev = (
        paddle.to_tensor(0, dtype=grad.dtype),
        loss,
        grad,
        gtd,
    )
    done = False
    ls_iter = 0
    while ls_iter < max_ls:
        # check conditions
        if loss_new > (loss + c1 * alpha * gtd) or (
            ls_iter > 1 and loss_new >= f_prev
        ):
            bracket = [t_prev, alpha]
            bracket_f = [f_prev, loss_new]
            bracket_g = [g_prev, grad_new.clone()]
            bracket_gtd = [gtd_prev, gtd_new]
            break

        if paddle.abs(gtd_new) <= -c2 * gtd:
            bracket = [alpha]
            bracket_f = [loss_new]
            bracket_g = [grad_new]
            done = True
            break

        if gtd_new >= 0:
            bracket = [t_prev, alpha]
            bracket_f = [f_prev, loss_new]
            bracket_g = [g_prev, grad_new.clone()]
            bracket_gtd = [gtd_prev, gtd_new]
            break

        # interpolate
        min_step = alpha + 0.01 * (alpha - t_prev)
        max_step = alpha * 10
        tmp = alpha
        alpha = _cubic_interpolate(
            t_prev,
            f_prev,
            gtd_prev,
            alpha,
            loss_new,
            gtd_new,
            bounds=(min_step, max_step),
        )

        # next step
        t_prev = tmp
        f_prev = loss_new
        g_prev = grad_new.clone()
        gtd_prev = gtd_new

        loss_new, grad_new = obj_func(xk, alpha, d)
        ls_func_evals += 1
        gtd_new = grad_new.dot(d)
        ls_iter += 1

    # reached max number of iterations?
    if ls_iter == max_ls:
        bracket = [0, alpha]
        bracket_f = [loss, loss_new]
        bracket_g = [grad, grad_new]

    # zoom phase: we now have a point satisfying the criteria, or
    # a bracket around it. We refine the bracket until we find the
    # exact point satisfying the criteria
    insuf_progress = False
    # find high and low points in bracket
    low_pos, high_pos = (0, 1) if bracket_f[0] <= bracket_f[-1] else (1, 0)
    while not done and ls_iter < max_ls:
        # line-search bracket is so small
        if paddle.abs(bracket[1] - bracket[0]) * d_norm < tolerance_change:
            break

        # compute new trial value
        alpha = _cubic_interpolate(
            bracket[0],
            bracket_f[0],
            bracket_gtd[0],
            bracket[1],
            bracket_f[1],
            bracket_gtd[1],
        )

        # test that we are making sufficient progress:
        # in case `alpha` is so close to boundary, we mark that we are making
        # insufficient progress, and if
        #   + we have made insufficient progress in the last step, or
        #   + `alpha` is at one of the boundary,
        # we will move `alpha` to a position which is `0.1 * len(bracket)`
        # away from the nearest boundary point.

        eps = 0.1 * (max(bracket) - min(bracket))
        if min(max(bracket) - alpha, alpha - min(bracket)) < eps:
            # interpolation close to boundary
            if insuf_progress or alpha >= max(bracket) or alpha <= min(bracket):
                # evaluate at 0.1 away from boundary
                if paddle.abs(alpha - max(bracket)) < paddle.abs(
                    alpha - min(bracket)
                ):
                    alpha = max(bracket) - eps
                else:
                    alpha = min(bracket) + eps
                insuf_progress = False
            else:
                insuf_progress = True
        else:
            insuf_progress = False
        # Evaluate new point
        loss_new, grad_new = obj_func(xk, alpha, d)
        ls_func_evals += 1
        gtd_new = grad_new.dot(d)
        ls_iter += 1

        if (
            loss_new > (loss + c1 * alpha * gtd)
            or loss_new >= bracket_f[low_pos]
        ):
            # Armijo condition not satisfied or not lower than lowest point
            bracket[high_pos] = alpha
            bracket_f[high_pos] = loss_new
            # bracket_g[high_pos] = grad_new.clone(memory_format=torch.contiguous_format)
            bracket_g[high_pos] = grad_new.clone()
            bracket_gtd[high_pos] = gtd_new
            low_pos, high_pos = (
                (0, 1) if bracket_f[0] <= bracket_f[1] else (1, 0)
            )
        else:
            if paddle.abs(gtd_new) <= -c2 * gtd:
                # Wolfe conditions satisfied
                done = True
            elif gtd_new * (bracket[high_pos] - bracket[low_pos]) >= 0:
                # old high becomes new low
                bracket[high_pos] = bracket[low_pos]
                bracket_f[high_pos] = bracket_f[low_pos]
                bracket_g[high_pos] = bracket_g[low_pos]
                bracket_gtd[high_pos] = bracket_gtd[low_pos]

            # new point becomes new low
            bracket[low_pos] = alpha
            bracket_f[low_pos] = loss_new
            bracket_g[low_pos] = grad_new.clone()
            bracket_gtd[low_pos] = gtd_new

    # return stuff
    alpha = bracket[low_pos]
    loss_new = bracket_f[low_pos]
    grad_new = bracket_g[low_pos]
    return loss_new, grad_new, alpha, ls_func_evals
