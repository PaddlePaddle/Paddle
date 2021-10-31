# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
from ...autograd import vjp
from .bfgs_utils import (StopCounter,
                         StopCounterException,
                         vnorm_inf,
                         vnorm_p,
                         make_const,
                         update_state,
                         any_active,
                         active_state,
                         converged_state,
                         failed_state,
                         any_active_with_predicates,
                         all_active_with_predicates)


hz_default_params = {
    'eps': 1e-6,
    'delta': .1,
    'sigma': .9,
    'omiga': 1e-3,
    'rho': 5.0,
    'mu': .01,
    'theta': .5,
    'gamma': .66,
    'Delta': .7,
    'psi_0': .01,
    'psi_1': .1,
    'psi_2': 2
}


def initial(state):
    r"""Generates initial step size.
    
    Args:
        state (Tensor): the search state tensor.

    Returns:
        The tensor of initial step size.
    """
    params = state.params
    psi_0, psi_1, psi_2 = params['psi_0'], params['psi_1'], params['psi_2']

    # I0. if k = 0, generate the initial step size c using the following rules.
    #     (a) If x_0 is not 0, c = psi_0 * infnorm(x_0) / infnorm(g_0)
    #     (b) If f(x_0) is not 0, then c = psi_0 * |f(x_0)| / vnorm(g_0)^2
    #     (c) Otherwise, c = 1
    #
    # I1. If QuadStep is true, phi(psi_1 * a_k-1) <= phi(0), and the
    # quadratic interpolant
    #     q() matches phi(0), phi'(0) and phi(psi * a_k-1) is strongly convex
    #     with a minimizer a_q, then c = a_k
    #
    # I2. Otherwise, c = psi_2 * a_k-1
    if state.k == 0:
        x0, f0, g0, a0 = state.xk, state.fk, state.gk, state.ak
        
        if a0 is not None:
            return a0.broadcast_to(f0)

        if paddle.all(x0 == .0):
            c = psi_0 * paddle.abs(f0) / vnorm_p(g0)**2
            c = paddle.where(f0 == 0, paddle.ones_like(f0), c)
        else:
            c = psi_0 * vnorm_inf(x0) / vnorm_inf(g0)
    else:
        # (TODO) implements quadratic interpolant
        prev_ak = state.ak
        c = psi_2 * prev_ak
    
    return c


def bracket(state, phi, c, iter_count):
    r"""Generates opposite slope interval.
        
    Args:
        state (Tensor): the search state tensor.
        phi (Callable): the restricted function on the search line.
        c (Tensor): the initial step sizes.
        iter_count (BoundedCounter): the bounded counter that controls the  
            maximum number of iterations.
    
    Returns:
        [a, b]: left ends and right ends of the result intervels.
    """
    # B0. Initialize j = 0, c_0 = c
    #
    # B1. If phi'(c_j) >= 0, then return [c_j-1, c_j]
    #
    # B2. If phi'(c_j) < 0 and phi(c_j) > phi(0) + epsilon_k,
    #     then return Bisect([0, c_j])
    #
    # B3. Otherwise, c_j = c, c_j+1 = rho * c, j = j + 1, goto B1
    params = state.params
    eps, rho = params['eps'], params['rho']
    epsilon_k = eps * state.Ck

    # f0 = phi(0)
    f0 = state.fk

    # The following loop repeatedly applies B3 if condition allows
    expanding = True
    prev_c = make_const(c, .0)

    # f = phi(c), g = phi'(c)
    iter_count.increment()
    f, g = vjp(phi, c)
    
    while expanding:
        # Generates the B3 condition in a boolean tensor.
        B3_cond = paddle.logical_and(g < .0, f <= f0 + epsilon_k)

        # Sets [prev_c, c] to [c, rho*c] if B3 is true.
        prev_c = paddle.where(B3_cond, c, prev_c)
        c = paddle.where(B3_cond, rho * c, c)

        # Calculates function values and gradients for the new step size.
        iter_count.increment()
        f, g = vjp(phi, c)

        expanding = any_active_with_predicates(state.state, B3_cond)
    
    # (TODO) Line search stops on the still expanding step sizes and exceeding 
    # maximum iterations.
    
    # Narrows down the interval by recursively bisecting it. 
    a, b = bisect(state, phi, make_const(c, .0), c, iter_count)

    # Condition B1, that is, the rising right end
    B1_cond = g >= .0
    
    # Condition B2, that is, the hoisted falling right end  
    B2_cond = paddle.logical_and(g < .0, f > f0 + epsilon_k)

    # Sets [a, _] to [prev_c, _] if B1 holds, [a, _] if B2 holds
    a = paddle.where(B2_cond, a, prev_c)

    # Sets [_, b] to [_, c] if B1 holds, [_, b] if B2 holds
    b = paddle.where(B2_cond, b, c)

    # Invalidates the state in case neither B1 nor B2 holds.
    failed = paddle.logical_not(paddle.logical_or(B1_cond, B2_cond))
    state.state = update_state(state.state, failed, 'failed')

    return [a, b]


def bisect(state, phi, a, b, iter_count):
    r"""Bisects to locate opposite slope interval.
    
    Args:
        state (Tensor): the search state tensor.
        phi (Callable): the restricted function on the search line.
        a (Tensor): holds the left ends of the intervals.
        b (Tensor): holds the right ends of the intervals.
        iter_count (BoundedCounter): the bounded counter that controls the  
            maximum number of iterations.

    Returns:
        [a, b]: left ends and right ends of the result intervels.   
    """
    params = state.params
    theta, eps = params['theta'], params['eps']
    epsilon_k = eps * state.Ck

    f0 = state.fk

    # a. Let d = (1 - theta)*a + theta*b, if phi'(d) >= 0, then return [a, d]
    #
    # b. If phi'(d) < 0 and phi(d) > phi(0) + epsilon_k, then Bisect([a, d])
    #
    # c. If phi'(d) < 0 and phi(d) <= phi(0) + epsilon_k, then Bisect([d, b])
    while True:
        d = (1 - theta) * a + theta * b

        iter_count.increment()
        f, g = vjp(phi, d)
        
        # Updates the intervals
        c_cond = paddle.logical_and(g < .0, f <= f0 + epsilon_k)
        a = paddle.where(c_cond, d, a)
        b = paddle.where(c_cond, b, d)

        # If condition a is not all true then continue
        a_cond = g >= .0
        if all_active_with_predicates(state.state, a_cond):
            return [a, b]

    # Invalidates the state if a condition does not hold.
    failed = g < .0
    state.state = update_state(state.state, failed, 'failed')

    return [a, b]


def secant(phi, a, b):
    r"""Implements the secant function, a sub-procedure in secant2.

    The output value is the weighted average of the input values, where
    the weights are given by the slopes of `phi` at the inputs values. 

    Args:
        phi (Callable): the restricted function on the search line.
        a (Tensor): holds the left ends of the intervals.
        b (Tensor): holds the right ends of the intervals.

    Returns:
        [a, b]: left ends and right ends of the result intervels. 
    """
    #                 a * phi'(b) - b * phi'(a)
    # secant(a, b) = ---------------------------
    #                   phi'(b)  - phi'(a)
    
    fa, ga = vjp(phi, a)
    fb, gb = vjp(phi, b)

    c = (a * gb - b * ga) / (gb - ga)
    # (TODO) Handles divide by zero
    # if paddle.any(paddle.isinf(c)):
    #     c = paddle.where(paddle.isinf(c), paddle.zeros_like(c), c)
    return c


def secant2(state, phi, a, b, iter_count):
    r"""Implements the secant2 procedure in the Hager-Zhang method.

    Args:
        state (Tensor): the search state tensor.
        phi (Callable): the restricted function on the search line.
        a (Tensor): holds the left ends of the intervals.
        b (Tensor): holds the right ends of the intervals.
        iter_count (BoundedCounter): the bounded counter that controls the  
            maximum number of iterations.

    Returns:
        [a, b]: left ends and right ends of the result intervels. 
    """
    # This step uses a function called secant which generates a new point
    # from two existing points using the slopes at the two points.
    #
    # S1. Let c = secant(a, b) and [A, B] = update(a, b, c)
    #
    # S2. If c = B, then let c = secant(b, B), [a, b] = update(A, B, c)
    #
    # S3. If c = A, then let c = secant(a, A), [a, b] = update(A, B, c)
    #
    # S3. Otherwise, [a, b] = [A, B]
    c = secant(phi, a, b)

    A, B = update(state, phi, a, b, c, iter_count)
    
    # Boolean tensor each element of which holds the S2 condition 
    S2_cond = c == B
    
    # Boolean tensor each element of which holds the S3 condition 
    S3_cond = c == A

    # Generates secant's input in S2 and S3
    l = paddle.where(S2_cond, b, A)
    r = B

    l = paddle.where(S3_cond, a, l)
    r = paddle.where(S3_cond, A, r)

    # Outputs of S2 and S3
    a, b = update(state, phi, A, B, c, iter_count)

    # If S2 or S3, returns [a, b], otherwise returns [A, B]
    S2_or_S3 = paddle.logical_or(S2_cond, S3_cond)
    a = paddle.where(S2_or_S3, a, A)
    b = paddle.where(S2_or_S3, b, B)

    return [a, b]   


def stopping_condition(state, phi, c, phiprime_0,
                       phi_c=None, phiprime_c=None):
    r"""Tests T1/T2 condition in the Hager-Zhang paper.
    
    Args:
        state (Tensor): the search state tensor.
        phi (Callable): the restricted function on the search line.
        c (Tensor): the step size tensor.
        phiprime_0 (Tensor): the derivative of `phi` at 0.
        phi_c (Tensor, optional): the value of `phi(c)`. Default is None.
        phiprime_c (Tensor, optional): the derivative of `phi` at `c`.
            Default is None.
        params (Dict, optional): used to configure the HagerZhang method.
            Default is None.

    Returns:
        A boolean tensor holding the evaluated stopping condition for each
        function instance.
    """
    params = state.params
    delta, sigma, eps = params['delta'], params['sigma'], params['eps']
    epsilon_k = eps * state.Ck

    phi_0 = state.fk

    if phi_c is None or phiprime_c is None:
       phi_c, phiprime_c = vjp(phi, c)
    
    # T1 (Wolfe). 
    #   T1.1            phi(c) - phi(0) <= c * phi'(0)
    #   T1.2            phi'(c) >= sigma * phi'(0)
    #
    # T2 (Approximate Wolfe).
    #   T2.1            phi'(c) <= (2*sigma - 1) * phi'(0)
    #   T2.2            (T1.2)
    #   T2.3            phi(c) - phi(0) <= epsilon_k

    phi_dy = phi_c - phi_0
    T11_cond = phi_dy <= c * phiprime_0 
    T12_cond = phiprime_c >= sigma * phiprime_0    
    
    wolfe_cond = paddle.logical_and(T11_cond, T12_cond)

    T21_cond = phiprime_c <= (2 * sigma - 1) * phiprime_0 
    T22_cond = T12_cond
    T23_cond = phi_dy <= epsilon_k

    approx_wolfe_cond = paddle.logical_and(T21_cond, T22_cond, T23_cond)
    
    stopping = paddle.logical_or(wolfe_cond, approx_wolfe_cond)

    return stopping


def update(state, phi, a, b, c, iter_count):
    r"""Performs the update procedure in the Hager-Zhang method.

    Args:
        state (Tensor): the search state tensor.
        phi (Callable): the restricted function on the search line.
        a (Tensor): holds the left ends of the intervals.
        b (Tensor): holds the right ends of the intervals.
        c (Tensor): holds the new step sizes.
        iter_count (BoundedCounter): the bounded counter that controls the  
            maximum number of iterations.

    Returns:
        [a, b]: left ends and right ends of the result intervels. 
    """
    eps = state.params['eps']
    f0 = state.fk
    epsilon_k = eps * state.Ck

    # U0. If c is outside of (a, b), then return [a, b]
    #
    # U1. If phi'(c) >= 0, then return [a, c]
    #
    # U2. If phi'(c) < 0 and phi(c) <= phi(0) + epsilon_k, return [c, b]
    #
    # U3. If phi'(c) < 0 and phi(c) > phi(0) + epsilon_k,
    #     return Bisect([a, c])
    iter_count.increment()
    f, g = vjp(phi, c)

    # Bisects [a, c] for the U3 condition
    A, B = bisect(state, phi, a, c, iter_count)

    # Step 1: branches between U2 and U3 
    U23_cond = f <= f0 + epsilon_k
    A = paddle.where(U23_cond, c, A)
    B = paddle.where(U23_cond, b, B)

    # Step 2: updates for true U1
    U1_cond = g >= .0
    A = paddle.where(U1_cond, a, A)
    B = paddle.where(U1_cond, c, B)

    # Step 3: in case c is not in [a, b], keeps [a, b]
    U0_cond = (c > a) == (c > b)
    A = paddle.where(U0_cond, a, A)
    B = paddle.where(U0_cond, b, B)

    return [A, B]


def update_approx_inverse_hessian(state, Hk, sk, yk, enforce_curvature=False):
    r"""Updates the approximate inverse Hessian.
    
    Given the input displacement s_k and the change of gradients y_k,
    the inverse Hessian at the next iterate is approximated using the following
    formula:
    
        H_k+1 = (I - rho_k * s_k * T(y_k)) * H_k * (I - rho_k * y_k * T(s_k))
                + rho_k * s_k * T(s_k),
    
                            1
        where rho_k = ----------------.
                        T(s_k) * y_k

    Note, the symmetric positive definite property of H_k+1 requires
        
        T(s_k) * y_k > 0.
    
    This is the curvature condition. It's known that a line search result that 
    satisfies the strong Wolfe conditions is guaranteed to meet the curvature 
    condition.

    Args:
        
    """
    rho_k = .1 / paddle.dot(sk, yk)

    # Enforces the curvature condition before updating the inverse Hessian.
    if enforce_curvature:
        assert not any_active_with_predicates(rho_k <= 0)
    else:
        update_state(state.state, rho_k <= 0, 'failed')

    # By expanding the updating formula we obtain a sum of tensor products
    #
    #      H_k+1 = H_k 
    #              - rho * H_k * y_k * T(s_k)    ----- (2)
    #              - rho * s_k * T(y_k) * H_k    ----- (3)
    #              + rho * s_k * T(s_k)                            ----- (4)
    #              + rho * rho * (T(y_k) * H_y * y_k) s_k * T(s_k) ----- (5)
    #
    # Since H_k is symmetric, (3) is (2)'s transpose.
    prod_H_y = paddle.matmul(Hk, yk.unsqueeze(-1))
    
    term23 = prod_H_y * sk
    
    perm = list(range(Hk.dim()))
    perm[-1], perm[-2] = perm[-2], perm[-1] 
    
    # Sums terms (2) and (3) forgoing rho
    term23 += term23.transpose(perm)

    # Merges terms (4) and (5) forgoing rho
    term45 = sk + rho_k * paddle.dot(prod_H_y.unsqueeze(-1), yk) * sk
    term45 *= sk.unsqueeze(-1)

    # Updates H_k and obtain H_k+1
    new_Hk = Hk + rho_k * (term45 - term23)

    return new_Hk


def hz_linesearch(state,
                  func,
                  gtol,
                  xtol,
                  initial_step=None,
                  max_iters=20,
                  params=None):
    r"""
    Implements the Hager-Zhang line search method. This method can be used as
    a drop-in replacement of any standard line search algorithm in solving
    a non-linear optimization problem.

    Args:
        state (SearchState): the search state which this line search is invoked 
            with.
        func (Callable): the objective function for the optimization problem.
        gtol (Tensor): a scalar tensor specifying the gradient tolerance.
        xtol (Tensor): a scalar tensor specifying the input difference
            tolerance.
        initial_step (float, optional): the initial step size to use for the 
            line search. Default value is None.
        max_iters (int, optional): the maximum number of trials for locating
            the next step size. Default value is 10.
        params (Dict, optional): used to configure the HagerZhang method.
            Default is None. 

    Reference:
        Algorithm 851: CG_DESCENT, a conjugate gradient method with guaranteed 
        descent, W. W. Hager and H. Zhang, ACM Transactions on Mathematical 
        Software 32: 113-137, 2006

    The Hager-Zhang method successfully addresses two issues. The first is with 
    numerical stability, where the standard strong Wolfe (SW) condition fails 
    to test the difference of function values in the neighborhood of a minimum. 
    The Hager-Zhang method uses a relaxed form of SW and is able to check value 
    difference more reliably. The second issue is with convergence speed. A 
    special secant step is the key feature that helps achieve fast
    convergence.

    Following summarizes the essentials of the HZ line search algorithm.

    Some notations used in the description:
    
        - `f` denotes the objective function

        - `phi` is a function of step size `a`, restricting `f` on a line.
        
            phi(a) = f(x_k + a * d_k),

            where x_k is the k'th iterate, d_k is the line search direction, and a is the
            step size.

    **Termination conditions**

        T1. the standard SW condition.

            phi(a) - phi(0) >= delta * a * phi'(0),  sufficient decrease 
                                                     condition, delta is a
                                                     small positive number
            |phi'(a)| <= |sigma * phi'(0)|,          curvature condition,
                                                     where 0< delta <= sigma < 1

        T2. the approximate Wolf (AW) condition.

            sigma * phi'(0) <= phi'(a) <= (2*delta - 1) phi'(0),
                                                     where 0 < delta < .5 and 
                                                     delta <= sigma < 1
            phi(a) <= phi(0) + epsilon_k,            epsilon_k is error
                                                     estimation at iteration k

    **Termination strategy**

        At initial iterates which are fairly distant from the minimum, line
        search stops when a step size is found satisfying either T1 or T2. If 
        at some iteration k, the following switching condition holds, then the 
        iterates afterwards must use the AW condition to terminate the line 
        search.

            |f(x_k+1) - f(x_k)| <= omiga * C_k,

        where C_k is a weighted average of all the function values on the 
        located iterates so far.

    **Initial(k)**

        This procedure generates the initial step size.

        I0. if k = 0 and no starting point is given, then generate the initial 
        step size c using the following rules.
            (a) If x_0 is not 0, c = psi_0 * infnorm(x_0) / infnorm(g_0) 
            (b) If f(x_0) is not 0, then c = psi_0 * |f(x_0)| / vnorm(g_0)^2
            (c) Otherwise, c = 1
        
        I1. If QuadStep is true, phi(psi_1 * a_k-1) <= phi(0), and the 
        quadratic interpolant
            q() matches phi(0), phi'(0) and phi(psi * a_k-1) is strongly convex 
            with a minimizer a_q, then c = a_k
        
        I2. Otherwise, c = psi_2 * a_k-1

    **Bracket(c)**

        This procedure is used to generate an opposite slope interval, kicking 
        start a bracketing process that eventually narrows down to a point 
        satfisfying T1 or T2.

        B0. c_-1 = c

        B1. If phi'(c) >= 0, then return [c_-1, c]

        B2. If phi'(c) < 0 and phi(c) > phi(0) + epsilon_k, 
            then return Bisect([0, c])

        B3. Otherwise, c_-1 = c, c = rho * c, goto B1

    **Bisect([a, b])**

        a. Let d = (1 - theta)*a + theta*b, if phi'(d) >= 0, then return [a, d]

        b. If phi'(d) < 0 and phi(d) > phi(0) + epsilon_k, then Bisect([a, d])

        c. If phi'(d) < 0 and phi(d) <= phi(0) + epsilon_k, then Bisect([d, b])

    **Secant2(a, b)**

        This step uses a function called secant which generates a new point 
        from two existing points using the slopes at the two points.  
    
                        a * phi'(b) - b * phi'(a)
        secant(a, b) = ---------------------------
                          phi'(b)  - phi'(a)

        S1. Let c = secant(a, b) and [A, B] = update(a, b, c)

        S2. If c = B, then let c = secant(b, B), [a, b] = update(A, B, c)

        S3. If c = A, then let c = secant(a, A), [a, b] = update(A, B, c)

        S3. Otherwise, [a, b] = [A, B]

    **Update(a, b, c)**

        U0. If c is outside of (a, b), then return [a, b]

        U1. If phi'(c) >= 0, then return [a, c]

        U2. If phi'(c) < 0 and phi(c) <= phi(0) + epsilon_k, return [c, b]

        U3. If phi'(c) < 0 and phi(c) > phi(0) + epsilon_k, 
            return Bisect([a, c])

    """
    if params is None:
        state.params = hz_default_params

    gamma = state.params['gamma']

    # For each line search, the input location, function value, gradients and
    # the approximate inverse hessian are already present in the state date
    # struture. No need to recompute.
    k, xk, fk, gk, Hk = state.k, state.xk, state.fk, state.gk, state.Hk  

    # The negative inner product of approximate inverse hessian and gradient
    # gives the line search direction p_k. Immediately after p_k is calculated,
    # the directional derivative on p_k should be checked to make sure 
    # the p_k is a descending direction. If that's not the case, then sets
    # the line search state as failed for the corresponding batching element.
    pk = -paddle.dot(Hk, gk)

    def phi(a):
        r'''
        phi is used as the objective function restricted on the line search 
        secant.

        Args:
            a (Tensor): a scalar tensor, or a tensor of shape [...] in batching 
            mode, giving the step sizes alpha.
        '''
        if len(pk.shape) > 1:
            a = paddle.unsqueeze(a, axis=-1)

        return func(xk + a * pk)

    # derive is the directional derivative of f at x_k on the direction of p_k.
    # It's also the gradient of phi    
    deriv = paddle.dot(gk, pk)

    # Marks inputs with invalid function values and non-negative derivatives 
    invalid_input = paddle.logical_or(paddle.isinf(fk), deriv >= .0)
    state.state = update_state(state.state, invalid_input, 'failed')

    # L0. c = initial(k), [a0, b0] = bracket(c), and j = 0
    #
    # L1. [a, b] = secant2(aj, bj)
    #
    # L2. If b - a > gamma * (bj - aj), 
    #     then c = (a + b)/2 and [a, b] = update(a, b, c)
    #
    # L3. j = j + 1, [aj, bj] = [a, b], go to L1.

    # Initializes a bounded counter
    iter_count = StopCounter(max_iters)

    # Generates initial step sizes
    c = initial(state)

    try:
        # Generates the opposite slope interval
        a_j, b_j = bracket(state, phi, c, iter_count)

        ls_found = make_const(c, False, dtype='bool')
        ls_stepsize = c

        while True:
            # Applies secant2 to the located opposite slope interval
            a, b = secant2(state, phi, a_j, b_j, iter_count)

            # If interval does not shrink enough, then applies bisections 
            # repeatedly.
            L2_cond = (b - a) > gamma * (b_j - a_j)

            c = 0.5 * (a + b)
        
            # Halts the line search if the stopping conditions are all 
            # satisfied.
            not_found = paddle.logical_not(ls_found)
            stopped = stopping_condition(state, phi, c, deriv)
            new_stopped = paddle.logical_and(not_found, stopped)
            ls_stepsize = paddle.where(new_stopped, c, ls_stepsize)
            ls_found = paddle.logical_or(ls_found, stopped)

            if all_active_with_predicates(state.state, ls_found):
                break

            A, B = update(state, phi, a, b, c, iter_count)
            a = paddle.where(L2_cond, A, a)
            b = paddle.where(L2_cond, B, b)

            # Goes to next iteration
            a_j, b_j = a, b

    except StopCounterException:
        pass

    # Updates the state of the instances for which the line search failed.
    ls_failed = paddle.logical_not(ls_found)
    state.state = update_state(state.state, ls_failed, 'failed')
    
    # Uses the obtained search steps to generate next iterates.
    next_xk = xk + ls_stepsize * pk
    
    # Calculates displacement s_k = x_k+1 - x_k
    sk = next_xk - xk

    # Obtains the function values and gradients at x_k+1
    next_fk, next_gk = vjp(func, next_xk)
    
    # Calculates the gradient difference y_k = g_k+1 - g_k
    yk = next_fk - fk
    
    # Updates the approximate inverse hessian
    next_Hk = update_approx_inverse_hessian(state, Hk, sk, yk)

    state.xk = paddle.where(active_state(state.state), next_xk, xk)
    state.fk = paddle.where(active_state(state.state), next_fk, fk)
    state.gk = paddle.where(active_state(state.state), next_gk, gk)
    state.Hk = paddle.where(active_state(state.state), next_Hk, Hk)
    state.k = k + 1

    return

