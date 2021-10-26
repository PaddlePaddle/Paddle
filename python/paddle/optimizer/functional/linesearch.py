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
from .bfgs import SearchState
from .bfgs_utils import as_float_tensor, make_state, update_state, 
from .bfgs_utils import vnorm_inf, vnorm_p, any_predicates
from .bfgs_utils import any_active, active_state, converged_state, failed_state

_hz_params = {
    'eps': 1e-6,
    'delta': .1,
    'sigma': .9,
    'omiga': 1e-3,
    'rho': 5.0,
    'mu': 0.01,
    'theta': .5,
    'Delta': .7,
    'psi_0': 0.01,
    'psi_1': 0.1,
    'psi_2': 2
}


def initial(state, params):
    r"""Generates initial step size.
    
    Args:
        state (Tensor): the search state tensor.
        params(Dict): the control parameters used in the HagerZhang method.

    Returns:
        The tensor of initial step size.
    """
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

def bracket(state, phi, c, params, max_iters):
    r"""Generates opposite slope interval.
        
    Args:
        state (Tensor): the search state tensor.
        phi (Callable): the restricted function on the search line.
        c (Tensor): the initial step sizes.
        params (Dict): the control parameters used in the HagerZhang method.
        max_iters (int): the maximum iterations.
    
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
    eps, rho = params['eps'], params['rho']
    epsilon_k = eps * state.Ck

    # f0 = phi(0)
    f0 = state.fk

    # The following loop repeatedly applies B3 if condition allows
    iters = 0
    expanding = True
    prev_c = make_const(c, .0)
    # f = phi(c), g = phi'(c)
    f, g = vjp(phi, c)
    
    while expanding and iters < max_iters:
        # Generates the B3 condition in a boolean tensor.
        B3_cond = paddle.logical_and(g < .0, f <= f0 + epsilon_k)

        # Sets [prev_c, c] to [c, rho*c] if B3 is true.
        prev_c = paddle.where(B3_cond, c, prev_c)
        c = paddle.where(B3_cond, rho*c, c)

        # Calculates function values and gradients for the new step size.
        f, g = vjp(phi, c)

        expanding = any_active_with_predicates(state.state, B3_cond)
        iters += 1
    
    # (TODO) Line search stops on the still expanding step sizes and exceeding 
    # maximum iterations.
    
    # Narrows down the interval by recursively bisecting it. 
    a, b = bisect(state, phi, make_const(c, .0), c, params)

    # Condition B1, that is, the rising right end
    B1_cond = g >= .0
    
    # Condition B2, that is, 
    B2_cond = paddle.logical_and(g < .0, f > f0 + epsilon_k)

    # Sets [a, _] to [prev_c, _] if B1 holds, [a, _] if B2 holds
    a = paddle.where(B2_cond, a, prev_c)

    # Sets [_, b] to [_, c] if B1 holds, [_, b] if B2 holds
    b = paddle.where(B2_cond, b, c)

    # Invalidates the state in case neither B1 nor B2 holds.
    failed = paddle.logical_not(paddle.logical_or(B1_cond, B2_cond))
    state.state = update_state(state.state, failed, 'failed')

    return [a, b]

def bisect(state, phi, a, b, params):
    r"""Bisects to locate opposite slope interval.
    
    Args:
        state (Tensor): the search state tensor.
        phi (Callable): the restricted function on the search line.
        a (Tensor): holds the left ends of the intervals.
        b (Tensor): holds the right ends of the intervals.
        params(Dict): the control parameters used in the HagerZhang method.
    
    Returns:
        [a, b]: left ends and right ends of the result intervels.   
    """
    # a. Let d = (1 - theta)*a + theta*b, if phi'(d) >= 0, then return [a, d]

    # b. If phi'(d) < 0 and phi(d) > phi(0) + epsilon_k, then Bisect([a, d])

    # c. If phi'(d) < 0 and phi(d) <= phi(0) + epsilon_k, then Bisect([d, b])

def hz_linesearch(state,
                  func,
                  gtol,
                  xtol,
                  initial_step=None,
                  max_iters=10,
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
        params (Dict, optional): the control parameters used in the HagerZhang 
            method. Default is None. 

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
        params = _hz_params


    def phi(a):
        r'''
        phi is used as the objective function restricted on the line search 
        secant.

        Args:
            a (Tensor): a scalar tensor, or a tensor of shape [...] in batching 
            mode, giving the step sizes alpha.
        '''
        if len(p.shape) > 1:
            a = paddle.unsqueeze(a, axis=-1)

        return func(x + a*p)

    def bracket(ls_active, b, phi, phi0):
        r'''
        Find an interval satisfying the opposite slope condition.
        Args:
            ls_active (Tensor): boolean typed tensor of shape [...] indicating
                which part of the input to work on 
            b (Tensor): float typed tensor setting the right ends of the 
                intervals
        '''
        
        expansion = params['rho']
        eps = params['eps']
        
        # b maintains the current right end of the interval
        
        while True:
            phi_j, phiprime_j = vjp(phi, b)
            
            rising, falling = phiprime_j >= .0, phiprime_j < .0
            
            ceiling = phi_j > phi0 + eps 

            # Stop 
            b = ifelse_select(phiprime_j >= 0, b, expansion * b)

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

    # The directional derivative of f at x_k on the direction of p_k.
    # It's also the gradient of phi    
    deriv = dot(gk, pk)

    # Marks inputs with invalid function values and non-negative derivatives 
    invalid_input = paddle.logical_or(paddle.isinf(fk), deriv >= .0)
    state.state = update_state(state.state, invalid_input, 'failed')

    # Generates initial step size
    c = initial(state, params)

    # Generates the opposite slope interval
    a, b = bracket(state, phi, c, params, max_iters=max_iters)


    return 

def wolfe12():
    
