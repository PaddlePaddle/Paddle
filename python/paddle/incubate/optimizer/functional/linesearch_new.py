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
from paddle import dot, einsum
from .bfgs_utils import vjp
from .bfgs_utils import (StepCounter, StepCounterException, vnorm_inf, vnorm_p,
                         ternary, make_const, SearchState)

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


class HagerZhang(SearchState):
    r"""
    Implements the Hager-Zhang line search method. This method can be used as
    a drop-in replacement of any standard line search algorithm in solving
    a non-linear optimization problem.

    Args:
        func (Callable): the objective function for the optimization problem.
        initial_step (float, optional): the initial step size to use for the 
            line search. Default value is None.
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
    def __init__(self,
                 func,
                 bat,
                 x0,
                 f0,
                 g0,
                 H0,
                 gnorm,
                 a0=None,
                 lowerbound=None,
                 ls_iters=50,
                 params=hz_default_params):
        super(HagerZhang).__init__(bat,
                                   x0,
                                   f0,
                                   g0,
                                   H0,
                                   gnorm,
                                   ak=a0,
                                   lowerbound=lowerbound,
                                   ls_iters=ls_iters)
        self.func = func
        self.phi = None
        self.deriv = None
        self.set_params(params)

    def update_stop(self, c, phi_c=None, phiprime_c=None):
        r"""Tests T1/T2 condition in the Hager-Zhang paper.
        
        Args:
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
        delta, sigma, eps = (getattr(self, p) for p in ('delta',
                                                        'sigma',
                                                        'eps'))
        epsilon_k = eps * self.Ck

        phi_0 = self.fk
        phiprime_0 = self.phiprime0

        if phi_c is None or phiprime_c is None:
            phi_c, phiprime_c = self.func_and_deriv(self.phi, c)
        
        self.stop_lowerbound |= self.is_lowerbound(phi_c)
        self.stop_blowup |= self.is_blowup(phi_c, phiprime_c)

        # T1 (Wolfe). 
        #   T1.1            phi(c) - phi(0) <= delta * c * phi'(0)
        #   T1.2            phi'(c) >= sigma * phi'(0)
        #
        # T2 (Approximate Wolfe).
        #   T2.1            phi'(c) <= (2*delta - 1) * phi'(0)
        #   T2.2            (T1.2)
        #   T2.3            phi(c) - phi(0) <= epsilon_k

        phi_diff = phi_c - phi_0
        T11_cond = phi_diff <= delta * c * phiprime_0
        T12_cond = phiprime_c >= sigma * phiprime_0

        wolfe_cond = T11_cond & T12_cond

        T21_cond = phiprime_c <= (2 * delta - 1) * phiprime_0
        T22_cond = T12_cond
        T23_cond = phi_diff <= epsilon_k

        approx_wolfe_cond = T21_cond & T22_cond & T23_cond

        wolfe = wolfe_cond | approx_wolfe_cond
        self.stop_wolfe |= wolfe
        self.stop = self.stop_wolfe | self.stop_blowup | self.stop_lowerbound

    def should_stop(self):
        return not self.any_active_with_predicates(self.stop)

    def initial(self):
        r"""Generates the initial step size.

        Returns:
            The tensor of initial step size.
        """
        psi_0, psi_1, psi_2 = (getattr(self, p) for p in ('psi_0',
                                                          'psi_1',
                                                          'psi_2'))

        # I0. if k = 0, generate the initial step size c using the following 
        # rules.
        #     (a) If x_0 is not 0, c = psi_0 * infnorm(x_0) / infnorm(g_0)
        #     (b) If f(x_0) is not 0, then c = psi_0 * |f(x_0)| / vnorm(g_0)^2
        #     (c) Otherwise, c = 1
        #
        # I1. If QuadStep is true, phi(psi_1 * a_k-1) <= phi(0), and the
        # quadratic interpolant q() matches phi(0), phi'(0) and
        # phi(psi * a_k-1) is strongly convex with a minimizer a_q, then c = a_k
        #
        # I2. Otherwise, c = psi_2 * a_k-1
        if self.k == 0:
            x0, f0, g0, a0 = self.xk, self.fk, self.gk, self.ak

            if a0 is not None:
                return a0.broadcast_to(f0)

            if paddle.all(x0 == .0):
                c = psi_0 * paddle.abs(f0) / vnorm_p(g0)**2
                c = ternary(f0 == 0, paddle.ones_like(f0), c)
            else:
                c = psi_0 * vnorm_inf(x0) / vnorm_inf(g0)

            self.ak = c
        else:
            # (TODO) implements quadratic interpolant
            prev_ak = self.ak
            c = psi_2 * prev_ak

        return c

    def bisect(self, a, b, ifcond):
        r"""Bisects to locate opposite slope interval.

        Args:
            a (Tensor): holds the left ends of the intervals.
            b (Tensor): holds the right ends of the intervals.
            ifcond (Tensor): boolean tensor that holds the if-converse condition.

        Returns:
            [a, b]: left ends and right ends of the result intervels.   
        """
        theta, eps = (getattr(self, p) for p in ('theta', 'eps'))
        epsilon_k = eps * self.Ck

        f0 = self.fk

        # a. Let d = (1 - theta)*a + theta*b, if phi'(d) >= 0, then return [a, d]
        #
        # b. If phi'(d) < 0 and phi(d) > phi(0) + epsilon_k, then Bisect([a, d])
        #
        # c. If phi'(d) < 0 and phi(d) <= phi(0) + epsilon_k, then Bisect([d, b])

        # falling = make_const(f0, True, dtype='bool')
        bisect_pred = ifcond
        while self.any_active_with_predicates(bisect_pred & ~self.stop):
            # d = (1.0 - theta) * a + theta * b
            d = a + theta * (b - a)

            f, g = self.func_and_deriv(self.phi, d)

            falling = g < .0
            lo = f < f0 + epsilon_k

            # Condition a.
            b = ternary(bisect_pred & ~falling, d, b)

            # Condition b and c.
            bisect_pred = bisect_pred & falling
            a = ternary(bisect_pred & lo, d, a)
            b = ternary(bisect_pred & ~lo, d, b)
            print(f'bisect a: {a.numpy()}    b: {b.numpy()}')
            
            self.update_stop(a)
            self.update_stop(b)
        return a, b

    def bracket(self, phi, c, phi_c=None, phiprime_c=None):
        r"""Generates opposite slope interval.
            
        Args:
            phi (Callable): the restricted function on the search line.
            c (Tensor): the initial step sizes.
        
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
        eps, rho = (getattr(self, p) for p in ('eps', 'rho'))
        epsilon_k = eps * self.Ck

        # f0 = phi(0)
        f0 = self.fk

        # The following loop repeatedly applies (B3) if condition allows
        prev_c = make_const(c, .0)

        # f = phi(c), g = phi'(c)
        if phi_c is None or phiprime_c is None:
            f, g = self.func_and_deriv(phi, c)
            self.update_stop(c, f, g)
        else:
            f, g = phi_c, phiprime_c
        
        # Initializes condition B3
        B3_cond = (g < .0) & (f < f0 + epsilon_k)

        expanding_pred = B3_cond & ~self.stop

        while self.any_active_with_predicates(expanding_pred):
            # Sets [prev_c, c] to [c, rho*c] if B3 is true.
            prev_c = ternary(expanding_pred, c, prev_c)
            c = ternary(expanding_pred, rho * c, c)
            print(f'expanding  c: {c.numpy()}')
            # Calculates the function values and gradients.
            f, g = self.func_and_deriv(self.phi, c)
            self.update_stop(c, f, g)
            B3_cond = (g < .0) & (f < f0 + epsilon_k)
            expanding_pred &= B3_cond & ~self.stop

        # Narrows down the interval by recursively bisecting it.
        B1_cond = g >= .0
        B2_cond = ~B1_cond & (f >= f0 + epsilon_k)
        a, b = self.bisect(make_const(c, .0), c, B2_cond)
        # Sets [a, _] to [prev_c, _] if B1 holds, [a, _] if B2 holds
        a = ternary(B1_cond, prev_c, a)
        b = ternary(B1_cond, c, b)
        return a, b

    def secant(self, a, b):
        r"""Implements the secant function, a sub-procedure in secant2.
    
        The output value is the weighted average of the input values, where
        the weights are given by the slopes of `phi` at the inputs values. 
    
        Args:
            a (Tensor): holds the left ends of the intervals.
            b (Tensor): holds the right ends of the intervals.
    
        Returns:
            [a, b]: left ends and right ends of the result intervels. 
        """
        #                 a * phi'(b) - b * phi'(a)
        # secant(a, b) = ---------------------------
        #                   phi'(b)  - phi'(a)
    
        fa, ga = self.func_and_deriv(self.phi, a)
        fb, gb = self.func_and_deriv(self.phi, b)
    
        c = (a * gb - b * ga) / (gb - ga)

        self.update_stop(c)
        return c
    
    def update(self, a, b, c, ifcond):
        r"""Performs the update procedure in the Hager-Zhang method.
    
        Args:
            a (Tensor): holds the left ends of the intervals.
            b (Tensor): holds the right ends of the intervals.
            c (Tensor): holds the new step sizes.

        Returns:
            [a, b]: left ends and right ends of the result intervels. 
        """
        eps = getattr(self, 'eps')
        f0 = self.fk
        epsilon_k = eps * self.Ck
    
        # U0. If c is outside of (a, b), then return [a, b]
        #
        # U1. If phi'(c) >= 0, then return [a, c]
        #
        # U2. If phi'(c) < 0 and phi(c) <= phi(0) + epsilon_k, return [c, b]
        #
        # U3. If phi'(c) < 0 and phi(c) > phi(0) + epsilon_k,
        #     return Bisect([a, c])
        f, g = self.func_and_deriv(self.phi, c)
    
        # Early returns on U0
        U0_cond = (c > a) == (c > b)
        selected0 = ifcond & U0_cond
        selected0_c = ifcond & ~U0_cond
        if not self.any_active_with_predicates(selected0_c):
            return a, b
    
        # Early returns if U1 holds
        U1_cond = (g >= .0)
        selected1 = selected0_c & U1_cond
        selected1_c = selected0_c & ~U1_cond
        b = ternary(selected1, c, b)
        if not self.any_active_with_predicates(selected1_c):
            return a, b
    
        # It's tricky to handle iterative tensor algorithms when control 
        # dependence is involved. If naively running two branches in parallel 
        # before merging the two control paths, the `invalid` path may have 
        # never ended. We use an if-converse predicate to overcome this issue.
        U2_cond = f > f0 + epsilon_k
        selected2 = selected1_c & U2_cond
        selected2_c = selected1_c & ~U2_cond
        a = ternary(selected2, c, a)
        if not self.any_active_with_predicates(selected2_c):
            return a, b

        # Bisects [a, c] for the U3 condition
        selected3 = selected2_c
        A, B = self.bisect(a, c, selected3)
    
        # U3 
        a = ternary(selected3, A, a)
        b = ternary(selected3, B, b)
        return a, b

    def secant2(self, a, b):
        r"""Implements the secant2 procedure in the Hager-Zhang method.
    
        Args:
            a (Tensor): holds the left ends of the intervals.
            b (Tensor): holds the right ends of the intervals.
    
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
        c = self.secant(a, b)
    
        A, B = self.update(a, b, c, ~self.stop)
    
        # Boolean tensor each element of which holds the S2 condition 
        S2_cond = c == B
        S3_cond = c == A
        selected_S2 = ~self.stop & S2_cond
        selected_S3 = ~self.stop & S3_cond
        selected_S23 =  ~self.stop & (S2_cond | S3_cond)
        c = ternary(selected_S2, self.secant(b, c), c)
        c = ternary(selected_S3, self.secant(a, c), c)
    
        a, b = self.update(A, B, c, selected_S2 | selected_S3)

        a = ternary(selected_S23, a, A)
        b = ternary(selected_S23, b, B)

        return a, b
    

    def linesearch(self, pk):
        gamma = getattr(self, 'gamma')
        Delta = getattr(self, 'Delta')

        # For each line search, the input location, function value, gradients 
        # and the approximate inverse hessian are already present in the state 
        # date struture. No need to recompute.
        bat, xk, fk, gk = self.bat, self.xk, self.fk, self.gk

        # Updates C_k, the weighted average of the absolute function values, 
        # used for assessing the relative change of function values over 
        # succesive iterates.
        Qk, Ck = self.Qk, self.Ck
        Qk = 1 + Qk * Delta
        Ck = Ck + (paddle.abs(fk) - Ck) / Qk
        self.Qk, self.Ck = Qk, Ck
    
        def phi(a):
            r'''
            phi is the objective function projected on the direction `pk`.

            Args:
                a (Tensor): a scalar tensor, or a tensor of shape [...] in batching 
                mode, giving the step sizes alpha.
            '''
            if len(pk.shape) > 1:
                a = paddle.unsqueeze(a, axis=-1)

            return self.func(xk + a * pk)

        # deriv is the directional derivative of f at x_k on the direction of 
        # p_k. It's also the gradient of phi    
        deriv = einsum('...i, ...i', gk, pk) if bat else dot(gk, pk)

        self.phi = phi
        self.phiprime0 = deriv

        self.ls_step_counter.reset()

        # Makes early decisions in case some instances are found to be converged
        # or failed. The status `stop_lowerbound` and `stop_blowup` are monotone
        # w.r.t. logical_or.
        self.stop_lowerbound |= self.is_lowerbound(fk)
        self.stop_blowup |= self.is_blowup(fk, deriv)
        rising = deriv >= .0
        print(f'deriv >= 0:  {rising.numpy()}')
        print(f'islowerbound : {fk.numpy()}')
        self.update_state(self.stop_lowerbound, 'converged')
        self.update_state(self.stop_blowup, 'blowup')
        self.update_state(rising, 'failed')

        # L0. c = initial(k), [a0, b0] = bracket(c), and j = 0
        #
        # L1. [a, b] = secant2(aj, bj)
        #
        # L2. If b - a > gamma * (bj - aj),
        #     then c = (a + b)/2 and [a, b] = update(a, b, c)
        #
        # L3. j = j + 1, [aj, bj] = [a, b], go to L1.

        # Initializes the stopping flags. `stop_lowerbound` and `stop_blowup`
        # are carried over from the previous linesearch.
        self.stop_wolfe = make_const(fk, False, dtype='bool')
        # Generates initial step sizes
        c = self.initial()
        print(f'ls  \nInitial step size: {c.numpy()}')

        import traceback
        try:
            # Initial stopping test. Those already converged instances are 
            # likely to succeed.
            self.update_stop(c)

            # Finds the first opposite-sloped bracket 
            a_j, b_j = self.bracket(phi, c)
            # Continues if there's line search still active
            while not self.should_stop():
                # Applies secant2 to the located intervals
                a, b = self.secant2(a_j, b_j)

                print(f'secant2 a: {a.numpy()}     b: {b.numpy()}')
                next_c = a + 0.5 * (b - a)
                c = ternary(self.stop, c, next_c)

                self.update_stop(c)

                # If interval does not shrink enough, then applies bisect
                # repeatedly.
                L2_cond = (b - a) > gamma * (b_j - a_j)

                selected_L2 = ~self.stop & L2_cond

                if self.any_active_with_predicates(selected_L2):
                    A, B = self.update(a, b, c, selected_L2)
                    a = ternary(selected_L2, A, a)
                    b = ternary(selected_L2, B, b)

                # Goes to next iteration
                a_j, b_j = a, b
        except StepCounterException as count_e:
            # traceback.print_exc()
            pass

        # Changes state due to failed line search
        self.update_state(~self.stop, 'failed')

        print(f'state {self.state}')
        print(f'ak {self.ak}')
        print(f'ls_stepsize {c}')
        # Writes back the obtained step size to the search state.
        next_ak = ternary(self.active_state() & self.stop, c, self.ak)
        print(f'ak2 {next_ak}')
        return next_ak
