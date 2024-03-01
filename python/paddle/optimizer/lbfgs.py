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

from collections import defaultdict
from functools import reduce

import paddle

from ..base import framework
from .optimizer import Optimizer

__all__ = []


def dot(x, y):
    r"""
    NOTE: This is a temporary workaround for unstable result computed by `paddle.dot`,
    which will be reverted when the problem is fixed."
    """
    return (x * y).sum(axis=-1)


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

    reference: https://github.com/pytorch/pytorch
    """

    d_norm = d.abs().max()
    grad = grad.clone()
    # evaluate objective and gradient using initial step
    loss_new, grad_new = obj_func(xk, alpha, d)
    ls_func_evals = 1
    gtd_new = dot(grad_new, d)

    # bracket an interval containing a point satisfying the Wolfe criteria
    t_prev, f_prev, g_prev, gtd_prev = (0, loss, grad, gtd)
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
        gtd_new = dot(grad_new, d)
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
        bracket_ls = bracket[1] - bracket[0]
        if not isinstance(bracket_ls, paddle.Tensor):
            bracket_ls = paddle.to_tensor(bracket_ls, dtype=gtd_new.dtype)
        if paddle.abs(bracket_ls) * d_norm < tolerance_change:
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
        gtd_new = dot(grad_new, d)
        ls_iter += 1

        if (
            loss_new > (loss + c1 * alpha * gtd)
            or loss_new >= bracket_f[low_pos]
        ):
            # Armijo condition not satisfied or not lower than lowest point
            bracket[high_pos] = alpha
            bracket_f[high_pos] = loss_new
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


class LBFGS(Optimizer):
    r"""
    The L-BFGS is a quasi-Newton method for solving an unconstrained optimization problem over a differentiable function.
    Closely related is the Newton method for minimization. Consider the iterate update formula:

    .. math::
        x_{k+1} = x_{k} + H_k \nabla{f_k}

    If :math:`H_k` is the inverse Hessian of :math:`f` at :math:`x_k`, then it's the Newton method.
    If :math:`H_k` is symmetric and positive definite, used as an approximation of the inverse Hessian, then
    it's a quasi-Newton. In practice, the approximated Hessians are obtained
    by only using the gradients, over either whole or part of the search
    history, the former is BFGS, the latter is L-BFGS.

    Reference:
        Jorge Nocedal, Stephen J. Wright, Numerical Optimization, Second Edition, 2006. pp179: Algorithm 7.5 (L-BFGS).

    Args:
        learning_rate (float, optional): learning rate .The default value is 1.
        max_iter (int, optional): maximal number of iterations per optimization step.
            The default value is 20.
        max_eval (int, optional): maximal number of function evaluations per optimization
            step. The default value is max_iter * 1.25.
        tolerance_grad (float, optional): termination tolerance on first order optimality
            The default value is 1e-5.
        tolerance_change (float, optional): termination tolerance on function
            value/parameter changes. The default value is 1e-9.
        history_size (int, optional): update history size. The default value is 100.
        line_search_fn (string, optional): either 'strong_wolfe' or None. The default value is strong_wolfe.
        parameters (list|tuple, optional): List/Tuple of ``Tensor`` names to update to minimize ``loss``. \
            This parameter is required in dygraph mode. The default value is None.
        weight_decay (float|WeightDecayRegularizer, optional): The strategy of regularization. \
            It canbe a float value as coeff of L2 regularization or \
            :ref:`api_paddle_regularizer_L1Decay`, :ref:`api_paddle_regularizer_L2Decay`.
            If a parameter has set regularizer using :ref:`api_paddle_ParamAttr` already, \
            the regularization setting here in optimizer will be ignored for this parameter. \
            Otherwise, the regularization setting here in optimizer will take effect. \
            Default None, meaning there is no regularization.
        grad_clip (GradientClipBase, optional): Gradient clipping strategy, it's an instance of \
            some derived class of ``GradientClipBase`` . There are three clipping strategies \
            ( :ref:`api_paddle_nn_ClipGradByGlobalNorm` , :ref:`api_paddle_nn_ClipGradByNorm` , \
            :ref:`api_paddle_nn_ClipGradByValue` ). Default None, meaning there is no gradient clipping.
        name (str, optional): Normally there is no need for user to set this property.
            For more information, please refer to :ref:`api_guide_Name`.
            The default value is None.

    Return:
        loss (Tensor): the final loss of closure.

    Examples:
        .. code-block:: python

            >>> import paddle
            >>> import numpy as np

            >>> paddle.disable_static()
            >>> np.random.seed(0)
            >>> np_w = np.random.rand(1).astype(np.float32)
            >>> np_x = np.random.rand(1).astype(np.float32)

            >>> inputs = [np.random.rand(1).astype(np.float32) for i in range(10)]
            >>> # y = 2x
            >>> targets = [2 * x for x in inputs]

            >>> class Net(paddle.nn.Layer):
            ...     def __init__(self):
            ...         super().__init__()
            ...         w = paddle.to_tensor(np_w)
            ...         self.w = paddle.create_parameter(shape=w.shape, dtype=w.dtype, default_initializer=paddle.nn.initializer.Assign(w))
            ...
            ...     def forward(self, x):
            ...         return self.w * x
            ...
            >>> net = Net()
            >>> opt = paddle.optimizer.LBFGS(learning_rate=1, max_iter=1, max_eval=None, tolerance_grad=1e-07, tolerance_change=1e-09, history_size=100, line_search_fn='strong_wolfe', parameters=net.parameters())
            >>> def train_step(inputs, targets):
            ...     def closure():
            ...         outputs = net(inputs)
            ...         loss = paddle.nn.functional.mse_loss(outputs, targets)
            ...         print('loss: ', loss.item())
            ...         opt.clear_grad()
            ...         loss.backward()
            ...         return loss
            ...     opt.step(closure)
            ...
            >>> for input, target in zip(inputs, targets):
            ...     input = paddle.to_tensor(input)
            ...     target = paddle.to_tensor(target)
            ...     train_step(input, target)
    """

    def __init__(
        self,
        learning_rate=1.0,
        max_iter=20,
        max_eval=None,
        tolerance_grad=1e-7,
        tolerance_change=1e-9,
        history_size=100,
        line_search_fn=None,
        parameters=None,
        weight_decay=None,
        grad_clip=None,
        name=None,
    ):
        if max_eval is None:
            max_eval = max_iter * 5 // 4

        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.max_eval = max_eval
        self.tolerance_grad = tolerance_grad
        self.tolerance_change = tolerance_change
        self.history_size = history_size
        self.line_search_fn = line_search_fn

        if isinstance(parameters, paddle.Tensor):
            raise TypeError(
                "parameters argument given to the optimizer should be "
                "an iterable of Tensors or dicts, but got " + type(parameters)
            )

        self.state = defaultdict(dict)

        super().__init__(
            learning_rate=1.0,
            parameters=parameters,
            weight_decay=weight_decay,
            grad_clip=grad_clip,
            name=name,
        )

        if not isinstance(self._parameter_list[0], dict):
            self._params = self._parameter_list
        else:
            for idx, param_group in enumerate(self._param_groups):
                self._params = param_group['params']

        self._numel_cache = None

    def state_dict(self):
        r"""Returns the state of the optimizer as a :class:`dict`.

        Return:
            state, a dict holding current optimization state. Its content
            differs between optimizer classes.

        Examples:
            .. code-block:: python

                >>> import paddle

                >>> paddle.disable_static()

                >>> net = paddle.nn.Linear(10, 10)
                >>> opt = paddle.optimizer.LBFGS(
                ...     learning_rate=1,
                ...     max_iter=1,
                ...     max_eval=None,
                ...     tolerance_grad=1e-07,
                ...     tolerance_change=1e-09,
                ...     history_size=100,
                ...     line_search_fn='strong_wolfe',
                ...     parameters=net.parameters(),
                >>> )

                >>> def train_step(inputs, targets):
                ...     def closure():
                ...         outputs = net(inputs)
                ...         loss = paddle.nn.functional.mse_loss(outputs, targets)
                ...         opt.clear_grad()
                ...         loss.backward()
                ...         return loss
                ...
                ...     opt.step(closure)
                ...
                >>> inputs = paddle.rand([10, 10], dtype="float32")
                >>> targets = paddle.to_tensor([2 * x for x in inputs])

                >>> n_iter = 0
                >>> while n_iter < 20:
                ...     loss = train_step(inputs, targets)
                ...     n_iter = opt.state_dict()["state"]["func_evals"]
                ...     print("n_iter:", n_iter)
                ...
        """

        packed_state = {}
        for k, v in self.state.items():
            packed_state.update({k: v})

        return {'state': packed_state}

    def _numel(self):
        # compute the number of all parameters
        if self._numel_cache is None:
            self._numel_cache = reduce(
                lambda total, p: total + p.numel(), self._params, 0
            )
        return self._numel_cache

    # flatten grad of all parameters
    def _gather_flat_grad(self):
        views = []
        for p in self._params:
            if p.grad is None:
                view = paddle.zeros_like(p).reshape([-1])
            else:
                view = p.grad.reshape([-1])
            views.append(view)
        return paddle.concat(views, axis=0)

    # compute xk = xk + alpha * direction
    def _add_grad(self, alpha, direction):
        offset = 0
        for p in self._params:
            numel = reduce(lambda x, y: x * y, p.shape) if p.shape != [] else 1
            p = paddle.assign(
                p.add(
                    direction[offset : offset + numel].reshape(p.shape) * alpha
                ),
                p,
            )
            offset += numel
        assert offset == self._numel()

    def _clone_param(self):
        return [p.clone() for p in self._params]

    def _set_param(self, params_data):
        for p, pdata in zip(self._params, params_data):
            paddle.assign(pdata, p)

    def _directional_evaluate(self, closure, x, alpha, d):
        self._add_grad(alpha, d)
        loss = float(closure())
        flat_grad = self._gather_flat_grad()
        self._set_param(x)
        return loss, flat_grad

    @framework.non_static_only
    def step(self, closure):
        """Performs a single optimization step.

        Args:
            closure (callable): A closure that reevaluates the model
            and returns the loss.

        Examples:
            .. code-block:: python

                >>> import paddle

                >>> paddle.disable_static()

                >>> inputs = paddle.rand([10, 10], dtype="float32")
                >>> targets = paddle.to_tensor([2 * x for x in inputs])

                >>> net = paddle.nn.Linear(10, 10)
                >>> opt = paddle.optimizer.LBFGS(
                ...     learning_rate=1,
                ...     max_iter=1,
                ...     max_eval=None,
                ...     tolerance_grad=1e-07,
                ...     tolerance_change=1e-09,
                ...     history_size=100,
                ...     line_search_fn='strong_wolfe',
                ...     parameters=net.parameters(),
                >>> )

                >>> def closure():
                ...     outputs = net(inputs)
                ...     loss = paddle.nn.functional.mse_loss(outputs, targets)
                ...     print("loss:", loss.item())
                ...     opt.clear_grad()
                ...     loss.backward()
                ...     return loss
                ...
                >>> opt.step(closure)
        """

        with paddle.no_grad():
            # Make sure the closure is always called with grad enabled
            closure = paddle.enable_grad()(closure)

            learning_rate = self.learning_rate
            max_iter = self.max_iter
            max_eval = self.max_eval
            tolerance_grad = self.tolerance_grad
            tolerance_change = self.tolerance_change
            line_search_fn = self.line_search_fn
            history_size = self.history_size
            state = self.state
            state.setdefault('func_evals', 0)
            state.setdefault('n_iter', 0)

            # evaluate initial f(x) and df/dx
            orig_loss = closure()
            loss = float(orig_loss)

            current_evals = 1
            state['func_evals'] += 1

            flat_grad = self._gather_flat_grad()
            opt_cond = flat_grad.abs().max() <= tolerance_grad

            # optimal condition
            if opt_cond:
                return orig_loss

            # tensors cached in state (for tracing)
            d = state.get('d')
            alpha = state.get('alpha')
            old_yk = state.get('old_yk')
            old_sk = state.get('old_sk')
            ro = state.get('ro')
            H_diag = state.get('H_diag')
            prev_flat_grad = state.get('prev_flat_grad')
            prev_loss = state.get('prev_loss')

            n_iter = 0
            # optimize for a max of max_iter iterations
            while n_iter < max_iter:
                # keep track of nb of iterations
                n_iter += 1
                state['n_iter'] += 1

                ############################################################
                # compute gradient descent direction
                ############################################################
                if state['n_iter'] == 1:
                    d = flat_grad.neg()
                    old_yk = []
                    old_sk = []
                    ro = []
                    H_diag = paddle.to_tensor(1.0, dtype=orig_loss.dtype)
                else:
                    # do lbfgs update (update memory)
                    y = flat_grad.subtract(prev_flat_grad)
                    s = d.multiply(paddle.to_tensor(alpha, dtype=d.dtype))
                    ys = dot(y, s)
                    if ys > 1e-10:
                        # updating memory
                        if len(old_yk) == history_size:
                            # shift history by one (limited-memory)
                            old_yk.pop(0)
                            old_sk.pop(0)
                            ro.pop(0)

                        # store new direction/step
                        old_yk.append(y)
                        old_sk.append(s)
                        ro.append(1.0 / ys)

                        # update scale of initial Hessian approximation
                        H_diag = ys / dot(y, y)  # (y*y)

                    # compute the approximate (L-BFGS) inverse Hessian
                    # multiplied by the gradient
                    num_old = len(old_yk)

                    if 'al' not in state:
                        state['al'] = [None] * history_size
                    al = state['al']

                    # iteration in L-BFGS loop collapsed to use just one buffer
                    q = flat_grad.neg()
                    for i in range(num_old - 1, -1, -1):
                        al[i] = dot(old_sk[i], q) * ro[i]
                        paddle.assign(q.add(old_yk[i] * (-al[i])), q)

                    # multiply by initial Hessian
                    # r/d is the final direction
                    d = r = paddle.multiply(q, H_diag)
                    for i in range(num_old):
                        be_i = dot(old_yk[i], r) * ro[i]
                        paddle.assign(r.add(old_sk[i] * (al[i] - be_i)), r)

                if prev_flat_grad is None:
                    prev_flat_grad = flat_grad.clone()
                else:
                    paddle.assign(flat_grad, prev_flat_grad)
                prev_loss = loss

                ############################################################
                # compute step length
                ############################################################
                # reset initial guess for step size
                if state['n_iter'] == 1:
                    alpha = (
                        min(1.0, 1.0 / flat_grad.abs().sum()) * learning_rate
                    )
                else:
                    alpha = learning_rate

                # directional derivative
                gtd = dot(flat_grad, d)

                # directional derivative is below tolerance
                if gtd > -tolerance_change:
                    break

                # optional line search: user function
                ls_func_evals = 0
                if line_search_fn is not None:
                    # perform line search, using user function
                    if line_search_fn != "strong_wolfe":
                        raise RuntimeError("only 'strong_wolfe' is supported")
                    else:
                        x_init = self._clone_param()

                        def obj_func(x, alpha, d):
                            return self._directional_evaluate(
                                closure, x, alpha, d
                            )

                        loss, flat_grad, alpha, ls_func_evals = _strong_wolfe(
                            obj_func, x_init, alpha, d, loss, flat_grad, gtd
                        )
                    self._add_grad(alpha, d)
                    opt_cond = flat_grad.abs().max() <= tolerance_grad
                else:
                    # no line search, simply move with fixed-step
                    self._add_grad(alpha, d)
                    if n_iter != max_iter:
                        with paddle.enable_grad():
                            loss = float(closure())
                        flat_grad = self._gather_flat_grad()
                        opt_cond = flat_grad.abs().max() <= tolerance_grad
                        ls_func_evals = 1

                # update func eval
                current_evals += ls_func_evals
                state['func_evals'] += ls_func_evals

                # optimal condition
                if opt_cond:
                    break

                # lack of progress
                if (d * alpha).abs().max() <= tolerance_change:
                    break

                if abs(loss - prev_loss) < tolerance_change:
                    break

                # check conditions
                if current_evals >= max_eval:
                    break

                if n_iter == max_iter:
                    break

            state['d'] = d
            state['alpha'] = alpha
            state['old_yk'] = old_yk
            state['old_sk'] = old_sk
            state['ro'] = ro
            state['H_diag'] = H_diag
            state['prev_flat_grad'] = prev_flat_grad
            state['prev_loss'] = prev_loss

        return orig_loss

    def minimize(
        self, loss, startup_program=None, parameters=None, no_grad_set=None
    ):
        """Empty method. LBFGS optimizer does not use this way to minimize ``loss``. Please refer 'Examples' of LBFGS() above for usage."""
        raise NotImplementedError(
            "LBFGS optimizer does not use this way to minimize loss. Please refer 'Examples' of LBFGS() for usage."
        )
