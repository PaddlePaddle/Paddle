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
from paddle.optimizer import Optimizer
from paddle.utils import deprecated

from .line_search_dygraph import _strong_wolfe


@deprecated(since="2.5.0", update_to="paddle.optimizer.LBFGS", level=1)
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
            :ref:`api_fluid_regularizer_L1Decay`, :ref:`api_fluid_regularizer_L2Decay`.
            If a parameter has set regularizer using :ref:`api_fluid_ParamAttr` already, \
            the regularization setting here in optimizer will be ignored for this parameter. \
            Otherwise, the regularization setting here in optimizer will take effect. \
            Default None, meaning there is no regularization.
        grad_clip (GradientClipBase, optional): Gradient cliping strategy, it's an instance of \
            some derived class of ``GradientClipBase`` . There are three cliping strategies \
            ( :ref:`api_fluid_clip_GradientClipByGlobalNorm` , :ref:`api_fluid_clip_GradientClipByNorm` , \
            :ref:`api_fluid_clip_GradientClipByValue` ). Default None, meaning there is no gradient clipping.
        name (str, optional): Normally there is no need for user to set this property.
            For more information, please refer to :ref:`api_guide_Name`.
            The default value is None.

    Return:
        loss (Tensor): the final loss of closure.

    Examples:
        .. code-block:: python

            import paddle
            import numpy as np
            from paddle.incubate.optimizer import LBFGS

            paddle.disable_static()
            np.random.seed(0)
            np_w = np.random.rand(1).astype(np.float32)
            np_x = np.random.rand(1).astype(np.float32)

            inputs = [np.random.rand(1).astype(np.float32) for i in range(10)]
            # y = 2x
            targets = [2 * x for x in inputs]

            class Net(paddle.nn.Layer):
                def __init__(self):
                    super().__init__()
                    w = paddle.to_tensor(np_w)
                    self.w = paddle.create_parameter(shape=w.shape, dtype=w.dtype, default_initializer=paddle.nn.initializer.Assign(w))

                def forward(self, x):
                    return self.w * x

            net = Net()
            opt = LBFGS(learning_rate=1, max_iter=1, max_eval=None, tolerance_grad=1e-07, tolerance_change=1e-09, history_size=100, line_search_fn='strong_wolfe', parameters=net.parameters())
            def train_step(inputs, targets):
                def closure():
                    outputs = net(inputs)
                    loss = paddle.nn.functional.mse_loss(outputs, targets)
                    print('loss: ', loss.item())
                    opt.clear_grad()
                    loss.backward()
                    return loss
                opt.step(closure)


            for input, target in zip(inputs, targets):
                input = paddle.to_tensor(input)
                target = paddle.to_tensor(target)
                train_step(input, target)

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
            numel = reduce(lambda x, y: x * y, p.shape)
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

    def step(self, closure):
        """Performs a single optimization step.
        Args:
            closure (callable): A closure that reevaluates the model
                and returns the loss.
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
                    ys = y.dot(s)
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
                        H_diag = ys / y.dot(y)  # (y*y)

                    # compute the approximate (L-BFGS) inverse Hessian
                    # multiplied by the gradient
                    num_old = len(old_yk)

                    if 'al' not in state:
                        state['al'] = [None] * history_size
                    al = state['al']

                    # iteration in L-BFGS loop collapsed to use just one buffer
                    q = flat_grad.neg()
                    for i in range(num_old - 1, -1, -1):
                        al[i] = old_sk[i].dot(q) * ro[i]
                        paddle.assign(q.add(old_yk[i] * (-al[i])), q)

                    # multiply by initial Hessian
                    # r/d is the final direction
                    d = r = paddle.multiply(q, H_diag)
                    for i in range(num_old):
                        be_i = old_yk[i].dot(r) * ro[i]
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
                gtd = flat_grad.dot(d)

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
