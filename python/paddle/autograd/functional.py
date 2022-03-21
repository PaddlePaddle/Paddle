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

import contextlib
import typing
import paddle
from paddle.static import gradients
import functools

from paddle.fluid import framework
from paddle.fluid.dygraph import grad
from paddle.tensor import reshape, to_tensor, zeros_like
from paddle.tensor.creation import assign


def vjp(func, xs, v=None):
    r"""Computes the Vector-Jacobian product, a functional form of
    reverse mode automatic differentiation.

    Args:
        func(Callable): A function that takes ``xs`` as inputs parameter and
            returns a sequence of Tensors or a Tensor.
        xs(Tensor|Sequence[Tensor]): Used as positional arguments to evaluate
            ``func``. ``xs`` is accepted as one tensor or a sequence of tensors.
        v(Tensor|Sequence[Tensor]|None, optional): The cotangent vector invovled
            in the VJP computation. `v` matches the size and shape of
            ``func`` 's output. Defaults to None, which is equivalent to all
            ones the same size of ``func`` 's output.

    Returns:
        output(tuple):
            func_out(Tensor|List[Tensor]): The output of ``func(xs)`` .
            vjp(Tensor|List[Tensor]): The pullback results of ``v`` on ``func`` .

    Examples:

      .. code-block:: python

        def func(x):
          return paddle.matmul(x, x)

        x = paddle.ones(shape=[2, 2], dtype='float32')
        output, inputs_grad = vjp(func, x)
        print(inputs_grad)
        # [Tensor(shape=[2, 2], dtype=float32, place=CUDAPlace(0), stop_gradient=True,
        #        [[4., 4.],
        #         [4., 4.]])]

        v = paddle.to_tensor([[1.0, 0.0], [0.0, 0.0]])
        output, inputs_grad = vjp(func, x, v)
        print(inputs_grad)
        # [Tensor(shape=[2, 2], dtype=float32, place=CUDAPlace(0), stop_gradient=True,
        #        [[2., 1.],
        #         [1., 0.]])]

        output, inputs_grad = vjp(func, x, v, create_graph=True)
        print(inputs_grad)
        # [Tensor(shape=[2, 2], dtype=float32, place=CUDAPlace(0), stop_gradient=False,
        #        [[2., 1.],
        #         [1., 0.]])]

        y = paddle.ones(shape=[2, 2], dtype='float32')
        def func_unused(x, y):
          return paddle.matmul(x, x)

        output, inputs_grad = vjp(func, [x, y], v)
        # ValueError: (InvalidArgument) The 1-th input does not appear in the backward graph.
        # Please check the input variable or set allow_unused=True to get None result.
        # [Hint: Expected allow_unused_ == true, but received allow_unused_:0 != true:1.]

        output, inputs_grad = vjp(func, [x, y], v, allow_unused=True)
        print(inputs_grad)
        # [Tensor(shape=[2, 2], dtype=float32, place=CUDAPlace(0), stop_gradient=True,
        #        [[2., 1.],
        #         [1., 0.]]), None]
    """
    _check_inputs(func, xs, v)

    # _grad_preprocess will detach element of xs from computing graph when
    # stop_gradient=True. Therefore, must execute this method before
    # calling func for avoiding breaking the dependencies between xs and ys
    xs, v = _grad_preprocess(xs), _grad_preprocess(v)
    ys = func(*xs) if isinstance(xs, typing.Sequence) else func(xs)
    _check_v_shape(v, ys)

    return ys, _grad(ys, xs, v)


def jvp(func, xs, v=None):
    r"""
    Computes the Jacobian-Vector product for a function at the given
    inputs and a vector in the tangent space induced by the inputs.

    Args:
        func(Callable): The ``func`` takes as input a tensor or a list/tuple
            of tensors and returns a tensor or a list/tuple of tensors.
        inputs(Tensor|Sequence[Tensor]): Used as positional arguments to
            evaluate ``func``.  The ``inputs`` is accepted as one tensor or a
            sequence of tensors.
        v(Tensor|Sequence[Tensor]|None, optional): The tangent vector invovled
            in the JVP computation. The ``v`` matches the size and shape of
            ``inputs``. ``v`` is Optional if ``func`` returns a single tensor.
            Default value is None and in this case is equivalent to all ones
            the same size of ``inputs``.

    Returns:
        output(tuple):
            func_out(Tensor|Sequence[Tensor]): The output of ``func(xs)`` .
            jvp(Tuple[Tensor]): The pullback results of ``v`` on ``func`` .

    Examples:

    .. code-block:: python

        def func(x):
          return paddle.matmul(x, x)

        x = paddle.ones(shape=[2, 2], dtype='float32')

        output, inputs_grad = jvp(func, x)
        print(inputs_grad)
        # [Tensor(shape=[2, 2], dtype=float32, place=CUDAPlace(0), stop_gradient=True,
        #        [[2., 2.],
        #         [2., 2.]])]

        v = paddle.to_tensor([[1.0, 0.0], [0.0, 0.0]])
        output, inputs_grad = vjp(func, x, v)
        print(inputs_grad)
        # [Tensor(shape=[2, 2], dtype=float32, place=CUDAPlace(0), stop_gradient=True,
        #        [[1., 1.],
        #         [0., 0.]])]

    """
    _check_inputs(func, xs, v)
    # _grad_preprocess will detach element of xs from computing graph when
    # stop_gradient=True. Therefore, must execute this method before
    # calling func for avoiding breaking the graph dependencies between xs and
    # ys.
    xs, v = _grad_preprocess(xs), _grad_preprocess(v)
    ys = func(*xs) if isinstance(xs, typing.Sequence) else func(xs)
    _check_v_shape(v, xs)
    return ys, _double_backward_trick(ys, xs, v)


def _double_backward_trick(ys, xs, v):
    """Double backward trick for computing ``jvp`` by ``vjp``

    see details: https://j-towns.github.io/2017/06/12/A-new-trick.html
    """
    # In theory, it can be any random value, using zeros at this palce is just
    # for computing conveniently.
    zeros_ys = [paddle.zeros_like(y) for y in _as_tensors(ys)] if isinstance(
        ys, typing.Sequence) else paddle.zeros_like(ys)
    first_order_grad = _grad(ys, xs, zeros_ys)
    return _grad(first_order_grad, zeros_ys, v)


@framework.dygraph_only
def vhp(func, inputs, v=None, create_graph=False, allow_unused=False):
    '''
    .. note::
        **This API is ONLY available in the imperative mode.**

    This function computes the product between a vector ``v`` and the
    Hessian matrix of `func` with respect to `inputs`.

    Parameters:
        func (function): a Python function that takes a Tensor or a Tensor
            list/tuple as inputs and returns a Tensor with a single element.
        inputs (Tensor|list(Tensor)|tuple(Tensor)): the input Tensor or
            Tensor list/tuple of the function ``func``.
        v (Tensor|list(Tensor)|tuple(Tensor)|None, optional): the vector used
            to compute vector hessian product. ``v`` should have same shape
            and dtype with ``inputs``. If ``v`` is None, it will be set as
            Tensor|list(Tensor) with all elements 1. Defaults to "None".
        create_graph (bool, optional): whether to create the gradient graphs
            of the computing process. When it is True, higher order derivatives
            are supported to compute; when it is False, the gradient graphs of
            the computing process would be discarded. Defaults to ``False``.
        allow_unused (bool, optional): whether to raise error or return None if
            some Tensors of `inputs` are unreachable in the graph. Error would
            be raised if allow_unused=False, and None would be returned as
            their gradients if allow_unused=True. Default False.
    Returns:
        output (tuple): tuple with:
            func_output (Tensor): output of ``func(inputs)``
            vhp (list(Tensor)): result of the vector hessian product
            with the same shape and dtype as the inputs.
    Examples 1:
        .. code-block:: python
            import paddle
            def func(x):
                return paddle.sum(paddle.matmul(x, x))

            x = paddle.ones(shape=[2, 2], dtype='float32')
            x.stop_gradient = False
            vx = paddle.ones(shape=[2, 2], dtype='float32') * 2
            vhp_rslt = paddle.autograd.vhp(func, x, v=vx)
            print(vhp_rslt)
            # (Tensor(shape=[1], dtype=float32, place=CUDAPlace(0), stop_gradient=False,
            #        [8.]),
            #  Tensor(shape=[2, 2], dtype=float32, place=CUDAPlace(0), stop_gradient=True,
            #        [[8., 8.],
            #         [8., 8.]]))

    Examples 2:
        .. code-block:: python
            import paddle
            def func(x):
                return paddle.sum(paddle.matmul(x, x))

            x = paddle.ones(shape=[2, 2], dtype='float32')
            x.stop_gradient = False
            vhp_rslt = paddle.autograd.vhp(func, x)
            print(vhp_rslt)
            # (Tensor(shape=[1], dtype=float32, place=CUDAPlace(0), stop_gradient=False,
            #        [8.]),
            #  Tensor(shape=[2, 2], dtype=float32, place=CUDAPlace(0), stop_gradient=True,
            #        [[4., 4.],
            #         [4., 4.]]))

    Examples 3:
        .. code-block:: python
            import paddle
            def func(x, y):
                return paddle.sum(paddle.matmul(x, x))

            x = paddle.ones(shape=[2, 2], dtype='float32')
            x.stop_gradient = False
            y = paddle.ones(shape=[2, 2], dtype='float32')
            y.stop_gradient = False
            vx = paddle.ones(shape=[2, 2], dtype='float32') * 2
            vy = paddle.ones(shape=[2, 2], dtype='float32') * 3
            vhp_rslt = paddle.autograd.vhp(
                func, [x, y], v=[vx, vy], allow_unused=True)
            print(vhp_rslt)
            # (Tensor(shape=[1], dtype=float32, place=CUDAPlace(0), stop_gradient=False,
            #        [8.]),
            # [Tensor(shape=[2, 2], dtype=float32, place=CUDAPlace(0), stop_gradient=True,
            #        [[8., 8.],
            #         [8., 8.]]), None])
    '''
    xs = _as_tensors(inputs, "inputs")
    if v is not None:
        v = _as_tensors(v, "v")

    with gradient_scope(
            xs, v, create_graph=create_graph,
            allow_unused=allow_unused) as [xs, v, grad_fn, return_fn]:
        outputs = func(*xs)
        ys = _as_tensors(outputs, "outputs")
        assert len(ys) == 1 and isinstance(
            ys[0], paddle.fluid.framework.Variable
        ) and ys[0].shape == [
            1
        ], "The function to compute vhp should return a Tensor with a single element"
        jac = grad_fn(ys, xs, create_graph=True)
        vhp = grad_fn(jac, xs, v)
        outputs, vhp = return_fn(outputs), return_fn(vhp)
    return outputs, vhp


class Jacobian(object):
    r"""
    Computes the Jacobian matrix of function `func`, which may take as input
    single or multiple tensor typed arguments and output a single tensor or
    multiple tensors.

    In case `func` is multi-input and multi-output, i.e.,

    func: Callable[[Tensor, ...], [Tensor, ...]]

    `func` is treated as a vector valued function with all its inputs flattened
    into a single one dimensional tensor, or a two dimensional tensor with the
    first dimension retained as the batching dimension. The same rule applies to
    the function outputs.

    Once the Jacobian J is constructed, there are four ways to retrieve the
    partial derivatives.

    - J[:], retrieving the full matrix.

    - J[:, j], retrieving the partial derivatives w.r.t. the j'th input
    variable.

    - J[i, :], retrieving the partial derivatives w.r.t. the i'th output
    variable.

    - J[i, j], retrieving the partial derivatives w.r.t. the i'th output
    variable and the j'th input variable.

    Examples:

        .. code-block:: python
            import paddle
            import numpy as np

            def func(xs):
                x, y = xs
                return paddle.matmul(x, y)

            main = fluid.Program()
            startup = fluid.Program()
            with fluid.program_guard(main, startup):
                x = paddle.static.data(name='x', shape=[2, 2], dtype='float32')
                JJ = paddle.autograd.functional.Jacobian(func, [x, x])
                nrow, ncol = JJ.shape()
                full_jacobian = JJ[:]
            place = fluid.CUDAPlace(0)
            exe = fluid.Executor(place)
            exe.run(startup)

            feeds = {'x': np.array([[2., 2.], [2., 1.]]).astype('float32')}
            jacobian = exe.run(main, feed=feeds, fetch_list=[full_jacobian])[0]
            print(jacobian)
            # [[4. 2. 2. 0. 4. 2. 2. 0.]
            #  [2. 3. 0. 2. 2. 3. 0. 2.]
            #  [2. 0. 3. 2. 2. 0. 3. 2.]
            #  [0. 2. 2. 2. 0. 2. 2. 2.]]
    """

    def __init__(self, func, xs, is_batched=False, batched_last=False):
        if not is_batched:
            self._jacobian = _JacobianNoBatch(func, xs)
        elif batched_last:
            self._jacobian = _JacobianBatchLast(func, xs)
        else:
            self._jacobian = _JacobianBatchFirst(func, xs)

    def __getitem__(self, indexes):
        return self._jacobian[indexes]

    @property
    def shape(self):
        return self._jacobian.shape


class _Jacobian(object):
    """The base class for computing Jacobian matrix.
    """

    def __init__(self, func, xs):
        self._xs = _grad_preprocess(xs)
        self._ys = func(*_as_tensors(self._xs))
        self._flatten_xs = self._flatten(_as_tensors(self._xs))
        self._flatten_ys = self._flatten(_as_tensors(self._ys))
        self._cache = {}

    @property
    def shape(self):
        raise NotImplementedError

    @property
    def _row_axis(self):
        raise NotImplementedError

    def _flatten(self, xs):
        raise NotImplementedError

    def _stack(self, values):
        raise NotImplementedError

    def __getitem__(self, indexes):
        row_index = _multi_index(indexes, self.shape)[self._row_axis]
        if isinstance(row_index, int):
            rows = (row_index, )
        else:
            rows = sorted(
                tuple(range(row_index.start, row_index.stop, row_index.step)))
        values = []
        for row in rows:
            value = self._cache.get(row)
            if value is None:
                value = self._evaluate(row)
                self._cache[row] = value
            values.append(value)
        return self._stack(values)[indexes]

    def _evaluate(self, row):
        """Lazy evaluation at one row."""
        raise NotImplementedError


class _JacobianNoBatch(_Jacobian):
    """Compute Jacobian matrix without batch.
    Suppose the mapping is :math:`f: R^M \to R^N`, the output shape is 
    ``(N, M)`` .
    """

    def __init__(self, func, xs):
        super(_JacobianNoBatch, self).__init__(func, xs)

    @property
    def shape(self):
        return (self._flatten_ys.shape[0], self._flatten_xs.shape[0])

    @property
    def _row_axis(self):
        return 0

    def _flatten(self, xs):
        return paddle.concat(tuple(x.reshape((-1, )) for x in xs))

    def _stack(self, rows):
        return paddle.stack(rows)

    def _evaluate(self, row_index):
        return self._flatten(_grad(
            self._flatten_ys[row_index],
            self._xs, ))


class _JacobianBatchLast(_Jacobian):
    """Compute Jacobian matrix with batch at last axis.
    Suppose the mapping is :math:`f: R^{M,B} \to R^{N,B}`, the output shape is 
    ``(N, M, B)`` .
    """

    def __init__(self, func, xs):
        super(_JacobianBatchLast, self).__init__(func, xs)

    @property
    def shape(self):
        return (self._flatten_ys.shape[0], self._flatten_xs.shape[0],
                self._flatten_xs.shape[1])

    @property
    def _row_axis(self):
        return 0

    def _flatten(self, xs):
        return paddle.concat(
            tuple(x.reshape((-1, x.shape[-1])) for x in _as_tensors(xs)), 0)

    def _stack(self, rows):
        return paddle.stack(rows, 0)

    def _evaluate(self, row):
        return self._flatten(_grad(self._flatten_ys[row, :], self._xs))


class _JacobianBatchFirst(_Jacobian):
    """Compute Jacobian matrix with batch at first axis.
    Suppose the mapping is :math:`f: R^{B,M} \to R^{B,N}`, the output shape is 
    ``(B, N, M)`` .
    """

    def __init__(self, func, xs):
        super(_JacobianBatchFirst, self).__init__(func, xs)

    @property
    def shape(self):
        return (self._flatten_xs.shape[0], self._flatten_ys.shape[1],
                self._flatten_xs.shape[1])

    @property
    def _row_axis(self):
        return 1

    def _flatten(self, xs):
        return paddle.concat(
            tuple(x.reshape((x.shape[0], -1)) for x in _as_tensors(xs)), 1)

    def _stack(self, rows):
        return paddle.stack(rows, 1)

    def _evaluate(self, row_index):
        return self._flatten(_grad(self._flatten_ys[:, row_index], self._xs))


def _multi_index(indexes, shape):
    """A tool for parsing N-dimensional index into a standard format.

    Currently supporting following input format:
        * ([positive|negative|slice], ...), the right-most elements can be 
            omited.

    The standard format after converted is slice tuple which contains N elements:
        * ([positive|slice], ..., [positive|slice])

    Notes: 
        Ellipsis indexes such as ``(..., i), (i, ...)`` is not supported.

    Args:
        indexes (tuple): The input indexes.
        shape (tuple): The input shape.

    Returns:
        tuple: The standard format index as the above description.
    """
    indexes = indexes if isinstance(indexes, typing.Sequence) else (indexes, )
    if any(isinstance(i, type(Ellipsis)) for i in indexes):
        raise IndexError('Ellipsis index currently is not supported.')
    # Fill the right-most elements.
    indexes = indexes + (slice(0, None, None), ) * (len(shape) - len(indexes))
    # Convert to positive index.
    positive_indexes = []
    for i, index in enumerate(indexes):
        if isinstance(index, slice):
            index = slice(index.start or 0, index.stop or shape[i],
                          index.step or 1)
            positive_indexes.append(
                slice(
                    index.start + shape[i] if index.start < 0 else index.start,
                    index.stop + shape[i] if index.stop < 0 else index.stop,
                    # Negative step means index backward, no need to convert to
                    # positive interger.
                    index.step))
        elif isinstance(index, int):
            positive_indexes.append(index + shape[i] if index < 0 else index)
        else:
            raise TypeError(f'Not supported index type {index}.')
    return positive_indexes


class Hessian(object):
    def __init__(self, func, xs, is_batched=False, batched_last=False):
        def _jacobian(xs):
            return Jacobian(
                func, xs, is_batched=is_batched, batched_last=batched_last)

        self.symbolic = Jacobian(
            f_x, xs, is_batched=is_batched, batched_last=batched_last)

    def __getitem__(self, indexes):
        return self.symbolic[indexes]

    def shape(self):
        return self.symbolic.shape()


def _as_tensors(xs):
    return (xs, ) if isinstance(xs, framework.Variable) else xs


def _stack_tensor_or_return_none(origin_list):
    assert len(origin_list) > 0, "Can't not stack an empty list"
    return paddle.stack(
        origin_list, axis=0) if isinstance(
            origin_list[0], paddle.fluid.framework.Variable) else None


def _replace_none_with_zero_tensor(xs, refs):
    if xs is None:
        return paddle.zeros_like(refs)
    elif isinstance(xs, typing.Sequence):
        return tuple(
            paddle.zeros_like(refs[i]) if x is None else x
            for i, x in enumerate(xs))
    else:
        return xs


def _grad(ys, xs, v=None):
    """A gradient function that can be used in dynamic graph and static graph.

    The ``grad`` combines ``paddle.grad`` used in dynamic graph and
    ``paddle.static.gradients`` used in static graph, and do following changes:

    * The ``allow_unused`` flag is removed and set defaults to true internally,
        outputs with none will be replace by zero.
    * The ``create_graph`` flag is removed and set defaults to true internally,
        only makes sense in dynamic graph.
    * When ``xs`` is a Tensor, ``paddle.grad`` still return a list of Tensor.
        It's not consistent. So, ``_grad`` unpack the return value to a Tensor
        when ``xs`` is a Tensor.

    Args:
        ys (Tensor|Sequence[Tensor]): The output tensor or tensor sequence of
            the graph to compute gradients.
        xs (Tensor|Sequence[Tensor]): The input tensor or tensor sequence of the graph to
            compute gradients. The returned values of this API are the
            gradients of inputs .
        v (Tensor|Sequence[Tensor]|None,optional): The initial gradient values
            of outputs . If grad_outputs is None, the initial gradient values of
            outputs would be Tensors filled with 1; if grad_outputs is not None,
            it must have the same length as outputs , and in this case, the
            initial gradient value of the i-th outputs would be: (1) a Tensor
            filled with 1 when the i-th element of grad_outputs is None;
            (2) the i-th element of grad_outputs when the i-th element of
            grad_outputs is a Tensor. Default None.

    Returns:
        Tensor|Tuple[Tensor]: A tuple of Tensors, whose length is the same as the
            Tensor number inside inputs, and the i-th returned Tensor is the sum
            of gradients of outputs with respect to the i-th inputs.
    """
    if paddle.fluid.framework.in_dygraph_mode():
        xs_grad = paddle.grad(ys, xs, v, create_graph=True, allow_unused=True)
    else:
        xs_grad = paddle.static.gradients(ys, xs, v)

    if isinstance(xs, paddle.fluid.framework.Variable):
        xs_grad = xs_grad[0]

    return _replace_none_with_zero_tensor(xs_grad, xs)


def _grad_preprocess(xs):
    # If v is treated as constant in the outer scope, its gradient is guaranteed
    # not to be taken beyond this scope. Within this scope, however, v's gradient
    # may be computed. We only need to detach v in this case.
    # Otherwise, v's gradient is valid, and is subject to update beyond this scope.
    # In this case we must not confuse the gradient in the outer scope with the
    # inner one's. Moreover, we need to make sure that the result from the inner
    # scope can flow back to the outer scope. This can be satisfied by extending
    # the original variable with a duplication operation v1 = v so that v still
    # maintains the complete lineage.
    if isinstance(xs, typing.Sequence):
        return tuple(_cutoff_dependency(x) for x in xs)
    else:
        return _cutoff_dependency(xs)


def _cutoff_dependency(x):
    if x is None:  # x maybe none because grad input's v defaults to none.
        return x
    if not x.stop_gradient:
        return paddle.assign(x)
    else:
        x = x.detach()
        x.stop_gradient = False
        return x
    return x


def _check_inputs(func, xs, v=None):
    if not callable(func):
        raise TypeError(f"Expected 'fun' is Callable, but got {type(func)}.")

    if not isinstance(xs, (framework.Variable, typing.Sequence)):
        raise TypeError(f"Expected 'xs' is a Tensor|Sequence[Tensor],"
                        f"but got {type(xs)}.")
    if isinstance(xs, typing.Sequence) and not all(
            isinstance(x, framework.Variable) for x in xs):
        raise TypeError("All elements of 'xs' shoule be Tensor.")

    if not isinstance(v, (framework.Variable, typing.Sequence, type(None))):
        raise TypeError(
            f"Expected 'v' is Tensor|List[Tensor]|None, but got {type(v)}.")

    if isinstance(v, typing.Sequence) and not all(
            isinstance(e, framework.Variable) for e in v):
        raise TypeError("All elements of 'xs' shoule be Tensor.")


def _check_v_shape(v, refs):
    if isinstance(v, typing.Sequence):
        if len(refs) != len(v):
            raise RuntimeError(f"The argument v is a tuple of invalid length:"
                               f"should be {len(refs)} but got {len(v)}.")

        for index, (element_v, element_ref) in enumerate(zip(v, refs)):
            if element_v.shape != element_ref.shape:
                raise RuntimeError(
                    f"The v[{index}] has invalid shape: should "
                    f"be {element_ref.shape} but got {element_v.shape}.")
