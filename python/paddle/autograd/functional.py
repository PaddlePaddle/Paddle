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

import functools
import typing

import paddle
from paddle.fluid import framework


def vjp(func, xs, v=None):
    r"""Computes the Vector-Jacobian product, a functional form of
    reverse mode automatic differentiation.

    Warning:
        This API is in beta, the signatures could be changed in future version.

    Args:
        func(Callable): A function that takes ``xs`` as inputs parameter and
            returns a sequence of Tensors or a Tensor.
        xs(Tensor|Sequence[Tensor]): Used as positional arguments to evaluate
            ``func``. ``xs`` is accepted as one Tensor or a sequence of Tensors.
        v(Tensor|Sequence[Tensor]|None, optional): The cotangent vector invovled
            in the VJP computation. ``v`` matches the size and shape of
            ``func`` 's output. Defaults to None, which is equivalent to all
            ones the same size of ``func`` 's output.

    Returns:
        output(tuple):
        
            - func_out(Tensor|tuple[Tensor]): The output of ``func(xs)`` .
            - vjp(Tensor|tuple[Tensor]): The vjp result.

    Examples:

        .. code-block:: python

            import paddle

            def func(x):
                return paddle.matmul(x, x)

            x = paddle.ones(shape=[2, 2], dtype='float32')
            _, vjp_result = paddle.incubate.autograd.vjp(func, x)
            print(vjp_result)
            # Tensor(shape=[2, 2], dtype=float32, place=Place(gpu:0), stop_gradient=False,
            #        [[4., 4.],
            #         [4., 4.]])

            v = paddle.to_tensor([[1.0, 0.0], [0.0, 0.0]])
            _, vjp_result = paddle.incubate.autograd.vjp(func, x, v)
            print(vjp_result)
            # Tensor(shape=[2, 2], dtype=float32, place=Place(gpu:0), stop_gradient=False,
            #        [[2., 1.],
            #         [1., 0.]])
    """
    _check_inputs(func, xs, v)

    # ``_seprate`` breaks the dependencies between ``xs`` and other
    # variables. See more ``_seprate`` .
    xs, v = _separate(xs), _separate(v)
    ys = func(*xs) if isinstance(xs, typing.Sequence) else func(xs)
    _check_v_shape(v, ys)

    return ys, _grad(ys, xs, v)


def jvp(func, xs, v=None):
    r"""
    Computes the Jacobian-Vector product for a function at the given
    inputs and a vector in the tangent space induced by the inputs.

    Warning:
        This API is in beta, the signatures could be changed in future version.

    Args:
        func(Callable): The ``func`` takes as input a Tensor or a Sequence
            of Tensors and returns a Tensor or a Sequence of Tensors.
        xs(Tensor|Sequence[Tensor]): Used as positional arguments to
            evaluate ``func``.  The ``xs`` is accepted as one Tensor or a
            Sequence of Tensors.
        v(Tensor|Sequence[Tensor]|None, Optional): The tangent vector invovled
            in the JVP computation. The ``v`` matches the size and shape of
            ``xs`` . Default value is None and in this case is equivalent to 
            all ones the same size of ``xs`` .

    Returns:
        output(tuple):

            - func_out(Tensor|tuple[Tensor]): The output of ``func(xs)`` .
            - jvp(Tensor|tuple[Tensor]): The jvp result.

    Examples:

        .. code-block:: python

            import paddle


            def func(x):
                return paddle.matmul(x, x)


            x = paddle.ones(shape=[2, 2], dtype='float32')
            _, jvp_result = paddle.incubate.autograd.jvp(func, x)
            print(jvp_result)
            # Tensor(shape=[2, 2], dtype=float32, place=Place(gpu:0), stop_gradient=False,
            #        [[4., 4.],
            #         [4., 4.]])
            v = paddle.to_tensor([[1.0, 0.0], [0.0, 0.0]])
            _, jvp_result = paddle.incubate.autograd.jvp(func, x, v)
            print(jvp_result)
            # Tensor(shape=[2, 2], dtype=float32, place=Place(gpu:0), stop_gradient=False,
            #        [[2., 1.],
            #         [1., 0.]])

    """
    _check_inputs(func, xs, v)
    # ``_seprate`` breaks the dependencies between ``xs`` and other
    # variables. See more ``_seprate`` .
    xs, v = _separate(xs), _separate(v)
    ys = func(*xs) if isinstance(xs, typing.Sequence) else func(xs)
    _check_v_shape(v, xs)
    return ys, _double_backward_trick(ys, xs, v)


def _double_backward_trick(ys, xs, v):
    """Double backward trick for computing ``jvp`` by ``vjp``
    see details: https://j-towns.github.io/2017/06/12/A-new-trick.html
    """
    # The value of ys_grad is not important, it can be any random value in 
    # theory, but it's required to set stop_gradient=False.
    ys_grad = _zeros_like_with_grad(ys)
    xs_grad = _grad(ys, xs, ys_grad)
    return _grad(xs_grad, ys_grad, v)


def _zeros_like_with_grad(xs):
    """Create a zero or zeros sequence Tensor like ``xs`` with a flag 
    ``stop_graident=False`` .
    """
    if not isinstance(xs, typing.Sequence):
        ys = paddle.zeros_like(xs)
        ys.stop_gradient = False
    else:
        ys = []
        for x in xs:
            y = paddle.zeros_like(x)
            y.stop_gradient = False
            ys.append(y)
    return ys


class Jacobian(object):
    r"""
    Computes the Jacobian matrix of a given function.

    If the function has multiple inputs and multiple outputs, during internal 
    implementation, all input tensors are concatenated after being flatten, 
    the batch dimension is retained, and the output is subject to the same 
    processing rules.

    Once the Jacobian ``J`` is constructed, you can use a multidimensional index 
    to retrieve the submatrix of ``J``, as same as slicing a Tensor. The 
    submatrix is lazily evaluated along row axis, and will be cached once 
    evaluated.

    For examples, supposing ``is_batched=True``, you can retrieve the submatrix 
    by following methods:

        * J[:], retrieving the full matrix.
        * J[:, :, j], retrieving the partial derivatives w.r.t. the j'th input
          variable.
        * J[:, i, :], retrieving the partial derivatives w.r.t. the i'th output
          variable.
        * J[:, i, j], retrieving the partial derivatives w.r.t. the i'th output
          variable and the j'th input variable.

    Notes:

        Eclipsis index is not supported currently.

    Warning:
        This API is in beta, the signatures could be changed in future version.

    Args:

        func (Callable): A python function that takes a Tensor or a sequence of 
            Tensors as inputs(the first dimension is batch size) and
            returns a Tensor  a sequence of Tensors.
        xs (Tensor|Sequence[Tensor]): The input to the function ``func`` .
        is_batched (bool): If true, the first axis is batch axis. Defaults to 
            False.

    Returns:

        Jacobian (Object): A python object retains the Jacobian matrix.

    Examples:

        .. code-block:: python

            import paddle


            def func(x, y):
                return paddle.matmul(x, y)


            x = paddle.to_tensor([[1., 2.], [3., 4.]])
            J = paddle.incubate.autograd.Jacobian(func, [x, x])
            print(J[:, :])
            # Tensor(shape=[4, 8], dtype=float32, place=Place(gpu:0), stop_gradient=False,
            #        [[1., 3., 0., 0., 1., 0., 2., 0.],
            #         [2., 4., 0., 0., 0., 1., 0., 2.],
            #         [0., 0., 1., 3., 3., 0., 4., 0.],
            #         [0., 0., 2., 4., 0., 3., 0., 4.]])

            print(J[0, :])
            # Tensor(shape=[8], dtype=float32, place=Place(gpu:0), stop_gradient=False,
            #        [1., 3., 0., 0., 1., 0., 2., 0.])
            print(J[:, 0])
            # Tensor(shape=[4], dtype=float32, place=Place(gpu:0), stop_gradient=False,
            #        [1., 2., 0., 0.])

    """

    def __init__(self, func, xs, is_batched=False):
        if not is_batched:
            self._jacobian = _JacobianNoBatch(func, xs)
        else:
            self._jacobian = _JacobianBatchFirst(func, xs)

    def __getitem__(self, indexes):
        return self._jacobian[indexes]

    @property
    def shape(self):
        """The shape of flattened Jacobian matrix.
        """
        return self._jacobian.shape


class Hessian(object):
    """
    Computes the Hessian matrix  with a given ``func`` with respect to ``xs`` .

    If the function has multiple inputs, during internal implementation, 
    all input tensors are concatenated after being flatten, the batch dimension 
    is retained.

    The Hessian submatrix is lazily evaluated, and can be retrieved with a 
    multidimensional indexes. See details ``Jacobian`` .

    Warning:
        This API is in beta, the signatures could be changed in future version.

    Args:
        func (Callable): A python function that takes a Tensor or a Tensor
            sequence as inputs and returns a Tensor with shape 
            ``[batch_size, 1]`` with batch or ``[1]`` without batch.
        xs (Tensor|Sequence(Tensor)): The input Tensor or Tensor sequence of 
            the function ``func``.
        is_batched (bool): If true, the first axis is batch axis. Defaults to 
            False.

    Returns:

        Hessian (Object): A python object retains the Hessian matrix.


    Examples:

    .. code-block:: python

        import paddle


        def reducer(x):
            return paddle.sum(x * x)


        x = paddle.rand([2, 2])
        h = paddle.incubate.autograd.Hessian(reducer, x)
        print(h[:])
        # Tensor(shape=[4, 4], dtype=float32, place=Place(gpu:0), stop_gradient=False,
        #        [[2., 0., 0., 0.],
        #         [0., 2., 0., 0.],
        #         [0., 0., 2., 0.],
        #         [0., 0., 0., 2.]])
    """

    def __init__(self, func, xs, is_batched=False):
        def _jac_func(*xs):
            jac = Jacobian(func, xs, is_batched=is_batched)
            if (is_batched and jac.shape[1] != 1) or (not is_batched and
                                                      jac.shape[0] != 1):
                raise RuntimeError(
                    "The function given to Hessian shoud return as single element Tensor or batched single element Tensor."
                )
            return jac[:, 0, :] if is_batched else jac[0, :]

        self.symbolic = Jacobian(_jac_func, xs, is_batched=is_batched)

    def __getitem__(self, indexes):
        return self.symbolic[indexes]

    @property
    def shape(self):
        """The shape of flattened Hessian matrix.
        """
        return self.symbolic.shape


class _Jacobian(object):
    """The base class for computing Jacobian matrix.

    ``_Jacobian`` implementes the core logic of multidimensional index and lazy 
    evaluation for Jacobian matrix, subclass only need to overwrite following 
    methods:

        * ``_lazy_axis()``,  return the axis along which will be lazy 
            evaluating.
        * ``_flatten(xs)``, flattens the inputs ``xs``.
        * ``_evaluate(index)``, evaluates one slice along ``_lazy_axis`` .

    Notes:

        Because currently PaddlePaddle only support reverse differentiation by 
        ``paddle.grad``, so lazy evaluation is only supported along the row of 
        Jacobian matrix, which means that slicing along row will get better 
        performance.

    """

    def __init__(self, func, xs):
        self._xs = _separate(xs)
        self._ys = func(*_as_tensors(self._xs))
        self._flatten_xs = self._flatten(_as_tensors(self._xs))
        self._flatten_ys = self._flatten(_as_tensors(self._ys))
        self._cache = {}

    @property
    def shape(self):
        raise NotImplementedError

    @property
    def _lazy_axis(self):
        """"The axis of lazily evaluated."""
        raise NotImplementedError

    def _lazy_indexes(self, indexes):
        idx = indexes[self._lazy_axis]
        return (idx, ) if isinstance(
            idx, int) else tuple(range(idx.start, idx.stop, idx.step))

    def _flatten(self, xs):
        raise NotImplementedError

    def _shifted_indexes(self, indexes, lazy_axis_size=0):
        idx = indexes[self._lazy_axis]
        shifted_lazy_axis_idx = 0 if isinstance(
            idx, int) else slice(0, lazy_axis_size, 1)
        return indexes[:self._lazy_axis] + (shifted_lazy_axis_idx,
                                            ) + indexes[self._lazy_axis + 1:]

    def __getitem__(self, indexes):
        indexes = _multi_index(indexes, self.shape)

        if isinstance(indexes[self._lazy_axis], int):
            other_indexes = indexes[:self._lazy_axis] + \
                indexes[self._lazy_axis+1:]
            return self._cached_evaluate(indexes[self._lazy_axis])[
                other_indexes]
        lazy_indexes = self._lazy_indexes(indexes)
        part_jac = paddle.stack(
            [self._cached_evaluate(i) for i in lazy_indexes],
            axis=self._lazy_axis)
        return part_jac[self._shifted_indexes(indexes, len(lazy_indexes))]

    def _cached_evaluate(self, k):
        v = self._cache.get(k)
        if v is None:
            v = self._evaluate(k)
            self._cache[k] = v
        return v

    def _evaluate(self, index):
        """Evaluate one slice at along lazy axis."""
        raise NotImplementedError


class _JacobianNoBatch(_Jacobian):
    """Compute Jacobian matrix without batch dimension.
    Suppose the mapping is :math:`f: R^M \to R^N`, the output shape is 
    ``(N, M)`` .
    """

    def __init__(self, func, xs):
        super(_JacobianNoBatch, self).__init__(func, xs)

    @property
    def shape(self):
        return (self._flatten_ys.shape[0], self._flatten_xs.shape[0])

    @property
    def _lazy_axis(self):
        return 0

    def _flatten(self, xs):
        return paddle.concat(tuple(x.reshape((-1, )) for x in xs))

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
    def _lazy_axis(self):
        return 0

    def _flatten(self, xs):
        return paddle.concat(
            tuple(x.reshape((-1, x.shape[-1])) for x in _as_tensors(xs)), 0)

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
    def _lazy_axis(self):
        return 1

    def _flatten(self, xs):
        return paddle.concat(
            tuple(x.reshape((x.shape[0], -1)) for x in _as_tensors(xs)), 1)

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
    return tuple(positive_indexes)


def _as_tensors(xs):
    return (xs, ) if isinstance(xs, framework.Variable) else xs


def _stack_tensor_or_return_none(origin_list):
    assert len(origin_list) > 0, "Can't not stack an empty list"
    return paddle.stack(
        origin_list, axis=0) if isinstance(
            origin_list[0], paddle.fluid.framework.Variable) else None


def _replace_none_with_zero_tensor(xs, refs):
    if xs is None:
        xs = paddle.zeros_like(refs)
        xs.stop_gradient = refs.stop_gradient
        return xs
    elif isinstance(xs, typing.Sequence):
        return tuple(
            _replace_none_with_zero_tensor(x, refs[i])
            for i, x in enumerate(xs))
    else:
        return xs


def _grad(ys, xs, v=None):
    """A gradient function that can be used in dynamic graph and static graph.

    The ``grad`` combines ``paddle.grad`` used in dynamic graph and
    ``paddle.static.gradients`` used in static graph, and do following changes:

    * The ``allow_unused`` flag is removed and set defaults to true internally,
        none in outputs will be replaced by zero tensor.
    * The ``create_graph`` flag is removed and set defaults to true internally,
        only makes sense in dynamic graph.
    * When xs is a single Tensor, ``paddle.grad`` returns a list which only 
        contains one Tensor. It may confuse users, thus in this case we improve 
        to return a single Tensor in _grad interface.

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
        Tensor|tuple[Tensor]: Tensor or a tuple of Tensors, whose length is the 
            same as the Tensor number inside inputs, and the i-th returned 
            Tensor is the sum of gradients of outputs with respect to the i-th 
            inputs.
    """
    if paddle.fluid._non_static_mode():
        xs_grad = paddle.grad(ys, xs, v, create_graph=True, allow_unused=True)
    else:
        xs_grad = paddle.static.gradients(ys, xs, v)

    if isinstance(xs, paddle.fluid.framework.Variable):
        xs_grad = xs_grad[0]

    return _replace_none_with_zero_tensor(xs_grad, xs)


def _separate(xs):
    """
    ``_separate`` separates ``xs`` from the computation graph through ``clone`` 
    or ``deteach`` .

    Interally, ``paddle.grad(xs, ys)`` is stateful API implemented based on 
    computional graph, which will reduce gradients along all path from ys to xs.

    However, funcional autograd API such as ``vjp``, ``jvp`` is stateless, and 
    only compute gradients with a given ``func`` .

    For example, given a ``func`` :math:`y0=f(x0)`, supposing forward path is:
    ``x0 -> y0``, ``x0 -> x1 -> y0`` .
    ``paddle.grad(y0, x0)`` will reduce gradients along ``y0->x0`` and 
    ``y0->x1->x0``, and ``vjp`` only need reduce along ``y0->x0``.

    So, it's needed to clone or detach xs for breaking the dependencies with 
    other variables.

    Examples:

        .. code-block:: python

            import paddle
            from paddle.autograd.functional import _separate


            def func(x, y):
                return x * y


            x = paddle.ones((1,))
            x.stop_gradient = False

            y = func(x, x)
            print(paddle.grad(y, x))
            # [Tensor(shape=[1], dtype=float32, place=Place(gpu:0), stop_gradient=True,
            #        [2.])]

            x1, x2 = _separate((x, x))
            y = func(x1, x2)
            print(paddle.grad(y, x1))
            # [Tensor(shape=[1], dtype=float32, place=Place(gpu:0), stop_gradient=True,
            #        [1.])]

    """
    if isinstance(xs, typing.Sequence):
        return tuple(_single_separate(x) for x in xs)
    else:
        return _single_separate(xs)


def _single_separate(x):
    if x is None:  # x maybe none because grad input's v defaults to none.
        return x
    if not x.stop_gradient:
        return paddle.clone(x)
    else:  # use detach to share memory when no need gradients.
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
            f"Expected 'v' is Tensor|Sequence[Tensor]|None, but got {type(v)}.")

    if isinstance(v, typing.Sequence) and not all(
            isinstance(e, framework.Variable) for e in v):
        raise TypeError("All elements of 'xs' shoule be Tensor.")


def _check_v_shape(v, refs):
    if v is None:
        return

    v, refs = _as_tensors(v), _as_tensors(refs)
    if len(refs) != len(v):
        raise RuntimeError(f"The argument v is a tuple of invalid length:"
                           f"should be {len(refs)} but got {len(v)}.")

    for index, (element_v, element_ref) in enumerate(zip(v, refs)):
        if element_v.shape != element_ref.shape:
            raise RuntimeError(
                f"The v[{index}] has invalid shape: should "
                f"be {element_ref.shape} but got {element_v.shape}.")


@framework.dygraph_only
def jacobian(func, inputs, create_graph=False, allow_unused=False):
    ''' 
    .. note::
        **This API is ONLY available in the imperative mode.**

    This function computes the Jacobian matrix of `func` with respect to `inputs`.

    Parameters:
        func (function): a Python function that takes a Tensor or a Tensor
            list/tuple as inputs and returns a Tensor or a Tensor tuple.
        inputs (Tensor|list(Tensor)|tuple(Tensor)): the input Tensor or 
            Tensor list/tuple of the function ``func``.
        create_graph (bool, optional): whether to create the gradient graphs
            of the computing process. When it is True, higher order derivatives
            are supported to compute; when it is False, the gradient graphs of
            the computing process would be discarded. Defaults to ``False``.
        allow_unused (bool, optional): whether to raise error or return None if
            some Tensors of `inputs` are unreachable in the graph. Error would
            be raised if allow_unused=False, and None would be returned as
            their gradients if allow_unused=True. Default False.
    Returns:
        Jacobian (Tensor or nested tuple of Tensors): if function ``func``
        takes a Tensor as inputs and returns a Tensor as outputs, Jacobian
        will be a single Tensor containing the Jacobian matrix for the
        linearized inputs and outputs. If one of the inputs and outputs is
        a Tensor, and another is a Tensor list/tuple, then the Jacobian will
        be a tuple of Tensors. If both of inputs and outputs are Tensor
        list/tuple, then the Jacobian will be a tuple of tuple of Tensors
        where ``Jacobian[i][j]`` will contain the Jacobian matrix of the
        linearized ``i``th output and ``j``th input and will have same
        dtype and device as the corresponding input. ``Jacobian[i][j]`` will
        have as size ``m * n``, where ``m`` and ``n`` denote the numbers of
        elements of ``i``th output and ``j``th input respectively.


    Examples 1:
        .. code-block:: python

            import paddle

            def func(x):
                return paddle.matmul(x, x)

            x = paddle.ones(shape=[2, 2], dtype='float32')
            x.stop_gradient = False
            jacobian = paddle.autograd.jacobian(func, x)
            print(jacobian)
            # Tensor(shape=[4, 4], dtype=float32, place=CUDAPlace(0), stop_gradient=True,
            #        [[2., 1., 1., 0.],
            #         [1., 2., 0., 1.],
            #         [1., 0., 2., 1.],
            #         [0., 1., 1., 2.]])

    Examples 2:
        .. code-block:: python

            import paddle

            def func(x, y):
                return paddle.matmul(x, y)

            x = paddle.ones(shape=[2, 2], dtype='float32')
            y = paddle.ones(shape=[2, 2], dtype='float32') * 2
            x.stop_gradient = False
            y.stop_gradient = False
            jacobian = paddle.autograd.jacobian(func, [x, y], create_graph=True)
            print(jacobian)
            # (Tensor(shape=[4, 4], dtype=float32, place=CUDAPlace(0), stop_gradient=False,
            #        [[2., 2., 0., 0.],
            #         [2., 2., 0., 0.],
            #         [0., 0., 2., 2.],
            #         [0., 0., 2., 2.]]), 
            #  Tensor(shape=[4, 4], dtype=float32, place=CUDAPlace(0), stop_gradient=False,
            #        [[1., 0., 1., 0.],
            #         [0., 1., 0., 1.],
            #         [1., 0., 1., 0.],
            #         [0., 1., 0., 1.]]))

    Examples 3:
        .. code-block:: python

            import paddle

            def func(x, y):
                return paddle.matmul(x, y), x * x

            x = paddle.ones(shape=[2, 2], dtype='float32')
            y = paddle.ones(shape=[2, 2], dtype='float32') * 2
            x.stop_gradient = False
            y.stop_gradient = False
            jacobian = paddle.autograd.jacobian(func, [x, y], allow_unused=True)
            print(jacobian)
            # ((Tensor(shape=[4, 4], dtype=float32, place=CUDAPlace(0), stop_gradient=True,
            #        [[2., 2., 0., 0.],
            #         [2., 2., 0., 0.],
            #         [0., 0., 2., 2.],
            #         [0., 0., 2., 2.]]),
            #   Tensor(shape=[4, 4], dtype=float32, place=CUDAPlace(0), stop_gradient=True,
            #        [[1., 0., 1., 0.],
            #         [0., 1., 0., 1.],
            #         [1., 0., 1., 0.],
            #         [0., 1., 0., 1.]])),
            #  (Tensor(shape=[4, 4], dtype=float32, place=CUDAPlace(0), stop_gradient=True,
            #        [[2., 0., 0., 0.],
            #         [0., 2., 0., 0.],
            #         [0., 0., 2., 0.],
            #         [0., 0., 0., 2.]]), None))

    '''
    inputs = _as_tensors(inputs)
    outputs = _as_tensors(func(*inputs))
    fin_size = len(inputs)
    fout_size = len(outputs)
    flat_outputs = tuple(
        paddle.reshape(
            output, shape=[-1]) for output in outputs)
    jacobian = tuple()
    for i, flat_output in enumerate(flat_outputs):
        jac_i = list([] for _ in range(fin_size))
        for k in range(len(flat_output)):
            row_k = paddle.grad(
                flat_output[k],
                inputs,
                create_graph=create_graph,
                retain_graph=True,
                allow_unused=allow_unused)
            for j in range(fin_size):
                jac_i[j].append(
                    paddle.reshape(
                        row_k[j], shape=[-1])
                    if isinstance(row_k[j], paddle.Tensor) else None)
        jacobian += (tuple(
            _stack_tensor_or_return_none(jac_i_j) for jac_i_j in jac_i), )
    if fin_size == 1 and fout_size == 1:
        return jacobian[0][0]
    elif fin_size == 1 and fout_size != 1:
        return tuple(jacobian[i][0] for i in range(fout_size))
    elif fin_size != 1 and fout_size == 1:
        return jacobian[0]
    else:
        return jacobian


@framework.dygraph_only
def batch_jacobian(func, inputs, create_graph=False, allow_unused=False):
    ''' 
    .. note::
        **This API is ONLY available in the imperative mode.**

    This function computes the batch Jacobian matrix of `func` with respect to `inputs`.
    Noted that the first dimension of inputs is batch size.

    Parameters:
        func (function): a Python function that takes a Tensor or a Tensor
            list/tuple as inputs(the first dimension is batch size) and 
            returns a Tensor or a Tensor tuple.
        inputs (Tensor|list(Tensor)|tuple(Tensor)): the input Tensor or 
            Tensor list/tuple of the function ``func``, Noted that
            the first dimension of inputs is batch size.
        create_graph (bool, optional): whether to create the gradient graphs
            of the computing process. When it is True, higher order derivatives
            are supported to compute; when it is False, the gradient graphs of
            the computing process would be discarded. Defaults to ``False``.
        allow_unused (bool, optional): whether to raise error or return None if
            some Tensors of `inputs` are unreachable in the graph. Error would
            be raised if allow_unused=False, and None would be returned as
            their gradients if allow_unused=True. Default False.
    Returns:
        Jacobian (Tensor or nested tuple of Tensors): if function ``func``
        takes a Tensor as inputs and returns a Tensor as outputs, Jacobian
        will be a single Tensor containing the Jacobian matrix for the
        linearized inputs and outputs. If one of the inputs and outputs is
        a Tensor, and another is a Tensor list/tuple, then the Jacobian will
        be a tuple of Tensors. If both of inputs and outputs are Tensor
        list/tuple, then the Jacobian will be a tuple of tuple of Tensors.
        Noted that the first dimension of inputs is batch size.

        For example,
        the inputs shape and outputs shape of function ``func` is [batch_size, num] 
        and [batch_size, num] respectively, then the Jacobian will be a Tensor with
        a shape of [num, batch_size * num], where ``Jacobian[i][j]`` will contain 
        the Jacobian matrix of the ``i``th column output and the ``j``th input and 
        will have same dtype and device as the corresponding input.
        Other situations can be deduced by analogy.

    Examples 1:
        .. code-block:: python

            import paddle

            x = paddle.ones(shape=(4, 2), dtype='float64')
            weight = paddle.ones(shape=(2, 4), dtype='float64')
            y = paddle.ones(shape=(4, 2), dtype='float64')

            def func(x):
                return paddle.matmul(paddle.matmul(x, weight), y)

            x.stop_gradient = False
            batch_jacobian = paddle.autograd.batch_jacobian(func, x)
            print(batch_jacobian)
            # Tensor(shape=[2, 8], dtype=float64, place=CUDAPlace(0), stop_gradient=True,
            #      [[4., 4., 4., 4., 4., 4., 4., 4.],
            #       [4., 4., 4., 4., 4., 4., 4., 4.]])

    Examples 2:
        .. code-block:: python

            import paddle

            x = paddle.ones(shape=(4, 2), dtype='float64')
            weight = paddle.ones(shape=(2, 4), dtype='float64')
            y = paddle.ones(shape=(4, 2), dtype='float64')

            def func(x):
                return paddle.matmul(paddle.matmul(x, weight), y), x * x

            x.stop_gradient = False
            batch_jacobian = paddle.autograd.batch_jacobian(func, x) 
            print(batch_jacobian)    
            # (Tensor(shape=[2, 8], dtype=float64, place=CUDAPlace(0), stop_gradient=True,
            #       [[4., 4., 4., 4., 4., 4., 4., 4.],
            #        [4., 4., 4., 4., 4., 4., 4., 4.]]), Tensor(shape=[2, 8], dtype=float64, place=CUDAPlace(0), stop_gradient=True,
            #       [[2., 0., 2., 0., 2., 0., 2., 0.],
            #        [0., 2., 0., 2., 0., 2., 0., 2.]]))

    Examples 3:
        .. code-block:: python

            import paddle

            x = paddle.ones(shape=(4, 2), dtype='float64')
            weight = paddle.ones(shape=(2, 4), dtype='float64')
            y = paddle.ones(shape=(4, 2), dtype='float64')

            def func(x, y):
                return x * y

            x.stop_gradient = False
            y.stop_gradient = False
            batch_jacobian = paddle.autograd.batch_jacobian(func, [x, y])
            print(batch_jacobian)
            # (Tensor(shape=[2, 8], dtype=float64, place=CUDAPlace(0), stop_gradient=True,
            #       [[1., 0., 1., 0., 1., 0., 1., 0.],
            #        [0., 1., 0., 1., 0., 1., 0., 1.]]), Tensor(shape=[2, 8], dtype=float64, place=CUDAPlace(0), stop_gradient=True,
            #       [[1., 0., 1., 0., 1., 0., 1., 0.],
            #        [0., 1., 0., 1., 0., 1., 0., 1.]]))

    '''

    inputs = _as_tensors(inputs)
    outputs = _as_tensors(func(*inputs))

    batch_size = inputs[0].shape[0]
    for input in inputs:
        assert input.shape[
            0] == batch_size, "The first dimension of input should equals to the same batch size!"
    for output in outputs:
        assert output.shape[
            0] == batch_size, "The first dimension of output should equals to the same batch size!"
    fin_size = len(inputs)
    fout_size = len(outputs)
    flat_outputs = tuple(
        paddle.reshape(
            output, shape=[batch_size, -1]) for output in outputs)
    jacobian = tuple()
    for i, flat_output in enumerate(flat_outputs):
        jac_i = list([] for _ in range(fin_size))
        for k in range(flat_output.shape[1]):

            row_k = paddle.grad(
                flat_output[:, k],
                inputs,
                create_graph=create_graph,
                retain_graph=True,
                allow_unused=allow_unused)

            for j in range(fin_size):
                jac_i[j].append(
                    paddle.reshape(
                        row_k[j], shape=[-1])
                    if isinstance(row_k[j], paddle.Tensor) else None)
        jacobian += (tuple(
            _stack_tensor_or_return_none(jac_i_j) for jac_i_j in jac_i), )
    if fin_size == 1 and fout_size == 1:
        return jacobian[0][0]
    elif fin_size == 1 and fout_size != 1:
        return tuple(jacobian[i][0] for i in range(fout_size))
    elif fin_size != 1 and fout_size == 1:
        return jacobian[0]
    else:
        return jacobian


@framework.dygraph_only
def batch_hessian(func, inputs, create_graph=False, allow_unused=False):
    ''' 
    .. note::
        **This API is ONLY available in the imperative mode.**

    This function computes the batch Hessian matrix of `func` with respect to `inputs`.
    Noted that the first dimension of inputs is batch size.

    Parameters:
        func (function): a Python function that takes a Tensor or a Tensor
            list/tuple as inputs(the first dimension is batch size) and
            returns a Tensor with shape [batch_size, 1].
        inputs (Tensor|list(Tensor)|tuple(Tensor)): the input Tensor or 
            Tensor list/tuple of the function ``func``.
            Noted that the first dimension of inputs is batch size.
        create_graph (bool, optional): whether to create the gradient graphs
            of the computing process. When it is True, higher order derivatives
            are supported to compute; when it is False, the gradient graphs of
            the computing process would be discarded. Defaults to ``False``.
        allow_unused (bool, optional): whether to raise error or return None if
            some Tensors of `inputs` are unreachable in the graph. Error would
            be raised if allow_unused=False, and None would be returned as
            their gradients if allow_unused=True. Default False.
    Returns:
        Hessian (Tensor or a tuple of tuple of Tensors): if function ``func``
        takes a Tensor as ``inputs``, Hessian will be a single Tensor containing
        the Hessian matrix for the linearized ``inputs`` Tensor. If function
        ``func`` takes a Tensor list/tuple as ``inputs``, then the Hessian will
        be a tuple of tuple of Tensors. Noted that the first dimension of inputs 
        is batch size and the execution step is to obtain the result of the 
        first order differentiation, and then differentiate the batch input.

        For example,
        the inputs shape and outputs shape of function ``func` is [batch_size, num] 
        and [batch_size, 1] respectively, then the batched Hessian will be a Tensor with
        a shape of [num, batch_size * num].

        Why the final shape in this case is that?
        because batch_hessian will create a inner func(the wrapper of paddle.grad() func)
        to computes the sum of gradients of `outputs` with respect to each `inputs`,
        this inner func will get the first order differentiation and shape is [batch_size, num], 
        then call batch_jacobian to compute jacobian between the first order differentiation
        and the origin inputs. The final result ``Hessian[i][j]`` will contain the Jacobian 
        matrix of the ``i``th column output(Noted that this output means the first order 
        differentiation) and the ``j``th input and will have same dtype and device as the 
        corresponding input. Other situations can be deduced by analogy.


    Examples 1:
        .. code-block:: python

            import paddle

            x = paddle.ones(shape=(4, 2), dtype='float64')
            weight = paddle.ones(shape=(2, 4), dtype='float64')
            y = paddle.ones(shape=(4, 2), dtype='float64')

            def func(x):
                return paddle.matmul(x * x, weight)[:, 0:1]


            x.stop_gradient = False
            batch_hessian = paddle.autograd.batch_hessian(func, x)
            print(batch_hessian)
            # Tensor(shape=[2, 8], dtype=float64, place=CUDAPlace(0), stop_gradient=True,
            #      [[2., 0., 2., 0., 2., 0., 2., 0.],
            #       [0., 2., 0., 2., 0., 2., 0., 2.]])

    Examples 2:
        .. code-block:: python

            import paddle

            x = paddle.ones(shape=(4, 2), dtype='float64')
            weight = paddle.ones(shape=(2, 4), dtype='float64')
            y = paddle.ones(shape=(4, 2), dtype='float64')

            def func(x, y):
                return paddle.matmul(x * x * y * y, weight)[:, 0:1]

            x.stop_gradient = False
            y.stop_gradient = False
            batch_hessian = paddle.autograd.batch_hessian(func, [x, y])
            print(batch_hessian)
            # ((Tensor(shape=[2, 8], dtype=float64, place=CUDAPlace(0), stop_gradient=True,
            #        [[2., 0., 2., 0., 2., 0., 2., 0.],
            #         [0., 2., 0., 2., 0., 2., 0., 2.]]), 
            #   Tensor(shape=[2, 8], dtype=float64, place=CUDAPlace(0), stop_gradient=True,
            #        [[4., 0., 4., 0., 4., 0., 4., 0.],
            #         [0., 4., 0., 4., 0., 4., 0., 4.]])), 
            #  (Tensor(shape=[2, 8], dtype=float64, place=CUDAPlace(0), stop_gradient=True,
            #        [[4., 0., 4., 0., 4., 0., 4., 0.],
            #         [0., 4., 0., 4., 0., 4., 0., 4.]]), 
            #   Tensor(shape=[2, 8], dtype=float64, place=CUDAPlace(0), stop_gradient=True,
            #        [[2., 0., 2., 0., 2., 0., 2., 0.],
            #         [0., 2., 0., 2., 0., 2., 0., 2.]])))


    Examples 3:
        .. code-block:: python

            import paddle

            x = paddle.ones(shape=(4, 2), dtype='float64')
            weight = paddle.ones(shape=(2, 4), dtype='float64')
            y = paddle.ones(shape=(4, 2), dtype='float64')

            def func(x, y):
                return paddle.matmul(x * x, weight)[:, 0:1]

            x.stop_gradient = False
            y.stop_gradient = False
            batch_hessian = paddle.autograd.batch_hessian(func, [x, y], allow_unused=True)
            print(batch_hessian)
            # ((Tensor(shape=[2, 8], dtype=float64, place=CUDAPlace(0), stop_gradient=True,
            #        [[2., 0., 2., 0., 2., 0., 2., 0.],
            #         [0., 2., 0., 2., 0., 2., 0., 2.]]), None), (None, None))

    '''
    inputs = _as_tensors(inputs)
    outputs = func(*inputs)
    batch_size = inputs[0].shape[0]
    for input in inputs:
        assert input.shape[
            0] == batch_size, "The first dimension of input should equals to the same batch size!"
    assert isinstance(outputs, paddle.Tensor) and outputs.shape == [
        batch_size, 1
    ], "The function to compute batched Hessian matrix should return a Tensor of shape [batch_size, 1]"

    def jac_func(*ins):
        grad_inputs = paddle.grad(
            outputs,
            ins,
            create_graph=True,
            retain_graph=True,
            allow_unused=allow_unused)
        return tuple(
            _replace_none_with_zero_tensor(grad_inputs[i], inputs[i])
            for i in range(len(inputs)))

    return batch_jacobian(
        jac_func, inputs, create_graph=create_graph, allow_unused=allow_unused)


@framework.dygraph_only
def hessian(func, inputs, create_graph=False, allow_unused=False):
    ''' 
    .. note::
        **This API is ONLY available in the imperative mode.**

    This function computes the Hessian matrix of `func` with respect to `inputs`.

    Parameters:
        func (function): a Python function that takes a Tensor or a Tensor
            list/tuple as inputs and returns a Tensor with a single element.
        inputs (Tensor|list(Tensor)|tuple(Tensor)): the input Tensor or 
            Tensor list/tuple of the function ``func``.
        create_graph (bool, optional): whether to create the gradient graphs
            of the computing process. When it is True, higher order derivatives
            are supported to compute; when it is False, the gradient graphs of
            the computing process would be discarded. Defaults to ``False``.
        allow_unused (bool, optional): whether to raise error or return None if
            some Tensors of `inputs` are unreachable in the graph. Error would
            be raised if allow_unused=False, and None would be returned as
            their gradients if allow_unused=True. Default False.
    Returns:
        Hessian (Tensor or a tuple of tuple of Tensors): if function ``func``
        takes a Tensor as ``inputs``, Hessian will be a single Tensor containing
        the Hessian matrix for the linearized ``inputs`` Tensor. If function
        ``func`` takes a Tensor list/tuple as ``inputs``, then the Hessian will
        be a tuple of tuple of Tensors where ``Hessian[i][j]`` will contain the
        Hessian matrix of the ``i``th input and ``j``th input with size ``m * n``.
        Here ``m`` and ``n`` denote the number of elements of the ``i`` th input
        and the ``j`` th input respectively.

    Examples 1:
        .. code-block:: python

            import paddle

            def func(x):
                return paddle.sum(paddle.matmul(x, x))

            x = paddle.ones(shape=[2, 2], dtype='float32')
            x.stop_gradient = False
            hessian = paddle.autograd.hessian(func, x)
            print(hessian)
            # Tensor(shape=[4, 4], dtype=float32, place=CUDAPlace(0), stop_gradient=True,
            #        [[2., 1., 1., 0.],
            #         [1., 0., 2., 1.],
            #         [1., 2., 0., 1.],
            #         [0., 1., 1., 2.]])

    Examples 2:
        .. code-block:: python

            import paddle

            def func(x, y):
                return paddle.sum(paddle.matmul(x, y))

            x = paddle.ones(shape=[2, 2], dtype='float32')
            y = paddle.ones(shape=[2, 2], dtype='float32')
            x.stop_gradient = False
            y.stop_gradient = False
            hessian = paddle.autograd.hessian(func, [x, y])
            print(hessian)
            # ((Tensor(shape=[4, 4], dtype=float32, place=CUDAPlace(0), stop_gradient=True,
            #        [[0., 0., 0., 0.],
            #         [0., 0., 0., 0.],
            #         [0., 0., 0., 0.],
            #         [0., 0., 0., 0.]]),
            #   Tensor(shape=[4, 4], dtype=float32, place=CUDAPlace(0), stop_gradient=True,
            #        [[1., 1., 0., 0.],
            #         [0., 0., 1., 1.],
            #         [1., 1., 0., 0.],
            #         [0., 0., 1., 1.]])),
            #  (Tensor(shape=[4, 4], dtype=float32, place=CUDAPlace(0), stop_gradient=True,
            #        [[1., 0., 1., 0.],
            #         [1., 0., 1., 0.],
            #         [0., 1., 0., 1.],
            #         [0., 1., 0., 1.]]),
            #   Tensor(shape=[4, 4], dtype=float32, place=CUDAPlace(0), stop_gradient=True,
            #        [[0., 0., 0., 0.],
            #         [0., 0., 0., 0.],
            #         [0., 0., 0., 0.],
            #         [0., 0., 0., 0.]])))

    Examples 3:
        .. code-block:: python

            import paddle

            def func(x, y):
                return paddle.sum(paddle.matmul(x, x))

            x = paddle.ones(shape=[2, 2], dtype='float32')
            y = paddle.ones(shape=[2, 2], dtype='float32')
            x.stop_gradient = False
            y.stop_gradient = False
            hessian = paddle.autograd.hessian(func, [x, y], allow_unused=True)
            print(hessian)
            # ((Tensor(shape=[4, 4], dtype=float32, place=CUDAPlace(0), stop_gradient=True,
            #        [[2., 1., 1., 0.],
            #         [1., 0., 2., 1.],
            #         [1., 2., 0., 1.],
            #         [0., 1., 1., 2.]]), None), (None, None))

    '''
    inputs = _as_tensors(inputs)
    outputs = func(*inputs)
    assert isinstance(outputs, paddle.Tensor) and outputs.shape == [
        1
    ], "The function to compute Hessian matrix should return a Tensor with a single element"

    def jac_func(*ins):
        grad_inputs = paddle.grad(
            outputs,
            ins,
            create_graph=True,
            retain_graph=True,
            allow_unused=allow_unused)
        return tuple(
            _replace_none_with_zero_tensor(grad_inputs[i], inputs[i])
            for i in range(len(inputs)))

    return jacobian(
        jac_func, inputs, create_graph=create_graph, allow_unused=allow_unused)


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
            vhp_rslt = paddle.autograd.vhp(func, [x, y], v=[vx, vy], allow_unused=True)
            print(vhp_rslt)
            # (Tensor(shape=[1], dtype=float32, place=CUDAPlace(0), stop_gradient=False,
            #        [8.]),
            # [Tensor(shape=[2, 2], dtype=float32, place=CUDAPlace(0), stop_gradient=True,
            #        [[8., 8.],
            #         [8., 8.]]), None])
    '''
    xs = _as_tensors(inputs)
    if v is not None:
        v = _as_tensors(v)
    xs, v = _separate(xs), _separate(v)
    outputs = func(*xs)
    ys = _as_tensors(outputs)
    assert len(ys) == 1 and isinstance(
        ys[0], framework.Variable
    ) and ys[0].shape == [
        1
    ], "The function to compute vhp should return a Tensor with a single element"
    jac = _grad(ys, xs)
    vhp = _grad(jac, xs, v)
    return outputs, vhp
