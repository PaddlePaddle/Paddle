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

import typing

import paddle
from paddle.fluid import framework
from paddle.incubate.autograd import primapi, utils


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
    if paddle.fluid._non_static_mode() or not utils.prim_enabled():
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
    if paddle.fluid._non_static_mode() or not utils.prim_enabled():
        xs, v = _separate(xs), _separate(v)
    ys = func(*xs) if isinstance(xs, typing.Sequence) else func(xs)
    _check_v_shape(v, xs)

    if not paddle.fluid._non_static_mode() and utils.prim_enabled():
        return ys, primapi.forward_grad(ys, xs, v)
    else:
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


class Jacobian:
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
        """The shape of flattened Jacobian matrix."""
        return self._jacobian.shape


class Hessian:
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
            if (is_batched and jac.shape[1] != 1) or (
                not is_batched and jac.shape[0] != 1
            ):
                raise RuntimeError(
                    "The function given to Hessian shoud return as single element Tensor or batched single element Tensor."
                )
            return jac[:, 0, :] if is_batched else jac[0, :]

        self.symbolic = Jacobian(_jac_func, xs, is_batched=is_batched)

    def __getitem__(self, indexes):
        return self.symbolic[indexes]

    @property
    def shape(self):
        """The shape of flattened Hessian matrix."""
        return self.symbolic.shape


class _Jacobian:
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
        # Skip separating in prim mode temporarily, as detach and clone are not
        # primitive operators.
        if not paddle.fluid._non_static_mode() and utils.prim_enabled():
            self._xs = xs
        else:
            self._xs = _separate(xs)
        self._ys = func(*utils.as_tensors(self._xs))
        self._flatten_xs = self._flatten(utils.as_tensors(self._xs))
        self._flatten_ys = self._flatten(utils.as_tensors(self._ys))
        self._cache = {}

    @property
    def shape(self):
        raise NotImplementedError

    @property
    def _lazy_axis(self):
        """ "The axis of lazily evaluated."""
        raise NotImplementedError

    def _lazy_indexes(self, indexes):
        idx = indexes[self._lazy_axis]
        return (
            (idx,)
            if isinstance(idx, int)
            else tuple(range(idx.start, idx.stop, idx.step))
        )

    def _flatten(self, xs):
        raise NotImplementedError

    def _shifted_indexes(self, indexes, lazy_axis_size=0):
        idx = indexes[self._lazy_axis]
        shifted_lazy_axis_idx = (
            0 if isinstance(idx, int) else slice(0, lazy_axis_size, 1)
        )
        return (
            indexes[: self._lazy_axis]
            + (shifted_lazy_axis_idx,)
            + indexes[self._lazy_axis + 1 :]
        )

    def __getitem__(self, indexes):
        indexes = _multi_index(indexes, self.shape)

        if isinstance(indexes[self._lazy_axis], int):
            other_indexes = (
                indexes[: self._lazy_axis] + indexes[self._lazy_axis + 1 :]
            )
            return self._cached_evaluate(indexes[self._lazy_axis])[
                other_indexes
            ]
        lazy_indexes = self._lazy_indexes(indexes)
        # Using concat and reshape to replace stack operator temporarily, as
        # it is not a primitive operator.
        shape = list(self.shape)
        shape[self._lazy_axis] = len(lazy_indexes)
        part_jac = paddle.concat(
            [self._cached_evaluate(i) for i in lazy_indexes],
            axis=self._lazy_axis,
        ).reshape(shape)
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
        super().__init__(func, xs)

    @property
    def shape(self):
        return (self._flatten_ys.shape[0], self._flatten_xs.shape[0])

    @property
    def _lazy_axis(self):
        return 0

    def _flatten(self, xs):
        return paddle.concat(tuple(x.reshape((-1,)) for x in xs))

    def _evaluate(self, row_index):
        return self._flatten(
            _grad(
                self._flatten_ys[row_index],
                self._xs,
            )
        )


class _JacobianBatchFirst(_Jacobian):
    """Compute Jacobian matrix with batch at first axis.
    Suppose the mapping is :math:`f: R^{B,M} \to R^{B,N}`, the output shape is
    ``(B, N, M)`` .
    """

    def __init__(self, func, xs):
        super().__init__(func, xs)

    @property
    def shape(self):
        return (
            self._flatten_xs.shape[0],
            self._flatten_ys.shape[1],
            self._flatten_xs.shape[1],
        )

    @property
    def _lazy_axis(self):
        return 1

    def _flatten(self, xs):
        return paddle.concat(
            tuple(x.reshape((x.shape[0], -1)) for x in utils.as_tensors(xs)), 1
        )

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
    indexes = indexes if isinstance(indexes, typing.Sequence) else (indexes,)
    if any(isinstance(i, type(Ellipsis)) for i in indexes):
        raise IndexError('Ellipsis index currently is not supported.')
    # Fill the right-most elements.
    indexes = indexes + (slice(0, None, None),) * (len(shape) - len(indexes))
    # Convert to positive index.
    positive_indexes = []
    for i, index in enumerate(indexes):
        if isinstance(index, slice):
            index = slice(
                index.start or 0, index.stop or shape[i], index.step or 1
            )
            positive_indexes.append(
                slice(
                    index.start + shape[i] if index.start < 0 else index.start,
                    index.stop + shape[i] if index.stop < 0 else index.stop,
                    # Negative step means index backward, no need to convert to
                    # positive interger.
                    index.step,
                )
            )
        elif isinstance(index, int):
            positive_indexes.append(index + shape[i] if index < 0 else index)
        else:
            raise TypeError(f'Not supported index type {index}.')
    return tuple(positive_indexes)


def _replace_none_with_zero_tensor(xs, refs):
    if xs is None:
        xs = paddle.zeros_like(refs)
        xs.stop_gradient = refs.stop_gradient
        return xs
    elif isinstance(xs, typing.Sequence):
        return tuple(
            _replace_none_with_zero_tensor(x, refs[i]) for i, x in enumerate(xs)
        )
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
        # paddle.grad returns a list though the inputs is a signle Tensor. The
        # follow code snippet fixes the problem by return the first element of
        # xs_grad when the xs is a signle Tensor.
        xs_grad = paddle.grad(ys, xs, v, create_graph=True, allow_unused=True)
        if (
            isinstance(xs, paddle.fluid.framework.Variable)
            and isinstance(xs_grad, typing.Sequence)
            and len(xs_grad) > 0
        ):
            xs_grad = xs_grad[0]
    else:
        xs_grad = paddle.incubate.autograd.grad(ys, xs, v)
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
        return paddle.assign(x)
    else:  # use detach to share memory when no need gradients.
        x = x.detach()
        x.stop_gradient = False
        return x
    return x


def _check_inputs(func, xs, v=None):
    if not callable(func):
        raise TypeError(f"Expected 'fun' is Callable, but got {type(func)}.")

    if not isinstance(xs, (framework.Variable, typing.Sequence)):
        raise TypeError(
            f"Expected 'xs' is a Tensor|Sequence[Tensor],"
            f"but got {type(xs)}."
        )
    if isinstance(xs, typing.Sequence) and not all(
        isinstance(x, framework.Variable) for x in xs
    ):
        raise TypeError("All elements of 'xs' shoule be Tensor.")

    if not isinstance(v, (framework.Variable, typing.Sequence, type(None))):
        raise TypeError(
            f"Expected 'v' is Tensor|Sequence[Tensor]|None, but got {type(v)}."
        )

    if isinstance(v, typing.Sequence) and not all(
        isinstance(e, framework.Variable) for e in v
    ):
        raise TypeError("All elements of 'xs' shoule be Tensor.")


def _check_v_shape(v, refs):
    if v is None:
        return

    v, refs = utils.as_tensors(v), utils.as_tensors(refs)
    if len(refs) != len(v):
        raise RuntimeError(
            f"The argument v is a tuple of invalid length:"
            f"should be {len(refs)} but got {len(v)}."
        )

    for index, (element_v, element_ref) in enumerate(zip(v, refs)):
        if element_v.shape != element_ref.shape:
            raise RuntimeError(
                f"The v[{index}] has invalid shape: should "
                f"be {element_ref.shape} but got {element_v.shape}."
            )
