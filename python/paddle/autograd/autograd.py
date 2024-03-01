# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

from typing import Optional, Sequence, Tuple, Union

import paddle
from paddle.base import framework


def as_tensors(xs):
    if isinstance(xs, framework.Variable):
        return xs
    elif isinstance(xs, Sequence):
        return tuple(xs)
    else:
        return xs


class Jacobian:
    r"""Computes the Jacobian matrix of given xs and ys.

    Once the Jacobian ``J`` is constructed, you can use a multidimensional index
    to retrieve the submatrix of ``J``, as same as slicing a Tensor. The
    submatrix is lazily evaluated along row axis, and will be cached once
    evaluated.

    you can retrieve the submatrix by
    following methods:

        * J[:], retrieving the full matrix.
        * J[:, :, j], retrieving the partial derivatives w.r.t. the j'th input
          variable.
        * J[:, i, :], retrieving the partial derivatives w.r.t. the i'th output
          variable.
        * J[:, i, j], retrieving the partial derivatives w.r.t. the i'th output
          variable and the j'th input variable.

    Notes:

        Ellipsis index is not supported currently.

    Args:

        ys (Tensor|Tuple[Tensor, ...]): The output derived from xs .
        xs (Tensor|Tuple[Tensor, ...]): The input tensor(s) .
        is_batched (bool): If true, the first axis is batch axis. Defaults to
            False.

    Returns:

        Jacobian (Object): A python object retains the Jacobian matrix.

    """

    def __init__(self, ys, xs, is_batched=False):
        if not is_batched:
            if not 0 <= len(xs.shape) <= 1:
                raise ValueError(
                    f"xs.ndim should be 0 or 1 when is_batched=False"
                    f" but got {len(xs.shape)}"
                )
            if not 0 <= len(ys.shape) <= 1:
                raise ValueError(
                    f"ys.ndim should be 0 or 1 when is_batched=False"
                    f" but got {len(ys.shape)}"
                )
            self._jacobian = _JacobianNoBatch(ys, xs)
        else:
            if not 1 <= len(ys.shape) <= 2:
                raise ValueError(
                    f"ys.ndim should be 1 or 2 when is_batched=True"
                    f" but got {len(ys.shape)}"
                )
            if not 1 <= len(xs.shape) <= 2:
                raise ValueError(
                    f"xs.ndim should be 1 or 2 when is_batched=True"
                    f" but got {len(xs.shape)}"
                )
            self._jacobian = _JacobianBatchFirst(ys, xs)

    @property
    def shape(self):
        """The shape of flattened Jacobian matrix."""
        return self._jacobian.shape

    def __getitem__(self, indexes):
        return self._jacobian[indexes]

    def __getattr__(self, __name: str):
        if __name == "shape":
            return getattr(self._jacobian, __name)
        if __name == "_evaluate_all":
            return getattr(self._jacobian, __name)
        return getattr(self._jacobian._evaluate_all(), __name)

    def __add__(self, other):
        lhs = self._evaluate_all()
        rhs = other._evaluate_all() if isinstance(other, Jacobian) else other
        return lhs + rhs

    def __sub__(self, other):
        lhs = self._evaluate_all()
        rhs = other._evaluate_all() if isinstance(other, Jacobian) else other
        return lhs - rhs

    def __mul__(self, other):
        lhs = self._evaluate_all()
        rhs = other._evaluate_all() if isinstance(other, Jacobian) else other
        return lhs * rhs

    def __div__(self, other):
        lhs = self._evaluate_all()
        rhs = other._evaluate_all() if isinstance(other, Jacobian) else other
        return lhs / rhs

    def __truediv__(self, other):
        lhs = self._evaluate_all()
        rhs = other._evaluate_all() if isinstance(other, Jacobian) else other
        return lhs / rhs

    def __pow__(self, other):
        lhs = self._evaluate_all()
        rhs = other._evaluate_all() if isinstance(other, Jacobian) else other
        return lhs**rhs

    def __mod__(self, other):
        lhs = self._evaluate_all()
        rhs = other._evaluate_all() if isinstance(other, Jacobian) else other
        return lhs % rhs

    def __floordiv__(self, other):
        lhs = self._evaluate_all()
        rhs = other._evaluate_all() if isinstance(other, Jacobian) else other
        return lhs // rhs

    def __matmul__(self, other):
        lhs = self._evaluate_all()
        rhs = other._evaluate_all() if isinstance(other, Jacobian) else other
        return lhs @ rhs

    def __eq__(self, other):
        lhs = self._evaluate_all()
        rhs = other._evaluate_all() if isinstance(other, Jacobian) else other
        return lhs == rhs

    def __ne__(self, other):
        lhs = self._evaluate_all()
        rhs = other._evaluate_all() if isinstance(other, Jacobian) else other
        return lhs != rhs

    def __lt__(self, other):
        lhs = self._evaluate_all()
        rhs = other._evaluate_all() if isinstance(other, Jacobian) else other
        return lhs < rhs

    def __le__(self, other):
        lhs = self._evaluate_all()
        rhs = other._evaluate_all() if isinstance(other, Jacobian) else other
        return lhs <= rhs

    def __gt__(self, other):
        lhs = self._evaluate_all()
        rhs = other._evaluate_all() if isinstance(other, Jacobian) else other
        return lhs > rhs

    def __ge__(self, other):
        lhs = self._evaluate_all()
        rhs = other._evaluate_all() if isinstance(other, Jacobian) else other
        return lhs >= rhs


class Hessian(Jacobian):
    pass


class _Jacobian:
    """The base class for computing Jacobian matrix.

    ``_Jacobian`` implements the core logic of multidimensional index and lazy
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

    def __init__(self, ys, xs):
        self.original_xs_shape = xs.shape
        self.original_ys_shape = ys.shape
        self._xs = xs
        self._ys = ys
        if len(self._ys.shape) == 0 and not self.is_batched:
            self._ys = self._ys.reshape(
                [
                    -1,
                ]
            )
        if len(self._ys.shape) == 1 and self.is_batched:
            self._ys = self._ys.reshape([-1, 1])

        self._flatten_xs = self._flatten(as_tensors(self._xs))
        self._flatten_ys = self._flatten(as_tensors(self._ys))
        self._cache = {}

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
        if self.is_batched is False:
            if len(self.shape) == 0:
                # xs and ys are both 0-D tensor
                raise IndexError("0-D tensor can not be indexed.")
            elif len(self.shape) == 1:
                # either ys or xs is 0-D tensor
                indexes = (
                    (0, indexes)
                    if len(self.original_ys_shape) == 0
                    else (indexes, 0)
                )
        else:
            if len(self.shape) == 1:
                # xs and ys are both 1-D tensor
                indexes = (indexes, 0, 0)
            elif len(self.shape) == 2:
                # either xs or ys is 1-D tensor
                if isinstance(indexes, slice):
                    indexes = (indexes, slice(None, None, None))
                else:
                    indexes = (
                        (indexes[0], 0, indexes[1])
                        if len(self.original_ys_shape) == 1
                        else (indexes[0], indexes[1], 0)
                    )

        indexes = _multi_index(indexes, self.inner_shape)

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
        shape = list(self.inner_shape)
        shape[self._lazy_axis] = len(lazy_indexes)
        part_jac = paddle.concat(
            [self._cached_evaluate(i) for i in lazy_indexes],
            axis=self._lazy_axis,
        ).reshape(shape)
        result = part_jac[self._shifted_indexes(indexes, len(lazy_indexes))]

        # squeeze redundant 1 in shape
        if len(result.shape) > len(self.shape):
            for _ in range(len(result.shape) - len(self.shape)):
                result = result.squeeze(-1)

        return result

    def _cached_evaluate(self, k):
        if k is None:
            return self._cached_evaluate(0).reshape([])
        v = self._cache.get(k)
        if v is None:
            v = self._evaluate(k)
            self._cache[k] = v
        return v

    def _evaluate(self, index):
        """Evaluate one slice at along lazy axis."""
        raise NotImplementedError

    def _evaluate_all(self):
        if len(self.shape) == 0:
            return self._cached_evaluate(None)
        else:
            return self[:]


class _JacobianNoBatch(_Jacobian):
    """Compute Jacobian matrix without batch dimension.
    Suppose the mapping is :math:`f: R^M \to R^N`, the output shape is
    ``(N, M)`` .
    """

    def __init__(self, ys, xs):
        self.is_batched = False
        super().__init__(ys, xs)
        # inner_shape is for convenient, it will regard 0-D tensor as 1-D tensor
        self.inner_shape = [
            *(self._flatten_ys.shape[0:1]),
            *(self._flatten_xs.shape[0:1]),
        ]
        self.shape = [
            *(self.original_ys_shape[0:1]),
            *(self.original_xs_shape[0:1]),
        ]

    @property
    def _lazy_axis(self):
        return 0

    def _flatten(self, xs):
        if not isinstance(xs, Sequence):
            return xs.reshape((-1,))
        return paddle.concat(tuple(x.reshape((-1,)) for x in xs))

    def _evaluate(self, row_index):
        return self._flatten(
            _grad_for_jacobian(
                self._flatten_ys[row_index],
                self._xs,
            )
        )


class _JacobianBatchFirst(_Jacobian):
    """Compute Jacobian matrix with batch at first axis.
    Suppose the mapping is :math:`f: R^{B,M} \to R^{B,N}`, the output shape is
    ``(B, N, M)`` .
    """

    def __init__(self, ys, xs):
        self.is_batched = True
        super().__init__(ys, xs)
        # inner_shape is for convenient, it will regard 0-D tensor as 1-D tensor
        self.inner_shape = [
            *(self._flatten_xs.shape[0:1]),
            *(self._flatten_ys.shape[1:2]),
            *(self._flatten_xs.shape[1:2]),
        ]
        self.shape = [
            *(self._flatten_xs.shape[0:1]),
            *(self.original_ys_shape[1:2]),
            *(self.original_xs_shape[1:2]),
        ]

    @property
    def _lazy_axis(self):
        return 1

    def _flatten(self, xs):
        if not isinstance(xs, Sequence):
            return xs.reshape((xs.shape[0], -1))
        return paddle.concat(
            tuple(x.reshape((x.shape[0], -1)) for x in as_tensors(xs)), 1
        )

    def _evaluate(self, row_index):
        return self._flatten(
            _grad_for_jacobian(self._flatten_ys[:, row_index], self._xs)
        )


def _multi_index(indexes, shape):
    """A tool for parsing N-dimensional index into a standard format.

    Currently supporting following input format:
        * ([positive|negative|slice], ...), the right-most elements can be
            omitted.

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
    indexes = indexes if isinstance(indexes, Sequence) else (indexes,)
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
                    # positive integer.
                    index.step,
                )
            )
        elif isinstance(index, int):
            positive_indexes.append(index + shape[i] if index < 0 else index)
        else:
            raise TypeError(f'Not supported index type {index}.')
    return tuple(positive_indexes)


def jacobian(
    ys: Union[paddle.Tensor, Tuple[paddle.Tensor, ...]],
    xs: Union[paddle.Tensor, Tuple[paddle.Tensor, ...]],
    batch_axis: Optional[int] = None,
) -> Union[Tuple[Tuple[Jacobian, ...], ...], Tuple[Jacobian, ...], Jacobian]:
    r"""
    Computes the Jacobian of the dependent variable ``ys`` versus the independent
    variable ``xs``.

    Where ``ys`` represents the output of ``xs`` after a certain operation, ``ys`` and
    ``xs`` can be Tensor or tuple of Tensors, ``batch_axis`` indicates the position of
    the batch dimension of the parameter data.

    When the input is a tuple Tensors, the returned result is a ``Jacobian`` object with
    the same number of nesting levels as ``xs``, and each Jacobian has the same shape as
    The ``xs`` tuples are identical in one-to-one correspondence.

    - When ``batch_axis=None``, only 0-dimensional Tensor or 1-dimensional Tensor is
      supported, assuming the shape of ``xs`` is ``[N, ]``, the shape of ``ys`` is
      ``[M, ]``, then the output Jacobian matrix shape is ``[M, N]``.

    - When ``batch_axis=0``, only 1-dimensional Tensor or 2-dimensional Tensor is
      supported, assuming the shape of ``xs`` is ``[B, N]``, The shape of ``ys`` is
      ``[B, M]``, then the output Jacobian matrix shape is ``[B, M, N]``.

    After the ``Jacobian`` object is created, the actual calculation process does not
    occur, but the lazy evaluation method is used for calculation. It can be
    multi-dimensional indexed to obtain the entire Jacobian matrix or sub-matrix, and
    the actual calculation will be performed at this time the value is calculated and
    the result is returned. At the same time, in the actual evaluation process, the
    calculated sub-matrix will be cached to avoid duplicate calculations in the
    subsequent indexing process.

    For example, assuming ``Jacobian`` instance ``J`` has shape ``[B, M, N]``, assuming
    ``M > 4`` , then ``J[:, 1:4:1, :]`` means to get the values from row ``1`` to row
    ``3`` of ``J``. In actual calculation, only the rows ``1`` to ``3`` are evaluated,
    and the calculation results of ``1`` to ``3`` will be cached at the granularity of
    the row, and will be used next time. When obtaining one or more rows of results
    above, the already calculated parts will not be recalculated.

    Args:

        ys (Union[paddle.Tensor, Tuple[paddle.Tensor, ...]]): Output or tuple of outputs derived from xs.
        xs (Union[paddle.Tensor, Tuple[paddle.Tensor, ...]]): Input or tuple of inputs.
        batch_axis (Optional[int], optional): Index of batch axis. Defaults to None.

    Returns:

        Union[Tuple[Tuple[Jacobian, ...], ...], Tuple[Jacobian, ...], Jacobian]: Jacobian(s) of ys derived from xs.

    Examples:

        .. code-block:: python

            >>> import paddle

            >>> x1 = paddle.randn([3, ])
            >>> x2 = paddle.randn([3, ])
            >>> x1.stop_gradient = False
            >>> x2.stop_gradient = False

            >>> y = x1 + x2

            >>> J = paddle.autograd.jacobian(y, (x1, x2))
            >>> J_y_x1 = J[0][:] # evaluate result of dy/dx1
            >>> J_y_x2 = J[1][:] # evaluate result of dy/dx2

            >>> print(J_y_x1.shape)
            [3, 3]
            >>> print(J_y_x2.shape)
            [3, 3]
    """

    if batch_axis is not None and batch_axis != 0:
        raise ValueError(
            f"batch_axis should be None or 0, but got {batch_axis}."
        )

    # TODO(HydrogenSulfate): support batch_axis > 0
    is_batched = batch_axis is not None
    if isinstance(ys, Sequence) and isinstance(xs, Sequence):
        _jacobian = tuple(
            tuple(Jacobian(_ys, _xs, is_batched) for _xs in xs) for _ys in ys
        )
    elif isinstance(ys, Sequence) and not isinstance(xs, Sequence):
        _jacobian = tuple(Jacobian(_ys, xs, is_batched) for _ys in ys)
    elif not isinstance(ys, Sequence) and isinstance(xs, Sequence):
        _jacobian = tuple(Jacobian(ys, _xs, is_batched) for _xs in xs)
    else:
        _jacobian = Jacobian(ys, xs, is_batched)

    return _jacobian


def hessian(
    ys: paddle.Tensor,
    xs: Union[paddle.Tensor, Tuple[paddle.Tensor, ...]],
    batch_axis: Optional[int] = None,
) -> Union[Tuple[Tuple[Hessian, ...], ...], Hessian]:
    r"""
    Computes the Jacobian of the dependent variable ``ys`` versus the independent
    variable ``xs``.

    Among them, ``ys`` means the output of ``xs`` after a certain operation, ``ys`` can
    only be a single Tensor, ``xs`` can be a Tensor or a Tensor tuple, and
    ``batch_axis`` means The position of the batch dimension of the parameter data.

    When the input ``xs`` is a Tensor tuple, the returned result is a ``Hessian`` tuple,
    assuming that the internal shape of the ``xs`` tuple is composed of ``([M1, ], [M2, ])``, the shape of the returned
    result consists of ``(([M1, M1], [M1, M2]), ([M2, M1], [M2, M2]))``

    - When ``batch_axis=None``, only 0-dimensional Tensor or 1-dimensional Tensor is
      supported, assuming that the shape of ``xs`` is ``[N, ]``, and the shape of ``ys`` is ``[ ]`` (0-dimensional Tensor), the final output is a single Hessian matrix whose shape is ``[N, N]``.

    - When ``batch_axis=0``, only 1-dimensional Tensor or 2-dimensional Tensor is
      supported, assuming that the shape of ``xs`` is ``[B, N]``, and the shape of ``ys`` is ``[B, ]``, the final output Jacobian matrix shape is ``[B, N, N]``.

    After the ``Hessian`` object is created, the complete calculation process does not
    occur, but a partial lazy evaluation method is used for calculation. It can be
    multi-dimensionally indexed to obtain the entire Hessian matrix or sub-matrix. At
    this time, the actual Evaluates the computation and returns the result. At the same
    time, in the actual evaluation process, the calculated sub-matrix will be cached to
    avoid repeated calculations in the subsequent indexing process.

    Args:

        ys (paddle.Tensor): Output derived from xs which contain one element.
        xs (Union[paddle.Tensor, Tuple[paddle.Tensor, ...]]): Input or tuple of inputs.
        batch_axis (Optional[int], optional): Index of batch axis. Defaults to None.

    Returns:

        Union[Tuple[Tuple[Hessian, ...], ...], Tuple[Hessian, ...], Hessian]: Hessian(s) of ys derived from xs.

    Examples:

        .. code-block:: python

            >>> import paddle

            >>> x1 = paddle.randn([3, ])
            >>> x2 = paddle.randn([4, ])
            >>> x1.stop_gradient = False
            >>> x2.stop_gradient = False

            >>> y = x1.sum() + x2.sum()

            >>> H = paddle.autograd.hessian(y, (x1, x2))
            >>> H_y_x1_x1 = H[0][0][:] # evaluate result of ddy/dx1x1
            >>> H_y_x1_x2 = H[0][1][:] # evaluate result of ddy/dx1x2
            >>> H_y_x2_x1 = H[1][0][:] # evaluate result of ddy/dx2x1
            >>> H_y_x2_x2 = H[1][1][:] # evaluate result of ddy/dx2x2

            >>> print(H_y_x1_x1.shape)
            [3, 3]
            >>> print(H_y_x1_x2.shape)
            [3, 4]
            >>> print(H_y_x2_x1.shape)
            [4, 3]
            >>> print(H_y_x2_x2.shape)
            [4, 4]
    """

    if batch_axis is None:
        if ys.numel() > 1:
            raise ValueError(
                f"Only support ys.numel()({ys.numel()})==1 when batch_axis is None."
            )
        ys = ys.reshape(())
    elif isinstance(batch_axis, int):
        if ys[0].numel() > 1:
            raise ValueError(
                f"Only support ys[0].numel()({ys.numel()})==1 when batch_axis is int"
            )
        # TODO(HydrogenSulfate): support batch_axis > 0
        if batch_axis != 0:
            raise ValueError("Only support batch_axis=0 yet.")
        ys = ys.reshape((-1,))
    else:
        raise ValueError(
            f"batch_axis should be None or int, but got {type(batch_axis)}."
        )

    _jacobian = jacobian(ys, xs, batch_axis)
    if not isinstance(xs, Sequence):
        hessian = jacobian(_jacobian, xs, batch_axis)

        # change classname to Hessian instead of Jacobian.
        hessian.__class__ = Hessian
    else:
        hessian = tuple(jacobian(_j, xs, batch_axis) for _j in _jacobian)

        # change classname to Hessian instead of Jacobian.
        for i in range(len(hessian)):
            for j in range(len(hessian[0])):
                hessian[i][j].__class__ = Hessian

    return hessian


def _replace_none_with_zero_tensor(xs, refs):
    if xs is None:
        xs = paddle.zeros_like(refs)
        xs.stop_gradient = refs.stop_gradient
        return xs
    elif isinstance(xs, Sequence):
        return tuple(
            _replace_none_with_zero_tensor(x, refs[i]) for i, x in enumerate(xs)
        )
    else:
        return xs


def _grad_for_jacobian(ys, xs, v=None):
    """A gradient function that can be used in dynamic graph and static graph.

    The ``grad`` combines ``paddle.grad`` used in dynamic graph and
    ``paddle.static.gradients`` used in static graph, and do following changes:

    * The ``allow_unused`` flag is removed and set defaults to true internally,
        none in outputs will be replaced by zero tensor.
    * The ``create_graph`` flag is removed and set defaults to true internally,
        only makes sense in dynamic graph.
    * When xs is a single Tensor, ``paddle.grad`` returns a list which only
        contains one Tensor. It may confuse users, thus in this case we improve
        to return a single Tensor in _grad_for_jacobian interface.

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
    if paddle.in_dynamic_mode():
        # paddle.grad returns a list though the inputs is a single Tensor. The
        # follow code snippet fixes the problem by return the first element of
        # xs_grad when the xs is a single Tensor.
        xs_grad = paddle.grad(ys, xs, v, create_graph=True, allow_unused=True)
        if (
            isinstance(xs, paddle.base.framework.Variable)
            and isinstance(xs_grad, Sequence)
            and len(xs_grad) > 0
        ):
            xs_grad = xs_grad[0]
    else:
        xs_grad = paddle.static.gradients(ys, xs, v)
        if (
            isinstance(xs, framework.Variable)
            and isinstance(xs_grad, Sequence)
            and len(xs_grad) > 0
        ):
            xs_grad = xs_grad[0]
    return _replace_none_with_zero_tensor(xs_grad, xs)
