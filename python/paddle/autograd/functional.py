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
from ..fluid import framework
from ..fluid import Tensor
from ..fluid.dygraph import grad
from ..nn.initializer import assign
from ..tensor import reshape, zeros_like, to_tensor
from .utils import _check_tensors, _stack_tensor_or_return_none, _replace_none_with_zero_tensor


def to_tensorlist(tl):
    if not isinstance(tl, list):
        if isinstance(tl, tuple):
            tl = list(tl)
        else:
            tl = [tl]
    for t in tl:
        assert isinstance(t, Tensor) or t is None, (
            f'{t} is expected to be paddle.Tensor or None, but found {type(t)}.'
        )
    return tl


@contextlib.contextmanager
def gradient_scope(*var_lists, create_graph=False, allow_unused=False):
    def grad_fn(ys, xs, v, create_graph=create_graph):
        assert len(ys) == len(v), (
            f'`v` is expected to be of the same size as the output. '
            f'Here the output is {ys}, and `v` is {v}.')
        if allow_unused:
            ys = [
                to_tensor(
                    [0.0], stop_gradient=False) if y is None else y for y in ys
            ]
        return grad(
            ys, xs, v, create_graph=create_graph, allow_unused=allow_unused)

    def return_fn(out):
        if isinstance(out, Tensor):
            if not create_graph:
                out = out.detach()
            return out
        if isinstance(out, list):
            return list(return_fn(x) for x in out)
        elif isinstance(out, tuple):
            return tuple(return_fn(x) for x in out)
        else:
            assert out is None
            return out

    def process(vl):
        out = []
        # If v is treated as constant in the outer scope, its gradient is guaranteed
        # not to be taken beyond this scope. Within this scope, however, v's gradient
        # may be computed. We only need to detach v in this case.
        # Otherwise, v's gradient is valid, and is subject to update beyond this scope.
        # In this case we must not confuse the gradient in the outer scope with the
        # inner one's. Moreover, we need to make sure that the result from the inner
        # scope can flow back to the outer scope. This can be satisfied by extending
        # the original variable with a duplication operation v1 = v so that v still
        # maintains the complete lineage.
        for v in vl:
            if v is None:
                out.append(v)
                continue
            if create_graph and not v.stop_gradient:
                v = assign(v)
            else:
                v = v.detach()
                v.stop_gradient = False
            out.append(v)
        return out

    try:
        var_lists = [process(vl) for vl in var_lists]
        bundle = var_lists + [grad_fn, return_fn]
        yield bundle
    finally:
        pass


@framework.dygraph_only
def vjp(func, inputs, v=None, create_graph=False, allow_unused=False):
    r"""Computes the Vector-Jacobian product, a functional form of
    reverse mode automatic differentiation.

    Args:
        func(Callable): `func` takes as input a tensor or a list
            of tensors and returns a tensor or a list of tensors.
        inputs(list[Tensor]|Tensor): used as positional arguments
            to evaluate `func`. `inputs` is accepted as one tensor
            or a list of tensors.
        v(list[Tensor]|Tensor, optional): the cotangent vector
            invovled in the VJP computation. `v` matches the size
            and shape of `func`'s output. Default value is None
            and in this case is equivalent to all ones the same size
            of `func`'s output.
        create_graph(bool, optional): if `True`, gradients can
            be evaluated on the results. If `False`, taking gradients
            on the results is invalid. Default value is False.
        allow_unused(bool, optional): In case that some Tensors of
            `inputs` do not contribute to the computation of the output.
            If `allow_unused` is False, an error will be raised,
            Otherwise, the gradients of the said inputs are returned
            None. Default value is False.

    Returns:
        output(tuple):
            func_out: the output of `func(inputs)`
            vjp(list[Tensor]|Tensor): the pullback results of `v` on `func`

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
    xs, v = to_tensorlist(inputs), to_tensorlist(v)

    with gradient_scope(
            xs, v, create_graph=create_graph,
            allow_unused=allow_unused) as [xs, v, grad_fn, return_fn]:
        outputs = func(*xs)
        ys = to_tensorlist(outputs)
        grads = grad_fn(ys, xs, v)
        outputs, grads = return_fn(outputs), return_fn(grads)

    return outputs, grads


@framework.dygraph_only
def jvp(func, inputs, v=None, create_graph=False, allow_unused=False):
    r"""
    Computes the Jacobian-Vector product for a function at the given
    inputs and a vector in the tangent space induced by the inputs.

    .. note::
        **This API is ONLY available in imperative mode.**

    Args:
        func(Callable): `func` takes as input a tensor or a list
            of tensors and returns a tensor or a list of tensors.
        inputs(list[Tensor]|Tensor): used as positional arguments
            to evaluate `func`. `inputs` is accepted as one tensor
            or a list of tensors.
        v(list[Tensor]|Tensor, optional): the tangent vector
            invovled in the JVP computation. `v` matches the size
            and shape of `inputs`. `v` is Optional if `func` returns
            a single tensor. Default value is None and in this case
            is equivalent to all ones the same size of `inputs`.
        create_graph(bool, optional): if `True`, gradients can
            be evaluated on the results. If `False`, taking gradients
            on the results is invalid. Default value is False.
        allow_unused(bool, optional): In case that some Tensors of
            `inputs` do not contribute to the computation of the output.
            If `allow_unused` is False, an error will be raised,
            Otherwise, the gradients of the said inputs are returned
            None. Default value is False.

    Returns:
        output(tuple):
            func_out: the output of `func(inputs)`
            jvp(list[Tensor]|Tensor): the pullback results of `v` on `func`

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
    xs, v = to_tensorlist(inputs), to_tensorlist(v)

    with gradient_scope(
            xs, v, create_graph=create_graph,
            allow_unused=allow_unused) as [xs, v, grad_fn, return_fn]:
        outputs = func(*xs)
        ys = to_tensorlist(outputs)
        ys_grad = [zeros_like(y) for y in ys]
        xs_grad = grad_fn(ys, xs, ys_grad, create_graph=True)
        ys_grad = grad_fn(xs_grad, ys_grad, v)
        outputs, ys_grad = return_fn(outputs), return_fn(ys_grad)

    return outputs, ys_grad


@framework.dygraph_only
def jacobian(func, inputs, create_graph=False, allow_unused=False):
    ''' 
    .. note::
        **This API is ONLY available in imperative mode.**

    This API computes the Jacobian matrix of `func` with respect to `inputs`.

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
    inputs = _check_tensors(inputs, "inputs")
    outputs = _check_tensors(func(*inputs), "outputs")
    fin_size = len(inputs)
    fout_size = len(outputs)
    flat_outputs = tuple(
        reshape(output, shape=[-1]) for output in outputs)
    jacobian = tuple()
    for i, flat_output in enumerate(flat_outputs):
        jac_i = list([] for _ in range(fin_size))
        for k in range(len(flat_output)):
            row_k = grad(
                flat_output[k],
                inputs,
                create_graph=create_graph,
                retain_graph=True,
                allow_unused=allow_unused)
            for j in range(fin_size):
                jac_i[j].append(
                    reshape(row_k[j], shape=[-1])
                    if isinstance(row_k[j], Tensor) else None)
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
def hessian(func, inputs, create_graph=False, allow_unused=False):
    ''' 
    .. note::
        **This API is ONLY available in imperative mode.**

    This API computes the Hessian matrix of `func` with respect to `inputs`.

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
    inputs = _check_tensors(inputs, "inputs")
    outputs = func(*inputs)
    assert isinstance(outputs, Tensor) and outputs.shape == [
        1
    ], "The function to compute Hessian matrix should return a Tensor with a single element"

    def jac_func(*ins):
        grad_inputs = grad(
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
