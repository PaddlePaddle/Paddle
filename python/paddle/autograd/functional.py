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

from paddle.fluid import framework
from .utils import _check_tensors, _stack_tensor_or_return_none, _replace_none_with_zero_tensor, _stop_gradient_pre_process
import paddle


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
    inputs = _stop_gradient_pre_process(inputs)
    outputs = _check_tensors(func(*inputs), "outputs")
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
    inputs = _stop_gradient_pre_process(inputs)
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
