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
import paddle


def _check_tensors(in_out_list, name):
    assert in_out_list is not None, "{} should not be None".format(name)

    if isinstance(in_out_list, (list, tuple)):
        assert len(in_out_list) > 0, "{} connot be empyt".format(name)
        for each_var in in_out_list:
            assert isinstance(
                each_var,
                paddle.Tensor), "Elements of {} must be paddle.Tensor".format(
                    name)
        return in_out_list
    else:
        assert isinstance(
            in_out_list,
            paddle.Tensor), "{} must be Tensor or list of Tensor".format(name)
        return [in_out_list]


def _stack_tensor_or_return_none(origin_list):
    assert len(origin_list) > 0, "Can't not stack an empty list"
    return paddle.stack(
        origin_list, axis=0) if isinstance(origin_list[0],
                                           paddle.Tensor) else None


@framework.dygraph_only
def jacobian(func, inputs, create_graph=False, allow_unused=False):
    ''' 
    .. note::
        **This API is ONLY available in Dygraph mode.**

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
        ``i``th output and ``j``th input and will have as size the
        concatenation of the sizes of the corresponding output and the
        corresponding input and will have same dtype and device as the
        corresponding input.
    '''
    inputs = _check_tensors(inputs, "inputs")
    outputs = _check_tensors(func(*inputs), "outputs")
    fin_size = len(inputs)
    fout_size = len(outputs)
    flat_outputs = tuple(
        paddle.reshape(
            output, shape=[-1]) for output in outputs)
    if fin_size == 1 and fout_size == 1:
        flat_output = flat_outputs[0]
        jac = []
        for k in range(len(flat_output)):
            row_k = paddle.grad(
                flat_output[k],
                inputs[0],
                create_graph=create_graph,
                retain_graph=True,
                allow_unused=allow_unused)
            jac.append(
                paddle.reshape(
                    row_k[0], shape=[-1])
                if isinstance(row_k[0], paddle.Tensor) else None)
        return _stack_tensor_or_return_none(jac)
    elif fin_size == 1 and fout_size != 1:
        jacobian = tuple()
        for i, flat_output in enumerate(flat_outputs):
            jac = []
            for k in range(len(flat_output)):
                row_k = paddle.grad(
                    flat_output[k],
                    inputs[0],
                    create_graph=create_graph,
                    retain_graph=True,
                    allow_unused=allow_unused)
                jac.append(
                    paddle.reshape(
                        row_k[0], shape=[-1])
                    if isinstance(row_k[0], paddle.Tensor) else None)
            jacobian += (_stack_tensor_or_return_none(jac), )
        return jacobian
    elif fin_size != 1 and fout_size == 1:
        flat_output = flat_outputs[0]
        jac = list([] for _ in range(fin_size))
        for k in range(len(flat_output)):
            row_k = paddle.grad(
                flat_output[k],
                inputs,
                create_graph=create_graph,
                retain_graph=True,
                allow_unused=allow_unused)
            for j in range(fin_size):
                jac[j].append(
                    paddle.reshape(
                        row_k[j], shape=[-1])
                    if isinstance(row_k[j], paddle.Tensor) else None)
        return tuple(
            _stack_tensor_or_return_none(jac[j]) for j in range(fin_size))
    else:
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
                _stack_tensor_or_return_none(jac_i[j])
                for j in range(fin_size)), )
        return jacobian
