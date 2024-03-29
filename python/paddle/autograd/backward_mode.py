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
from paddle.base import core, framework
from paddle.base.backward import gradients_with_optimizer  # noqa: F401

__all__ = []


@framework.dygraph_only
def backward(tensors, grad_tensors=None, retain_graph=False):
    """
    Compute the backward gradients of given tensors.

    Args:
        tensors(list of Tensors): the tensors which the gradient to be computed. The tensors can not contain the same tensor.

        grad_tensors(list of Tensors of None, optional): the init gradients of the `tensors`` .If not None, it must have the same length with ``tensors`` ,
            and if any of the elements is None, then the init gradient is the default value which is filled with 1.0.
            If None, all the gradients of the ``tensors`` is the default value which is filled with 1.0.
            Defaults to None.

        retain_graph(bool, optional): If False, the graph used to compute grads will be freed. If you would
            like to add more ops to the built graph after calling this method( :code:`backward` ), set the parameter
            :code:`retain_graph` to True, then the grads will be retained. Thus, setting it to False is much more memory-efficient.
            Defaults to False.

    Returns:
        NoneType: None


    Examples:
        .. code-block:: python

            >>> import paddle
            >>> x = paddle.to_tensor([[1, 2], [3, 4]], dtype='float32', stop_gradient=False)
            >>> y = paddle.to_tensor([[3, 2], [3, 4]], dtype='float32')

            >>> grad_tensor1 = paddle.to_tensor([[1,2], [2, 3]], dtype='float32')
            >>> grad_tensor2 = paddle.to_tensor([[1,1], [1, 1]], dtype='float32')

            >>> z1 = paddle.matmul(x, y)
            >>> z2 = paddle.matmul(x, y)

            >>> paddle.autograd.backward([z1, z2], [grad_tensor1, grad_tensor2], True)
            >>> print(x.grad)
            Tensor(shape=[2, 2], dtype=float32, place=Place(cpu), stop_gradient=False,
            [[12., 18.],
             [17., 25.]])


            >>> x.clear_grad()

            >>> paddle.autograd.backward([z1, z2], [grad_tensor1, None], True)
            >>> print(x.grad)
            Tensor(shape=[2, 2], dtype=float32, place=Place(cpu), stop_gradient=False,
            [[12., 18.],
             [17., 25.]])

            >>> x.clear_grad()

            >>> paddle.autograd.backward([z1, z2])
            >>> print(x.grad)
            Tensor(shape=[2, 2], dtype=float32, place=Place(cpu), stop_gradient=False,
            [[10., 14.],
             [10., 14.]])


    """

    def check_tensors(in_out_list, name):
        assert in_out_list is not None, f"{name} should not be None"

        if isinstance(in_out_list, (list, tuple)):
            assert len(in_out_list) > 0, f"{name} cannot be empty"
            for each_var in in_out_list:
                assert isinstance(
                    each_var, (paddle.Tensor, core.eager.Tensor)
                ), f"Elements of {name} must be paddle.Tensor"
            return in_out_list
        else:
            assert isinstance(
                in_out_list, (paddle.Tensor, core.eager.Tensor)
            ), f"{name} must be Tensor or list of Tensor"
            return [in_out_list]

    tensors = check_tensors(tensors, "tensors")

    assert len(tensors) == len(
        set(tensors)
    ), "The argument 'tensors' of paddle.autograd.backward contains duplicate paddle.Tensor object."

    if grad_tensors is not None:
        if not isinstance(grad_tensors, (list, tuple)):
            grad_tensors = [grad_tensors]

        for each_tensor in grad_tensors:
            if each_tensor is not None:
                assert isinstance(
                    each_tensor, (paddle.Tensor, core.eager.Tensor)
                ), "The argument 'grad_tensors' of paddle.autograd.backward is invalid, it can be 'None', 'paddle.Tensor' or 'list[None/paddle.Tensor]'."
    else:
        grad_tensors = []

    if len(grad_tensors) > 0:
        assert len(tensors) == len(
            grad_tensors
        ), "The length of grad_tensors must be equal to tensors"

    assert isinstance(retain_graph, bool), "retain_graph must be True or False"

    core.eager.run_backward(tensors, grad_tensors, retain_graph)
