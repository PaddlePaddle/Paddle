#   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

from .. import core as core
from .. import framework as framework
from ..dygraph.parallel import scale_loss
import numpy as np


def monkey_patch_eagertensor():
    def __str__(self):
        from paddle.tensor.to_string import eager_tensor_to_string
        return eager_tensor_to_string(self)

    @framework.dygraph_only
    def backward(self, grad_tensor=None, retain_graph=False):
        """
        Run backward of current Graph which starts from current Tensor.

        The new gradient will accumulat on previous gradient.

        You can clear gradient by ``Tensor.clear_grad()`` .

        Args:
            grad_tensor(Tensor, optional): initial gradient values of the current Tensor. If `grad_tensor` is None, 
            the initial gradient values of the current Tensor would be Tensor filled with 1.0; 
            if `grad_tensor` is not None, it must have the same length as the current Tensor.
            Teh default value is None.

            retain_graph(bool, optional): If False, the graph used to compute grads will be freed. If you would
                like to add more ops to the built graph after calling this method( :code:`backward` ), set the parameter
                :code:`retain_graph` to True, then the grads will be retained. Thus, seting it to False is much more memory-efficient.
                Defaults to False.
        Returns:
            NoneType: None

        Examples:
            .. code-block:: python

                import paddle
                x = paddle.to_tensor(5., stop_gradient=False)
                for i in range(5):
                    y = paddle.pow(x, 4.0)
                    y.backward()
                    print("{}: {}".format(i, x.grad))
                # 0: [500.]
                # 1: [1000.]
                # 2: [1500.]
                # 3: [2000.]
                # 4: [2500.]

                x.clear_grad()
                print("{}".format(x.grad))
                # 0.

                grad_tensor=paddle.to_tensor(2.)
                for i in range(5):
                    y = paddle.pow(x, 4.0)
                    y.backward(grad_tensor)
                    print("{}: {}".format(i, x.grad))
                # 0: [1000.]
                # 1: [2000.]
                # 2: [3000.]
                # 3: [4000.]
                # 4: [5000.]

        """
        if framework.in_dygraph_mode():
            if grad_tensor is not None:
                assert isinstance(
                    grad_tensor, core.eager.EagerTensor
                ), "The type of grad_tensor must be paddle.Tensor"
                assert grad_tensor.shape == self.shape, \
                    "Tensor shape not match, Tensor of grad_tensor [ {} ] with shape {} mismatch Tensor [ {} ] with shape {}".format(
                    grad_tensor.name, grad_tensor.shape, self.name, self.shape)
                grad_tensor = [grad_tensor]
            else:
                grad_tensor = []

            if core.is_compiled_with_xpu() or core.is_compiled_with_npu():
                # TODO(liuyuhui): Currently only for xpu. Will be removed in the future.
                scaled_loss = scale_loss(self)
                core.eager.run_backward([scaled_loss], grad_tensor,
                                        retain_graph)
            else:
                core.eager.run_backward([self], grad_tensor, retain_graph)
        else:
            raise ValueError(
                "Variable.backward() is only available in DyGraph mode")

    @framework.dygraph_only
    def gradient(self):
        """
        .. warning::
          This API will be deprecated in the future, it is recommended to use
          :code:`x.grad` which returns the tensor value of the gradient.

        Get the Gradient of Current Tensor.

        Returns:
            ndarray: Numpy value of the gradient of current Tensor

        Examples:
            .. code-block:: python

                import paddle

                x = paddle.to_tensor(5., stop_gradient=False)
                y = paddle.pow(x, 4.0)
                y.backward()
                print("grad of x: {}".format(x.gradient()))
                # [500.]

        """
        if self.grad is None:
            return None
        # TODO(wanghuancoder) support SELECTED_ROWS
        return self.grad.numpy()

    if hasattr(core, "eager"):
        setattr(core.eager.EagerTensor, "__str__", __str__)
        setattr(core.eager.EagerTensor, "backward", backward)
        setattr(core.eager.EagerTensor, "gradient", gradient)
