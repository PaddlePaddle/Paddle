# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

import inspect
import numpy as np

import paddle
from .. import framework
from .. import core
from .. import unique_name
from ..framework import Variable, Parameter, ParamBase
from .base import switch_to_static_graph
from .math_op_patch import monkey_patch_math_varbase
from .parallel import scale_loss


def monkey_patch_varbase():
    @switch_to_static_graph
    def _to_static_var(self, to_parameter=False, **kwargs):
        """
        **Notes**:
            **This API is ONLY available in Dygraph mode**

        Transform a VarBase into static Variable with same attributes. It's a low level interface used
        in dy2static and shall not be called directly.

        Args:
            to_parameter (bool): It takes effect only if the input a VarBase. If set True,
                                 the VarBase will be converted into framework.Parameters. Otherwise, it will
                                 be converted into framework.Variable. Default False.

        Examples:
            .. code-block:: python

                import paddle.fluid as fluid
                from paddle.fluid.dygraph.base import to_variable
                import numpy as np

                data = np.ones([3, 1024], dtype='float32')
                with fluid.dygraph.guard():
                    var_base = to_variable(data)
                    static_var = var_base._to_static_var()

        """

        # Note: getattr(self, attr, None) will call x.grad=x.gradient(), but gradient() only available in dygraph. 
        # It will fail. So, for propery in dygraph only, should not let it getattr(self, attr, None).
        attr_not_need_keys = ['grad']
        if isinstance(self, ParamBase):
            attr_kwargs = self.__dict__.copy()
        else:
            attr_names = []
            for name in dir(self):
                if name not in attr_not_need_keys and not (
                        inspect.ismethod(getattr(self, name)) or
                        name.startswith('_')):
                    attr_names.append(name)
            attr_kwargs = {name: getattr(self, name) for name in attr_names}

        attr_keys = ['block', 'shape', 'dtype', 'type', 'name', 'persistable']
        for attr in attr_keys:
            attr_kwargs[attr] = getattr(self, attr, None)

        attr_kwargs.update(kwargs)

        if to_parameter or isinstance(self, ParamBase):
            del attr_kwargs['persistable']
            static_var = Parameter(**attr_kwargs)
        else:
            static_var = Variable(**attr_kwargs)
        return static_var

    # TODO(jiabin): move this to cplusplus end if we find some performance issue on it
    @framework.dygraph_only
    def set_value(self, value):
        """
        **Notes**:
            **This API is ONLY available in Dygraph mode**

        Set a new value for this Variable.

        Args:
            value (Variable|np.ndarray): the new value.

        Examples:
            .. code-block:: python

                import paddle.fluid as fluid
                from paddle.fluid.dygraph.base import to_variable
                from paddle.fluid.dygraph import Linear
                import numpy as np

                data = np.ones([3, 1024], dtype='float32')
                with fluid.dygraph.guard():
                    linear = fluid.dygraph.Linear(1024, 4)
                    t = to_variable(data)
                    linear(t)  # call with default weight
                    custom_weight = np.random.randn(1024, 4).astype("float32")
                    linear.weight.set_value(custom_weight)  # change existing weight
                    out = linear(t)  # call with different weight

        """
        assert isinstance(value, (np.ndarray, core.VarBase)), \
            "Variable set_value function, arguments type only support Variable, numpy, VarBase"

        value_np = value
        if isinstance(value, core.VarBase):
            value_np = value.numpy()

        self_tensor_np = self.numpy()

        assert self_tensor_np.shape == value_np.shape, \
            "Variable Shape not match, Variable [ {} ] need tensor with shape {} but load set tensor with shape {}".format(
                self.name, self_tensor_np.shape, value_np.shape)

        assert self_tensor_np.dtype == value_np.dtype, \
            "Variable dtype not match, Variable [ {} ] need tensor with dtype {}  but load tensor with dtype {}".format(
                self.name, self_tensor_np.dtype, value_np.dtype)

        self.value().get_tensor().set(value_np,
                                      framework._current_expected_place())

    @framework.dygraph_only
    def backward(self, retain_graph=False):
        """
        Run backward of current Graph which starts from current Tensor.

        The new gradient will accumulat on previous gradient.

        You can clear gradient by ``Tensor.clear_grad()`` .

        Args:
            retain_graph(bool, optional): If False, the graph used to compute grads will be freed. If you would
                like to add more ops to the built graph after calling this method( :code:`backward` ), set the parameter
                :code:`retain_graph` to True, then the grads will be retained. Thus, seting it to False is much more memory-efficient.
                Defaults to False.

        Returns:
            NoneType: None

        Examples:
            .. code-block:: python

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

        """
        if framework.in_dygraph_mode():
            if paddle.distributed.get_world_size() > 1:
                scaled_loss = scale_loss(self)
                scaled_loss._run_backward(framework._dygraph_tracer(),
                                          retain_graph)
            else:
                self._run_backward(framework._dygraph_tracer(), retain_graph)
        else:
            raise ValueError(
                "Variable.backward() is only available in DyGraph mode")

    @framework.dygraph_only
    def gradient(self):
        """
        Get the Gradient of Current Tensor.

        Returns:
            ndarray: Numpy value of the gradient of current Tensor

        Examples:
            .. code-block:: python

                import paddle

                x = paddle.to_tensor(5., stop_gradient=False)
                y = paddle.pow(x, 4.0)
                y.backward()
                print("grad of x: {}".format(x.grad))
                # [500.]

        """
        if self._grad_ivar() is None:
            return None

        new_ivar = self._grad_ivar()._copy_to(core.CPUPlace(), True)
        if self._grad_ivar().type == core.VarDesc.VarType.SELECTED_ROWS:
            return (np.array(new_ivar.value().get_selected_rows().get_tensor()),
                    np.array(new_ivar.value().get_selected_rows().rows()))
        else:
            return np.array(new_ivar.value().get_tensor())

    @property
    def grad(self):
        """
        The alias of gradient().
        """

        return self.gradient()

    def clear_grad(self):
        """
        The alias of clear_gradient().
        """
        self.clear_gradient()

    @property
    def inplace_version(self):
        """
        The inplace version of current Tensor.
        The version number is incremented whenever the current Tensor is modified through an inplace operation.

        **Notes: This is a read-only property**

        Examples:
          .. code-block:: python

            import paddle
            var = paddle.ones(shape=[4, 2, 3], dtype="float32")
            print(var.inplace_version)  # 0

            var[1] = 2.2
            print(var.inplace_version)  # 1

        """
        return self._inplace_version()

    def __str__(self):
        """
        Convert a VarBase object to a readable string.

        Returns(str): A readable string.

        Examples:
            .. code-block:: python

                import paddle
                x = paddle.rand([2, 5])
                print(x)
                
                # Tensor(shape=[2, 5], dtype=float32, place=CPUPlace,
                #        [[0.30574632, 0.55739117, 0.30902600, 0.39413780, 0.44830436],
                #         [0.79010487, 0.53972793, 0.09495186, 0.44267157, 0.72112119]])
        """
        from paddle.tensor.to_string import to_string
        return to_string(self)

    def __deepcopy__(self, memo):
        """
        Deep copy Tensor, it will always performs Tensor copy.

        Examples:
            .. code-block:: python

                import paddle
                import copy
                x = paddle.to_tensor(2.)
                y = copy.deepcopy(x)
                
                print(x)
                # Tensor(shape=[1], dtype=float32, place=CPUPlace, stop_gradient=True,
                #        [2.])

                print(y)
                # Tensor(shape=[1], dtype=float32, place=CPUPlace, stop_gradient=True,
                #        [2.])

        """
        if not self.is_leaf:
            raise RuntimeError(
                "Only Leaf Tensor support the deepcopy at the moment, non-Leaf Tensors contains graph information that does't support deepcopy"
            )
        new_varbase = core.VarBase()
        new_varbase.name = self.name + unique_name.generate("_deepcopy")
        memo[id(self)] = new_varbase
        new_varbase.copy_(self, True)
        return new_varbase

    @property
    def block(self):
        return framework.default_main_program().global_block()

    def __nonzero__(self):
        numel = np.prod(self.shape)
        assert numel == 1, "When Variable is used as the condition of if/while , Variable can only contain one element."
        tensor = self.value().get_tensor()
        assert tensor._is_initialized(), "tensor not initialized"
        return bool(np.all(tensor.__array__() > 0))

    def __bool__(self):
        return self.__nonzero__()

    for method_name, method in (
        ("__bool__", __bool__), ("__nonzero__", __nonzero__),
        ("_to_static_var", _to_static_var), ("set_value", set_value),
        ("block", block), ("backward", backward), ("clear_grad", clear_grad),
        ("inplace_version", inplace_version), ("grad", grad),
        ("gradient", gradient), ("__str__", __str__), ("__repr__", __str__),
        ("__deepcopy__", __deepcopy__), ("__module__", "paddle"),
        ("__name__", "Tensor")):
        setattr(core.VarBase, method_name, method)

    # patch math methods for varbase
    monkey_patch_math_varbase()
