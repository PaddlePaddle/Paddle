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

from .. import framework
from .. import core
from . import BackwardStrategy
from ..framework import Variable, _getitem_impl_
from .. import unique_name
import numpy as np
from .math_op_patch import monkey_patch_math_varbase


def monkey_patch_varbase():
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
    def backward(self, backward_strategy=None):
        """
        **Notes**:
            **This API is ONLY available in Dygraph mode**

        Run backward of current Graph which starts from current Variable

        Args:
            backward_strategy( :ref:`api_fluid_dygraph_BackwardStrategy` ): The Backward Strategy to run backward

        Returns:
            NoneType: None

        Examples:
            .. code-block:: python

                import paddle.fluid as fluid
                import numpy as np

                x = np.ones([2, 2], np.float32)
                with fluid.dygraph.guard():
                    inputs2 = []
                    for _ in range(10):
                        tmp = fluid.dygraph.base.to_variable(x)
                        # if we don't set tmp's stop_gradient as False then, all path to loss will has no gradient since
                        # there is no one need gradient on it.
                        tmp.stop_gradient=False
                        inputs2.append(tmp)
                    ret2 = fluid.layers.sums(inputs2)
                    loss2 = fluid.layers.reduce_sum(ret2)
                    backward_strategy = fluid.dygraph.BackwardStrategy()
                    backward_strategy.sort_sum_gradient = True
                    loss2.backward(backward_strategy)

        """
        if framework.in_dygraph_mode():
            if backward_strategy is None:
                backward_strategy = BackwardStrategy()
                backward_strategy.sort_sum_gradient = False

            self._run_backward(backward_strategy, framework._dygraph_tracer())
        else:
            raise ValueError(
                "Variable.backward() is only available in DyGraph mode")

    @framework.dygraph_only
    def gradient(self):
        """
        **Notes**:
            **This API is ONLY available in Dygraph mode**

        Get the Gradient of Current Variable

        Returns:
            ndarray: Numpy value of the gradient of current Variable

        Examples:
            .. code-block:: python

                import paddle.fluid as fluid
                import numpy as np

                x = np.ones([2, 2], np.float32)
                with fluid.dygraph.guard():
                    inputs2 = []
                    for _ in range(10):
                        tmp = fluid.dygraph.base.to_variable(x)
                        tmp.stop_gradient=False
                        inputs2.append(tmp)
                    ret2 = fluid.layers.sums(inputs2)
                    loss2 = fluid.layers.reduce_sum(ret2)
                    backward_strategy = fluid.dygraph.BackwardStrategy()
                    backward_strategy.sort_sum_gradient = True
                    loss2.backward(backward_strategy)
                    print(loss2.gradient())

        """
        if self._grad_ivar() is None:
            raise ValueError(
                "%s has no grad, Please set Variable.stop_gradient=False, or "
                "check if this is the first and only variable need grad, if so, please set its pre-Variable's "
                "stop_gradient=False, to make sure it has gradient " %
                self.name)
        new_ivar = self._grad_ivar()._copy_to(core.CPUPlace(), True)
        if self._grad_ivar().type == core.VarDesc.VarType.SELECTED_ROWS:
            return (np.array(new_ivar.value().get_selected_rows().get_tensor()),
                    np.array(new_ivar.value().get_selected_rows().rows()))
        else:
            return np.array(new_ivar.value().get_tensor())

    def __str__(self):
        return self.to_string(True)

    @property
    def block(self):
        return framework.default_main_program().global_block()

    def to_string(self, throw_on_error, with_details=False):
        """
        Get debug string.

        Args:

            throw_on_error (bool): True if raise an exception when self is not initialized.

            with_details (bool): more details about variables and parameters (e.g. trainable, optimize_attr, ...) will be printed when with_details is True. Default value is False;

        Returns:
            str: The debug string.

        Examples:
            .. code-block:: python

                import paddle.fluid as fluid

                cur_program = fluid.Program()
                cur_block = cur_program.current_block()
                new_variable = cur_block.create_var(name="X",
                                                    shape=[-1, 23, 48],
                                                    dtype='float32')
                print(new_variable.to_string(True))
                print("=============with detail===============")
                print(new_variable.to_string(True, True))
        """
        if framework.in_dygraph_mode():
            # TODO(panyx0718): add more dygraph debug info.
            tensor = self.value().get_tensor()
            if tensor._is_initialized():
                return 'name %s, dtype: %s shape: %s %s' % (
                    self.name, self.dtype, self.shape, str(tensor))
            else:
                return 'name %s, shape: %s, not inited' % (self.name,
                                                           self.shape)

    def __getitem__(self, item):
        if not isinstance(item, tuple):
            item = [item]

        decrease_axis = []
        slice_axis = []
        slice_start = []
        slice_end = []
        reverse_axis = []

        for dim, slice_item in enumerate(item):
            if isinstance(slice_item, slice):
                start = slice_item.start
                end = slice_item.stop
                step = slice_item.step if slice_item.step else 1

                assert (step == 1 or step == -1)

                if step == -1:
                    reverse_axis.append(dim)
                    assert (start is None and end is None)

                if start is None and end is None:
                    continue

                if start is None:
                    start = 0

                if end is None:
                    end = 10000000

                slice_axis.append(dim)
                slice_start.append(start)
                slice_end.append(end)
            else:
                # int
                decrease_axis.append(dim)
                slice_axis.append(dim)
                slice_start.append(slice_item)
                slice_end.append(slice_item + 1
                                 if slice_item != -1 else 10000000)

        out = self
        if len(slice_axis) > 0:
            # append slice_op here
            inputs = {'Input': [out]}
            attrs = {
                'axes': slice_axis,
                'starts': slice_start,
                'ends': slice_end,
                'decrease_axis': decrease_axis
            }
            outs = core.ops.slice(inputs, attrs)
            out = outs['Out'][0]

        if len(reverse_axis) > 0:
            inputs = {'X': [out]}
            attrs = {'axis': reverse_axis}
            outs = core.ops.reverse(inputs, attrs)
            out = outs['Out'][0]

        return out

    for method_name, method in (("set_value", set_value), ("block", block),
                                ("backward", backward), ("gradient", gradient),
                                ("__str__", __str__), ("to_string", to_string),
                                ("__getitem__", __getitem__)):
        setattr(core.VarBase, method_name, method)

    # patch math methods for varbase
    monkey_patch_math_varbase()
