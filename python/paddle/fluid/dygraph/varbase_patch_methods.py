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
from ..framework import Variable, is_variable
from .. import unique_name
import numpy as np


def monkey_patch_varbase():
    @framework.dygraph_only
    def set_value(
            self, value
    ):  # TODO(jiabin): move this to cplusplus end if we find some performance issue on it
        """
        **Notes**:
            **This API is ONLY avaliable in Dygraph mode**

        Set a new value for this Variable.

        Args:
            value (Variable|np.ndarray): the new value.

        Examples:
            .. code-block:: python

                import paddle.fluid as fluid
                from paddle.fluid.dygraph.base import to_variable
                from paddle.fluid.dygraph import FC
                import numpy as np

                data = np.ones([3, 32, 32], dtype='float32')
                with fluid.dygraph.guard():
                    fc = fluid.dygraph.FC("fc", 4)
                    t = to_variable(data)
                    fc(t)  # call with default weight
                    custom_weight = np.random.randn(1024, 4).astype("float32")
                    fc.weight.set_value(custom_weight)  # change existing weight
                    out = fc(t)  # call with different weight

        """
        assert isinstance(value, (np.ndarray, core.VarBase)), \
            "Variable set_value function, arguments type only support Variable, numpy, VarBase"

        value_np = value
        if isinstance(value, core.VarBase):
            value_np = value.numpy()

        self_tensor_np = self.numpy()

        assert self_tensor_np.shape == value_np.shape, \
            "Variable Shape not match, Variable [ {} ] need tensor with shape {} but load set tensor with shape {}".format( self.name, self_tensor_np.shape, value_np.shape)

        assert self_tensor_np.dtype == value_np.dtype, \
            "Variable dtype not match, Variable [ {} ] need tensor with dtype {}  but load tensor with dtype {}".format( self.name, self_tensor_np.dtype, value_np.dtype)

        self.value().get_tensor().set(value_np,
                                      framework._current_expected_place())

    @framework.dygraph_only
    def backward(self, backward_strategy=None):
        """
        **Notes**:
            **This API is ONLY avaliable in Dygraph mode**

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
                "Variable.backward() is only avaliable in DyGraph mode")

    @framework.dygraph_only
    def gradient(self):
        """
        **Notes**:
            **This API is ONLY avaliable in Dygraph mode**

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
            raise ValueError("%s has no grad, Please set Variable.stop_gradient=False, or " \
                             "check if this is the first and only variable need grad, if so, please set its pre-Variable's " \
                             "stop_gradient=False, to make sure it has gradient " % self.name)
        if not self._grad_ivar().value().get_tensor()._is_initialized():
            raise ValueError(
                "%s's Grad is Empty, Please check if it has no data in" %
                self.name)
        return self._grad_ivar().numpy()

    def __str__(self):
        return self.to_string(True)

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
        """
        Slice the variable.

        Args:
            item(int/slice/tuple) : the index.

        Returns:
            Sliced variable
        """

        if not isinstance(item, tuple):
            item = [item]

        decrease_axis = []
        slice_axis = []
        slice_start = []
        slice_end = []
        slice_step = []
        use_strided_slice = False
        reverse_axis = []

        def fill_constant(shape, value, force_cpu=False, out=None):
            framework.default_main_program().global_block().append_op(
                type='fill_constant',
                inputs={},
                outputs={'Out': [out]},
                attrs={
                    'shape': shape,
                    'dtype': out.dtype,
                    'value': float(value),
                    'force_cpu': force_cpu
                },
                stop_gradient=True)
            out.stop_gradient = True
            return out

        for dim, slice_item in enumerate(item):
            if isinstance(slice_item, slice):
                start = slice_item.start
                end = slice_item.stop
                step = slice_item.step

                if start is None and end is None and step is None:
                    continue

                if step is None:
                    step = 1

                if start is None and end is None:
                    assert (step == -1)
                    reverse_axis.append(dim)
                    continue

                if start is None:
                    start = 0

                if end is None:
                    end = 10000000

                if step != 1:
                    use_strided_slice = True

                slice_axis.append(dim)
                slice_start.append(start)
                slice_end.append(end)
                slice_step.append(step)
            else:
                decrease_axis.append(dim)
                slice_axis.append(dim)
                slice_start.append(slice_item)
                slice_step.append(1)
                if is_variable(slice_item, Variable):
                    temp_1 = framework.default_main_program().global_block(
                    ).create_var(dtype='int32')
                    fill_constant([1], 1, force_cpu=True, out=temp_1)
                    temp_end = framework.default_main_program().global_block(
                    ).create_var(dtype='int32')
                    framework.default_main_program().global_block().append_op(
                        type='elementwise_add',
                        inputs={'X': slice_item,
                                'Y': temp_1},
                        outputs={'Out': temp_end},
                        attrs={'axis': -1})
                    slice_end.append(temp_end)
                else:
                    slice_end.append(slice_item + 1
                                     if slice_item != -1 else 10000000)

        def contain_var(one_list):
            for ele in one_list:
                if is_variable(ele, Variable):
                    return True
            return False

        def get_new_list_tensor(old_list):
            new_list_tensor = []
            for dim in old_list:
                if is_variable(dim, Variable):
                    dim.stop_gradient = True
                    new_list_tensor.append(dim)
                else:
                    assert (isinstance(dim, int))
                    temp_out = framework.default_main_program().global_block(
                    ).create_var(dtype='int32')
                    fill_constant([1], dim, force_cpu=True, out=temp_out)
                    new_list_tensor.append(temp_out)
            return new_list_tensor

        inputs = {'Input': [self]}
        attrs = {
            'axes': slice_axis,
            'starts': [],
            'ends': [],
            'decrease_axis': decrease_axis
        }
        if (use_strided_slice == True):
            attrs['strides'] = []
        infer_flags = list(1 for i in range(len(slice_axis)))
        # starts
        if not contain_var(slice_start):
            attrs['starts'] = slice_start
        else:
            inputs['StartsTensorList'] = get_new_list_tensor(slice_start)
            for i, dim in enumerate(slice_start):
                if is_variable(dim, Variable):
                    attrs['starts'].append(-1)
                    infer_flags[i] = -1
                else:
                    attrs['starts'].append(dim)
        # ends
        if not contain_var(slice_end):
            attrs['ends'] = slice_end
        else:
            inputs['EndsTensorList'] = get_new_list_tensor(slice_end)
            for i, dim in enumerate(slice_end):
                if is_variable(dim, Variable):
                    attrs['ends'].append(-1)
                    infer_flags[i] = -1
                else:
                    attrs['ends'].append(dim)
        # strides
        if use_strided_slice == True:
            if not contain_var(slice_step):
                attrs['strides'] = slice_step
            else:
                inputs['StridesTensorList'] = get_new_list_tensor(slice_step)
                for i, dim in enumerate(slice_step):
                    if is_variable(dim, Variable):
                        attrs['strides'].append(-1)
                        infer_flags[i] = -1
                    else:
                        attrs['strides'].append(dim)
        # infer_flags
        attrs['infer_flags'] = infer_flags

        out = self
        if use_strided_slice == False and len(slice_axis) > 0:
            # append slice_op here
            slice_out_var = framework.default_main_program().global_block(
            ).create_var(
                name=unique_name.generate_with_ignorable_key(self.name +
                                                             "_slice"),
                dtype=self.dtype)

            framework.default_main_program().global_block().append_op(
                type="slice",
                inputs=inputs,
                outputs={'Out': [slice_out_var]},
                attrs=attrs)

            out = slice_out_var
        elif use_strided_slice == True and len(slice_axis) > 0:
            strided_slice_out_var = framework.default_main_program(
            ).global_block().create_var(
                name=unique_name.generate_with_ignorable_key(self.name +
                                                             "_strided_slice"),
                dtype=self.dtype)
            framework.default_main_program().global_block().append_op(
                type="strided_slice",
                inputs=inputs,
                outputs={'Out': [strided_slice_out_var]},
                attrs=attrs)

            out = strided_slice_out_var

        if len(reverse_axis) > 0:
            reverse_out_var = framework.default_main_program().global_block(
            ).create_var(
                name=unique_name.generate_with_ignorable_key(self.name +
                                                             "_slice_reverse"),
                dtype=self.dtype)
            framework.default_main_program().global_block().append_op(
                type="reverse",
                inputs={'X': out},
                outputs={'Out': [reverse_out_var]},
                attrs={'axis': reverse_axis})

            out = reverse_out_var

        return out

    for method_name, method in (("set_value", set_value),
                                ("backward", backward), ("gradient", gradient),
                                ("__str__", __str__), ("to_string", to_string),
                                ("__getitem__", __getitem__)):
        setattr(core.VarBase, method_name, method)
