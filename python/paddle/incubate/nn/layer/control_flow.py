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

import paddle

from ....fluid.data_feeder import check_variable_and_dtype
from ....fluid.framework import static_only
from ....fluid.layers import LayerHelper
from ....fluid.layers.control_flow import (
    ConditionalBlock,
    ConditionalBlockGuard,
)

__all__ = [
    'Switch',
    'Print',
]


@static_only
def Print(
    input,
    first_n=-1,
    message=None,
    summarize=20,
    print_tensor_name=True,
    print_tensor_type=True,
    print_tensor_shape=True,
    print_tensor_layout=True,
    print_tensor_lod=True,
    print_phase='both',
):
    '''
    :api_attr: Static Graph

    **Print operator**

    This creates a print op that will print when a tensor is accessed.

    Wraps the tensor passed in so that whenever that a tensor is accessed,
    the message `message` is printed, along with the current value of the
    tensor `t`.

    Args:
        input (Tensor): A Tensor to print.
        first_n (int, optional): Only log `first_n` number of times. Default: -1.
        message (str, optional): A string message to print as a prefix. Default: None.
        summarize (int, optional): Number of elements in the tensor to be print. If
                it's value is -1, then all elements in the tensor will be print.
        print_tensor_name (bool, optional): Print the tensor name. Default: True.
        print_tensor_type (bool, optional): Print the tensor type. Defaultt: True.
        print_tensor_shape (bool, optional): Print the tensor shape. Default: True.
        print_tensor_layout (bool, optional): Print the tensor layout. Default: True.
        print_tensor_lod (bool, optional): Print the tensor lod. Default: True.
        print_phase (str, optional): Which phase to displace, including 'forward',
                'backward' and 'both'. Default: 'both'. If set to 'backward', will
                only print the gradients of input tensor; If set to 'both', will
                both print the input tensor itself and the gradients of input tensor.

    Returns:
        Tensor: Output tensor.

    NOTES:
        The input and output are two different Tensor, and in the
        following process, you should use the output Tensor but not the input,
        otherwise, the print layer doesn't have backward.

    Examples:
        .. code-block:: python

           import paddle

           paddle.enable_static()

           x = paddle.full(shape=[2, 3], fill_value=3, dtype='int64')
           out = paddle.incubate.nn.Print(x, message="The content of input layer:")

           main_program = paddle.static.default_main_program()
           exe = paddle.static.Executor(place=paddle.CPUPlace())
           res = exe.run(main_program, fetch_list=[out])
           # Variable: fill_constant_1.tmp_0
           #   - message: The content of input layer:
           #   - lod: {}
           #   - place: CPUPlace
           #   - shape: [2, 3]
           #   - layout: NCHW
           #   - dtype: long
           #   - data: [3 3 3 3 3 3]
    '''
    check_variable_and_dtype(
        input,
        'input',
        ['float32', 'float64', 'int32', 'int64', 'bool'],
        'fluid.layers.Print',
    )

    helper = LayerHelper('print' + "_" + input.name, **locals())
    output = helper.create_variable_for_type_inference(input.dtype)
    helper.append_op(
        type='print',
        inputs={'In': input},
        outputs={'Out': output},
        attrs={
            'first_n': first_n,
            'summarize': summarize,
            'message': message or "",
            'print_tensor_name': print_tensor_name,
            'print_tensor_type': print_tensor_type,
            'print_tensor_shape': print_tensor_shape,
            'print_tensor_layout': print_tensor_layout,
            'print_tensor_lod': print_tensor_lod,
            'print_phase': print_phase.upper(),
        },
    )
    return output


class Switch:
    """
    :api_attr: Static Graph

    This class is used to implement Switch branch control function.
    Switch branch contains several case branches and one default branch.
    Switch control flow checks whether the case branch conditions are satisfied in turn,
    and only executes the statement after the first case branch that satisfies the conditions.
    If there is no case branch that satisfies the condition,
    only the statement following the default branch is executed.

    Note:
        A new OP :ref:`api_fluid_layers_case` is highly recommended instead of ``Switch`` if the shape of parameter ``cond`` is [1].
        OP :ref:`api_fluid_layers_case` is easier to use and is called with less code but does the same thing as ``Switch`` .

    Member Functions:
        case(condition): The case branch of Switch whose parameter cond is a scalar Variable of bool type. Only if the cond of the current case branch is True and the cond of the previous case branch is False, the statement after the case branch will be executed, and the statement after the case branch will not be executed.

        default(): The default branch of Switch. When cond of all case branches is False, the statement after default branch is executed.

    Case and default functions can only be used inside the scope of Switch, as shown below:

    .. code-block:: python

        '''
        with fluid.layers.Switch() as switch:
            with switch.case(cond1):
                i = fluid.layers.fill_constant(shape=[1], dtype='int64', value=1)
            with switch.case(cond2):
                i = fluid.layers.fill_constant(shape=[1], dtype='int64', value=2)
            with switch.default():
                i = fluid.layers.fill_constant(shape=[1], dtype='int64', value=0)
        '''

    Args:
        name(str, optional): The default value is None.  Normally there is no need for user to set this property.  For more information, please refer to :ref:`api_guide_Name` .

    Examples:
        .. code-block:: python

            import paddle
            import paddle.fluid as fluid

            lr = paddle.static.create_global_var(
                shape=[1],
                value=0.0,
                dtype='float32',
                persistable=True,
                name="learning_rate")
            zero_var = fluid.layers.fill_constant(
                shape=[1], dtype='float32', value=0.0)
            one_var = fluid.layers.fill_constant(
                shape=[1], dtype='float32', value=1.0)
            two_var = fluid.layers.fill_constant(
                shape=[1], dtype='float32', value=2.0)

            global_step = paddle.incubate.nn.functional.autoincreased_step_counter(
                counter_name='@LR_DECAY_COUNTER@', begin=0, step=1)

            with paddle.incubate.nn.Switch() as switch:
                with switch.case(global_step == zero_var):
                    fluid.layers.assign(input=one_var, output=lr)
                with switch.default():
                    fluid.layers.assign(input=two_var, output=lr)

            exe = fluid.Executor(fluid.CPUPlace())
            exe.run(fluid.default_startup_program())

            res = exe.run(fluid.default_main_program(), feed={}, fetch_list=[lr])
            print(res) # [array([1.], dtype=float32)]
    """

    def __init__(self, name=None):
        self.helper = LayerHelper('switch', name=name)
        self.inside_scope = False
        self.pre_not_conditions = []

    def case(self, condition):
        if not self.inside_scope:
            raise ValueError("case should be called inside with")

        check_variable_and_dtype(
            condition,
            'condition',
            ['bool'],
            'the member function case of fluid.layers.Switch',
        )

        if len(self.pre_not_conditions) == 0:
            cond_block = ConditionalBlock([condition], is_scalar_condition=True)
            not_cond = paddle.logical_not(x=condition)
            self.pre_not_conditions.append(not_cond)
        else:
            pre_cond_num = len(self.pre_not_conditions)
            pre_not_cond = self.pre_not_conditions[pre_cond_num - 1]
            new_not_cond = paddle.logical_and(
                x=pre_not_cond, y=paddle.logical_not(x=condition)
            )
            self.pre_not_conditions.append(new_not_cond)
            cond_block = ConditionalBlock(
                [paddle.logical_and(x=pre_not_cond, y=condition)],
                is_scalar_condition=True,
            )

        return ConditionalBlockGuard(cond_block)

    def default(self):
        pre_cond_num = len(self.pre_not_conditions)
        if pre_cond_num == 0:
            raise ValueError("there should be at least one condition")
        cond_block = ConditionalBlock(
            [self.pre_not_conditions[pre_cond_num - 1]],
            is_scalar_condition=True,
        )
        return ConditionalBlockGuard(cond_block)

    def __enter__(self):
        """
        set flag that now is inside switch.block {}
        :return:
        """
        self.inside_scope = True
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.inside_scope = False
        if exc_type is not None:
            return False  # re-raise exception

        return True
