#   Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

from paddle.common_ops_import import LayerHelper
from paddle.fluid.data_feeder import check_variable_and_dtype
from paddle.fluid.framework import static_only


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
        input (Variable): A Tensor to print.
        summarize (int): Number of elements in the tensor to be print. If it's
                value is -1, then all elements in the tensor will be print.
        message (str): A string message to print as a prefix.
        first_n (int): Only log `first_n` number of times.
        print_tensor_name (bool, optional): Print the tensor name. Default: True.
        print_tensor_type (bool, optional): Print the tensor type. Defaultt: True.
        print_tensor_shape (bool, optional): Print the tensor shape. Default: True.
        print_tensor_layout (bool, optional): Print the tensor layout. Default: True.
        print_tensor_lod (bool, optional): Print the tensor lod. Default: True.
        print_phase (str): Which phase to displace, including 'forward',
                'backward' and 'both'. Default: 'both'. If set to 'backward', will
                only print the gradients of input tensor; If set to 'both', will
                both print the input tensor itself and the gradients of input tensor.

    Returns:
        Variable: Output tensor.

    NOTES:
        The input and output are two different variables, and in the
        following process, you should use the output variable but not the input,
        otherwise, the print layer doesn't have backward.

    Examples:
        .. code-block:: python

           import paddle

           paddle.enable_static()

           x = paddle.full(shape=[2, 3], fill_value=3, dtype='int64')
           out = paddle.static.Print(x, message="The content of input layer:")

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
        'paddle.static.Print',
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
    