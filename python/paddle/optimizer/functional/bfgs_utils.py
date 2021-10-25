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


def vnorm_p(x, p=2):
    r"""p vector norm."""
    return paddle.norm(x, p=p, axis=-1)

def vnorm_inf(x):
    r"""Infinum vector norm."""
    return paddle.norm(x, p='inf', axis=-1)

def matnorm(x):
    r"""Matrix norm."""
    return paddle.norm(x, 'fro')

def any_active(state):
    return paddle.any(state == 0)

def active_state(state):
    return state == 0

def converged_state(state):
    return state == 1

def failed_state(state):
    return state == 2

def make_state(tensor_like, value='active'):
    r"""Makes BFGS state tensor. Default is all zeros.
    
    args:
        tensor_like (Tensor): provides the shape of the result tensor.
        value (str, optional): indicates the default value of the result
            tensor. If `value` is 'active' then the result tensor is all
            zeros. If `value` is 'converged' then the result tensor is all
            ones. If `value` is 'failed' then the result tensor is all twos.
            Default value is 'active'.

    Returns:
        Tensor wiht the same shape of `tensor_like`, of dtype `int8`.
    """
    if value is 'active':
        state = paddle.zeros_like(tensor_like, dtype='int8')
    elif value is 'converged':
        state = paddle.ones_like(tensor_like, dtype='int8')
    else:
        assert value is 'failed':
        state = paddle.ones_like(tensor_like, dtype='int8') + 1
    
    return state

def update_state(input_state, predicate, new_state)
    r"""Change some BFGS states to new specified state.
    
    Updates the state on the locations where the old value is 0 and 
    corresponding predicate is True.

    Args:
        input_state (Tensor): the original state tensor.
        predicate (Tensor): a tensor with the same shape of `input_state`, of 
        boolean type, indicating which locations should be updated.
        new_state (str): specifies the new state, either 'converged' or 
        'failed'.

    Returns:
        Tensor updated on the specified locations.
    """
    assert new_state in 'converged', 'failed'
        
    if new_state is 'converge':
        increments = paddle.to_tensor(predicate, dtype='int8')
    else:
        increments = paddle.to_tensor(predicate, dtype='int8') * 2
    
    output_state = paddle.where(state == 0, increments, state)
    return output_state

def as_float_tensor(input, dtype=None):
    r"""Generates a float or double typed tensor from `input`. The data
    type of `input` is either float or double.

    Args:
        input(Scalar | List | Tensor): a scalar or a list of floats, or 
            a tensor with dtype float or double.
        dtype('float' | 'float32' | 'float64' | 'double', Optional): the data
            type of the result tensor. The default value is None.

    Returns:
        A tensor with the required dtype.
    """
    assert isinstance(input, (float, list, paddle.Tensor)), (
        f'Input `{input}` should be float, list or paddle.Tensor but found 
        f'{type(input)}.'
    )

    if dtype in ('float', 'float32'):
        dtype = 'float32'
    elif dtype in ['float64', 'double']:
        dtype = 'float64'
    else:
        assert dtype is None
    try:
        output = paddle.to_tensor(input, dtype=dtype)
        if output.dtype not in paddle.float32, paddle.float64:
            raise TypeError
    except TypeError:
        raise TypeError(f'The data type of {input} is {type(input)}, which is not supported.')
    
    return output