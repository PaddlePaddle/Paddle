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

import numpy as np
import paddle
from ...autograd import vjp as _vjp


def ternary(cond, x, y):
    
    expanding_dim = x.dim() - cond.dim()
    assert expanding_dim >= 0

    for _ in range(expanding_dim):
        cond = cond.unsqueeze(-1)

    if cond.shape != x.shape:
        cond = cond.broadcast_to(x.shape)

    return paddle.where(cond, x, y)

def vjp(f, x, v=None, create_graph=False):
    r"""A single tensor version of VJP.
    
    Args:
        f (Callable): the objective function.
        x (Tensor): the input tensor.

    Returns:
        (fval, gval):
            fval: a tensor that holds the function value.
            gval: a tensor that holds the function gradients.  
    """
    assert isinstance(x, paddle.Tensor), (
        f'This BFGS optimizer applies to function of a single input tensor. '
        f'The input however is a {type(x)}.'
    )
    fval, gval = _vjp(f, x, v=v, create_graph=create_graph)
    assert isinstance(fval, paddle.Tensor), (
        f'This BFGS optimizer only supports function returning a single output '
        f'tensor. However, the function result is a {type(fval)}.'
    )

    return fval, gval[0]

def vnorm_p(x, p=2):
    r"""p vector norm."""
    return paddle.norm(x, p=p, axis=-1)

def vnorm_inf(x):
    r"""Infinum vector norm."""
    return paddle.norm(x, p=np.inf, axis=-1)

def matnorm(x):
    r"""Matrix norm."""
    return paddle.norm(x, 'fro')

def any_active(state):
    return paddle.any(state == 0)

def all_active_with_predicates(state, *predicates):
    r"""Tests whether all active states also satisfies the predicates.
    
    Args:
        state (Tensor): the search state of dtype int. For each element, 0 
            represents active state.
        predicates (List[Tensor]): a list of boolean typed tensors of the
            same shape with `state`.
    
    Returns:
        A scalar boolean tensor. True if the predicates are true for every
        active state. Otherwise False.
    """
    active_preds = active = active_state(state)
    for p in predicates:
        active_preds = paddle.logical_and(active_preds, p)
  
    return paddle.all(active == active_preds)

def any_active_with_predicates(state, *predicates):
    r"""Tests whether there's any active state also satisfies all the
    predicates.
    
    Args:
        state (Tensor): the search state of dtype int. For each element, 0 
            represents active state.
        predicates (List[Tensor]): a list of boolean typed tensors of the
            same shape with `state`.
    
    Returns:
        A scalar boolean tensor. True if any element in `state` is active and
        the corresponding predicate values are all True. Otherwise False.
    """
    active_preds = active_state(state)
    for p in predicates:
        active_preds = paddle.logical_and(active_preds, p)

    return paddle.any(active_preds)

def active_state(state):
    return state == 0

def converged_state(state):
    return state == 1

def failed_state(state):
    return state == 2

def make_const(tensor_like, value, dtype=None):
    r"""Makes a tensor filled with specified constant value.
    
    Args:
        tensor_like (Tensor): uses this tensor's shape and dtype to build
            the output tensor.
        value (float|boolean|int): fills the output tensor with this value.
        dtype (Optional): specifies as the output tensor's dtype. Default is
            None, in which case the output uses `tensor_like`'s dtype.
    
    Returns:
        The generated tensor with constant value `value` with the desired 
        dtype. 
    """
    if dtype is None:
        dtype = tensor_like.dtype
    return paddle.to_tensor(value, dtype).broadcast_to(tensor_like.shape)

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
        Tensor wiht the same shape of `tensor_like`, of dtype `int`.
    """
    # (FIXME) int8 is preferably a better choice but we found it's not 
    # consistently supported across Paddle APIs so we used int32 instead.  
    if value is 'active':
        state = paddle.zeros_like(tensor_like, dtype='int32')
    elif value is 'converged':
        state = paddle.ones_like(tensor_like, dtype='int32')
    else:
        assert value is 'failed'
        state = paddle.ones_like(tensor_like, dtype='int32') + 1
    
    return state

def update_state(input_state, predicate, new_state):
    r"""Updates the state on the locations where the old value is 0 and 
    corresponding predicate is True.

    Args:
        input_state (Tensor): the original state tensor.
        predicate (Tensor): a tensor with the same shape of `input_state`, of 
        boolean type, indicating which locations should be updated.
        new_state ('failed' | 'converged'): specifies the new state, either 
            'converged' or 'failed'.

    Returns:
        Tensor updated on the specified locations.
    """
    assert new_state in ('converged', 'failed')
        
    if new_state is 'converged':
        increments = paddle.to_tensor(predicate, dtype='int32')
    else:
        increments = paddle.to_tensor(predicate, dtype='int32') * 2
    
    output_state = paddle.where(input_state == 0, increments, input_state)
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
    assert isinstance(input, (int, float, list, paddle.Tensor)), (
        f'Input `{input}` should be float, list or paddle.Tensor but found ' 
        f'{type(input)}.'
    )

    if dtype in ('float', 'float32', paddle.float32):
        dtype = 'float32'
    elif dtype in ['float64', 'double', paddle.float64]:
        dtype = 'float64'
    else:
        assert dtype is None
    try:
        output = paddle.to_tensor(input, dtype=dtype)
        if output.dtype not in (paddle.float32, paddle.float64):
            raise TypeError
    except TypeError:
        raise TypeError(f'The data type of {input} is {type(input)}, which is not supported.')
    
    return output


class StopCounterException(Exception):
    r"""raises this Exception on the event that increments a stopped 
    StopCounter.
    """
    pass


class StopCounter(object):
    r"""Defines a counter with a predefined end count.
    """

    def __init__(self, end):
        self.count = 0
        assert isinstance(end, int) and end > 0
        self.end = end
    
    def increment(self):
        r"""Increments the counter."""

        if self.count < self.end:
            self.count += 1
        else:
            raise StopCounterException()