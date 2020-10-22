# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
import numpy as np
from paddle.fluid.layers import core
from paddle.fluid.data_feeder import convert_dtype, check_variable_and_dtype, check_type, check_dtype

__all__ = ['set_printoptions']


class PrintOptions(object):
    precision = 8
    threshold = 1000
    edgeitems = 3
    linewidth = 80
    sci_mode = False


DEFAULT_PRINT_OPTIONS = PrintOptions()


def set_printoptions(precision=None,
                     threshold=None,
                     edgeitems=None,
                     sci_mode=None):
    """Set the printing options for Tensor.
    NOTE: The function is similar with numpy.set_printoptions()

    Args:
        precision (int, optional): Number of digits of the floating number, default 8.
        threshold (int, optional): Total number of elements printed, default 1000.
        edgeitems (int, optional): Number of elements in summary at the begining and end of each dimension, defalt 3.
        sci_mode (bool, optional): Format the floating number with scientific notation or not, default False.
    
    Returns:
        None.

    Examples:
        .. code-block:: python

            import paddle

            paddle.seed(10)
            a = paddle.rand([10, 20])
            paddle.set_printoptions(4, 100, 3)
            print(a)
            
            '''
            Tensor: dygraph_tmp_0
            - place: CPUPlace
            - shape: [10, 20]
            - layout: NCHW
            - dtype: float32
            - data: [[0.2727, 0.5489, 0.8655, ..., 0.2916, 0.8525, 0.9000],
                    [0.3806, 0.8996, 0.0928, ..., 0.9535, 0.8378, 0.6409],
                    [0.1484, 0.4038, 0.8294, ..., 0.0148, 0.6520, 0.4250],
                    ...,
                    [0.3426, 0.1909, 0.7240, ..., 0.4218, 0.2676, 0.5679],
                    [0.5561, 0.2081, 0.0676, ..., 0.9778, 0.3302, 0.9559],
                    [0.2665, 0.8483, 0.5389, ..., 0.4956, 0.6862, 0.9178]]
            '''
    """
    kwargs = {}

    if precision is not None:
        check_type(precision, 'precision', (int), 'set_printoptions')
        DEFAULT_PRINT_OPTIONS.precision = precision
        kwargs['precision'] = precision
    if threshold is not None:
        check_type(threshold, 'threshold', (int), 'set_printoptions')
        DEFAULT_PRINT_OPTIONS.threshold = threshold
        kwargs['threshold'] = threshold
    if edgeitems is not None:
        check_type(edgeitems, 'edgeitems', (int), 'set_printoptions')
        DEFAULT_PRINT_OPTIONS.edgeitems = edgeitems
        kwargs['edgeitems'] = edgeitems
    if sci_mode is not None:
        check_type(sci_mode, 'sci_mode', (bool), 'set_printoptions')
        DEFAULT_PRINT_OPTIONS.sci_mode = sci_mode
        kwargs['sci_mode'] = sci_mode
    #TODO(zhiqiu): support linewidth
    core.set_printoptions(**kwargs)


def _to_sumary(var):
    edgeitems = DEFAULT_PRINT_OPTIONS.edgeitems

    if len(var.shape) == 0:
        return var
    elif len(var.shape) == 1:
        if var.shape[0] > 2 * edgeitems:
            return np.concatenate([var[:edgeitems], var[-edgeitems:]])
        else:
            return var
    else:
        # recursively handle all dimensions
        if var.shape[0] > 2 * edgeitems:
            begin = [x for x in var[:edgeitems]]
            end = [x for x in var[-edgeitems:]]
            return np.stack([_to_sumary(x) for x in (begin + end)])
        else:
            return np.stack([_to_sumary(x) for x in var])


def _format_item(np_var, max_width=0, signed=False):
    if np_var.dtype == np.float32 or np_var.dtype == np.float64 or np_var.dtype == np.float16:
        if DEFAULT_PRINT_OPTIONS.sci_mode:
            item_str = '{{:.{}e}}'.format(
                DEFAULT_PRINT_OPTIONS.precision).format(np_var)
        elif np.ceil(np_var) == np_var:
            item_str = '{:.0f}.'.format(np_var)
        else:
            item_str = '{{:.{}f}}'.format(
                DEFAULT_PRINT_OPTIONS.precision).format(np_var)
    else:
        item_str = '{}'.format(np_var)

    if max_width > len(item_str):
        if signed:  # handle sign character for tenosr with negative item
            if np_var < 0:
                return item_str.ljust(max_width)
            else:
                return ' ' + item_str.ljust(max_width - 1)
        else:
            return item_str.ljust(max_width)
    else:  # used for _get_max_width
        return item_str


def _get_max_width(var):
    max_width = 0
    signed = False
    for item in list(var.flatten()):
        if (not signed) and (item < 0):
            signed = True
        item_str = _format_item(item)
        max_width = max(max_width, len(item_str))

    return max_width, signed


def _format_tensor(var, sumary, indent=0, max_width=0, signed=False):
    edgeitems = DEFAULT_PRINT_OPTIONS.edgeitems

    if len(var.shape) == 0:
        # currently, shape = [], i.e., scaler tensor is not supported.
        # If it is supported, it should be formatted like this.
        return _format_item(var.item(0), max_width, signed)
    elif len(var.shape) == 1:
        if sumary and var.shape[0] > 2 * edgeitems:
            items = [
                _format_item(item, max_width, signed)
                for item in list(var)[:DEFAULT_PRINT_OPTIONS.edgeitems]
            ] + ['...'] + [
                _format_item(item, max_width, signed)
                for item in list(var)[-DEFAULT_PRINT_OPTIONS.edgeitems:]
            ]
        else:
            items = [
                _format_item(item, max_width, signed) for item in list(var)
            ]
        s = ', '.join(items)
        return '[' + s + ']'
    else:
        # recursively handle all dimensions
        if sumary and var.shape[0] > 2 * edgeitems:
            vars = [
                _format_tensor(x, sumary, indent + 1, max_width, signed)
                for x in var[:edgeitems]
            ] + ['...'] + [
                _format_tensor(x, sumary, indent + 1, max_width, signed)
                for x in var[-edgeitems:]
            ]
        else:
            vars = [
                _format_tensor(x, sumary, indent + 1, max_width, signed)
                for x in var
            ]

        return '[' + (',' + '\n' * (len(var.shape) - 1) + ' ' *
                      (indent + 1)).join(vars) + ']'


def to_string(var, prefix='Tensor'):
    np_var = var.numpy()

    indent = len(prefix) + 1

    _template = "{prefix}(shape={shape}, dtype={dtype}, place={place}, stop_gradient={stop_gradient},\n{indent}{data})"

    tensor = var.value().get_tensor()
    if not tensor._is_initialized():
        return "Tensor(Not initialized)"

    if len(var.shape) == 0:
        size = 0
    else:
        size = 1
        for dim in var.shape:
            size *= dim

    sumary = False
    if size > DEFAULT_PRINT_OPTIONS.threshold:
        sumary = True

    max_width, signed = _get_max_width(_to_sumary(np_var))

    data = _format_tensor(
        np_var, sumary, indent=indent, max_width=max_width, signed=signed)

    return _template.format(
        prefix=prefix,
        shape=var.shape,
        dtype=convert_dtype(var.dtype),
        place=var._place_str,
        stop_gradient=var.stop_gradient,
        indent=' ' * indent,
        data=data)
