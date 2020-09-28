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

__all__ = ['set_printoptions', ]


class PrintOptions(object):
    precision = 4
    threshold = 100
    edgeitems = 3
    linewidth = 75


DEFAULT_PRINT_OPTIONS = PrintOptions()


def set_printoptions(precision=None, threshold=None, edgeitems=None):
    """Set the printing options for Tensor.
    NOTE: The function is similar with numpy.set_printoptions()

    Args:
        precision (int, optional): Number of digits of the floating number, default 6.
        threshold (int, optional): Total number of elements printed, default 1000.
        edgeitems (int, optional): Number of elements in summary at the begining and end of each dimension, defalt 3.
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

    core.set_printoptions(**kwargs)


def _to_sumary(var):
    edgeitems = DEFAULT_PRINT_OPTIONS.edgeitems

    if len(var.shape) == 0:
        return var
    elif len(var.shape) == 1:
        if var.shape[0] > 2 * edgeitems:
            return paddle.concat([var[:edgeitems], var[-edgeitems:]])
        else:
            return var
    else:
        # recursively handle all dimensions
        if var.shape[0] > 2 * edgeitems:
            begin = [x for x in var[:edgeitems]]
            end = [x for x in var[-edgeitems:]]
            return paddle.stack([_to_sumary(x) for x in (begin + end)])
        else:
            return paddle.stack([_to_sumary(x) for x in var])


def _format_item(var):
    if var.dtype == np.float32 or var.dtype == np.float64:
        return '{{:.{}f}}'.format(DEFAULT_PRINT_OPTIONS.precision).format(var)
    else:
        return '{}'.format(var)


def _format_tensor(var, sumary, indent=0):
    edgeitems = DEFAULT_PRINT_OPTIONS.edgeitems

    if len(var.shape) == 0:
        return _format_item(var.numpy.items(0))
    elif len(var.shape) == 1:
        if sumary and var.shape[0] > 2 * edgeitems:
            items = [
                _format_item(item)
                for item in list(var.numpy())[:DEFAULT_PRINT_OPTIONS.edgeitems]
            ] + ['...'] + [
                _format_item(item)
                for item in list(var.numpy())[-DEFAULT_PRINT_OPTIONS.edgeitems:]
            ]
        else:
            items = [_format_item(item) for item in list(var.numpy())]
        #elements_per_line = max(1, int(math.floor((PRINT_OPTS.linewidth - indent) / (element_length))))
        s = ', '.join(items)
        return '[' + s + ']'
    else:
        # recursively handle all dimensions
        if sumary and var.shape[0] > 2 * edgeitems:
            vars = [
                _format_tensor(x, sumary, indent + 1) for x in var[:edgeitems]
            ] + ['...'
                 ] + [_format_tensor(x, indent + 1) for x in var[-edgeitems:]]
        else:
            vars = [_format_tensor(x, sumary, indent + 1) for x in var]

        return '[' + (',' + '\n' * (len(var.shape) - 1) + ' ' *
                      (indent + 1)).join(vars) + ']'


def to_string(var):

    if len(var.shape) == 0:
        size = 0
    else:
        size = 1
        for dim in var.shape:
            size *= dim

    sumary = False
    if size > DEFAULT_PRINT_OPTIONS.threshold:
        sumary = True

    return _format_tensor(var, sumary)
