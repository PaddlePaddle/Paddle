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
    precision = 4
    threshold = 1000
    edgeitems = 3
    sci_mode = False


DEFAULT_PRINT_OPTIONS = PrintOptions()


def set_printoptions(precision=None,
                     threshold=None,
                     edgeitems=None,
                     sci_mode=None):
    """Set the printing options for Tensor.
    NOTE: The function is similar with numpy.set_printoptions()

    Args:
        precision (int, optional): Number of digits of the floating number, default 6.
        threshold (int, optional): Total number of elements printed, default 1000.
        edgeitems (int, optional): Number of elements in summary at the begining and end of each dimension, defalt 3.
        sci_mode (bool, optional): Format the floating number with scientific notation or not, default False.
    
    Returns:
        None.

    Examples:
        .. code-block:: python

            import paddle

            paddle.manual_seed(10)
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


def _format_item(np_var):
    if np_var.dtype == np.float32 or np_var.dtype == np.float64 or np_var.dtype == np.float16:
        if DEFAULT_PRINT_OPTIONS.sci_mode:
            return '{{:.{}e}}'.format(DEFAULT_PRINT_OPTIONS.precision).format(
                np_var)
        elif np.ceil(np_var) == np_var:
            return '{:.0f}.'.format(np_var)
        else:
            return '{{:.{}f}}'.format(DEFAULT_PRINT_OPTIONS.precision).format(
                np_var)
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
            ] + ['...'] + [
                _format_tensor(x, sumary, indent + 1) for x in var[-edgeitems:]
            ]
        else:
            vars = [_format_tensor(x, sumary, indent + 1) for x in var]

        return '[' + (',' + '\n' * (len(var.shape) - 1) + ' ' *
                      (indent + 1)).join(vars) + ']'


def to_string(var):
    _template = "  - place: {place}\n  - shape: {shape}\n  - layout: {layout}\n  - dtype: {dtype}\n  - data: {data}"

    tensor = var.value().get_tensor()
    if not tensor._is_initialized():
        return "Not initialized"

    if len(var.shape) == 0:
        size = 0
    else:
        size = 1
        for dim in var.shape:
            size *= dim

    sumary = False
    if size > DEFAULT_PRINT_OPTIONS.threshold:
        sumary = True

    data = _format_tensor(var, sumary, indent=10)

    return _template.format(
        place=var._place_str,
        shape=var.shape,
        layout=tensor._layout(),
        dtype=convert_dtype(var.dtype),
        data=data)
