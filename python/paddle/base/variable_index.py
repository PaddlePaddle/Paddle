#   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

import itertools
import warnings
from functools import reduce

import numpy as np

import paddle

from . import core, unique_name

MAX_INTEGER = 2**31 - 1


def is_list_tuple(index, contain_type):
    def _is_list_tuple(item):
        if not (isinstance(item, (list, tuple)) or type(item) == contain_type):
            return False
        if isinstance(item, (tuple, list)):
            for s in item:
                if not _is_list_tuple(s):
                    return False
        return True

    if not isinstance(index, (tuple, list)):
        return False
    for s in index:
        if not _is_list_tuple(s):
            return False
    return True


def get_list_index_shape(var_dims, index_dims):
    var_dims_size = len(var_dims)
    index_dims_size = len(index_dims)

    out_dims_size = var_dims_size - index_dims[0] + index_dims_size - 1

    out_dims_shape = [1] * out_dims_size

    out_dims_shape[: index_dims_size - 1] = index_dims[1:]

    out_dims_shape[index_dims_size - 1 :] = var_dims[index_dims[0] :]
    return out_dims_shape


class SliceInfo:
    def __init__(self):
        self.pre_shape = None
        self.indexes = []
        self.dtype = None

    def update(self, index):
        if is_list_tuple(index, int) or isinstance(
            index, (paddle.base.Variable, np.ndarray)
        ):
            # convert index to Tensor
            if not isinstance(index, paddle.base.Variable):
                index = paddle.assign(index)

            if self.dtype is None:
                self.dtype = index.dtype
            else:
                if index.dtype != self.dtype:
                    raise IndexError(
                        "Data type of Tensor/List index should be same. The current data type is {}, but the previous data type is {}.".format(
                            index.dtype, self.dtype
                        )
                    )

            self.indexes.append(index)

            if self.pre_shape is None:
                self.pre_shape = index.shape
            else:
                if self.pre_shape != index.shape:
                    # broadcast
                    cur_shape = paddle.broadcast_shape(
                        self.pre_shape, index.shape
                    )
                    for i in range(len(self.indexes)):
                        self.indexes[i] = paddle.broadcast_to(
                            self.indexes[i], cur_shape
                        )
                self.pre_shape = self.indexes[-1].shape
        else:
            raise ValueError(
                f"Index should be list/tuple of int or Tensor, but received {index}."
            )

    def shape_stride(self, shape):
        s = [1] * len(shape)
        for i in range(len(shape) - 2, -1, -1):
            s[i] = shape[i + 1] * s[i + 1]

        return s

    def numel(self, shape):
        return reduce(lambda x, y: x * y, shape, 1)

    def get_offset_stride(self, tensor_shape):
        for index in self.indexes:
            if not isinstance(index, paddle.base.Variable):
                raise ValueError(
                    f"only support list/tensor index, but received {type(index)}."
                )

        if len(self.indexes) <= len(tensor_shape) or len(self.indexes) == 1:
            shape = paddle.stack(self.indexes)
            axes = list(range(1, len(self.pre_shape) + 1)) + [
                0,
            ]

        else:
            raise ValueError(
                "too many indices for tensor: tensor is {}-dimensional, but {} were indexed".format(
                    len(tensor_shape), self.pre_shape[0]
                )
            )

        shape_transpose = paddle.transpose(shape, axes)
        return shape_transpose

    def get_item(self, tensor):
        shape_transpose = self.get_offset_stride(tensor.shape)
        index = paddle.assign(shape_transpose)
        return paddle.gather_nd(tensor, index)

    def set_item(self, tensor_origin, value):
        if not isinstance(value, paddle.base.Variable):
            value = paddle.assign(value)
        tensor_type = None

        if tensor_origin.dtype in [
            core.VarDesc.VarType.FP32,
            core.VarDesc.VarType.FP64,
        ]:
            tensor = tensor_origin
        else:
            tensor_type = tensor_origin.dtype
            tensor = tensor_origin.astype(core.VarDesc.VarType.FP32)

        if value.dtype != tensor.dtype:
            value = value.astype(tensor.dtype)

        shape_transpose = self.get_offset_stride(tensor_origin.shape)
        index = paddle.assign(shape_transpose)

        gather_tensor_shape = get_list_index_shape(
            tensor.shape,
            [
                len(self.indexes),
            ]
            + list(self.indexes[-1].shape),
        )

        value_dims_bd = [
            1,
        ] * len(gather_tensor_shape)
        value_dims_bd[-len(value.shape) :] = list(value.shape)

        for i in range(len(gather_tensor_shape)):
            if not (
                len(value_dims_bd) == 0
                or value_dims_bd[i] == gather_tensor_shape[i]
                or value_dims_bd[i] == 1
            ):
                raise ValueError(
                    f"{value.shape} can not broadcast into {gather_tensor_shape}"
                )

        value_broadcast = paddle.broadcast_to(value, gather_tensor_shape)

        value_1d = value_broadcast.reshape(
            [-1] + gather_tensor_shape[len(index.shape) - 1 :]
        )

        index_1d = index.reshape([-1, index.shape[-1]])

        tensor_stride = paddle.assign(
            self.shape_stride(tensor.shape[: index.shape[-1]])
        )
        inds = []
        for i in range(index_1d.shape[0]):
            temp = (index_1d[i] * tensor_stride).sum()
            inds.append(temp)
        index_1d = paddle.stack(inds).reshape([-1])
        t_reshape = tensor.reshape([-1] + list(tensor.shape[index.shape[-1] :]))
        out = paddle.scatter(t_reshape, index_1d, value_1d)
        if tensor_type is not None:
            out = out.astype(tensor_type)
        tensor_origin = _setitem_impl_(
            tensor_origin, ..., out.reshape(tensor_origin.shape)
        )

        return tensor_origin


def replace_ellipsis(var, item):
    from .framework import Variable

    # Use slice(None) to replace Ellipsis.
    # For var, var.shape = [3,4,5,6]
    #
    #   var[..., 1:2] -> var[:, :, :, 1:2]
    #   var[0, ...] -> var[0]
    #   var[0, ..., 1:2] -> var[0, :, :, 1:2]

    item = list(item)

    # Remove Variable to skip bug when counting Ellipsis
    item_remove_var = [
        ele
        for ele in item
        if not isinstance(ele, (Variable, paddle.pir.OpResult, np.ndarray))
        and ele is not None
    ]
    ell_count = item_remove_var.count(Ellipsis)
    if ell_count == 0:
        return item
    elif ell_count > 1:
        raise IndexError("An index can only have a single ellipsis ('...')")

    ell_idx = item.index(Ellipsis)

    if ell_idx == len(item) - 1:
        return item[:-1]
    else:
        item[ell_idx : ell_idx + 1] = [slice(None)] * (
            len(var.shape) - len(item) + item.count(None) + 1
        )

    return item


def replace_ndarray_and_range(item):
    new_item = []
    for slice_item in item:
        if isinstance(slice_item, np.ndarray):
            new_item.append(paddle.assign(slice_item))
        elif isinstance(slice_item, range):
            new_item.append(list(slice_item))
        else:
            new_item.append(slice_item)
    return new_item


def replace_none(item):
    new_item = []
    none_axes = []
    for i, slice_item in enumerate(item):
        if slice_item is None:
            none_axes.append(i)
        else:
            new_item.append(slice_item)
    return new_item, none_axes


def is_integer_or_scalar_tensor(ele):
    from .framework import Variable

    if isinstance(ele, int):
        return True
    elif isinstance(ele, Variable):
        # NOTE(zoooo0820): For compatibility, if FLAGS_set_to_1d is set to True,
        # 1-D tensor is still treated as a scalar, which means basic indexing.
        # This will be removed in future.
        if paddle.get_flags('FLAGS_set_to_1d')['FLAGS_set_to_1d']:
            if len(ele.shape) == 1 and ele.shape[0] == 1:
                warnings.warn(
                    "1-D Tensor will be treat as advanced indexing in future version. Currently, 1-D Tensor means a scalar, not vector, and please modify it to 0-D Tensor. If advanced indexing is needed, please use `export FLAGS_set_to_1d=False` to set the flag."
                )
                return True
        if len(ele.shape) == 0 and ele.dtype != paddle.bool:
            return True
    elif isinstance(ele, paddle.pir.OpResult):
        if len(ele.shape) == 0 and ele.dtype != paddle.base.libpaddle.BOOL:
            return True
    return False


def is_bool_tensor(ele):
    from .framework import Variable

    if isinstance(ele, Variable) and ele.dtype == paddle.bool:
        return True
    elif (
        isinstance(ele, paddle.pir.OpResult)
        and ele.dtype == paddle.base.libpaddle.BOOL
    ):
        return True
    return False


def deal_attrs(attrs, attr, attr_name, tensor_attr_name, inputs, infer_flags):
    from .framework import Variable

    if paddle.utils._contain_var(attr):
        inputs[tensor_attr_name] = paddle.utils._convert_to_tensor_list(
            attr, dtype="int64"
        )
        for i, dim in enumerate(attr):
            if isinstance(dim, (Variable, paddle.pir.OpResult)):
                attrs[attr_name].append(-1)
                infer_flags[i] = -1
            else:
                attrs[attr_name].append(dim)
    else:
        attrs[attr_name] = attr


# the item is a tensor of bool
def get_value_for_bool_tensor(var, item):
    if len(item.shape) > len(var.shape):
        raise IndexError(
            "The dims of bool index doesn't match indexed array, "
            "the dims of bool index except to be equal or less "
            f"than {len(var.shape)}, but received {len(item.shape)}."
        )
    i = 0
    item_shape = item.shape
    while i < len(item.shape):
        dim_len = item_shape[i]
        if dim_len != -1 and var.shape[i] != -1 and dim_len != var.shape[i]:
            raise IndexError(
                "The dimension of bool index doesn't match indexed array along "
                "dimension {}, the target dimension is {}, but received {}.".format(
                    i, var.shape[i], dim_len
                )
            )
        i += 1
    empty_shape = [0] + list(var.shape[i:])

    def idx_not_empty(var, item):
        bool_2_idx = paddle.nonzero(item)
        return paddle.gather_nd(var, bool_2_idx)

    return paddle.static.nn.cond(
        item.any(),
        lambda: idx_not_empty(var, item),
        lambda: paddle.empty(empty_shape, var.dtype),
    )


def _setitem_for_tensor_array(var, item, value):
    """branches for tensor array setitem operation.
    A item can be a:
    (1) int/Variable, which is a simple number/variable such as [1], [-2]
    (2) Slice, which is represented by bounds such as [2:-1]
    (3) Tuple, which includes the above two cases such as [2:-1, 1]
    If item is case (1), we perform paddle.tensor.array_write,
    in other cases, we raise a NotImplementedError.
    """

    from .framework import Variable

    assert (
        not paddle.in_dynamic_mode()
    ), "setitem for tensor_array must be called in static graph mode."
    if isinstance(item, (Variable, int)):
        from paddle.jit.dy2static.variable_trans_func import to_static_variable
        from paddle.tensor import array_write

        item = paddle.cast(to_static_variable(item), dtype='int64')
        value = to_static_variable(value)
        return array_write(x=value, i=item, array=var)
    else:
        raise NotImplementedError(
            "Only support __setitem__ by Int/Variable in tensor_array, but gets {}".format(
                type(item)
            )
        )


def _setitem_impl_(var, item, value):
    from paddle.base import core

    from .framework import Variable, default_main_program

    if var.type == core.VarDesc.VarType.LOD_TENSOR_ARRAY:
        return _setitem_for_tensor_array(var, item, value)

    inputs = {'Input': var}

    # 1. Parse item
    if not isinstance(item, tuple):
        item = (item,)

    decrease_axes = []
    axes = []
    starts = []
    ends = []
    steps = []

    item = replace_ndarray_and_range(item)
    item = replace_ellipsis(var, item)
    item, none_axes = replace_none(item)
    slice_info = SliceInfo()
    dim = 0
    for _, slice_item in enumerate(item):
        if is_integer_or_scalar_tensor(slice_item) and not is_bool_tensor(
            slice_item
        ):
            decrease_axes.append(dim)
            start = slice_item
            end = slice_item + 1 if slice_item != -1 else MAX_INTEGER
            step = 1

        elif isinstance(slice_item, slice):
            start = slice_item.start
            end = slice_item.stop
            step = slice_item.step

            if start is None and end is None and step is None:
                dim += 1
                continue

            step = 1 if step is None else step

            if not isinstance(step, Variable) and step == 0:
                raise ValueError(
                    "When assign a value to a paddle.Tensor, step can not be 0, "
                    f"but received step is {step}."
                )

            if isinstance(step, Variable) and (start is None or end is None):
                raise ValueError(
                    "When assign a value to a paddle.Tensor, it's not supported that "
                    "the start or end is None when the type of step is paddle.Tensor."
                )

            if start is None:
                start = 0 if step > 0 else MAX_INTEGER

            if end is None:
                end = MAX_INTEGER if step > 0 else (0 - MAX_INTEGER)
        elif isinstance(slice_item, list):
            if is_list_tuple(slice_item, int):
                slice_info.update(slice_item)
                continue

            for i in slice_item:
                if not isinstance(i, bool):
                    raise TypeError(f"Doesn't support {type(i)} in index list.")

            if len(item) != 1:
                raise IndexError(
                    "When index contains a bool list, its length must be 1, but received {}.".format(
                        len(item)
                    )
                )

            idx_tensor = paddle.assign(slice_item)
            return set_value_for_bool_tensor(var, idx_tensor, value)

        elif isinstance(slice_item, Variable):
            if slice_item.dtype == core.VarDesc.VarType.BOOL:
                if len(item) != 1:
                    raise IndexError(
                        "When index contains a bool tensor, its length must be 1, but received {}.".format(
                            len(item)
                        )
                    )
                return set_value_for_bool_tensor(var, slice_item, value)
            else:
                slice_info.update(slice_item)
                continue
        else:
            raise IndexError(
                "Valid index accept int, slice, ellipsis, None, list of bool, Variable, "
                "but received {}.".format(slice_item)
            )

        axes.append(dim)
        starts.append(start)
        ends.append(end)
        steps.append(step)

        dim += 1
    if slice_info.indexes:
        if len(slice_info.indexes) != len(item):
            raise IndexError(
                "Valid index accept int or slice or ellipsis or list, but received {}.".format(
                    item
                )
            )
        return slice_info.set_item(var, value)
    attrs = {
        'axes': axes,
        'starts': starts,
        'ends': ends,
        'steps': steps,
        'decrease_axes': decrease_axes,
        'none_axes': none_axes,
    }

    if paddle.utils._contain_var(starts):
        inputs['StartsTensorList'] = paddle.utils._convert_to_tensor_list(
            starts
        )
        del attrs['starts']
    if paddle.utils._contain_var(ends):
        inputs['EndsTensorList'] = paddle.utils._convert_to_tensor_list(ends)
        del attrs['ends']
    if paddle.utils._contain_var(steps):
        inputs['StepsTensorList'] = paddle.utils._convert_to_tensor_list(steps)
        del attrs['steps']

    # 2. Parse value
    dtype = var.dtype
    attrs['dtype'] = dtype

    from .data_feeder import convert_dtype

    #  2.1 value is an integer, float or complex
    if isinstance(value, (bool, int, float, complex)):
        value = np.array([value]).astype(convert_dtype(dtype))

    #  2.2 value is a np.ndarray
    if isinstance(value, np.ndarray):
        shape = list(value.shape)
        values = value.ravel().tolist()
        attrs["values"] = values
        attrs["shape"] = shape

    elif isinstance(value, (Variable, core.eager.Tensor)):
        inputs["ValueTensor"] = value
    else:
        raise TypeError(
            "Only support to assign an integer, float, numpy.ndarray or "
            f"paddle.Tensor to a paddle.Tensor, but received {type(value)}"
        )

    if paddle.in_dynamic_mode():
        var._bump_inplace_version()
        output = var
    else:
        helper = paddle.base.layer_helper.LayerHelper('set_value', **locals())
        if helper.main_program.current_block_idx != 0:
            # not in global block, we should create a global variable.
            output = helper._create_global_variable_for_type_inference(
                dtype=var.dtype
            )
        else:
            output = helper.create_variable_for_type_inference(dtype=var.dtype)

    cur_block = default_main_program().current_block()
    cur_block.append_op(
        type="set_value",
        inputs=inputs,
        outputs={'Out': output},
        attrs=attrs,
        inplace_map={"Input": "Out"},
    )

    if not paddle.in_dynamic_mode():
        # map var to the new output
        from paddle.jit.dy2static.program_translator import ProgramTranslator

        ProgramTranslator.get_instance()._inplace_map.add(
            cur_block.program, var.desc.id(), output
        )

    return output


# the item is a tensor of bool
def set_value_for_bool_tensor(var, item, value):
    if len(item.shape) > len(var.shape):
        raise IndexError(
            "The dims of bool index doesn't match indexed array, "
            "the dims of bool index except to be equal or less "
            f"than {len(var.shape)}, but received {len(item.shape)}."
        )
    for i, dim_len in enumerate(item.shape):
        if dim_len != -1 and var.shape[i] != -1 and dim_len != var.shape[i]:
            raise IndexError(
                "The dimension of bool index doesn't match indexed array along "
                "dimension {}, the target dimension is {}, but received {}.".format(
                    i, var.shape[i], dim_len
                )
            )

    def idx_not_empty(var, item, value):
        from ..tensor import gather_nd, scatter_nd_add
        from .framework import Variable

        if not isinstance(value, Variable):
            value = paddle.assign(value).cast(var.dtype)

        idx = paddle.nonzero(item)
        gather_val = gather_nd(var, idx)
        gather_val_new = value - gather_val
        out = scatter_nd_add(var, idx, gather_val_new)
        var = _setitem_impl_(var, ..., out)
        return var

    def idx_is_empty(var):
        return var

    from paddle.static.nn import cond

    # If all the bool index is False, just do nothing
    var = cond(
        item.any(),
        lambda: idx_not_empty(var, item, value),
        lambda: idx_is_empty(var),
    )

    return var


def deal_advanced_index(ori_tensor, indices, is_for_setitem):
    """
    Transpose origin Tensor and advanced indices to the front.

    Returns:
        transed_tensor (Tensor): transposed tensor, corresbonding with advanced indices
        transed_index (List): advanced indices transed to the front
        trans_back_dim (List): order of axes to transpose back to original order. Only used in __setitem__.
        pos_of_new_dim (int):  axis of new dim in the result. Only used in __getitem__.
        rank_of_new_dim (int): rank of new dim in the result. Only used in __getitem__.
    """
    transed_dim = []
    transed_index = []

    # These flags indicates whether the result get by gather_nd requires a second transpose.
    # Only used in __getitem__.
    pos_of_new_dim = MAX_INTEGER
    rank_of_new_dim = 1

    for i, indice in enumerate(indices):
        if indice is not None:
            if not is_for_setitem:
                if i == 0:
                    # case 1: advanced indices at axis 0, the new dim will be at first.
                    pos_of_new_dim = 0
                if i > 0 and len(transed_dim) > 0 and transed_dim[-1] != i - 1:
                    # case 2: there are not adjacent advanced indices, the new dim will be at first.
                    pos_of_new_dim = 0
                else:
                    pos_of_new_dim = min(pos_of_new_dim, i)
                rank_of_new_dim = max(rank_of_new_dim, indice[1].ndim)
            transed_dim.append(i)
            transed_index.append(indice[1])
    for i in range(ori_tensor.ndim):
        if indices[i] is None:
            transed_dim.append(i)
    transed_tensor = ori_tensor.transpose(transed_dim)

    trans_back_dim = np.argsort(transed_dim).tolist() if is_for_setitem else []

    return (
        transed_tensor,
        transed_index,
        trans_back_dim,
        pos_of_new_dim,
        rank_of_new_dim,
    )


def parse_index(x, indices):
    advanced_index = [None] * 2 * len(x.shape)  # content is (dim, index)
    # for set_value / slice / strided_slice OP
    decrease_axes = []
    axes = []
    starts = []
    ends = []
    steps = []
    use_strided_slice = False
    has_advanced_index = False

    if not isinstance(indices, tuple):
        indices = (indices,)

    indices = replace_ndarray_and_range(indices)
    indices = replace_ellipsis(x, indices)
    indices, none_axes = replace_none(indices)

    is_tensor_array = (
        hasattr(x, "desc")
        and x.desc.type() == core.VarDesc.VarType.LOD_TENSOR_ARRAY
    )

    estimated_dim = 0
    for dim, slice_item in enumerate(indices):
        start, end, step = None, None, None
        if is_integer_or_scalar_tensor(slice_item):
            if (
                not is_tensor_array
                and isinstance(slice_item, int)
                and x.shape[dim] is not None
                and x.shape[dim] >= 0
                and slice_item >= x.shape[dim]
            ):
                # For python, if users write a, b = var, the __getitem__
                # method will iterate through 0, 1, 2 ... until __getitem__
                # throws an IndexError, then stop. The var[0], var[1] will
                # be given to a, b respectively. If more values are given,
                # the unpack size would cause error.
                # We raises IndexError here to support grammar like `a, b = var`
                raise IndexError(
                    "slice_item %d at dim %d should be >= 0 and < x.shape[%d]: %d"
                    % (slice_item, dim, dim, x.shape[dim])
                )
            # not calculate result to reduce call times for slice OP.
            decrease_axes.append(dim)
            start = slice_item
            step = 1
            end = slice_item + 1 if slice_item != -1 else MAX_INTEGER
        elif isinstance(slice_item, bool):
            # single bool is advanced-indexing
            none_axes.append(dim)
            estimated_dim += 1
            advanced_index[estimated_dim] = (
                estimated_dim,
                paddle.to_tensor(slice_item),
            )
            has_advanced_index = True
        elif isinstance(slice_item, slice):
            start = slice_item.start
            end = slice_item.stop
            step = slice_item.step
            estimated_dim += 1

            if start is None and end is None and step is None:
                continue

            step = 1 if step is None else step
            if start is None:
                start = 0 if step > 0 else MAX_INTEGER
            if end is None:
                end = MAX_INTEGER if step > 0 else -1

        elif isinstance(slice_item, (list, tuple)):
            advanced_index[estimated_dim] = (
                estimated_dim,
                paddle.to_tensor(slice_item),
            )

            if (
                advanced_index[estimated_dim][1].dtype == paddle.bool
                and len(slice_item) != x.shape[dim]
            ):
                raise IndexError(
                    "The shape of boolean index {} did not match indexed tensor {} along axis {}".format(
                        len(slice_item), x.shape[dim], dim
                    )
                )

            has_advanced_index = True
            estimated_dim += 1

        elif isinstance(
            slice_item, (paddle.base.Variable, paddle.pir.OpResult)
        ):
            # In this case, the Variable is not 0-dim Tensor and will be treated as advanced-indexing.
            if (
                slice_item.dtype == paddle.bool
                or slice_item.dtype == paddle.base.libpaddle.BOOL
            ):
                if slice_item.ndim == 0:
                    # 0-D bool Tensor, same as single PY-bool.
                    none_axes.append(dim)

                elif slice_item.shape[0] != x.shape[dim]:
                    raise IndexError(
                        "The shape of boolean index {} did not match indexed tensor {} along axis {}".format(
                            slice_item.shape[0], x.shape[dim], dim
                        )
                    )
            advanced_index[estimated_dim] = (estimated_dim, slice_item)
            has_advanced_index = True
            estimated_dim += 1

        else:
            raise IndexError(
                "Valid index accept int / bool / slice / ellipsis / list / Tuple / Ndarray / Tensor, but received {}.".format(
                    slice_item
                )
            )
        if not (start is None or end is None or step is None):
            starts.append(start)
            ends.append(end)
            steps.append(step)
            axes.append(dim)
            use_strided_slice = (
                True
                if (
                    isinstance(
                        step, (paddle.base.Variable, paddle.pir.OpResult)
                    )
                    or step != 1
                )
                else use_strided_slice
            )
    return (
        starts,
        ends,
        steps,
        axes,
        none_axes,
        decrease_axes,
        advanced_index,
        has_advanced_index,
        use_strided_slice,
    )


def _setitem_static(x, indices, values):
    """
    In dynamic mode, this function will modify the value at input tensor, returning same Tensor as input.
    But it will return a new Tensor with assigned value in static mode.

    Args:
        x(Tensor): Tensor to be set value.
        indices(int|slice|None|Tensor|List|Tuple...): Indices, used to indicate the position of the element to be fetched.
        values(Tensor|Number|Ndarray): values to be assigned to the x.
    """
    from .framework import Variable, default_main_program

    if x.type == paddle.base.core.VarDesc.VarType.LOD_TENSOR_ARRAY:
        return _setitem_for_tensor_array(x, indices, values)

    # step1: parsing the index and recording them
    (
        starts,
        ends,
        steps,
        axes,
        none_axes,
        decrease_axes,
        advanced_index,
        has_advanced_index,
        use_strided_slice,
    ) = parse_index(x, indices)

    inputs = {'Input': x}
    attrs = {
        'axes': axes,
        'starts': starts,
        'ends': ends,
        'steps': steps,
        'decrease_axes': decrease_axes,
        'none_axes': none_axes,
    }

    value_tensor = None
    StartsTensorList = None
    EndsTensorList = None
    StepsTensorList = None

    if paddle.utils._contain_var(starts):
        StartsTensorList = paddle.utils._convert_to_tensor_list(starts)
        inputs['StartsTensorList'] = StartsTensorList
        del attrs['starts']

    if paddle.utils._contain_var(ends):
        EndsTensorList = paddle.utils._convert_to_tensor_list(ends)
        inputs['EndsTensorList'] = EndsTensorList
        del attrs['ends']
    if paddle.utils._contain_var(steps):
        StepsTensorList = paddle.utils._convert_to_tensor_list(steps)
        inputs['StepsTensorList'] = StepsTensorList
        del attrs['steps']

    if not has_advanced_index:
        # step2. Parse values
        dtype = x.dtype
        attrs['dtype'] = dtype

        from .data_feeder import convert_dtype

        if isinstance(values, (bool, int, float, complex)):
            values = np.array([values]).astype(convert_dtype(dtype))

        if isinstance(values, np.ndarray):
            shape = list(values.shape)
            values = values.ravel().tolist()
            attrs["values"] = values
            attrs["shape"] = shape

        elif isinstance(values, Variable):
            values = values.astype(dtype)
            inputs["ValueTensor"] = values
            value_tensor = values

        else:
            raise TypeError(
                "Only support to assign an integer, float, numpy.ndarray or "
                f"paddle.Tensor to a paddle.Tensor, but received {type(values)}"
            )

        # step3.1: Only basic indexing, use OP set_value to set value.
        if paddle.in_dynamic_mode():
            return paddle._legacy_C_ops.set_value_(
                x,
                value_tensor,
                StartsTensorList,
                EndsTensorList,
                StepsTensorList,
                *itertools.chain.from_iterable(attrs.items()),
            )
        else:
            helper = paddle.base.layer_helper.LayerHelper(
                'set_value', **locals()
            )
            if helper.main_program.current_block_idx != 0:
                # not in global block, we should create a global variable.
                output = helper._create_global_variable_for_type_inference(
                    dtype=x.dtype
                )
            else:
                output = helper.create_variable_for_type_inference(
                    dtype=x.dtype
                )
            cur_block = default_main_program().current_block()
            cur_block.append_op(
                type="set_value",
                inputs=inputs,
                outputs={'Out': output},
                attrs=attrs,
                inplace_map={"Input": "Out"},
            )

            # map var to the new output
            paddle.jit.api.ProgramTranslator.get_instance()._inplace_map.add(
                cur_block.program, x.desc.id(), output
            )
            return output
    else:
        # step3.2: Case for there are advanced indexing.
        #   1. get __getitem__ result of basic indexing;
        #   2. transpose original tensor so that the axis with advanced indexing will come to the first;
        #   3. assign values to the sliced result by index_put OP;
        #   4. transpose back and assign the result to original tensor by set_value OP.

        sub_tensor = get_tensor_with_basic_indexing(
            x,
            axes,
            starts,
            ends,
            steps,
            decrease_axes,
            none_axes,
            use_strided_slice,
        )
        (
            transed_sub_tensor,
            adjusted_advanced_index,
            transback_dim,
            _,
            _,
        ) = deal_advanced_index(sub_tensor, advanced_index, True)
        if not isinstance(values, Variable):
            values = paddle.assign(values).astype(transed_sub_tensor.dtype)

        if values.dtype != transed_sub_tensor.dtype:
            values = values.astype(transed_sub_tensor.dtype)

        if paddle.in_dynamic_mode():
            # NOTE(zoooo0820): directly return result instead of another set_value, after backward bug fixed.
            transed_sub_tensor = transed_sub_tensor.index_put_(
                adjusted_advanced_index, values
            )
        else:
            transed_sub_tensor = transed_sub_tensor.index_put(
                adjusted_advanced_index, values
            )

        transback_sub_tensor = transed_sub_tensor.transpose(transback_dim)
        inputs["ValueTensor"] = transback_sub_tensor

        if paddle.in_dynamic_mode():
            x._bump_inplace_version()
            output = x
        else:
            helper = paddle.base.layer_helper.LayerHelper(
                'set_value', **locals()
            )
            if helper.main_program.current_block_idx != 0:
                # not in global block, we should create a global variable.
                output = helper._create_global_variable_for_type_inference(
                    dtype=x.dtype
                )
            else:
                output = helper.create_variable_for_type_inference(
                    dtype=x.dtype
                )
        cur_block = default_main_program().current_block()
        cur_block.append_op(
            type="set_value",
            inputs=inputs,
            outputs={'Out': output},
            attrs=attrs,
            inplace_map={"Input": "Out"},
        )

        if not paddle.in_dynamic_mode():
            # map var to the new output
            paddle.jit.api.ProgramTranslator.get_instance()._inplace_map.add(
                cur_block.program, x.desc.id(), output
            )
        return output


def get_tensor_with_basic_indexing(
    x, axes, starts, ends, steps, decrease_axes, none_axes, use_strided_slice
):
    from .dygraph.base import in_to_static_mode

    if in_to_static_mode() and hasattr(x, "is_view_var"):
        x.is_view_var = True

    if len(axes) == 0:
        out = x
    else:
        op_type = "strided_slice" if use_strided_slice else "slice"
        inputs = {'Input': [x]}
        attrs = {
            'axes': axes,
            'starts': [],
            'ends': [],
            'decrease_axis': decrease_axes,
        }
        if use_strided_slice:
            attrs['strides'] = []
        infer_flags = [1] * len(axes)
        deal_attrs(
            attrs, starts, "starts", "StartsTensorList", inputs, infer_flags
        )
        deal_attrs(attrs, ends, "ends", "EndsTensorList", inputs, infer_flags)
        deal_attrs(
            attrs, steps, "strides", "StridesTensorList", inputs, infer_flags
        )
        attrs['infer_flags'] = infer_flags

        from . import in_dynamic_or_pir_mode, in_pir_mode

        if in_dynamic_or_pir_mode():
            if "StartsTensorList" in inputs.keys():
                st = inputs['StartsTensorList']
            else:
                st = attrs['starts']
            if "EndsTensorList" in inputs.keys():
                end = inputs['EndsTensorList']
            else:
                end = attrs['ends']
            if "StridesTensorList" in inputs.keys():
                stride = inputs['StridesTensorList']
            else:
                stride = attrs['strides']
            if use_strided_slice:
                out = paddle._C_ops.strided_slice(x, axes, st, end, stride)
                if len(decrease_axes) > 0:
                    out = paddle._C_ops.squeeze(out, decrease_axes)
            else:
                if in_pir_mode():
                    if isinstance(st, (list, tuple)):
                        if paddle.utils._contain_var(st):
                            st = paddle.utils.get_int_tensor_list(st)
                    if isinstance(end, (list, tuple)):
                        if paddle.utils._contain_var(end):
                            end = paddle.utils.get_int_tensor_list(end)
                out = paddle._C_ops.slice(
                    x,
                    axes,
                    st,
                    end,
                    attrs['infer_flags'],
                    attrs['decrease_axis'],
                )
        else:
            from .framework import default_main_program

            target_block = default_main_program().current_block()

            slice_out_var = target_block.create_var(
                name=unique_name.generate_with_ignorable_key(
                    x.name + "_" + op_type
                ),
                dtype=x.dtype,
            )
            target_block.append_op(
                type=op_type,
                inputs=inputs,
                outputs={'Out': [slice_out_var]},
                attrs=attrs,
            )
            out = slice_out_var
    # NOTE(zoooo0820): When all axes are decreased, the output will be 1-D
    # with FLAGS_set_to_1d=True. In this case, one `None` should be pop out,
    # otherwise the output shape will be not correct.
    set_to_1d = paddle.get_flags('FLAGS_set_to_1d')['FLAGS_set_to_1d']
    if set_to_1d and len(decrease_axes) == len(x.shape):
        warnings.warn(
            "Warning: In Tensor '__getitem__', if the number of scalar elements in the index is equal to the rank of the Tensor, the output should be 0-D. In order to be consistent with the behavior of previous versions, it will be processed to 1-D. But it is not correct and will be removed in release 2.6. If 1-D is still wanted, please modify the index element from scalar to slice (e.g. 'x[i]' => 'x[i:i+1]')."
        )
        none_axes = none_axes[1:]

    if len(none_axes) > 0:
        # Deal with cases that decrease_axes is not empty
        # For example:
        # # x.shape: (2,3,4)
        # out = x[0, 0:2, None] # out.shape : (2, 1, 4)
        for idx, axis in enumerate(none_axes):
            l = len([i for i in decrease_axes if i < axis])
            new_axis = axis - l
            none_axes[idx] = new_axis

        out = paddle.unsqueeze(out, axis=none_axes)

    if in_to_static_mode() and hasattr(out, "is_view_var"):
        out.is_view_var = True
    return out


def _getitem_static(x, indices):
    """
    Args:
        x(Tensor): Tensor to be indexing.
        indices(int|slice|None|Tensor|List|Tuple...): Indices, used to indicate the position of the element to be fetched.
    """
    # step1: parsing the index and recording them
    (
        starts,
        ends,
        steps,
        axes,
        none_axes,
        decrease_axes,
        advanced_index,
        has_advanced_index,
        use_strided_slice,
    ) = parse_index(x, indices)

    # step2: Dealing with basic indexing
    out = get_tensor_with_basic_indexing(
        x,
        axes,
        starts,
        ends,
        steps,
        decrease_axes,
        none_axes,
        use_strided_slice,
    )

    # step3: Dealing with advanced indexing
    if has_advanced_index:
        (
            transed_tensor,
            adjusted_advanced_index,
            _,
            pos_of_new_dim,
            rank_of_new_dim,
        ) = deal_advanced_index(out, advanced_index, False)

        # TODO(zooooo0820): Replacing gather_nd to another advanded OP for handling of mixed indexes more efficiently
        if (
            len(adjusted_advanced_index) == 1
            and adjusted_advanced_index[0].dtype == paddle.bool
        ):
            # Note: now slice not support 0-size Tensor, so only one bool tensor can return empty 0-size.
            out = get_value_for_bool_tensor(
                transed_tensor, adjusted_advanced_index[0]
            )
        else:
            adjusted_advanced_index = parse_bool_and_broadcast_indices(
                adjusted_advanced_index
            )
            advanced_index_tensor = paddle.stack(
                adjusted_advanced_index, axis=-1
            )
            out = paddle.gather_nd(transed_tensor, advanced_index_tensor)

        if pos_of_new_dim != 0:
            perm = (
                list(range(pos_of_new_dim, pos_of_new_dim + rank_of_new_dim))
                + list(range(0, pos_of_new_dim))
                + list(range(pos_of_new_dim + rank_of_new_dim, out.ndim))
            )
            out = out.transpose(perm)

    return out


def parse_bool_and_broadcast_indices(indices):
    # deal with multiple Tensors and translating bool tensor to int tensor.
    # In static mode, bool-tensor cannot be broadcasted since its corressponding int tensor's shape cannot be infered.
    for i, indice in enumerate(indices):
        if indice.dtype == paddle.bool:
            indices[i] = paddle.nonzero(indice)[:, 0]
    if len(indices) > 1:
        indices = paddle.broadcast_tensors(indices)
    return indices
