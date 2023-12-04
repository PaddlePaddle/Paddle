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

import warnings

import numpy as np

import paddle

from . import core, unique_name

MAX_INTEGER = 2**31 - 1


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

    if type(ele) is int:
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

    bool_2_idx = paddle.nonzero(item)
    return paddle.gather_nd(var, bool_2_idx)


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
    dim = 0
    for i, slice_item in enumerate(indices):
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
            dim += 1
        elif isinstance(slice_item, bool):
            # single bool is advanced-indexing
            none_axes.append(dim)
            advanced_index[estimated_dim] = (
                estimated_dim,
                paddle.to_tensor([slice_item]),
            )
            has_advanced_index = True
            estimated_dim += 1
        elif isinstance(slice_item, slice):
            start = slice_item.start
            end = slice_item.stop
            step = slice_item.step
            estimated_dim += 1
            dim += 1
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
            dim += 1
        elif isinstance(slice_item, paddle.base.Variable):
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
            dim += 1
        elif isinstance(slice_item, paddle.pir.OpResult):
            # In this case, the Variable is not 0-dim Tensor and will be treated as advanced-indexing.
            if slice_item.dtype == paddle.pir.core.DataType.BOOL:
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
            dim += 1
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
            axes.append(dim - 1)
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
    from . import in_dynamic_or_pir_mode
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
    shape = None

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

        elif isinstance(values, (Variable, paddle.pir.Value)):
            values = values.astype(dtype)
            inputs["ValueTensor"] = values
            value_tensor = values

        else:
            raise TypeError(
                "Only support to assign an integer, float, numpy.ndarray or "
                f"paddle.Tensor to a paddle.Tensor, but received {type(values)}"
            )

        # step3.1: Only basic indexing, use OP set_value to set value.
        if in_dynamic_or_pir_mode():
            if value_tensor is None:
                return paddle._C_ops.set_value_(
                    x,
                    starts,
                    ends,
                    steps,
                    axes,
                    decrease_axes,
                    none_axes,
                    shape,
                    values,
                )
            else:
                return paddle._C_ops.set_value_with_tensor_(
                    x,
                    value_tensor,
                    starts,
                    ends,
                    steps,
                    axes,
                    decrease_axes,
                    none_axes,
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
        if not isinstance(values, (Variable, paddle.pir.Value)):
            values = paddle.assign(values).astype(transed_sub_tensor.dtype)

        if values.dtype != transed_sub_tensor.dtype:
            values = values.astype(transed_sub_tensor.dtype)

        if in_dynamic_or_pir_mode():
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

        if in_dynamic_or_pir_mode():
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

        if not in_dynamic_or_pir_mode():
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

            if len(adjusted_advanced_index) > 1:
                advanced_index_tensor = paddle.stack(
                    adjusted_advanced_index, axis=-1
                )
            else:
                # fast path for single bool tensor, since stack is much slower than unsuqeeze
                advanced_index_tensor = adjusted_advanced_index[0].unsqueeze(-1)

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
