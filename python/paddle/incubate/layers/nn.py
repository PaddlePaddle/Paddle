# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
"""
incubate layers just related to the neural network.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Literal, overload

import numpy as np

import paddle
from paddle import _C_ops, _legacy_C_ops
from paddle.base import core, unique_name
from paddle.base.data_feeder import (
    check_dtype,
    check_type,
    check_variable_and_dtype,
)
from paddle.base.framework import (
    Variable,
    convert_np_dtype_to_dtype_,
    in_dynamic_or_pir_mode,
)
from paddle.base.layer_helper import LayerHelper
from paddle.base.param_attr import ParamAttr
from paddle.framework import in_pir_mode

if TYPE_CHECKING:
    from paddle import Tensor
    from paddle._typing import DTypeLike, ParamAttrLike

__all__ = []


def fused_seqpool_cvm(
    input: Tensor,
    pool_type: Literal['sum'],
    cvm: Tensor,
    pad_value: float = 0.0,
    use_cvm: bool = True,
    cvm_offset: int = 2,
) -> Tensor:
    """
    :api_attr: Static Graph

    This OP is the fusion of sequence_pool and continuous_value_model op.

    **Note:** The Op only receives List of DenseTensor as input, only support SUM pooling now.

    Args:
        input(Tensor): Input is List of DenseTensor.
        pool_type(str): pooling type, only support SUM pooling now.
        cvm(Tensor): cvm Tensor.
        pad_value(float, optional): padding value of sequence pool. Default: 0.0.
        use_cvm(bool, optional): use cvm or not. Default: True.
        cvm_offset(int, optional): cvm offset. Default: 2, which means cvm contains show, click.

    Returns:
        Tensor : The tensor storing sequence pool and cvm of input.

    Examples:
        .. code-block:: python

            >>> import paddle
            >>> paddle.enable_static()

            >>> data = paddle.static.data(name='x', shape=[-1, 1], dtype='int64', lod_level=1)
            >>> data2 = paddle.static.data(name='y', shape=[-1, 1], dtype='int64', lod_level=1)
            >>> inputs = [data, data2]
            >>> embs = paddle.incubate.layers.nn._pull_box_sparse(input=inputs, size=11, is_distributed=True, is_sparse=True)

            >>> label = paddle.static.data(name="label", shape=[-1, 1], dtype="int64", lod_level=1)
            >>> ones = paddle.static.data(name="ones", shape=[-1, 1], dtype="int64", lod_level=1)
            >>> show_clk = paddle.cast(paddle.concat([ones, label], axis=1), dtype='float32')
            >>> show_clk.stop_gradient = True

            >>> cvms = paddle.incubate.layers.fused_seqpool_cvm(embs, 'sum', show_clk)


    """
    helper = LayerHelper('fused_seqpool_cvm', **locals())

    if pool_type.upper() != 'SUM':
        raise ValueError(
            "fused_seqpool_cvm only support SUM pooling now, and your type is: "
            + pool_type
        )

    check_type(input, 'input', list, 'fused_seqpool_cvm')
    if isinstance(input, list):
        for _input in input:
            check_variable_and_dtype(
                _input, 'input', ['float32'], 'fused_seqpool_cvm'
            )

    dtype = helper.input_dtype()
    inputs = helper.multiple_input()
    outs = [
        helper.create_variable_for_type_inference(dtype)
        for i in range(len(inputs))
    ]

    helper.append_op(
        type="fused_seqpool_cvm",
        inputs={"X": inputs, "CVM": cvm},
        outputs={"Out": outs},
        attrs={
            "pooltype": pool_type.upper(),
            "pad_value": pad_value,
            "use_cvm": use_cvm,
            "cvm_offset": cvm_offset,
        },
    )

    return outs


def search_pyramid_hash(
    input: Tensor,
    num_emb: int,
    space_len: int,
    pyramid_layer: int,
    rand_len: int,
    drop_out_percent: float,
    is_training: bool,
    use_filter: bool,
    white_list_len: int,
    black_list_len: int,
    seed: int,
    lr: float,
    param_attr: ParamAttrLike | None = None,
    param_attr_wl: ParamAttrLike | None = None,
    param_attr_bl: ParamAttrLike | None = None,
    name: str | None = None,
    distribute_update_vars: list[str] | None = None,
    dtype: DTypeLike = 'float32',
) -> Tensor:
    """
    **Pyramid hash embedding**

    Args:
        input (Tensor): DenseTensor<int32> Tensor contained the IDs' information.
        num_emb (int): The embedding size of output.
        space_len (int): The length of pyramid hash embedding space.
        pyramid_layer (int): The number of pyramid layers. It should be greater than 2.
        rand_len (int): The minimum length of pyramid hash cell.
        drop_out_percent (float): The probability of dropping out the input token randomly.
            It should satisfy: [0., 1.].
        is_training (bool): Whether in training or testing phrase.
        use_filter (bool): If set True, the white filter and black filter should be given by
            :attr:`param_attr_wl` and :attr:`param_attr_bl` .
        white_list_len (int): If set :math:`white_list_len>0` , white filter with shape [white_list_len, 1]
            should be provided by param_attr_wl.
        black_list_len (int): If set :math:`black_list_len>0` , black filter with shape [black_list_len, 1]
            should be provided by param_attr_bl.
        seed (int): The number of random seed.
        lr (float): The learning rate of weight created by :attr:`param_attr` with shape [space_len+rand_len, 1]
            in this layer.
        param_attr (ParamAttr|None, optional): To specify the weight parameter property. Default: None, which means the
            default weight parameter property is used. See usage for details in :ref:`api_paddle_ParamAttr` .
        param_attr_wl (ParamAttr|None, optional): Specified parameters of white filter. Default: None.
        param_attr_bl (ParamAttr|None, optional): Specified parameters of black filter. Default: None.
        distribute_update_vars(list[ParamAttr.name]|None, optional): Decided which params should be updated in distribute training.
            Used in Distribute Transpiler to create a trainer/server program. Default: None.
        name (str|None, optional): The default value is None.  Normally there is no need for user to set this property.
            For more information, please refer to :ref:`api_guide_Name` . Default: None.
        dtype (str, optional): The data type of output Tensor, float32. Default: float32.

    Returns:
        Tensor: DenseTensor of pyramid hash embedding.
    """
    helper = LayerHelper('search_pyramid_hash', **locals())

    w_shape = [space_len + rand_len, 1]
    w = helper.create_parameter(
        attr=param_attr, shape=w_shape, dtype=dtype, is_bias=False
    )
    w.stop_gradient = True

    input_vars = {'X': input, 'W': w}
    if white_list_len > 0:
        wl_shape = [white_list_len, 1]
        white_list = helper.create_parameter(
            attr=param_attr_wl, shape=wl_shape, dtype=dtype, is_bias=False
        )
        white_list.stop_gradient = True
        input_vars['WhiteList'] = white_list

    if black_list_len >= 0:
        bl_shape = [black_list_len, 1]
        black_list = helper.create_parameter(
            attr=param_attr_bl, shape=bl_shape, dtype=dtype, is_bias=False
        )
        black_list.stop_gradient = True
        input_vars['BlackList'] = black_list

    distribute_update_vars_str = ""
    if distribute_update_vars:
        assert isinstance(distribute_update_vars, list)
        special_name_list = []
        if param_attr:
            special_name_list.append(param_attr.name)
        if param_attr_wl:
            special_name_list.append(param_attr_wl.name)
        if param_attr_bl:
            special_name_list.append(param_attr_bl.name)
        for param in distribute_update_vars:
            if param not in special_name_list:
                raise ValueError(
                    f"Pyramid Hash layer didn't have parameter {param}"
                )
        distribute_update_vars_str = ",".join(distribute_update_vars)
    if in_dynamic_or_pir_mode():
        res, drop_pos = _C_ops.pyramid_hash(
            input_vars['X'],
            input_vars['W'],
            input_vars['WhiteList'],
            input_vars['BlackList'],
            num_emb,
            space_len,
            pyramid_layer,
            rand_len,
            drop_out_percent,
            int(is_training),
            use_filter,
            white_list_len,
            black_list_len,
            seed,
            lr,
            distribute_update_vars_str,
        )
        return res
    else:
        res = helper.create_variable_for_type_inference(dtype)
        drop_pos = helper.create_variable_for_type_inference(dtype)
        x_temp_out = helper.create_variable_for_type_inference(dtype)
        helper.append_op(
            type='pyramid_hash',
            inputs=input_vars,
            outputs={"Out": res, "X_Temp_Out": x_temp_out, 'DropPos': drop_pos},
            attrs={
                'num_emb': num_emb,
                'space_len': space_len,
                'pyramid_layer': pyramid_layer,
                'rand_len': rand_len,
                'drop_out_percent': drop_out_percent,
                'is_training': is_training,
                'use_filter': use_filter,
                'white_list_len': white_list_len,
                'black_list_len': black_list_len,
                'seed': seed,
                'lr': lr,
                'distribute_update_vars': distribute_update_vars_str,
            },
        )

        return res


def shuffle_batch(x: Tensor, seed: int | Tensor | None = None) -> Tensor:
    """
    This layer shuffle input tensor :attr:`x` . Normally, :attr:`x` is 2-D DenseTensor.

    :attr:`x` is a DenseTensor to be shuffled with shape :math:`[N_1, N_2, ..., N_k, D]` . Note that the last dim of input will not be shuffled.
    :math:`N_1 * N_2 * ... * N_k` numbers of elements with length :math:`D` will be shuffled randomly.

    Examples:

        .. code-block:: text

            Input:
              x.data = [[1, 2], [3, 4], [5, 6], [7, 8]]
              x.dims = [4, 2]

            Attrs:
              seed = 2019

            Output:
              Out.data =[[7, 8], [1, 2], [3, 4], [5, 6]]
              Out.dims = [4, 2]

    Args:
        x (Tensor): The input Tensor. The input Tensor is a N-D DenseTensor with type int, float32 or float64.
        seed (None|int|Tensor, optional): The start up seed. If set, seed will be set as the start up seed of shuffle engine.
            If not set(Default), start up seed of shuffle engine will be generated randomly. Default: None.

    Returns:
        Tensor: The shuffled DenseTensor with the same shape and lod as input.

    Examples:

        .. code-block:: python

            >>> import paddle
            >>> paddle.enable_static()
            >>> x = paddle.static.data(name="x", shape=[-1, 4])
            >>> out = paddle.incubate.layers.shuffle_batch(x)
    """
    helper = LayerHelper('shuffle_batch', **locals())

    out = helper.create_variable_for_type_inference(dtype=x.dtype)
    shuffle_idx = helper.create_variable_for_type_inference(dtype=np.int64)
    if seed is None and helper.main_program.random_seed != 0:
        seed = helper.main_program.random_seed
    if seed is None:
        seed = np.random.randint(-65536, 65535)
    op_attrs = {}
    if isinstance(seed, int):
        op_attrs["startup_seed"] = seed
        if in_pir_mode():
            seed = paddle.full([0], 0, "int64")
            out, _, _ = _C_ops.shuffle_batch(x, seed, op_attrs["startup_seed"])
            return out
        else:
            seed = helper.create_variable(
                name=unique_name.generate("shuffle_batch_seed"),
                dtype="int64",
                persistable=False,
            )
    if in_pir_mode():
        out, _, _ = _C_ops.shuffle_batch(x, seed, 0)
        return out
    helper.append_op(
        type='shuffle_batch',
        inputs={'X': x, 'Seed': seed},
        outputs={'Out': out, 'ShuffleIdx': shuffle_idx, 'SeedOut': seed},
        attrs=op_attrs,
    )
    return out


def partial_concat(
    input: list[Tensor], start_index: int = 0, length: int = -1
) -> Tensor:
    """
    **Partial Concat**
    This OP concatenates the inputs according to the start index and length. This
    OP exists in incubate layers, which means that it is not shown to the public.
    Only 2-D Tensor input is supported. Slice and concat can only be
    performed along the second dimension.

    .. code-block:: text

        Given:
            x = [[0, 1, 2],
                 [3, 4, 5]]
            y = [[6, 7 ,8],
                 [9, 10, 11]]
            output = partial_concat([x, y], start_index=0, length=2)

        We get:

            output = [[0, 1, 6, 7],
                      [3, 4, 9, 10]]

    Args:
        input(list): List of input Tensors with data type float32, float64, int32,
            int64, complex64, complex128.
        start_index(int32, optional): The start index of each instance for partial concatenation.
            Default is 0.
        length(int32, optional): The length of each instance for partial concatenation. Default is -1.
            Negative values for all elements after start_index.

    Returns:
        Tensor: A Tensor with the same data type as input's.

    Examples:
        .. code-block:: python

            >>> import paddle
            >>> x = paddle.randn(name="x", shape=[1,3], dtype="float32")
            >>> y = paddle.randn(name="y", shape=[1,3], dtype="float32")
            >>> concat = paddle.incubate.layers.partial_concat(
            ...     [x, y], start_index=0, length=2)
    """
    if not isinstance(input, list):
        warnings.warn(
            f"The type of input in partial_concat should be list, but received {type(input)}."
        )
        input = [input]
    for id, x in enumerate(input):
        check_variable_and_dtype(
            x,
            'input[' + str(id) + ']',
            [
                'float16',
                'float32',
                'float64',
                'uint16',
                'int32',
                'int64',
                'complex64',
                'complex128',
            ],
            'partial_concat',
        )
    check_type(start_index, 'start_index', (int), 'partial_concat')
    check_type(length, 'length', (int), 'partial_concat')
    inputs = {'X': input}
    attrs = {'start_index': start_index, 'length': length}
    helper = LayerHelper('partial_concat', **locals())
    out = helper.create_variable_for_type_inference(dtype=helper.input_dtype())
    helper.append_op(
        type='partial_concat',
        inputs=inputs,
        outputs={'Out': [out]},
        attrs=attrs,
    )
    return out


def partial_sum(
    input: list[Tensor], start_index: int = 0, length: int = -1
) -> Tensor:
    """
    **PartialSum**
    This Op can sum the vars by specifying the initial position(start_index) and length(length).
    This Op exists in incubate layers, which means that it is not shown to the public.
    Only 2-D Tensor input is supported. Slice and concat can only be
    performed along the second dimension.

    .. code-block:: text

        Given:
            x = [[0, 1, 2],
                 [3, 4, 5]]
            y = [[6, 7 ,8],
                 [9, 10, 11]]
            output = partial_sum([x, y], start_index=0, length=2)

        We get:

            output = [[6, 8],
                      [12, 14]]
    Args:
        input (list): List of input Tensors with data type float32, float64, int32,
            int64.
        start_index (int32, optional): The start index of each instance for partial sum. Default is 0.
        length (int32, optional): The length of each instance for partial sum. Default is -1.

    Returns:
        Tensor: A Tensor with the same data type as input's.

    Examples:
        .. code-block:: python

            >>> import paddle
            >>> paddle.enable_static()

            >>> x = paddle.static.data(name="x", shape=[2, 3], dtype="float32")
            >>> y = paddle.static.data(name="y", shape=[2, 3], dtype="float32")
            >>> sum = paddle.incubate.layers.partial_sum([x,y], start_index=0, length=2)
    """
    for id, x in enumerate(input):
        check_variable_and_dtype(
            x,
            'input[' + str(id) + ']',
            ['float32', 'float64', 'int32', 'int64'],
            'partial_sum',
        )

    inputs = {'X': input}
    attrs = {}
    attrs['start_index'] = start_index
    attrs['length'] = length
    helper = LayerHelper('partial_sum', **locals())
    out = helper.create_variable_for_type_inference(dtype=helper.input_dtype())
    helper.append_op(
        type='partial_sum', inputs=inputs, outputs={'Out': [out]}, attrs=attrs
    )
    return out


def tdm_child(
    x: Tensor,
    node_nums: int,
    child_nums: int,
    param_attr: ParamAttrLike | None = None,
    dtype: DTypeLike = 'int32',
) -> tuple[Tensor, Tensor]:
    """
    **Tdm Child**
     According to the input node_id on the given tree, return the corresponding child node_id and
      whether child is a leaf node by leaf_mask value.

    .. code-block:: text

        Given:
            tree[[0], [1, 2], [3, 4], [5, 6]] # A binary tree with seven nodes
            x = [[2], [3]]
            node_nums = 7
            child_nums = 2

        We get:
            child = [[5, 6],
                     [0, 0]]
            leaf_mask = [[1, 1],
                         [0, 0]]

    Args:
        x (Tensor): Tensor contained the node_id information, dtype support int32/int64.
        node_nums (int): Number of total nodes.
        child_nums (int): Maximum number of child nodes per node.
        param_attr (ParamAttr|None, optional): To specify the tdm-tree-info parameter property. Default: None, which means the
            default weight parameter property is used. See usage for details in: ref: `api_paddle_ParamAttr`, should
            has shape (node_nums, 3 + child_nums), dtype support int32/int64.
            The dimension[1] of tdm-tree-info contains the following:
            1. Item_id (int, shape(1)), if node is a leaf node, give its item_id corresponding to node_id, else give 0.
            2. Layer_id (int, shape(1)), indicates which layer the node is on.
            3. Parent_id (int, shape(1)), node's parent node.
            4. Child_id (int, shape(child_nums)), all child node's node_id of this node should be given.
            If the number of child nodes is insufficient, padding 0 until child nums equal to child_nums.
        dtype (str, optional): The data type of output child and leaf_mask, support int32/int64. Default: int32.

    Returns:
        tuple: A tuple including input node's child(Tensor) and leaf_mask(Tensor).
            If child is a leaf node, leaf_mask equal ot 1, otherwise equal to 0.

    Examples:
        .. code-block:: python

            >>> import paddle
            >>> import numpy as np
            >>> paddle.enable_static()

            >>> x = paddle.static.data(name="x", shape=[None, 1], dtype="int32", lod_level=1)
            >>> tree_info = [[0,0,0,1,2],
            ...             [0,1,0,3,4],[0,1,0,5,6],
            ...             [0,2,1,0,0],[1,2,1,0,0],[2,2,2,0,0],[3,2,2,0,0]]
            >>> tree_info_np = np.array(tree_info)
            >>> tree_info_np = np.reshape(tree_info_np, (7,5))
            >>> node_nums = 7
            >>> child_nums = 2
            >>> child, leaf_mask  = paddle.incubate.layers.tdm_child(x, node_nums, child_nums,
            ...                     param_attr=paddle.ParamAttr(
            ...                     initializer=paddle.nn.initializer.Assign(tree_info_np)))

    """
    helper = LayerHelper("tdm_child", **locals())
    check_dtype(
        dtype, 'dtype', ['int32', 'int64'], 'paddle.incubate.layers.tdm_child'
    )
    c_dtype = convert_np_dtype_to_dtype_(dtype)
    tree_info = helper.create_parameter(
        attr=helper.param_attr,
        shape=[node_nums, 3 + child_nums],
        dtype=dtype,
        default_initializer=paddle.nn.initializer.Constant(0),
    )
    tree_info.stop_gradient = True

    if in_pir_mode():
        return _C_ops.tdm_child(x, tree_info, child_nums, c_dtype)

    child = helper.create_variable_for_type_inference(dtype=dtype)
    leaf_mask = helper.create_variable_for_type_inference(dtype=dtype)

    helper.append_op(
        type='tdm_child',
        inputs={'X': x, 'TreeInfo': tree_info},
        outputs={'Child': child, 'LeafMask': leaf_mask},
        attrs={'child_nums': child_nums, 'dtype': c_dtype},
        stop_gradient=True,
    )
    return (child, leaf_mask)


@overload
def tdm_sampler(
    x: Tensor,
    neg_samples_num_list: list[int],
    layer_node_num_list: list[int],
    leaf_node_num: int,
    tree_travel_attr: ParamAttrLike | None = ...,
    tree_layer_attr: ParamAttrLike | None = ...,
    output_positive: bool = ...,
    output_list: Literal[True] = ...,
    seed: int = ...,
    tree_dtype: DTypeLike = ...,
    dtype: DTypeLike = ...,
) -> tuple[list[Tensor], list[Tensor], list[Tensor]]: ...


@overload
def tdm_sampler(
    x: Tensor,
    neg_samples_num_list: list[int],
    layer_node_num_list: list[int],
    leaf_node_num: int,
    tree_travel_attr: ParamAttrLike | None = ...,
    tree_layer_attr: ParamAttrLike | None = ...,
    output_positive: bool = ...,
    output_list: Literal[False] = ...,
    seed: int = ...,
    tree_dtype: DTypeLike = ...,
    dtype: DTypeLike = ...,
) -> tuple[Tensor, Tensor, Tensor]: ...


@overload
def tdm_sampler(
    x: Tensor,
    neg_samples_num_list: list[int],
    layer_node_num_list: list[int],
    leaf_node_num: int,
    tree_travel_attr: ParamAttrLike | None = ...,
    tree_layer_attr: ParamAttrLike | None = ...,
    output_positive: bool = ...,
    output_list: bool = ...,
    seed: int = ...,
    tree_dtype: DTypeLike = ...,
    dtype: DTypeLike = ...,
) -> (
    tuple[Tensor, Tensor, Tensor]
    | tuple[list[Tensor], list[Tensor], list[Tensor]]
): ...


def tdm_sampler(
    x,
    neg_samples_num_list,
    layer_node_num_list,
    leaf_node_num,
    tree_travel_attr=None,
    tree_layer_attr=None,
    output_positive=True,
    output_list=True,
    seed=0,
    tree_dtype='int32',
    dtype='int32',
):
    """
    **Tdm Sampler**
    According to the input positive samples at leaf node(x), do negative sampling layer by layer on the given tree.

    .. code-block:: text

        Given:
            tree[[0], [1, 2], [3, 4], [5, 6]] # A binary tree with seven nodes
            travel_list = [[1, 3], [1, 4], [2, 5], [2, 6]] # leaf node's travel path (exclude root node)
            layer_list = [[1, 2], [3, 4, 5, 6]] # two layer (exclude root node)

            x = [[0], [1], [2], [3]] # Corresponding to leaf node [[3], [4], [5], [6]]
            neg_samples_num_list = [0, 0] # negative sample nums = 0
            layer_node_num_list = [2, 4]
            leaf_node_num = 4
            output_list = False

        We get:
            out = [[1, 3], [1, 4], [2, 5], [2, 6]]
            labels = [[1, 1], [1, 1], [1, 1], [1, 1]]
            mask = [[1, 1], [1, 1], [1, 1], [1, 1]]

    Args:
        x (Tensor): Tensor contained the item_id(corresponding to leaf node) information, dtype support int32/int64.
        neg_samples_num_list (list(int)): Number of negative samples per layer.
        layer_node_num_list (list(int)): Number of nodes per layer, must has same shape with neg_samples_num_list.
        leaf_node_num (int): Number of leaf nodes.
        tree_travel_attr (ParamAttr|None, optional): To specify the tdm-travel parameter property. Default: None, which means the
            default weight parameter property is used. See usage for details in :ref:`api_paddle_ParamAttr`, should
            has shape (leaf_node_num, len(layer_node_num_list)), dtype support int32/int64.
        tree_layer_attr (ParamAttr|None, optional): To specify the tdm-layer parameter property. Default: None, which means the
            default weight parameter property is used. See usage for details in :ref:`api_paddle_ParamAttr`, should
            has shape (node_num, 1), dtype support int32/int64.
        output_positive (bool, optional): Whether to output positive samples (include label and mask )at the same time. Default: True.
        output_list (bool, optional): Whether to divide the output into layers and organize it into list format. Default: True.
        seed (int, optional): The number of random seed. Default: 0.
        tree_dtype (np.dtype|core.VarDesc.VarType|str, optional): The dtype of tdm-travel and tdm-layer, support int32/int64. Default: int32.
        dtype (np.dtype|core.VarDesc.VarType|str, optional): The dtype of output(sampling results, labels and masks). Default: int32.

    Returns:
        tuple: A tuple including sampling results, corresponding labels and masks. if output_positive = True, sampling
            result  will include both positive and negative samples. If sampling result is a positive sample, the label is 1,
            and if it is a negative sample, it is 0. If the tree is unbalanced, in order to ensure the consistency of the
            sampling result shape, the padding sample's mask = 0, the real sample's mask value = 1.
            If output_list = True, the result will organize into list format specified by layer information.
            Output Tensor have same type with tdm-travel and tdm-layer parameter(tree_dtype).

    Examples:
        .. code-block:: python

            >>> import paddle
            >>> import numpy as np
            >>> paddle.enable_static()

            >>> x = paddle.static.data(name="x", shape=[None, 1], dtype="int32", lod_level=1)
            >>> travel_list = [[1, 3], [1, 4], [2, 5], [2, 6]] # leaf node's travel path, shape(leaf_node_num, layer_num)
            >>> layer_list_flat = [[1], [2], [3], [4], [5], [6]] # shape(node_nums, 1)

            >>> neg_samples_num_list = [0, 0] # negative sample nums = 0
            >>> layer_node_num_list = [2, 4] #two layer (exclude root node)
            >>> leaf_node_num = 4

            >>> travel_array = np.array(travel_list)
            >>> layer_array = np.array(layer_list_flat)

            >>> sample, label, mask = paddle.incubate.layers.tdm_sampler(
            ...     x,
            ...     neg_samples_num_list,
            ...     layer_node_num_list,
            ...     leaf_node_num,
            ...     tree_travel_attr=paddle.ParamAttr(
            ...         initializer=paddle.nn.initializer.Assign(
            ...            travel_array)),
            ...     tree_layer_attr=paddle.ParamAttr(
            ...         initializer=paddle.nn.initializer.Assign(
            ...             layer_array)),
            ...     output_positive=True,
            ...     output_list=True,
            ...     seed=0,
            ...     tree_dtype='int32')

    """
    helper = LayerHelper("tdm_sampler", **locals())
    check_dtype(
        tree_dtype,
        'tree_dtype',
        ['int32', 'int64'],
        'paddle.incubate.layers.tdm_sampler',
    )
    check_dtype(
        dtype, 'dtype', ['int32', 'int64'], 'paddle.incubate.layers.tdm_sampler'
    )
    c_dtype = convert_np_dtype_to_dtype_(dtype)

    if len(neg_samples_num_list) != len(layer_node_num_list):
        raise ValueError(
            "The shape of negative samples list must match the shape of layers. "
            f"But received len of neg_samples_num_list: {len(neg_samples_num_list)},"
            f"and len of layer_node_num_list: {len(layer_node_num_list)}, please check your input."
        )
    assert leaf_node_num is not None, "leaf_node_num should not be None here."

    layer_nums = 0
    node_nums = 0
    tree_layer_offset = [0]
    for layer_idx, layer_node_num in enumerate(layer_node_num_list):
        layer_nums += 1
        node_nums += layer_node_num
        tree_layer_offset.append(node_nums)
        if neg_samples_num_list[layer_idx] >= layer_node_num_list[layer_idx]:
            raise ValueError(
                "The number of negative samples must be less than the number of nodes "
                f"in the layer {layer_idx}, But received negative nums {neg_samples_num_list[layer_idx]}, and num of node at layer {layer_idx} "
                f"is {layer_node_num_list[layer_idx]}, please check your input."
            )
    assert (
        leaf_node_num < node_nums
    ), "leaf_node_num must be less than total node nums."

    travel_shape = [leaf_node_num, layer_nums]
    travel = helper.create_parameter(
        attr=tree_travel_attr,
        shape=travel_shape,
        dtype=tree_dtype,
        default_initializer=paddle.nn.initializer.Constant(0),
    )

    layer_shape = [node_nums, 1]
    layer = helper.create_parameter(
        attr=tree_layer_attr,
        shape=layer_shape,
        dtype=tree_dtype,
        default_initializer=paddle.nn.initializer.Constant(0),
    )

    if in_dynamic_or_pir_mode():
        return _C_ops.tdm_sampler(
            x,
            travel,
            layer,
            output_positive,
            neg_samples_num_list,
            tree_layer_offset,
            seed,
            c_dtype,
        )

    out = helper.create_variable_for_type_inference(dtype=dtype)
    out.stop_gradient = True

    labels = helper.create_variable_for_type_inference(dtype=dtype)
    labels.stop_gradient = True

    mask = helper.create_variable_for_type_inference(dtype=dtype)
    mask.stop_gradient = True

    helper.append_op(
        type='tdm_sampler',
        inputs={"X": x, "Travel": travel, "Layer": layer},
        outputs={'Out': out, 'Labels': labels, 'Mask': mask},
        attrs={
            'neg_samples_num_list': neg_samples_num_list,
            'output_positive': output_positive,
            'layer_offset': tree_layer_offset,
            'seed': seed,
            'dtype': c_dtype,
        },
    )

    if output_list:
        output_list = []
        labels_list = []
        mask_list = []
        start_offset = 0
        positive_flag = 1
        if not output_positive:
            positive_flag = 0

        for layer_sample_num in neg_samples_num_list:
            end_offset = start_offset + layer_sample_num + positive_flag
            layer_samples = paddle.slice(
                out, axes=[1], starts=[start_offset], ends=[end_offset]
            )
            layer_labels = paddle.slice(
                labels, axes=[1], starts=[start_offset], ends=[end_offset]
            )
            layer_mask = paddle.slice(
                mask, axes=[1], starts=[start_offset], ends=[end_offset]
            )

            layer_samples = paddle.reshape(
                layer_samples, [-1, layer_sample_num + positive_flag, 1]
            )
            layer_samples.stop_gradient = True

            layer_labels = paddle.reshape(
                layer_labels, [-1, layer_sample_num + positive_flag, 1]
            )
            layer_labels.stop_gradient = True

            layer_mask = paddle.reshape(
                layer_mask, [-1, layer_sample_num + positive_flag, 1]
            )
            layer_mask.stop_gradient = True

            output_list.append(layer_samples)
            labels_list.append(layer_labels)
            mask_list.append(layer_mask)
            start_offset = end_offset

        out = output_list
        labels = labels_list
        mask = mask_list

    return (out, labels, mask)


def rank_attention(
    input: Tensor,
    rank_offset: Tensor,
    rank_param_shape: list[int],
    rank_param_attr: ParamAttrLike,
    max_rank: int = 3,
    max_size: int = 0,
) -> Tensor:
    """
    **Rank Attention layer**
    This Op can calculate rank attention between input and rank_param, and
    rank_param gives the organization of data. Notice: It currently supports
    GPU device.
    This Op exists in incubate layers, which means that it is not shown to the public.

    Args:
        input (Tensor): Tensor with data type float32, float64.
        rank_offset (Tensor): Tensor with data type int32.
        rank_para_shape (list[int]): The shape of rank_param.
        rank_param_attr (ParamAttr): Attribute initializer of rank_param.
        max_rank (int, optional): The max rank of input's ranks. Default is 3.
        max_size (int, optional): The max size of input's ranks. Default is 0.
    Returns:
        Tensor: A Tensor with the same data type as input's.

    Examples:
        .. code-block:: python

            >>> import paddle
            >>> paddle.enable_static()

            >>> input = paddle.static.data(name="input", shape=[None, 2], dtype="float32")
            >>> rank_offset = paddle.static.data(name="rank_offset", shape=[None, 7], dtype="int32")
            >>> out = paddle.incubate.layers.rank_attention(input=input,
            ...                                             rank_offset=rank_offset,
            ...                                             rank_param_shape=[18,3],
            ...                                             rank_param_attr=
            ...                                             paddle.ParamAttr(learning_rate=1.0,
            ...                                                              name="ubm_rank_param.w_0"),
            ...                                             max_rank=3,
            ...                                             max_size=0)
    """
    helper = LayerHelper('rank_attention', **locals())
    dtype = helper.input_dtype(input_param_name='input')
    input_shape = input.shape
    assert input_shape[1] * max_rank * max_rank == rank_param_shape[0]

    rank_param = helper.create_parameter(
        attr=rank_param_attr, shape=rank_param_shape, dtype=dtype
    )
    rank_param.stop_gradient = False

    output = helper.create_variable_for_type_inference(dtype)
    input_help = helper.create_variable_for_type_inference(
        dtype=dtype, stop_gradient=True
    )
    ins_rank = helper.create_variable_for_type_inference(
        dtype=dtype, stop_gradient=True
    )

    helper.append_op(
        type="rank_attention",
        inputs={"X": input, "RankOffset": rank_offset, "RankParam": rank_param},
        outputs={"Out": output, "InputHelp": input_help, "InsRank": ins_rank},
        attrs={"MaxRank": max_rank, "MaxSize": max_size},
    )
    return output


def batch_fc(
    input: Tensor,
    param_size: list[int],
    param_attr: ParamAttrLike,
    bias_size: list[int],
    bias_attr: ParamAttrLike,
    act: str | None = None,
) -> Tensor:
    """
    **Batch FC layer**
    This Op can calculate BatchFC. This is similar to matmul op,
    except that the bias and relu activation layers are added.
    Notice: It currently supports GPU device.
    This Op exists in incubate layers, which means that it is not shown to the public.

    Args:
        input (Tensor): Tensor with data type float32, float64.
        param_size (list[int]): The size of w.
        param_attr (ParamAttr): Attribute initializer of w.
        bias_size (list[int]): The size of bias.
        bias_attr (ParamAttr): Attribute initializer of bias.
        act (str, optional): Activation to be applied to the output of this layer. Default is None.

    Returns:
        Tensor: A Tensor with the same data type as input's.

    Examples:
        .. code-block:: python

            >>> import paddle
            >>> paddle.enable_static()

            >>> input = paddle.static.data(name="input", shape=[16, 2, 3], dtype="float32")
            >>> out = paddle.incubate.layers.batch_fc(input=input,
            ...                                     param_size=[16, 3, 10],
            ...                                     param_attr=
            ...                                     paddle.ParamAttr(learning_rate=1.0,
            ...                                                      name="w_0"),
            ...                                     bias_size=[16, 10],
            ...                                     bias_attr=
            ...                                     paddle.ParamAttr(learning_rate=1.0,
            ...                                                      name="b_0"),
            ...                                     act="relu")
    """

    helper = LayerHelper("batch_fc", **locals())
    check_type(input, 'input', (Variable), 'batch_fc')
    input_shape = input.shape
    assert input_shape[0] == param_size[0]
    assert input_shape[2] == param_size[1]
    assert param_size[2] == bias_size[1]
    assert input_shape[0] == bias_size[0]

    dtype = helper.input_dtype()
    check_dtype(dtype, 'input', ['float32', 'float64'], 'batch_fc')

    w = helper.create_parameter(
        attr=param_attr, shape=param_size, dtype=dtype, is_bias=False
    )
    b = helper.create_parameter(
        attr=bias_attr, shape=bias_size, dtype=dtype, is_bias=False
    )
    pre_act = helper.create_variable_for_type_inference(dtype)
    helper.append_op(
        type="batch_fc",
        inputs={"Input": input, "W": w, "Bias": b},
        outputs={"Out": pre_act},
    )
    return helper.append_activation(pre_act)


def correlation(
    x: Tensor,
    y: Tensor,
    pad_size: int,
    kernel_size: int,
    max_displacement: int,
    stride1: int,
    stride2: int,
    corr_type_multiply: int = 1,
) -> Tensor:
    """

    This operation compute correlation of two tensor.
    For more information of correlation, please refer to PWC-Net:
    CNNs for Optical Flow Using Pyramid, Warping, and Cost Volume
    <https://arxiv.org/pdf/1709.02371.pdf>_

    Args:
        x (Tensor): The input x is 4-D Tensor with shape [N, C, H, W]. The data type is float32 and float64.
        y (Tensor): The input y is 4-D Tensor with shape [N, C, H, W]. The data type is float32 and float64.
        pad_size (int): Pad size. The data type is int.
        max_displacement (int): Max displacement. The data type is int.
        stride1 (int): stride size of x. The data type is int.
        stride2 (int): stride size of y. The data type is int.
        corr_type_multiply (int, optional): The type of multiply. The data type is int. Default: 1.

    Returns:
        Tensor: The data type is same as input tensor.

    Examples:

        .. code-block:: python

            >>> import paddle
            >>> paddle.enable_static()
            >>> x1 = paddle.static.data(name='x1',
            ...                         shape=[2, 3, 4, 5],
            ...                         dtype="float32")
            >>> x2 = paddle.static.data(name='x2',
            ...                         shape=[2, 3, 4, 5],
            ...                         dtype="float32")


            >>> out = paddle.incubate.layers.correlation(
            ...                 x1,
            ...                 x2,
            ...                 pad_size=4,
            ...                 kernel_size=1,
            ...                 max_displacement=4,
            ...                 stride1=1,
            ...                 stride2=1)

    """

    if paddle.in_dynamic_mode():
        attrs = (
            "pad_size",
            pad_size,
            "kernel_size",
            kernel_size,
            "max_displacement",
            max_displacement,
            "stride1",
            stride1,
            "stride2",
            stride2,
            "corr_type_multiply",
            corr_type_multiply,
        )
        output = _legacy_C_ops.correlation(x, y, *attrs)
    else:
        helper = LayerHelper("correlation", **locals())
        output = helper.create_variable_for_type_inference(dtype=x.dtype)
        helper.append_op(
            type="correlation",
            inputs={"Input1": x, "Input2": y},
            attrs={
                "pad_size": pad_size,
                "kernel_size": kernel_size,
                "max_displacement": max_displacement,
                "stride1": stride1,
                "stride2": stride2,
                "corr_type_multiply": corr_type_multiply,
            },
            outputs={"Output": output},
        )
    return output


def fused_bn_add_act(
    x: Tensor,
    y: Tensor,
    momentum: float | Tensor = 0.9,
    epsilon: float = 1e-05,
    param_attr: ParamAttrLike | None = None,
    bias_attr: ParamAttrLike | None = None,
    moving_mean_name: str | None = None,
    moving_variance_name: str | None = None,
    act: str | None = None,
    name: str | None = None,
) -> Tensor:
    r"""
    This Op performs batch norm on input x, and adds the result to input y. Then
    it performs activation on the sum. The data format of inputs must be NHWC
    `[batch, in_height, in_width, in_channels]`.

    Args:
        x (Tensor): The rank of input tensor can be 2, 3, 4, 5. The data type
            is float16.
        y (Tensor): The rank of input tensor can be 2, 3, 4, 5. The data type
            is float16.
        momentum (float|Tensor, optional): The value used for the moving_mean and
            moving_var computation. This should be a float number or a 0-D Tensor with
            shape [] and data type as float32. The updated formula is:
            :math:`moving\_mean = moving\_mean * momentum + new\_mean * (1. - momentum)`
            :math:`moving\_var = moving\_var * momentum + new\_var * (1. - momentum)`
            Default is 0.9.
        epsilon (float, optional): A value added to the denominator for
            numerical stability. Default is 1e-05.
        param_attr (ParamAttr|None, optional): The parameter attribute for Parameter `scale`
            of batch_norm. If it is set to None or one attribute of ParamAttr, batch_norm
            will create ParamAttr as param_attr, the name of scale can be set in ParamAttr.
            If the Initializer of the param_attr is not set, the parameter is initialized
            with Xavier. Default: None.
        bias_attr (ParamAttr|None, optional): The parameter attribute for the bias of batch_norm.
            If it is set to None or one attribute of ParamAttr, batch_norm
            will create ParamAttr as bias_attr, the name of bias can be set in ParamAttr.
            If the Initializer of the bias_attr is not set, the bias is initialized zero.
            Default: None.
        moving_mean_name (str|None, optional): The name of moving_mean which store the global Mean. If it
            is set to None, batch_norm will save global mean with a random name, otherwise, batch_norm
            will save global mean with the string. Default: None.
        moving_variance_name (str|None, optional): The name of the moving_variance which store the global Variance.
            If it is set to None, batch_norm will save global variance with a random name, otherwise, batch_norm
            will save global variance with the string. Default: None.
        act (str|None, optional): Activation type, linear|relu|prelu|... Default: None.
        name (str:None, optional): For detailed information, please refer to :ref:`api_guide_Name`.
            Usually name is no need to set and None by default. Default: None.

    Examples:
        .. code-block:: python

            >>> # doctest: +REQUIRES(env:GPU)
            >>> import paddle
            >>> paddle.enable_static()

            >>> def build_program(main_program, startup_program):
            ...     with paddle.static.program_guard(main_program, startup_program):
            ...         x = paddle.static.data(name='x', shape=[-1, 1, 28, 28], dtype='float32')
            ...         y = paddle.static.data(name="y", shape=[-1, 1], dtype='int64')
            ...         conv1_1 = paddle.static.nn.conv2d(
            ...             input=x,
            ...             filter_size=3,
            ...             num_filters=32,
            ...             stride=1,
            ...             padding=1,
            ...             act=None,
            ...             bias_attr=False,
            ...            data_format='NHWC')
            ...         conv1_2 = paddle.static.nn.conv2d(
            ...             input=x,
            ...             filter_size=3,
            ...             num_filters=32,
            ...             stride=1,
            ...             padding=1,
            ...             act=None,
            ...             bias_attr=False,
            ...             data_format='NHWC')
            ...         bn = paddle.static.nn.batch_norm(
            ...            input=conv1_1,
            ...             act=None,
            ...             data_layout='NHWC')
            ...         fused_bn_add_act = paddle.incubate.layers.fused_bn_add_act(conv1_2, bn)
            ...         prediction = paddle.static.nn.fc(x=fused_bn_add_act, size=10, activation='softmax')
            ...         loss = paddle.nn.functional.cross_entropy(
            ...             input=prediction, label=y,
            ...             reduction='none', use_softmax=False
            ...         )
            ...         loss = paddle.mean(loss)
            ...         sgd = paddle.optimizer.SGD(learning_rate=0.001)
            ...         sgd = paddle.static.amp.decorate(
            ...             sgd, use_dynamic_loss_scaling=True, init_loss_scaling=128.0)
            ...         sgd.minimize(loss)
            ...
            ...     return x, y, loss

            >>> iters = 5
            >>> batch_size = 16
            >>> support_gpu = paddle.is_compiled_with_cuda()
            >>> if support_gpu:
            ...     main_program = paddle.static.Program()
            ...     startup_program = paddle.static.Program()
            ...     place = paddle.CUDAPlace(0)
            ...     x, y, loss = build_program(main_program, startup_program)
            ...
            ...     feeder = paddle.DataFeeder(feed_list=[x, y], place=place)
            ...     train_reader = paddle.batch(
            ...         paddle.dataset.mnist.train(), batch_size=batch_size)
    """
    helper = LayerHelper('fused_bn_add_act', **locals())

    check_variable_and_dtype(
        x, 'input', ['float16', 'float32', 'float64'], 'fused_bn_add_act'
    )
    check_variable_and_dtype(
        y, 'input', ['float16', 'float32', 'float64'], 'fused_bn_add_act'
    )
    bn_param_dtype = core.VarDesc.VarType.FP32

    x_shape = x.shape
    channel_num = x_shape[-1]
    param_shape = [channel_num]

    # create parameter
    scale = helper.create_parameter(
        attr=helper.param_attr,
        shape=param_shape,
        dtype=bn_param_dtype,
        default_initializer=paddle.nn.initializer.Constant(1.0),
    )
    bias = helper.create_parameter(
        attr=helper.bias_attr,
        shape=param_shape,
        dtype=bn_param_dtype,
        is_bias=True,
    )
    mean = helper.create_parameter(
        attr=ParamAttr(
            name=moving_mean_name,
            initializer=paddle.nn.initializer.Constant(0.0),
            trainable=False,
        ),
        shape=param_shape,
        dtype=bn_param_dtype,
    )
    mean.stop_gradient = True
    variance = helper.create_parameter(
        attr=ParamAttr(
            name=moving_variance_name,
            initializer=paddle.nn.initializer.Constant(1.0),
            trainable=False,
        ),
        shape=param_shape,
        dtype=bn_param_dtype,
    )
    variance.stop_gradient = True

    # create output
    # mean and mean_out share the same memory
    mean_out = mean
    # variance and variance out share the same memory
    variance_out = variance
    saved_mean = helper.create_variable_for_type_inference(
        dtype=bn_param_dtype, stop_gradient=True
    )
    saved_variance = helper.create_variable_for_type_inference(
        dtype=bn_param_dtype, stop_gradient=True
    )
    reserve_space = helper.create_variable_for_type_inference(
        dtype=core.VarDesc.VarType.FP16, stop_gradient=True
    )
    batch_norm_out = helper.create_variable_for_type_inference(
        core.VarDesc.VarType.FP16
    )

    inputs = {
        "X": x,
        "Z": y,
        "Scale": scale,
        "Bias": bias,
        "Mean": mean,
        "Variance": variance,
    }
    attrs = {"epsilon": epsilon, 'momentum': momentum}

    outputs = {
        "Y": batch_norm_out,
        "MeanOut": mean_out,
        "VarianceOut": variance_out,
        "SavedMean": saved_mean,
        "SavedVariance": saved_variance,
        "ReserveSpace": reserve_space,
    }

    helper.append_op(
        type="fused_bn_add_activation",
        inputs=inputs,
        outputs=outputs,
        attrs=attrs,
    )

    return batch_norm_out


def pow2_decay_with_linear_warmup(
    warmup_steps: float,
    total_steps: float,
    base_lr: float,
    end_lr: float,
    dtype: DTypeLike = 'float32',
    name: str | None = None,
) -> Tensor:
    if paddle.in_dynamic_mode():
        raise NotImplementedError(
            "pow2_decay_with_linear_warmup does not support dygraph mode yet."
        )

    helper = LayerHelper("pow2_decay_with_linear_warmup", **locals())
    lr = helper.create_global_variable(persistable=True, dtype=dtype, shape=[1])
    helper.set_variable_initializer(
        lr,
        paddle.nn.initializer.Constant(value=float(base_lr) / warmup_steps),
    )

    step = helper.create_global_variable(
        persistable=True, dtype='int64', shape=[1]
    )
    helper.set_variable_initializer(
        step, paddle.nn.initializer.Constant(value=0)
    )
    assert (
        warmup_steps <= total_steps
    ), "warmup_steps cannot be larger than total_steps"

    helper.append_op(
        type="pow2_decay_with_linear_warmup",
        inputs={"LearningRate": lr, "Step": step},
        outputs={"LearningRateOut": lr, "StepOut": step},
        attrs={
            "warmup_steps": warmup_steps,
            "total_steps": total_steps,
            "base_lr": base_lr,
            "end_lr": end_lr,
        },
    )
    return lr


def _pull_gpups_sparse(
    input, size, dtype='float32', is_distributed=False, is_sparse=False
):
    r"""
    **Pull GpuPS Sparse Layer**

    This layer is used to lookup embeddings of IDs, provided by :attr:`input`, in
    GpuPS lookup table. The result of this lookup is the embedding of each ID in the
    :attr:`input`.

    Args:
        input (Tensor): Input is a Tensor<int64>, which contains the IDs information.
        size (int|list of int): The embedding size parameter of each input, which indicates the size of
            each embedding vector respectively.
        dtype (str, optional): The dtype refers to the data type of output tensor. Only supportsfloat32 now. Default is float32.
        is_distributed (bool, optional): Whether to use distributed mode. Default is False.
        is_sparse (bool, optional): Whether to use sparse mode. Default is False.

    Returns:
        Tensor: The tensor storing the embeddings of the supplied inputs, whose size are indicated by size respectively.

    Examples:
        .. code-block:: python

            >>> import paddle.incubate as incubate
            >>> import paddle
            >>> paddle.enable_static()

            >>> slots = []
            >>> data_1 = paddle.static.data(name='sequence', shape=[-1,1], dtype='int64', lod_level=1)
            >>> slots.append(data_1)
            >>> data_2 = paddle.static.data(name='sequence', shape=[-1,1], dtype='int64', lod_level=1)
            >>> slots.append(data_2)
            >>> embs = incubate.layers.pull_gpups_sparse(input=slots, size=[11, 35])
    """
    helper = LayerHelper('pull_gpups_sparse', **locals())
    if dtype != 'float32':
        raise ValueError(
            "GpuPS only support float type embedding now, and your type is: "
            + dtype
        )
    helper.input_dtype()
    inputs = helper.multiple_input()
    outs = [
        helper.create_variable_for_type_inference(dtype)
        for i in range(len(inputs))
    ]
    w = helper.create_parameter(
        attr=helper.param_attr, shape=[size[0]], dtype=dtype, is_bias=False
    )
    helper.append_op(
        type='pull_gpups_sparse',
        inputs={'Ids': inputs, 'W': w},
        outputs={'Out': outs},
        attrs={
            'size': size,
            'is_distributed': is_distributed,
            'is_sparse': is_sparse,
        },
    )
    if len(outs) == 1:
        return outs[0]
    return outs


def _pull_box_sparse(
    input, size, dtype='float32', is_distributed=False, is_sparse=False
):
    r"""
    **Pull Box Sparse Layer**

    This layer is used to lookup embeddings of IDs, provided by :attr:`input`, in
    BoxPS lookup table. The result of this lookup is the embedding of each ID in the
    :attr:`input`.

    Args:
        input (Tensor): Input is a Tensor<int64>, which contains the IDs information.
        size (int): The embedding size parameter, which indicates the size of
            each embedding vector respectively.
        dtype (str, optional): The dtype refers to the data type of output tensor. Only supports float32 now. Default is float32.
        is_distributed (bool, optional): Whether to use distributed mode. Default is False.
        is_sparse (bool, optional): Whether to use sparse mode. Default is False.

    Returns:
        Tensor: The tensor storing the embeddings of the supplied inputs.

    Examples:
        .. code-block:: python

            >>> import paddle.incubate as incubate
            >>> import paddle
            >>> paddle.enable_static()

            >>> x = paddle.static.data(name='x', shape=[-1, 1], dtype='int64', lod_level=1)
            >>> y = paddle.static.data(name='y', shape=[-1, 1], dtype='int64', lod_level=1)
            >>> emb_x, emb_y = incubate.layers._pull_box_sparse([x, y], size=1)
    """
    helper = LayerHelper('pull_box_sparse', **locals())
    if dtype != 'float32':
        raise ValueError(
            "BoxPS only support float type embedding now, and your type is: "
            + dtype
        )
    helper.input_dtype()
    inputs = helper.multiple_input()
    outs = [
        helper.create_variable_for_type_inference(dtype)
        for i in range(len(inputs))
    ]
    w = helper.create_parameter(
        attr=helper.param_attr, shape=[size], dtype=dtype, is_bias=False
    )
    helper.append_op(
        type='pull_box_sparse',
        inputs={'Ids': inputs, 'W': w},
        outputs={'Out': outs},
        attrs={
            'size': size,
            'is_distributed': is_distributed,
            'is_sparse': is_sparse,
        },
    )
    if len(outs) == 1:
        return outs[0]
    return outs
