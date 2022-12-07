# Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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
All layers just related to the neural network.
"""
import os
import inspect
import warnings

import numpy as np

import paddle
from ..layer_helper import LayerHelper
from paddle.fluid.framework import _in_legacy_dygraph
from ..initializer import Normal, Constant
from ..framework import (
    Variable,
    OpProtoHolder,
    _non_static_mode,
    dygraph_only,
    _dygraph_tracer,
    default_main_program,
    _varbase_creator,
    static_only,
    _global_flags,
    _in_legacy_dygraph,
    in_dygraph_mode,
)
from ..framework import _current_expected_place
from .. import dygraph_utils
from ..param_attr import ParamAttr
from .layer_function_generator import (
    autodoc,
    templatedoc,
    _generate_doc_string_,
)
from .tensor import concat, assign, fill_constant, zeros, tensor_array_to_tensor
from . import utils
from .. import unique_name
from functools import reduce
from .. import core
from ...utils import deprecated
from ..data_feeder import (
    convert_dtype,
    check_variable_and_dtype,
    check_type,
    check_dtype,
)
from paddle.utils import deprecated
from paddle import _C_ops, _legacy_C_ops
from collections.abc import Iterable


__all__ = [
    'fc',
    'embedding',
    'linear_chain_crf',
    'crf_decoding',
    'conv2d',
    'pool2d',
    'batch_norm',
    'dropout',
    'split',
    'l2_normalize',
    'matmul',
    'row_conv',
    'layer_norm',
    'spectral_norm',
    'one_hot',
    'autoincreased_step_counter',
    'unsqueeze',
    'lod_reset',
    'relu',
    'elementwise_add',
    'elementwise_sub',
    'gaussian_random',
    'sampling_id',
    'clip',
    'clip_by_norm',
    'mean',
    'mul',
    'merge_selected_rows',
    'get_tensor_from_selected_rows',
    'unfold',
    'deformable_roi_pooling',
    'shard_index',
    'hard_swish',
    'mish',
    'uniform_random',
    'unbind',
]

OP_NAMEMAPPING = {
    'elementwise_max': 'maximum',
    'elementwise_min': 'minimum',
    'elementwise_pow': 'elementwise_pow',
    'elementwise_floordiv': 'floor_divide',
    'elementwise_add': 'add',
    'elementwise_sub': 'subtract',
    'elementwise_mul': 'multiply',
    'elementwise_div': 'divide',
    'elementwise_mod': 'remainder',
}


def _get_reduce_dim(dim, input):
    """
    Internal function for reduce_sum, reduce_mean, reduce_prod.
    It computes the attribute reduce_all value based on axis.
    """
    if dim is not None and not isinstance(dim, list):
        if isinstance(dim, (tuple, range)):
            dim = list(dim)
        elif isinstance(dim, int):
            dim = [dim]
        else:
            raise TypeError(
                "The type of dim must be int, list, tuple or range, but received {}".format(
                    type(dim)
                )
            )
    if dim is None:
        dim = []
    if dim == [] or len(dim) == len(input.shape):
        reduce_all = True
    else:
        reduce_all = False

    return reduce_all, dim


@dygraph_only
def _elementwise_op_in_dygraph(
    x, y, axis=-1, act=None, use_mkldnn=False, op_name=None
):
    def is_inplace(op_name):
        return op_name[-1] == "_"

    if op_name not in OP_NAMEMAPPING.keys() or axis != -1:
        op = getattr(_legacy_C_ops, op_name)
        out = op(x, y, 'axis', axis, 'use_mkldnn', use_mkldnn)
    else:
        if in_dygraph_mode():
            op = getattr(
                _C_ops,
                OP_NAMEMAPPING[op_name] if not is_inplace(op_name) else op_name,
            )
            out = op(x, y)

        if _in_legacy_dygraph():
            op = getattr(_legacy_C_ops, op_name)
            out = op(x, y, 'axis', axis, 'use_mkldnn', use_mkldnn)
    return dygraph_utils._append_activation_in_dygraph(
        out, act, use_mkldnn=use_mkldnn
    )


def fc(
    input,
    size,
    num_flatten_dims=1,
    param_attr=None,
    bias_attr=None,
    act=None,
    name=None,
):
    r"""
    :api_attr: Static Graph

    **Fully Connected Layer**

    This operator creates a fully connected layer in the network. It can take
    a Tensor(or LoDTensor) or a list of Tensor(or LoDTensor) as its inputs(see
    Args in detail). It creates a variable called weight for each input Tensor,
    which represents a fully connected weight matrix from each input unit to
    each output unit. The fully connected layer multiplies each input Tensor
    with its corresponding weight to produce an output Tensor with shape :math:`[M, size]` ,
    where M is batch size. If a list of Tensor is given, the results of
    multiple output Tensors with shape :math:`[M, size]` will be summed up. If :attr:`bias_attr`
    is not None, a bias variable will be created and added to the output.
    Finally, if :attr:`act` is not None, it will be applied to the output as well.

    When the input is a single Tensor(or LoDTensor):

    .. math::

        Out = Act({XW + b})

    When the input is a list of Tensor(or LoDTensor):

    .. math::

        Out = Act({\sum_{i=0}^{N-1}X_iW_i + b})

    In the above equation:

    * :math:`N`: Number of the input. N equals to len(input) if input is list of Variable.
    * :math:`X_i`: The i-th input tensor.
    * :math:`W_i`: The i-th weights matrix corresponding i-th input tensor.
    * :math:`b`: The bias parameter created by this layer (if needed).
    * :math:`Act`: The activation function.
    * :math:`Out`: The output Tensor.

    .. code-block:: text

        Case 1:
        Given a single Tensor data_1, and num_flatten_dims = 2:
            data_1.data = [[[0.1, 0.2],
                            [0.3, 0.4]]]
            data_1.shape = (1, 2, 2) # 1 is batch_size

            out = fluid.layers.fc(input=data_1, size=1, num_flatten_dims=2)

        Then output is:
            out.data = [[0.83234344], [0.34936576]]
            out.shape = (1, 2, 1)

        Case 2:
        Given a list of Tensor:
            data_1.data = [[[0.1, 0.2],
                           [0.3, 0.4]]]
            data_1.shape = (1, 2, 2) # 1 is batch_size

            data_2 = [[[0.1, 0.2, 0.3]]]
            data_2.shape = (1, 1, 3)

            out = fluid.layers.fc(input=[data_1, data_2], size=2)

        Then:
            out.data = [[0.18669507, 0.1893476]]
            out.shape = (1, 2)

    Args:
        input (Variable|list of Variable): A Tensor(or LoDTensor) with shape :math:`[N_1, N_2,..., N_k]` or
            a list of Tensor(or LoDTensor). The dimensions of the input Tensor is at least 2 and the data
            type should be float32 or float64.
        size(int): The number of output units in this layer, which also means the feature size of output
            Tensor(or LoDTensor).
        num_flatten_dims (int): The fc layer can accept an input Tensor with more than
            two dimensions. If this happens, the multidimensional tensor will first be flattened
            into a 2-D matrix. The parameter :attr:`num_flatten_dims` determines how the input
            Tensor is flattened: the first :attr:`num_flatten_dims` (inclusive, index starts from 1)
            dimensions will be flatten to form the first dimension of the final matrix (height of
            the matrix), and the rest :math:`rank(X) - num\_flatten\_dims` dimensions are flattened to
            form the second dimension of the final matrix (width of the matrix). For example, assuming that
            X is a 5-dimensional Tensor with a shape [2, 3, 4, 5, 6], and :attr:`num_flatten_dims` = 3.
            Then, the flattened matrix will have a shape [2 x 3 x 4, 5 x 6] = [24, 30]. Default: 1.
        param_attr (ParamAttr): To specify the weight parameter property. Default: None, which means the
            default weight parameter property is used. See usage for details in :ref:`api_fluid_ParamAttr` .
        bias_attr (ParamAttr): To specify the bias parameter property. Default: None, which means the
            default bias parameter property is used. See usage for details in :ref:`api_fluid_ParamAttr` .
        act (str): Activation to be applied to the output of this layer, such as tanh, softmax,
            sigmoid, relu. For more information, please refer to :ref:`api_guide_activations_en` . Default: None.
        name (str, optional): The default value is None.  Normally there is no need for user to set this property.
            For more information, please refer to :ref:`api_guide_Name` .

    Returns:
        Variable: Tensor or LoDTensor calculated by fc layer. The data type is same with input.

    Raises:
        ValueError: If dimensions of the input Tensor is less than 2.

    Examples:
        .. code-block:: python

          import paddle.fluid as fluid
          import paddle
          paddle.enable_static()
          # when input is single tensor
          data = fluid.data(name="data", shape=[-1, 32], dtype="float32")
          fc = fluid.layers.fc(input=data, size=1000, act="tanh")

          # when input are multiple tensors
          data_1 = fluid.data(name="data_1", shape=[-1, 32], dtype="float32")
          data_2 = fluid.data(name="data_2", shape=[-1, 36], dtype="float32")
          fc = fluid.layers.fc(input=[data_1, data_2], size=1000, act="tanh")
    """
    helper = LayerHelper("fc", **locals())
    check_type(input, 'input', (list, tuple, Variable), 'fc')
    if isinstance(input, (list, tuple)):
        for i, input_x in enumerate(input):
            check_type(input_x, 'input[' + str(i) + ']', Variable, 'fc')
    dtype = helper.input_dtype()
    check_dtype(
        dtype, 'input', ['float16', 'uint16', 'float32', 'float64'], 'fc'
    )
    mul_results = []
    for input_var, param_attr in helper.iter_inputs_and_params():
        input_shape = input_var.shape
        if num_flatten_dims == -1:
            num_flatten_dims = len(input_shape) - 1
        param_shape = [
            reduce(lambda a, b: a * b, input_shape[num_flatten_dims:], 1)
        ] + [size]

        w = helper.create_parameter(
            attr=param_attr, shape=param_shape, dtype=dtype, is_bias=False
        )
        tmp = helper.create_variable_for_type_inference(dtype)
        helper.append_op(
            type="mul",
            inputs={"X": input_var, "Y": w},
            outputs={"Out": tmp},
            attrs={"x_num_col_dims": num_flatten_dims, "y_num_col_dims": 1},
        )
        mul_results.append(tmp)

    if len(mul_results) == 1:
        pre_bias = mul_results[0]
    else:
        pre_bias = helper.create_variable_for_type_inference(dtype)
        helper.append_op(
            type="sum",
            inputs={"X": mul_results},
            outputs={"Out": pre_bias},
            attrs={"use_mkldnn": False},
        )
    # add bias
    pre_activation = helper.append_bias_op(pre_bias, dim_start=num_flatten_dims)
    # add activation
    return helper.append_activation(pre_activation)


@deprecated(since="2.0.0", update_to="paddle.nn.functional.embedding")
def embedding(
    input,
    size,
    is_sparse=False,
    is_distributed=False,
    padding_idx=None,
    param_attr=None,
    dtype='float32',
):
    r"""
    :api_attr: Static Graph

    **WARING:** This OP will be deprecated in a future release. This OP requires the
    last dimension of Tensor shape must be equal to 1. It is recommended to use
    fluid. :ref:`api_fluid_embedding` .

    The operator is used to lookup embeddings vector of ids provided by :attr:`input` .
    It automatically constructs a 2D embedding matrix based on the
    input :attr:`size` (vocab_size, emb_size) and :attr:`dtype` .

    This OP requires the last dimension of Tensor shape must be equal to 1. The shape
    of output Tensor is generated by replacing the last dimension of the input Tensor shape
    with emb_size.

    **Note:** The id in :attr:`input` must satisfy :math:`0 =< id < size[0]` ,
    otherwise the program will throw an exception and exit.

    .. code-block:: text

        Case 1:

        input is a Tensor. padding_idx = -1
            input.data = [[[1], [3]], [[2], [4]], [[4], [127]]]
            input.shape = [3, 2, 1]
        Given size = [128, 16]
        output is a Tensor:
            out.shape = [3, 2, 16]
            out.data = [[[0.129435295, 0.244512452, ..., 0.436322452],
                        [0.345421456, 0.524563927, ..., 0.144534654]],

                        [[0.345249859, 0.124939536, ..., 0.194353745],
                        [0.945345345, 0.435394634, ..., 0.435345365]],

                        [[0.945345345, 0.435394634, ..., 0.435345365],
                        [0.0,         0.0,         ..., 0.0        ]]]  # padding data
        The input padding_idx is less than 0, it is automatically converted to padding_idx = -1 + 128 = 127
        It will pad all-zero data when ids is 127.

        Case 2:

        input is a LoDTensor with 1-level LoD. padding_idx = 0
            input.lod = [[2, 3]]
            input.data = [[1], [3], [2], [4], [0]]
            input.shape = [5, 1]
        Given size = [128, 16]
        output is a LoDTensor:
            out.lod = [[2, 3]]
            out.shape = [5, 16]
            out.data = [[0.129435295, 0.244512452, ..., 0.436322452],
                        [0.345421456, 0.524563927, ..., 0.144534654],
                        [0.345249859, 0.124939536, ..., 0.194353745],
                        [0.945345345, 0.435394634, ..., 0.435345365],
                        [0.0,         0.0,         ..., 0.0        ]]  # padding data
        It will pad all-zero data when ids is 0.

    Args:
        input(Variable): A Tensor or LoDTensor with type int64, which contains the id information.
            The last dimension of Tensor shape must be equal to 1. The value of the input id should
            satisfy :math:`0<= id < size[0]` .
        size(tuple|list): The shape of lookup table parameter. It should have two elements which
            indicates the size of the dictionary of embeddings and the size of each embedding vector respectively.
        is_sparse(bool): The flag indicating whether to use sparse update. This parameter only
            affects the performance of the backwards gradient update. It is recommended to set
            True because sparse update is faster. But some optimizer does not support sparse update,
            such as :ref:`api_fluid_optimizer_AdadeltaOptimizer` , :ref:`api_fluid_optimizer_AdamaxOptimizer` ,
            :ref:`api_fluid_optimizer_DecayedAdagradOptimizer` , :ref:`api_fluid_optimizer_FtrlOptimizer` ,
            :ref:`api_fluid_optimizer_LambOptimizer` and :ref:`api_fluid_optimizer_LarsMomentumOptimizer` .
            In these case, is_sparse must be False. Default: False.
        is_distributed(bool): Whether to store the embedding matrix in a distributed manner. Only used
            in multi-machine distributed CPU training. Default: False.
        padding_idx(int|long|None): padding_idx needs to be in the interval [-vocab_size, vocab_size).
            If :math:`padding\_idx < 0`, the :math:`padding\_idx` will automatically be converted
            to :math:`vocab\_size + padding\_idx` . It will output all-zero padding data whenever lookup
            encounters :math:`padding\_idx` in id. And the padding data will not be updated while training.
            If set None, it makes no effect to output. Default: None.
        param_attr(ParamAttr): To specify the weight parameter property. Default: None, which means the
            default weight parameter property is used. See usage for details in :ref:`api_fluid_ParamAttr` . In addition,
            user-defined or pre-trained word vectors can be loaded with the :attr:`param_attr` parameter.
            The local word vector needs to be transformed into numpy format, and the shape of local word
            vector should be consistent with :attr:`size` . Then :ref:`api_fluid_initializer_NumpyArrayInitializer`
            is used to load custom or pre-trained word vectors. See code example 2 for details.
        dtype(str|core.VarDesc.VarType): It refers to the data type of output Tensor.
            It must be float32 or float64. Default: float32.

    Returns:
        Variable: Embedding Tensor or LoDTensor mapped by input. The data type is the same as :attr:`dtype` .

    Examples:
        .. code-block:: python

          import paddle.fluid as fluid
          import numpy as np
          import paddle
          paddle.enable_static()

          data = fluid.data(name='x', shape=[None, 1], dtype='int64')

          # example 1
          emb_1 = fluid.embedding(input=data, size=[128, 64])

          # example 2: load custom or pre-trained word vectors
          weight_data = np.random.random(size=(128, 100))  # word vectors with numpy format
          w_param_attrs = fluid.ParamAttr(
              name="emb_weight",
              learning_rate=0.5,
              initializer=fluid.initializer.NumpyArrayInitializer(weight_data),
              trainable=True)
          emb_2 = fluid.layers.embedding(input=data, size=(128, 100), param_attr=w_param_attrs, dtype='float32')
    """

    helper = LayerHelper('embedding', **locals())
    check_variable_and_dtype(
        input, 'input', ['int64'], 'fluid.layers.embedding'
    )
    check_dtype(
        dtype,
        'dtype',
        ['uint16', 'float16', 'float32', 'float64'],
        'fluid.layers.embedding',
    )

    if is_distributed:
        is_distributed = False
        warnings.warn(
            "is_distributed is go out of use, `fluid.contrib.layers.sparse_embedding` is your needed"
        )

    remote_prefetch = True if is_sparse else False

    w = helper.create_parameter(
        attr=helper.param_attr, shape=size, dtype=dtype, is_bias=False
    )
    tmp = helper.create_variable_for_type_inference(dtype)
    padding_idx = (
        -1
        if padding_idx is None
        else padding_idx
        if padding_idx >= 0
        else (size[0] + padding_idx)
    )
    helper.append_op(
        type='lookup_table',
        inputs={'Ids': input, 'W': w},
        outputs={'Out': tmp},
        attrs={
            'is_sparse': is_sparse,
            'is_distributed': is_distributed,
            'remote_prefetch': remote_prefetch,
            'padding_idx': padding_idx,
        },
    )
    return tmp


def _pull_sparse(
    input,
    size,
    table_id,
    accessor_class,
    name="embedding",
    ctr_label_name="",
    padding_id=0,
    dtype='float32',
    scale_sparse_grad=True,
):
    r"""
    **Pull Fleet Sparse Layer**

    This layer is used to lookup embeddings of IDs, provided by :attr:`input`, in
    Fleet lookup table. The result of this lookup is the embedding of each ID in the
    :attr:`input`.

    Args:
        input(Variable|list of Variable): Input is a Tensor<int64> Variable, which
            contains the IDs information.
        size(int): The embedding size parameter, which indicates the size of
            each embedding vector respectively.
        table_id(int): the fleet table id of this embedding.
        accessor_class(str): the pslib accessor of the table, default is DownpourCtrAccessor.
        ctr_label_name(str): the layer name of click.
        padding_id(int): the padding id during lookup, default is 0.
        dtype(str): The dtype refers to the data type of output tensor. Only supports
            float32 now.
        scale_sparse_grad(bool): whether to scale sparse gradient with batch size. default
            is True.

    Returns:
        Variable|list of Variable: The tensor variable storing the embeddings of the \
                  supplied inputs.

    Examples:
        .. code-block:: python

          import paddle.fluid as fluid
          data = fluid.layers.data(name='sequence', shape=[1], dtype='int64', lod_level=1)
          emb = fluid.layers.nn._pull_sparse(
              input=data, size=11, table_id=0, accessor_class="DownpourCtrAccessor")
    """
    helper = LayerHelper(name, **locals())
    inputs = helper.multiple_input()
    outs = [helper.create_variable_for_type_inference(dtype)]
    input_names = [i.name for i in inputs]
    attrs = {
        'EmbeddingDim': size,
        'TableId': table_id,
        'AccessorClass': accessor_class,
        'CtrLabelName': ctr_label_name,
        'PaddingId': padding_id,
        'ScaleSparseGrad': scale_sparse_grad,
        'InputNames': input_names,
        # this is only for compatible with embedding op
        'is_distributed': True,
    }
    # this is only for compatible with embedding op
    w, _ = helper.create_or_get_global_variable(
        name=name, shape=[size], dtype=dtype, is_bias=False, persistable=True
    )
    helper.append_op(
        type='pull_sparse',
        inputs={'Ids': inputs, 'W': w},
        outputs={'Out': outs},
        attrs=attrs,
    )
    if len(outs) == 1:
        return outs[0]
    return outs


def _pull_sparse_v2(
    input,
    size,
    table_id,
    accessor_class,
    name="embedding",
    ctr_label_name="",
    padding_id=0,
    dtype='float32',
    scale_sparse_grad=True,
):
    r"""
    **Pull Fleet Sparse Layer**

    This layer is used to lookup embeddings of IDs, provided by :attr:`input`, in
    Fleet lookup table. The result of this lookup is the embedding of each ID in the
    :attr:`input`.

    Args:
        input(Variable|list of Variable): Input is a Tensor<int64> Variable, which
            contains the IDs information.
        size(int): The embedding size parameter, which indicates the size of
            each embedding vector respectively.
        table_id(int): the pslib table id of this embedding.
        accessor_class(str): the fleet accessor of the table, default is DownpourCtrAccessor.
        ctr_label_name(str): the layer name of click.
        padding_id(int): the padding id during lookup, default is 0.
        dtype(str): The dtype refers to the data type of output tensor. Only supports
            float32 now.
        scale_sparse_grad(bool): whether to scale sparse gradient with batch size. default
            is True.

    Returns:
        Variable|list of Variable: The tensor variable storing the embeddings of the \
                  supplied inputs.

    Examples:
        .. code-block:: python

          import paddle.fluid as fluid
          data = fluid.layers.data(name='sequence', shape=[1], dtype='int64', lod_level=1)
          emb = fluid.layers.nn._pull_sparse_v2(
              input=data, size=11, table_id=0, accessor_class="DownpourCtrAccessor")
    """
    helper = LayerHelper(name, **locals())
    inputs = helper.multiple_input()
    outs = [helper.create_variable_for_type_inference(dtype)]
    input_names = [i.name for i in inputs]
    attrs = {
        'EmbeddingDim': size,
        'TableId': table_id,
        'AccessorClass': accessor_class,
        'CtrLabelName': ctr_label_name,
        'PaddingId': padding_id,
        'ScaleSparseGrad': scale_sparse_grad,
        'InputNames': input_names,
        # this is only for compatible with embedding op
        'is_distributed': True,
    }
    # this is only for compatible with embedding op
    w, _ = helper.create_or_get_global_variable(
        name=name, shape=[size], dtype=dtype, is_bias=False, persistable=True
    )
    helper.append_op(
        type='pull_sparse_v2',
        inputs={'Ids': inputs, 'W': w},
        outputs={'Out': outs},
        attrs=attrs,
    )
    if len(outs) == 1:
        return outs[0]
    return outs


def _pull_gpups_sparse(
    input, size, dtype='float32', is_distributed=False, is_sparse=False
):
    r"""
    **Pull GpuPS Sparse Layer**

    This layer is used to lookup embeddings of IDs, provided by :attr:`input`, in
    GpuPS lookup table. The result of this lookup is the embedding of each ID in the
    :attr:`input`.

    Args:
        input(Variable|list of Variable): Input is a Tensor<int64> Variable, which
            contains the IDs information.
        size(int|list of int): The embedding size parameter of each input, which indicates the size of
            each embedding vector respectively.
        dtype(str): The dtype refers to the data type of output tensor. Only supports
        float32 now.

    Returns:
        Variable|list of Variable: The tensor variable storing the embeddings of the \
                  supplied inputs, whose size are indicated by size respectively.

    Examples:
        .. code-block:: python

          import paddle.fluid as fluid
          slots = []
          data_1 = fluid.layers.data(name='sequence', shape=[1], dtype='int64', lod_level=1)
          slots.append(data_1)
          data_2 = fluid.layers.data(name='sequence', shape=[1], dtype='int64', lod_level=1)
          slots.append(data_2)
          embs = fluid.layers.pull_gpups_sparse(input=slots, size=[11, 35])
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
        input(Variable|list of Variable): Input is a Tensor<int64> Variable, which
            contains the IDs information.
        size(int): The embedding size parameter, which indicates the size of
            each embedding vector respectively.
        dtype(str): The dtype refers to the data type of output tensor. Only supports
        float32 now.

    Returns:
        Variable|list of Variable: The tensor variable storing the embeddings of the \
                  supplied inputs.

    Examples:
        .. code-block:: python

          import paddle.fluid as fluid
          data = fluid.layers.data(name='sequence', shape=[1], dtype='int64', lod_level=1)
          emb = fluid.layers.pull_box_sparse(input=data, size=[11])
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


@templatedoc()
def linear_chain_crf(input, label, param_attr=None, length=None):
    """
    :api_attr: Static Graph

    Linear Chain CRF.

    ${comment}

    Args:
        input(${emission_type}): ${emission_comment}
        label(${label_type}): ${label_comment}
        Length(${length_type}): ${length_comment}
        param_attr(ParamAttr): The attribute of the learnable parameter for transition parameter.

    Returns:
        output(${emission_exps_type}): ${emission_exps_comment} \n
        output(${transition_exps_type}): ${transition_exps_comment} \n
        output(${log_likelihood_type}): ${log_likelihood_comment} \n

    Examples:
        .. code-block:: python

            import paddle.fluid as fluid
            import numpy as np
            import paddle
            paddle.enable_static()

            #define net structure, using LodTensor
            train_program = fluid.Program()
            startup_program = fluid.Program()
            with fluid.program_guard(train_program, startup_program):
                input_data = fluid.data(name='input_data', shape=[-1,10], dtype='float32')
                label = fluid.data(name='label', shape=[-1,1], dtype='int')
                emission= fluid.layers.fc(input=input_data, size=10, act="tanh")
                crf_cost = fluid.layers.linear_chain_crf(
                    input=emission,
                    label=label,
                    param_attr=fluid.ParamAttr(
                    name='crfw',
                    learning_rate=0.01))
            use_cuda = False
            place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
            exe = fluid.Executor(place)
            exe.run(startup_program)
            #define data, using LoDTensor
            a = fluid.create_lod_tensor(np.random.rand(12,10).astype('float32'), [[3,3,4,2]], place)
            b = fluid.create_lod_tensor(np.array([[1],[1],[2],[3],[1],[1],[1],[3],[1],[1],[1],[1]]),[[3,3,4,2]] , place)
            feed1 = {'input_data':a,'label':b}
            loss= exe.run(train_program,feed=feed1, fetch_list=[crf_cost])
            print(loss)

            #define net structure, using padding
            train_program = fluid.Program()
            startup_program = fluid.Program()
            with fluid.program_guard(train_program, startup_program):
                input_data2 = fluid.data(name='input_data2', shape=[-1,10,10], dtype='float32')
                label2 = fluid.data(name='label2', shape=[-1,10,1], dtype='int')
                label_length = fluid.data(name='length', shape=[-1,1], dtype='int')
                emission2= fluid.layers.fc(input=input_data2, size=10, act="tanh", num_flatten_dims=2)
                crf_cost2 = fluid.layers.linear_chain_crf(
                    input=emission2,
                    label=label2,
                    length=label_length,
                    param_attr=fluid.ParamAttr(
                     name='crfw',
                     learning_rate=0.01))

            use_cuda = False
            place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
            exe = fluid.Executor(place)
            exe.run(startup_program)

            #define data, using padding
            cc=np.random.rand(4,10,10).astype('float32')
            dd=np.random.rand(4,10,1).astype('int64')
            ll=np.array([[3],[3],[4],[2]])
            feed2 = {'input_data2':cc,'label2':dd,'length':ll}
            loss2= exe.run(train_program,feed=feed2, fetch_list=[crf_cost2])
            print(loss2)
            #[array([[ 7.8902354],
            #        [ 7.3602567],
            #        [ 10.004011],
            #        [ 5.86721  ]], dtype=float32)]

            #you can use find_var to get transition parameter.
            transition=np.array(fluid.global_scope().find_var('crfw').get_tensor())
            print(transition)

    """
    check_variable_and_dtype(
        input, 'input', ['float32', 'float64'], 'linear_chain_crf'
    )
    check_variable_and_dtype(label, 'label', ['int64'], 'linear_chain_crf')
    helper = LayerHelper('linear_chain_crf', **locals())
    size = input.shape[2] if length else input.shape[1]
    transition = helper.create_parameter(
        attr=helper.param_attr,
        shape=[size + 2, size],
        dtype=helper.input_dtype(),
    )
    alpha = helper.create_variable_for_type_inference(
        dtype=helper.input_dtype()
    )
    emission_exps = helper.create_variable_for_type_inference(
        dtype=helper.input_dtype()
    )
    transition_exps = helper.create_variable_for_type_inference(
        dtype=helper.input_dtype()
    )
    log_likelihood = helper.create_variable_for_type_inference(
        dtype=helper.input_dtype()
    )
    this_inputs = {
        "Emission": [input],
        "Transition": transition,
        "Label": [label],
    }
    if length:
        this_inputs['Length'] = [length]
    helper.append_op(
        type='linear_chain_crf',
        inputs=this_inputs,
        outputs={
            "Alpha": [alpha],
            "EmissionExps": [emission_exps],
            "TransitionExps": transition_exps,
            "LogLikelihood": log_likelihood,
        },
    )

    return log_likelihood


@templatedoc()
def crf_decoding(input, param_attr, label=None, length=None):
    """
    :api_attr: Static Graph

    ${comment}

    Args:
        input(Tensor): ${emission_comment}

        param_attr (ParamAttr|None): To specify the weight parameter attribute.
            Default: None, which means the default weight parameter property is
            used. See usage for details in :ref:`api_paddle_fluid_param_attr_ParamAttr` .

        label(${label_type}, optional): ${label_comment}

        length(${length_type}, optional): ${length_comment}

    Returns:
        Tensor: ${viterbi_path_comment}

    Examples:
        .. code-block:: python

           import paddle
           paddle.enable_static()

           # LoDTensor-based example
           num_labels = 10
           feature = paddle.static.data(name='word_emb', shape=[-1, 784], dtype='float32', lod_level=1)
           label = paddle.static.data(name='label', shape=[-1, 1], dtype='int64', lod_level=1)
           emission = paddle.static.nn.fc(feature, size=num_labels)

           crf_cost = paddle.fluid.layers.linear_chain_crf(input=emission, label=label,
                     param_attr=paddle.ParamAttr(name="crfw"))
           crf_decode = paddle.static.nn.crf_decoding(input=emission,
                     param_attr=paddle.ParamAttr(name="crfw"))

           # Common tensor example
           num_labels, max_len = 10, 20
           feature = paddle.static.data(name='word_emb_pad', shape=[-1, max_len, 784], dtype='float32')
           label = paddle.static.data(name='label_pad', shape=[-1, max_len, 1], dtype='int64')
           length = paddle.static.data(name='length', shape=[-1, 1], dtype='int64')
           emission = paddle.static.nn.fc(feature, size=num_labels,
                                      num_flatten_dims=2)

           crf_cost = paddle.fluid.layers.linear_chain_crf(input=emission, label=label, length=length,
                     param_attr=paddle.ParamAttr(name="crfw_pad"))
           crf_decode = paddle.static.nn.crf_decoding(input=emission, length=length,
                     param_attr=paddle.ParamAttr(name="crfw_pad"))
    """
    check_variable_and_dtype(
        input, 'input', ['float32', 'float64'], 'crf_decoding'
    )
    helper = LayerHelper('crf_decoding', **locals())
    transition = helper.get_parameter(param_attr.name)
    viterbi_path = helper.create_variable_for_type_inference(
        dtype=core.VarDesc.VarType.INT64
    )
    inputs = {"Emission": [input], "Transition": transition, "Label": label}
    if length:
        inputs['Length'] = length
    helper.append_op(
        type='crf_decoding',
        inputs=inputs,
        outputs={"ViterbiPath": [viterbi_path]},
    )

    return viterbi_path


@deprecated(since="2.0.0", update_to="paddle.nn.functional.dropout")
def dropout(
    x,
    dropout_prob,
    is_test=None,
    seed=None,
    name=None,
    dropout_implementation="downgrade_in_infer",
):
    """

    Computes dropout.

    Drop or keep each element of `x` independently. Dropout is a regularization
    technique for reducing overfitting by preventing neuron co-adaption during
    training. The dropout operator randomly sets (according to the given dropout
    probability) the outputs of some units to zero, while others are remain
    unchanged.

    dropout op can be removed from the program to make the program more efficient.

    Args:
        x (Variable): The input tensor variable. The data type is float16 or float32 or float64.
        dropout_prob (float): Probability of setting units to zero.
        is_test (bool): A flag indicating whether it is in test phrase or not.
                        Default None, in dynamic graph, it use global tracer mode; in static graph, it means False.
        seed (int): A Python integer used to create random seeds. If this
                    parameter is set to None, a random seed is used.
                    NOTE: If an integer seed is given, always the same output
                    units will be dropped. DO NOT use a fixed seed in training.Default: None.
        name (str|None): A name for this layer(optional). If set None, the layer
                         will be named automatically.
        dropout_implementation(string): ['downgrade_in_infer'(default)|'upscale_in_train']

                                        1. downgrade_in_infer(default), downgrade the outcome at inference

                                           - train: out = input * mask
                                           - inference: out = input * (1.0 - dropout_prob)

                                           (mask is a tensor same shape with input, value is 0 or 1
                                           ratio of 0 is dropout_prob)
                                        2. upscale_in_train, upscale the outcome at training time

                                           - train: out = input * mask / ( 1.0 - dropout_prob )
                                           - inference: out = input

                                           (mask is a tensor same shape with input, value is 0 or 1
                                           ratio of 0 is dropout_prob)


    Returns:
        A Variable holding Tensor representing the dropout, has same shape and data type with `x`.

    Examples:

        .. code-block:: python

            import paddle
            import paddle.fluid as fluid

            paddle.enable_static()
            x = fluid.data(name="data", shape=[None, 32, 32], dtype="float32")
            dropped = fluid.layers.dropout(x, dropout_prob=0.5)
    """
    if not isinstance(dropout_prob, (float, int, Variable)):
        raise TypeError(
            "dropout_prob argument should be a number(int|float) or Variable"
        )
    # fast return for p == 0
    if isinstance(dropout_prob, (int, float)) and dropout_prob == 0:
        return x

    if _non_static_mode():
        if (
            seed is None or seed == 0
        ) and default_main_program().random_seed != 0:
            seed = default_main_program().random_seed
        if is_test is None:
            is_test = not _dygraph_tracer()._train_mode
        out, mask = _legacy_C_ops.dropout(
            x,
            'dropout_prob',
            dropout_prob,
            'is_test',
            is_test,
            'fix_seed',
            seed is not None,
            'seed',
            seed if seed is not None else 0,
            'dropout_implementation',
            dropout_implementation,
        )
        return out

    def get_attrs(prog, dropout_prob, is_test, seed):
        if (seed is None or seed == 0) and prog.random_seed != 0:
            seed = prog.random_seed
        if isinstance(dropout_prob, Variable) and not dropout_prob.shape != [1]:
            raise TypeError(
                "Required dropout_prob.shape == [1] if type(dropout_prob) is Variable, but received dropout_prob.shape = {}".format(
                    dropout_prob.shape
                )
            )
        attrs = {
            'dropout_prob': dropout_prob,
            'is_test': is_test,
            'fix_seed': seed is not None,
            'seed': seed if seed is not None else 0,
            'dropout_implementation': dropout_implementation,
        }
        return attrs

    helper = LayerHelper('dropout', **locals())
    check_variable_and_dtype(
        x, 'x', ['float16', 'float32', 'float64'], 'dropout'
    )

    out = helper.create_variable_for_type_inference(dtype=x.dtype)
    mask = helper.create_variable_for_type_inference(
        dtype=core.VarDesc.VarType.UINT8, stop_gradient=True
    )

    attrs = get_attrs(helper.main_program, dropout_prob, is_test, seed)

    helper.append_op(
        type='dropout',
        inputs={'X': [x]},
        outputs={'Out': [out], 'Mask': [mask]},
        attrs=attrs,
    )
    return out


def conv2d(
    input,
    num_filters,
    filter_size,
    stride=1,
    padding=0,
    dilation=1,
    groups=None,
    param_attr=None,
    bias_attr=None,
    use_cudnn=True,
    act=None,
    name=None,
    data_format="NCHW",
):
    r"""
    :api_attr: Static Graph

    The convolution2D layer calculates the output based on the input, filter
    and strides, paddings, dilations, groups parameters. Input and
    Output are in NCHW or NHWC format, where N is batch size, C is the number of
    channels, H is the height of the feature, and W is the width of the feature.
    Filter is in MCHW format, where M is the number of output image channels,
    C is the number of input image channels, H is the height of the filter,
    and W is the width of the filter. If the groups is greater than 1,
    C will equal the number of input image channels divided by the groups.
    Please refer to UFLDL's `convolution
    <http://ufldl.stanford.edu/tutorial/supervised/FeatureExtractionUsingConvolution/>`_
    for more details.
    If bias attribution and activation type are provided, bias is added to the
    output of the convolution, and the corresponding activation function is
    applied to the final result.

    For each input :math:`X`, the equation is:

    .. math::

        Out = \sigma (W \\ast X + b)

    Where:

    * :math:`X`: Input value, a tensor with NCHW or NHWC format.
    * :math:`W`: Filter value, a tensor with MCHW format.
    * :math:`\\ast`: Convolution operation.
    * :math:`b`: Bias value, a 2-D tensor with shape [M, 1].
    * :math:`\\sigma`: Activation function.
    * :math:`Out`: Output value, the shape of :math:`Out` and :math:`X` may be different.

    Example:

        - Input:

          Input shape: :math:`(N, C_{in}, H_{in}, W_{in})`

          Filter shape: :math:`(C_{out}, C_{in}, H_f, W_f)`

        - Output:

          Output shape: :math:`(N, C_{out}, H_{out}, W_{out})`

        Where

        .. math::

            H_{out}&= \\frac{(H_{in} + 2 * paddings[0] - (dilations[0] * (H_f - 1) + 1))}{strides[0]} + 1 \\\\
            W_{out}&= \\frac{(W_{in} + 2 * paddings[1] - (dilations[1] * (W_f - 1) + 1))}{strides[1]} + 1

    Args:
        input (Tensor): The input is 4-D Tensor with shape [N, C, H, W], the data type
            of input is float16 or float32 or float64.
        num_filters(int): The number of filter. It is as same as the output
            image channel.
        filter_size (int|tuple): The filter size. If filter_size
            is a tuple, it must contain two integers, (filter_size_height,
            filter_size_width). Otherwise, filter_size_height = filter_size_width =\
            filter_size.
        stride (int|tuple): The stride size. It means the stride in convolution.
            If stride is a tuple, it must contain two integers, (stride_height, stride_width).
            Otherwise, stride_height = stride_width = stride. Default: stride = 1.
        padding (string|int|list|tuple): The padding size. It means the number of zero-paddings
            on both sides for each dimension.If `padding` is a string, either 'VALID' or
            'SAME' which is the padding algorithm. If padding size is a tuple or list,
            it could be in three forms: `[pad_height, pad_width]` or
            `[pad_height_top, pad_height_bottom, pad_width_left, pad_width_right]`, and when
            `data_format` is `"NCHW"`, `padding` can be in the form `[[0,0], [0,0],
            [pad_height_top, pad_height_bottom], [pad_width_left, pad_width_right]]`.
            when `data_format` is `"NHWC"`, `pool_padding` can be in the form
            `[[0,0], [pad_height_top, pad_height_bottom], [pad_width_left, pad_width_right], [0,0]]`.
            Default: padding = 0.
        dilation (int|tuple): The dilation size. It means the spacing between the kernel
            points. If dilation is a tuple, it must contain two integers, (dilation_height,
            dilation_width). Otherwise, dilation_height = dilation_width = dilation.
            Default: dilation = 1.
        groups (int): The groups number of the Conv2d Layer. According to grouped
            convolution in Alex Krizhevsky's Deep CNN paper: when group=2,
            the first half of the filters is only connected to the first half
            of the input channels, while the second half of the filters is only
            connected to the second half of the input channels. Default: groups=1.
        param_attr (ParamAttr|None): The parameter attribute for learnable parameters/weights
            of conv2d. If it is set to None or one attribute of ParamAttr, conv2d
            will create ParamAttr as param_attr. If the Initializer of the param_attr
            is not set, the parameter is initialized with :math:`Normal(0.0, std)`,
            and the :math:`std` is :math:`(\\frac{2.0 }{filter\_elem\_num})^{0.5}`. Default: None.
        bias_attr (ParamAttr|bool|None): The parameter attribute for the bias of conv2d.
            If it is set to False, no bias will be added to the output units.
            If it is set to None or one attribute of ParamAttr, conv2d
            will create ParamAttr as bias_attr. If the Initializer of the bias_attr
            is not set, the bias is initialized zero. Default: None.
        use_cudnn (bool): Use cudnn kernel or not, it is valid only when the cudnn
            library is installed. Default: True
        act (str): Activation type, if it is set to None, activation is not appended.
            Default: None
        name(str|None): For detailed information, please refer
           to :ref:`api_guide_Name`. Usually name is no need to set and
           None by default.
        data_format (str, optional): Specify the data format of the input, and the data format of the output
            will be consistent with that of the input. An optional string from: `"NCHW"`, `"NHWC"`.
            The default is `"NCHW"`. When it is `"NCHW"`, the data is stored in the order of:
            `[batch_size, input_channels, input_height, input_width]`.

    Returns:
        A Tensor representing the conv2d, whose data type is the
        same with input. If act is None, the tensor storing the convolution
        result, and if act is not None, the tensor storing convolution
        and non-linearity activation result.

    Raises:
        ValueError: If the type of `use_cudnn` is not bool.
        ValueError: If `data_format` is not "NCHW" or "NHWC".
        ValueError: If the channel dimmention of the input is less than or equal to zero.
        ValueError: If `padding` is a string, but not "SAME" or "VALID".
        ValueError: If `padding` is a tuple, but the element corresponding to the input's batch size is not 0
            or the element corresponding to the input's channel is not 0.
        ShapeError: If the input is not 4-D Tensor.
        ShapeError: If the input's dimension size and filter's dimension size not equal.
        ShapeError: If the dimension size of input minus the size of `stride` is not 2.
        ShapeError: If the number of input channels is not equal to filter's channels * groups.
        ShapeError: If the number of output channels is not be divided by groups.

    Examples:
        .. code-block:: python

          import paddle
          paddle.enable_static()

          data = paddle.static.data(name='data', shape=[None, 3, 32, 32], dtype='float32')
          conv2d = paddle.static.nn.conv2d(input=data, num_filters=2, filter_size=3, act="relu")
          print(conv2d.shape) # [-1, 2, 30, 30]
    """

    check_variable_and_dtype(
        input, 'input', ['float16', 'float32', 'float64'], 'conv2d'
    )
    if len(input.shape) != 4:
        raise ValueError(
            "Input size should be 4, "
            "but received {}".format(len(input.shape))
        )
    num_channels = input.shape[1]
    if not isinstance(use_cudnn, bool):
        raise ValueError(
            "Attr(use_cudnn) should be True or False. Received "
            "Attr(use_cudnn): %s. " % str(use_cudnn)
        )

    if data_format not in ["NCHW", "NHWC"]:
        raise ValueError(
            "Attr(data_format) should be 'NCHW' or 'NHWC'. Received "
            "Attr(data_format): %s." % str(data_format)
        )

    channel_last = data_format == "NHWC"
    num_channels = input.shape[3] if channel_last else input.shape[1]
    if num_channels < 0:
        raise ValueError(
            "The channel dimmention of the input(%s) should be defined. "
            "Received: %s." % (str(input.shape), str(num_channels))
        )
    assert param_attr is not False, "param_attr should not be False here."

    if groups is None:
        num_filter_channels = num_channels
    elif groups <= 0:
        raise ValueError(
            "the groups of input must be greater than 0, "
            "but received the groups of input is {}".format(groups)
        )
    else:
        if num_channels % groups != 0:
            raise ValueError(
                "the channel of input must be divisible by groups,"
                "received: the channel of input is {}, the shape of input is {}"
                ", the groups is {}".format(num_channels, input.shape, groups)
            )
        num_filter_channels = num_channels // groups

    l_type = 'conv2d'
    if (
        num_channels == groups
        and num_filters % num_channels == 0
        and not use_cudnn
    ):
        l_type = 'depthwise_conv2d'

    if (
        num_channels == groups
        and num_filters % num_channels == 0
        and core.is_compiled_with_rocm()
    ):
        l_type = 'depthwise_conv2d'

    # NPU only supports depthwise_conv2d when  "input_channel = output_channel = groups"
    if core.is_compiled_with_npu():
        if num_channels == groups and num_channels == num_filters:
            l_type = 'depthwise_conv2d'
        else:
            l_type = 'conv2d'

    helper = LayerHelper(l_type, **locals())
    dtype = helper.input_dtype()

    filter_size = utils.convert_to_list(filter_size, 2, 'filter_size')
    stride = utils.convert_to_list(stride, 2, 'stride')
    dilation = utils.convert_to_list(dilation, 2, 'dilation')

    # padding
    def _update_padding(padding, data_format):
        def is_list_or_tuple(ele):
            if isinstance(ele, list) or isinstance(ele, tuple):
                return True
            return False

        if is_list_or_tuple(padding) and len(padding) == 4:
            if is_list_or_tuple(padding[0]) and (data_format == "NCHW"):
                if not (padding[0] == [0, 0] and padding[1] == [0, 0]):
                    raise ValueError(
                        "Non-zero padding(%s) in the batch or channel dimensions "
                        "is not supported." % str(padding)
                    )
                padding = padding[2:4]
                padding = [ele for a_list in padding for ele in a_list]
            elif is_list_or_tuple(padding[0]) and (data_format == "NHWC"):
                if not (padding[0] == [0, 0] and padding[3] == [0, 0]):
                    raise ValueError(
                        "Non-zero padding(%s) in the batch or channel dimensions "
                        "is not supported." % str(padding)
                    )
                padding = padding[1:3]
                padding = [ele for a_list in padding for ele in a_list]
            padding = utils.convert_to_list(padding, 4, 'padding')
            if utils._is_symmetric_padding(padding, 2):
                padding = [padding[0], padding[2]]

        else:
            padding = utils.convert_to_list(padding, 2, 'padding')

        return padding

    padding_algorithm = "EXPLICIT"
    if isinstance(padding, str):
        padding = padding.upper()
        if padding not in ["SAME", "VALID"]:
            raise ValueError(
                "Unknown padding: '%s'. It can only be 'SAME' or 'VALID'."
                % str(padding)
            )
        if padding == "VALID":
            padding_algorithm = "VALID"
            padding = [0, 0]
        elif padding == "SAME":
            padding_algorithm = "SAME"
            padding = [0, 0]

    padding = _update_padding(padding, data_format)

    filter_shape = [num_filters, int(num_filter_channels)] + filter_size

    def _get_default_param_initializer():
        filter_elem_num = filter_size[0] * filter_size[1] * num_channels
        if filter_elem_num <= 0:
            raise ValueError(
                "Invalid filter number, excepted number is larger than 0, but"
                " received {}, please check the input shape and "
                "filter size.".format(filter_elem_num)
            )
        std = (2.0 / filter_elem_num) ** 0.5
        return Normal(0.0, std, 0)

    filter_param = helper.create_parameter(
        attr=helper.param_attr,
        shape=filter_shape,
        dtype=dtype,
        default_initializer=_get_default_param_initializer(),
    )

    pre_bias = helper.create_variable_for_type_inference(dtype)

    if (
        core.is_compiled_with_cuda()
        and paddle.fluid.get_flags("FLAGS_conv2d_disable_cudnn")[
            "FLAGS_conv2d_disable_cudnn"
        ]
    ):
        use_cudnn = False

    helper.append_op(
        type=l_type,
        inputs={
            'Input': input,
            'Filter': filter_param,
        },
        outputs={"Output": pre_bias},
        attrs={
            'strides': stride,
            'paddings': padding,
            'dilations': dilation,
            'groups': groups,
            'use_cudnn': use_cudnn,
            'use_mkldnn': False,
            'fuse_relu_before_depthwise_conv': False,
            "padding_algorithm": padding_algorithm,
            "data_format": data_format,
        },
    )

    if data_format == 'NCHW':
        pre_act = helper.append_bias_op(pre_bias, dim_start=1, dim_end=2)
    else:
        pre_act = helper.append_bias_op(pre_bias, dim_start=3, dim_end=4)

    return helper.append_activation(pre_act)


@templatedoc()
def pool2d(
    input,
    pool_size=-1,
    pool_type="max",
    pool_stride=1,
    pool_padding=0,
    global_pooling=False,
    use_cudnn=True,
    ceil_mode=False,
    name=None,
    exclusive=True,
    data_format="NCHW",
):
    """

    ${comment}

    Args:
        input (Variable): The input tensor of pooling operator which is a 4-D tensor with
                          shape [N, C, H, W]. The format of input tensor is `"NCHW"` or
                          `"NHWC"`, where `N` is batch size, `C` is the number of channels,
                          `H` is the height of the feature, and `W` is the width of the
                          feature. The data type if float32 or float64.
        pool_size (int|list|tuple): The pool kernel size. If pool kernel size is a tuple or list,
            it must contain two integers, (pool_size_Height, pool_size_Width).
            Otherwise, the pool kernel size will be a square of an int.
        pool_type: ${pooling_type_comment}
        pool_stride (int|list|tuple): The pool stride size. If pool stride size is a tuple or list,
            it must contain two integers, (pool_stride_Height, pool_stride_Width).
            Otherwise, the pool stride size will be a square of an int.
        pool_padding (string|int|list|tuple): The pool padding. If `pool_padding` is a string, either 'VALID' or
            'SAME' which is the padding algorithm. If pool padding size is a tuple or list,
            it could be in three forms: `[pad_height, pad_width]` or
            `[pad_height_top, pad_height_bottom, pad_width_left, pad_width_right]`, and when `data_format` is `"NCHW"`,
            `pool_padding` can be in the form `[[0,0], [0,0], [pad_height_top, pad_height_bottom], [pad_width_left, pad_width_right]]`.
            when `data_format` is `"NHWC"`, `pool_padding` can be in the form
            `[[0,0], [pad_height_top, pad_height_bottom], [pad_width_left, pad_width_right], [0,0]]`.
            Otherwise, the pool padding size will be a square of an int.
        global_pooling (bool): ${global_pooling_comment}
        use_cudnn (bool): ${use_cudnn_comment}
        ceil_mode (bool): ${ceil_mode_comment}
        name(str, optional): For detailed information, please refer
                             to :ref:`api_guide_Name`. Usually name is no need to set and
                             None by default.
        exclusive (bool): Whether to exclude padding points in average pooling
                          mode, default is `true`.
        data_format (string): The data format of the input and output data. An optional string from: `"NCHW"`, `"NHWC"`.
                The default is `"NCHW"`. When it is `"NCHW"`, the data is stored in the order of:
                `[batch_size, input_channels, input_height, input_width]`.

    Returns:
        Variable: The output tensor of pooling result. The data type is same as input tensor.

    Raises:
        ValueError: If `pool_type` is not "max" nor "avg".
        ValueError: If `global_pooling` is False and `pool_size` is -1.
        TypeError: If `use_cudnn` is not a bool value.
        ValueError: If `data_format` is not "NCHW" or "NHWC".
        ValueError: If `pool_padding` is a string, but not "SAME" or "VALID".
        ValueError: If `pool_padding` is "VALID", but `ceil_mode` is True.
        ValueError: If `pool_padding` is a list or tuple, but the elements in the batch or channel dimensions are non-zero.
        ShapeError: If the input is not a 4-D or 5-D Tensor.
        ShapeError: If the dimension of input minus the size of `pool_stride` is not 2.
        ShapeError: If the size of `pool_size` and `pool_stride` is not equal.
        ShapeError: If the output's shape calculated is not greater than 0.


    Examples:

        .. code-block:: python

          import paddle.fluid as fluid
          import paddle

          paddle.enable_static()

          data = fluid.data(name='data', shape=[None, 3, 32, 32], dtype='float32')

          # max pool2d
          pool2d = fluid.layers.pool2d(
            input = data,
            pool_size = 2,
            pool_type = "max",
            pool_stride = 1,
            global_pooling=False)

          # average pool2d
          pool2d = fluid.layers.pool2d(
            input = data,
            pool_size = 2,
            pool_type = "avg",
            pool_stride = 1,
            global_pooling=False)

          # global average pool2d
          pool2d = fluid.layers.pool2d(
            input = data,
            pool_size = 2,
            pool_type = "avg",
            pool_stride = 1,
            global_pooling=True)

          # Attr(pool_padding) is a list with 4 elements, Attr(data_format) is "NCHW".
          out_1 = fluid.layers.pool2d(
            input = data,
            pool_size = 3,
            pool_type = "avg",
            pool_stride = 1,
            pool_padding = [1, 2, 1, 0],
            data_format = "NCHW")

          # Attr(pool_padding) is a string, Attr(data_format) is "NCHW".
          out_2 = fluid.layers.pool2d(
            input = data,
            pool_size = 3,
            pool_type = "avg",
            pool_stride = 1,
            pool_padding = "VALID",
            data_format = "NCHW")
    """
    if pool_type not in ["max", "avg"]:
        raise ValueError(
            "Unknown Attr(pool_type): '%s'. It can only be 'max' or 'avg'.",
            str(pool_type),
        )

    if global_pooling is False and pool_size == -1:
        raise ValueError(
            "When Attr(global_pooling) is False, Attr(pool_size) must be passed "
            "and be a valid value. Received pool_size: %s." % str(pool_size)
        )

    if not isinstance(use_cudnn, bool):
        raise TypeError(
            "Attr(use_cudnn) should be True or False. Received "
            "Attr(use_cudnn): %s." % str(use_cudnn)
        )

    if data_format not in ["NCHW", "NHWC"]:
        raise ValueError(
            "Attr(data_format) should be 'NCHW' or 'NHWC'. Received "
            "Attr(data_format): %s." % str(data_format)
        )

    pool_size = utils.convert_to_list(pool_size, 2, 'pool_size')
    pool_stride = utils.convert_to_list(pool_stride, 2, 'pool_stride')

    def update_padding(padding, data_format):
        def is_list_or_tuple(ele):
            if isinstance(ele, list) or isinstance(ele, tuple):
                return True
            return False

        if is_list_or_tuple(padding) and len(padding) == 4:
            if is_list_or_tuple(padding[0]) and (data_format == "NCHW"):
                if not (padding[0] == [0, 0] and padding[1] == [0, 0]):
                    raise ValueError(
                        "Non-zero pool_padding(%s) in the batch or channel dimensions "
                        "is not supported." % str(padding)
                    )
                padding = padding[2:4]
                padding = [ele for a_list in padding for ele in a_list]
            elif is_list_or_tuple(padding[0]) and (data_format == "NHWC"):
                if not (padding[0] == [0, 0] and padding[3] == [0, 0]):
                    raise ValueError(
                        "Non-zero pool_padding(%s) in the batch or channel dimensions "
                        "is not supported." % str(padding)
                    )
                padding = padding[1:3]
                padding = [ele for a_list in padding for ele in a_list]
            padding = utils.convert_to_list(padding, 4, 'padding')

            if utils._is_symmetric_padding(padding, 2):
                padding = [padding[0], padding[2]]
        else:
            padding = utils.convert_to_list(padding, 2, 'padding')

        return padding

    padding_algorithm = "EXPLICIT"
    if isinstance(pool_padding, str):
        pool_padding = pool_padding.upper()
        if pool_padding not in ["SAME", "VALID"]:
            raise ValueError(
                "Unknown Attr(pool_padding): '%s'. It can only be 'SAME' or 'VALID'."
                % str(pool_padding)
            )
        if pool_padding == "VALID":
            padding_algorithm = "VALID"
            pool_padding = [0, 0]
            if ceil_mode is not False:
                raise ValueError(
                    "When Attr(pool_padding) is \"VALID\", Attr(ceil_mode) must be False. "
                    "Received ceil_mode: True."
                )
        elif pool_padding == "SAME":
            padding_algorithm = "SAME"
            pool_padding = [0, 0]

    pool_padding = update_padding(pool_padding, data_format)
    if in_dygraph_mode():
        input = input._use_gpudnn(use_cudnn)
        return _C_ops.pool2d(
            input,
            pool_size,
            pool_stride,
            pool_padding,
            ceil_mode,
            exclusive,
            data_format,
            pool_type,
            global_pooling,
            False,
            padding_algorithm,
        )
    op_type = 'pool2d'
    helper = LayerHelper(op_type, **locals())
    dtype = helper.input_dtype()
    pool_out = helper.create_variable_for_type_inference(dtype)

    helper.append_op(
        type=op_type,
        inputs={"X": input},
        outputs={"Out": pool_out},
        attrs={
            "pooling_type": pool_type,
            "ksize": pool_size,
            "global_pooling": global_pooling,
            "strides": pool_stride,
            "paddings": pool_padding,
            "padding_algorithm": padding_algorithm,
            "use_cudnn": use_cudnn,
            "ceil_mode": ceil_mode,
            "use_mkldnn": False,
            "exclusive": exclusive,
            "data_format": data_format,
        },
    )

    return pool_out


def batch_norm(
    input,
    act=None,
    is_test=False,
    momentum=0.9,
    epsilon=1e-05,
    param_attr=None,
    bias_attr=None,
    data_layout='NCHW',
    in_place=False,
    name=None,
    moving_mean_name=None,
    moving_variance_name=None,
    do_model_average_for_mean_and_var=True,
    use_global_stats=False,
):
    r"""
    :api_attr: Static Graph

    **Batch Normalization Layer**

    Can be used as a normalizer function for convolution or fully_connected operations.
    The required data format for this layer is one of the following:

    1. NHWC `[batch, in_height, in_width, in_channels]`

    2. NCHW `[batch, in_channels, in_height, in_width]`

    Refer to `Batch Normalization: Accelerating Deep Network Training by Reducing
    Internal Covariate Shift <https://arxiv.org/pdf/1502.03167.pdf>`_
    for more details.

    :math:`input` is the input features over a mini-batch.

    ..  math::

        \\mu_{\\beta} &\\gets \\frac{1}{m} \\sum_{i=1}^{m} x_i \\qquad &//\\
        \ mini-batch\ mean \\\\
        \\sigma_{\\beta}^{2} &\\gets \\frac{1}{m} \\sum_{i=1}^{m}(x_i - \\
        \\mu_{\\beta})^2 \\qquad &//\ mini-batch\ variance \\\\
        \\hat{x_i} &\\gets \\frac{x_i - \\mu_\\beta} {\\sqrt{\\
        \\sigma_{\\beta}^{2} + \\epsilon}} \\qquad &//\ normalize \\\\
        y_i &\\gets \\gamma \\hat{x_i} + \\beta \\qquad &//\ scale\ and\ shift

        moving\_mean = moving\_mean * momentum + mini-batch\_mean * (1. - momentum) \\\\
        moving\_var = moving\_var * momentum + mini-batch\_var * (1. - momentum)


    moving_mean is global mean and moving_var is global variance.

    When use_global_stats = True, the :math:`\\mu_{\\beta}`
    and :math:`\\sigma_{\\beta}^{2}` are not the statistics of one mini-batch.
    They are global (or running) statistics. (It usually got from the
    pre-trained model.)
    The training and testing (or inference) have the same behavior:

    ..  math::

        \\hat{x_i} &\\gets \\frac{x_i - \\mu_\\beta} {\\sqrt{\\
        \\sigma_{\\beta}^{2} + \\epsilon}}  \\\\
        y_i &\\gets \\gamma \\hat{x_i} + \\beta

    Note:
        if build_strategy.sync_batch_norm=True, the batch_norm in network will use
        sync_batch_norm automatically.
        `is_test = True` can only be used in test program and inference program, `is_test` CANNOT be set to True in train program, if you want to use global status from pre_train model in train program, please set `use_global_stats = True`.

    Args:
        input(Tensor): The rank of input Tensor can be 2, 3, 4, 5. The data type
            is float16 or float32 or float64.
        act(string, Default None): Activation type, linear|relu|prelu|...
        is_test (bool, Default False): A flag indicating whether it is in
            test phrase or not.
        momentum(float|Tensor, Default 0.9): The value used for the moving_mean and
            moving_var computation. This should be a float number or a Tensor with
            shape [1] and data type as float32. The updated formula is:
            :math:`moving\_mean = moving\_mean * momentum + new\_mean * (1. - momentum)`
            :math:`moving\_var = moving\_var * momentum + new\_var * (1. - momentum)`
            Default is 0.9.
        epsilon(float, Default 1e-05): A value added to the denominator for
            numerical stability. Default is 1e-5.
        param_attr(ParamAttr|None): The parameter attribute for Parameter `scale`
             of batch_norm. If it is set to None or one attribute of ParamAttr, batch_norm
	     will create ParamAttr as param_attr, the name of scale can be set in ParamAttr.
	     If the Initializer of the param_attr is not set, the parameter is initialized
	     with Xavier. Default: None.
        bias_attr(ParamAttr|None): The parameter attribute for the bias of batch_norm.
             If it is set to None or one attribute of ParamAttr, batch_norm
	     will create ParamAttr as bias_attr, the name of bias can be set in ParamAttr.
	     If the Initializer of the bias_attr is not set, the bias is initialized zero.
	     Default: None.
        data_layout (str, optional): Specify the data format of the input, and the data format of the output
             will be consistent with that of the input. An optional string from: `"NCHW"`, `"NHWC"`.
             The default is `"NCHW"`. When it is `"NCHW"`, the data is stored in the order of:
             `[batch_size, input_channels, input_height, input_width]`.
        in_place(bool, Default False): Make the input and output of batch norm reuse memory.
        name(str|None): For detailed information, please refer to :ref:`api_guide_Name`.
            Usually name is no need to set and None by default.
        moving_mean_name(str, Default None): The name of moving_mean which store the global Mean. If it
            is set to None, batch_norm will save global mean with a random name, otherwise, batch_norm
            will save global mean with the string.
        moving_variance_name(str, Default None): The name of the moving_variance which store the global Variance.
            If it is set to None, batch_norm will save global variance with a random name, otherwise, batch_norm
            will save global variance with the string.
        do_model_average_for_mean_and_var(bool, Default True): Whether parameter mean and variance should do model
            average when model average is enabled.
        use_global_stats(bool, Default False): Whether to use global mean and
            variance. In inference or test mode, set use_global_stats to true
            or is_test to true, and the behavior is equivalent.
            In train mode, when setting use_global_stats True, the global mean
            and variance are also used during train period.
    Returns:
        A Tensor which is the result after applying batch normalization on the input,
        has same shape and data type with input.

    Examples:

        .. code-block:: python

            import paddle

            paddle.enable_static()
            x = paddle.static.data(name='x', shape=[3, 7, 3, 7], dtype='float32')
            hidden1 = paddle.static.nn.fc(x=x, size=200)
            print(hidden1.shape)
            # [3, 200]
            hidden2 = paddle.static.nn.batch_norm(input=hidden1)
            print(hidden2.shape)
            # [3, 200]
    """
    assert (
        bias_attr is not False
    ), "bias_attr should not be False in batch_norm."
    helper = LayerHelper('batch_norm', **locals())

    check_variable_and_dtype(
        input, 'input', ['float16', 'float32', 'float64'], 'batch_norm'
    )
    dtype = helper.input_dtype()

    # use fp32 for bn parameter
    if dtype == core.VarDesc.VarType.FP16:
        dtype = core.VarDesc.VarType.FP32

    input_shape = input.shape
    if data_layout == 'NCHW':
        channel_num = input_shape[1]
    else:
        if data_layout == 'NHWC':
            channel_num = input_shape[-1]
        else:
            raise ValueError("unsupported data layout:" + data_layout)

    param_shape = [channel_num]

    # create parameter
    scale = helper.create_parameter(
        attr=helper.param_attr,
        shape=param_shape,
        dtype=dtype,
        default_initializer=Constant(1.0),
    )
    bias = helper.create_parameter(
        attr=helper.bias_attr, shape=param_shape, dtype=dtype, is_bias=True
    )

    mean = helper.create_parameter(
        attr=ParamAttr(
            name=moving_mean_name,
            initializer=Constant(0.0),
            trainable=False,
            do_model_average=do_model_average_for_mean_and_var,
        ),
        shape=param_shape,
        dtype=dtype,
    )
    mean.stop_gradient = True

    variance = helper.create_parameter(
        attr=ParamAttr(
            name=moving_variance_name,
            initializer=Constant(1.0),
            trainable=False,
            do_model_average=do_model_average_for_mean_and_var,
        ),
        shape=param_shape,
        dtype=dtype,
    )
    variance.stop_gradient = True

    # create output
    # mean and mean_out share the same memory
    mean_out = mean
    # variance and variance_out share the same memory
    variance_out = variance

    if in_dygraph_mode():
        inputs_has_MomemtumTensor = False
        attrs_has_momentum = False
        tmp_tensor_type = core.eager.Tensor
        if isinstance(momentum, tmp_tensor_type):
            inputs_has_MomemtumTensor = True
        else:
            attrs_has_momentum = True

        attrs_ = ()
        if attrs_has_momentum:
            attrs_ = (
                'momentum',
                momentum,
                'epsilon',
                epsilon,
                'is_test',
                is_test,
                'data_layout',
                data_layout,
                'use_mkldnn',
                False,
                'fuse_with_relu',
                False,
                'use_global_stats',
                use_global_stats,
            )
        else:
            attrs_ = (
                'epsilon',
                epsilon,
                'is_test',
                is_test,
                'data_layout',
                data_layout,
                'use_mkldnn',
                False,
                'fuse_with_relu',
                False,
                'use_global_stats',
                use_global_stats,
            )
        if inputs_has_MomemtumTensor:
            batch_norm_out, _, _, _, _, _ = _legacy_C_ops.batch_norm(
                input,
                scale,
                bias,
                mean,
                variance,
                momentum,
                mean_out,
                variance_out,
                *attrs_,
            )
        else:
            batch_norm_out, _, _, _, _, _ = _legacy_C_ops.batch_norm(
                input,
                scale,
                bias,
                mean,
                variance,
                None,
                mean_out,
                variance_out,
                *attrs_,
            )

        return dygraph_utils._append_activation_in_dygraph(
            batch_norm_out, act=act, use_mkldnn=False
        )

    saved_mean = helper.create_variable_for_type_inference(
        dtype=dtype, stop_gradient=True
    )
    saved_variance = helper.create_variable_for_type_inference(
        dtype=dtype, stop_gradient=True
    )
    reserve_space = None
    if not is_test:
        reserve_space = helper.create_variable_for_type_inference(
            dtype=helper.input_dtype(), stop_gradient=True
        )

    batch_norm_out = (
        input if in_place else helper.create_variable_for_type_inference(dtype)
    )

    inputs = {
        "X": input,
        "Scale": scale,
        "Bias": bias,
        "Mean": mean,
        "Variance": variance,
        "MeanOut": mean_out,
        "VarianceOut": variance_out,
    }
    attrs = {
        "epsilon": epsilon,
        "is_test": is_test,
        "data_layout": data_layout,
        "use_mkldnn": False,
        "fuse_with_relu": False,
        "use_global_stats": use_global_stats,
    }
    if isinstance(momentum, Variable):
        inputs['MomemtumTensor'] = momentum
    else:
        attrs['momentum'] = momentum

    outputs = {
        "Y": batch_norm_out,
        "MeanOut": mean_out,
        "VarianceOut": variance_out,
        "SavedMean": saved_mean,
        "SavedVariance": saved_variance,
    }
    if reserve_space is not None:
        outputs["ReserveSpace"] = reserve_space

    helper.append_op(
        type="batch_norm", inputs=inputs, outputs=outputs, attrs=attrs
    )

    return helper.append_activation(batch_norm_out)


@templatedoc()
def layer_norm(
    input,
    scale=True,
    shift=True,
    begin_norm_axis=1,
    epsilon=1e-05,
    param_attr=None,
    bias_attr=None,
    act=None,
    name=None,
):
    r"""
    :api_attr: Static Graph

    **Layer Normalization Layer**

    The API implements the function of the Layer Normalization Layer and can be applied to mini-batch input data.
    Refer to `Layer Normalization <https://arxiv.org/pdf/1607.06450v1.pdf>`_

    The formula is as follows:

    ..  math::

        \\mu & = \\frac{1}{H}\\sum_{i=1}^{H} x_i

        \\sigma & = \\sqrt{\\frac{1}{H}\sum_{i=1}^{H}{(x_i - \\mu)^2} + \\epsilon}

        y & = f(\\frac{g}{\\sigma}(x - \\mu) + b)

    - :math:`x`: the vector representation of the summed inputs to the neurons in that layer.
    - :math:`H`: the number of hidden units in a layers
    - :math:`\\epsilon`: the small value added to the variance to prevent division by zero.
    - :math:`g`: the trainable scale parameter.
    - :math:`b`: the trainable bias parameter.

    Args:
        input(Tensor): A multi-dimension ``Tensor`` , and the data type is float32 or float64.
        scale(bool, optional): Whether to learn the adaptive gain :math:`g` after
            normalization. Default: True.
        shift(bool, optional): Whether to learn the adaptive bias :math:`b` after
            normalization. Default: True.
        begin_norm_axis(int, optional): The normalization will be performed along
            dimensions from :attr:`begin_norm_axis` to :attr:`rank(input)`.
            Default: 1.
        epsilon(float, optional): The small value added to the variance to prevent
            division by zero. Default: 1e-05.
        param_attr(ParamAttr, optional): The parameter attribute for the learnable
            gain :math:`g`. If :attr:`scale` is False, :attr:`param_attr` is
            omitted. If :attr:`scale` is True and :attr:`param_attr` is None,
            a default :code:`ParamAttr` would be added as scale. The
            :attr:`param_attr` is initialized as 1 if it is added. Default: None.
        bias_attr(ParamAttr, optional): The parameter attribute for the learnable
            bias :math:`b`. If :attr:`shift` is False, :attr:`bias_attr` is
            omitted. If :attr:`shift` is True and :attr:`param_attr` is None,
            a default :code:`ParamAttr` would be added as bias. The
            :attr:`bias_attr` is initialized as 0 if it is added. Default: None.
        act(str, optional): Activation to be applied to the output of layer normalization.
                  Default: None.
        name(str): The default value is None.  Normally there is no need for user to set this property.  For more information, please refer to :ref:`api_guide_Name` .

    Returns:
        Tensor: ``Tensor``  indicating the normalized result, the data type is the same as  ``input`` , and the return dimension is the same as  ``input`` .

    Examples:

        .. code-block:: python

            import paddle
            paddle.enable_static()
            x = paddle.static.data(name='x', shape=[8, 32, 32], dtype='float32')
            output = paddle.static.nn.layer_norm(input=x, begin_norm_axis=1)
            print(output.shape)  # [8, 32, 32]
    """
    assert (
        _non_static_mode() is not True
    ), "please use LayerNorm instead of layer_norm in dygraph mode!"
    helper = LayerHelper('layer_norm', **locals())
    check_variable_and_dtype(
        input, 'input', ['float32', 'float64'], 'layer_norm'
    )
    dtype = helper.input_dtype()

    # create intput and parameters
    inputs = {'X': input}
    input_shape = input.shape
    param_shape = [reduce(lambda x, y: x * y, input_shape[begin_norm_axis:])]
    if scale:
        assert (
            param_attr is not False
        ), "param_attr should not be False when using scale."
        scale = helper.create_parameter(
            attr=helper.param_attr,
            shape=param_shape,
            dtype=dtype,
            default_initializer=Constant(1.0),
        )
        inputs['Scale'] = scale
    else:
        if param_attr:
            warnings.warn("param_attr is only available with scale is True.")
    if shift:
        assert (
            bias_attr is not False
        ), "bias_attr should not be False when using shift."
        bias = helper.create_parameter(
            attr=helper.bias_attr, shape=param_shape, dtype=dtype, is_bias=True
        )
        inputs['Bias'] = bias
    else:
        if bias_attr:
            warnings.warn("bias_attr is only available with shift is True.")

    # create output
    mean_out = helper.create_variable_for_type_inference(
        dtype=dtype, stop_gradient=True
    )
    variance_out = helper.create_variable_for_type_inference(
        dtype=dtype, stop_gradient=True
    )
    layer_norm_out = helper.create_variable_for_type_inference(dtype)

    helper.append_op(
        type="layer_norm",
        inputs=inputs,
        outputs={
            "Y": layer_norm_out,
            "Mean": mean_out,
            "Variance": variance_out,
        },
        attrs={"epsilon": epsilon, "begin_norm_axis": begin_norm_axis},
    )

    return helper.append_activation(layer_norm_out)


@templatedoc()
def spectral_norm(weight, dim=0, power_iters=1, eps=1e-12, name=None):
    r"""
    :api_attr: Static Graph

    **Spectral Normalization Layer**

    This operation calculates the spectral normalization value of weight parameters of
    fc, conv1d, conv2d, conv3d layers which should be 2-D, 3-D, 4-D, 5-D
    Parameters. Output tensor will be in same shape with input tensor.
    Calculations are showed as follows.

    Step 1:
    Generate vector U in shape of [H], and V in shape of [W].
    While H is the :attr:`dim` th dimension of the input weights,
    and W is the product result of remaining dimensions.

    Step 2:
    :attr:`power_iters` should be a positive integer, do following
    calculations with U and V for :attr:`power_iters` rounds. Calculations
    as follows:

    .. math::

        \mathbf{v} := \\frac{\mathbf{W}^{T} \mathbf{u}}{\|\mathbf{W}^{T} \mathbf{u}\|_2}

        \mathbf{u} := \\frac{\mathbf{W}^{T} \mathbf{v}}{\|\mathbf{W}^{T} \mathbf{v}\|_2}

    Step 3:
    Calculate :math:`\sigma(\mathbf{W})` and normalize weight values.

    .. math::

        \sigma(\mathbf{W}) = \mathbf{u}^{T} \mathbf{W} \mathbf{v}

        \mathbf{W} = \\frac{\mathbf{W}}{\sigma(\mathbf{W})}


    Refer to `Spectral Normalization <https://arxiv.org/abs/1802.05957>`_ .

    Args:
        weight(Tensor): ${weight_comment}
        dim(int): ${dim_comment}
        power_iters(int): ${power_iters_comment}
        eps(float): ${eps_comment}
        name(str, optional): For detailed information, please refer
                             to :ref:`api_guide_Name`. Usually name is no need to set and
                             None by default.

    Returns:
        Tensor: A tensor of weight parameters after spectral normalization.
                  The data type and shape is same as input tensor.

    Examples:
       .. code-block:: python

            import paddle

            paddle.enable_static()
            weight = paddle.static.data(name='weight', shape=[2, 8, 32, 32], dtype='float32')
            x = paddle.static.nn.spectral_norm(weight=weight, dim=1, power_iters=2)
            print(x.shape) # [2, 8, 32, 32]
    """
    helper = LayerHelper('spectral_norm', **locals())
    check_variable_and_dtype(
        weight, 'weight', ['float32', 'float64'], 'spectral_norm'
    )
    check_type(dim, 'dim', int, 'spectral_norm')
    check_type(power_iters, 'power_iters', int, 'spectral_norm')
    check_type(eps, 'eps', float, 'spectral_norm')
    dtype = weight.dtype

    # create intput and parameters
    input_shape = weight.shape
    assert weight.numel() > 0, "Any dimension of input cannot be equal to 0."
    assert dim < len(input_shape), (
        "The input `dim` should be less than the "
        "rank of `weight`, but received dim="
        "{}".format(dim)
    )
    h = input_shape[dim]
    w = np.prod(input_shape) // h

    u = helper.create_parameter(
        attr=ParamAttr(),
        shape=[h],
        dtype=dtype,
        default_initializer=Normal(0.0, 1.0),
    )
    u.stop_gradient = True
    v = helper.create_parameter(
        attr=ParamAttr(),
        shape=[w],
        dtype=dtype,
        default_initializer=Normal(0.0, 1.0),
    )
    v.stop_gradient = True

    if in_dygraph_mode():
        return _C_ops.spectral_norm(weight, u, v, dim, power_iters, eps)

    inputs = {'Weight': weight}
    inputs['U'] = u
    inputs['V'] = v

    # create output
    out = helper.create_variable(dtype=dtype)

    helper.append_op(
        type="spectral_norm",
        inputs=inputs,
        outputs={
            "Out": out,
        },
        attrs={
            "dim": dim,
            "power_iters": power_iters,
            "eps": eps,
        },
    )

    return out


def reduce_sum(input, dim=None, keep_dim=False, name=None):
    """

    Computes the sum of tensor elements over the given dimension.

    Args:
        input (Variable): The input variable which is a Tensor, the data type is float32,
            float64, int32, int64.
        dim (list|int, optional): The dimensions along which the sum is performed. If
            :attr:`None`, sum all elements of :attr:`input` and return a
            Tensor variable with a single element, otherwise must be in the
            range :math:`[-rank(input), rank(input))`. If :math:`dim[i] < 0`,
            the dimension to reduce is :math:`rank + dim[i]`.
        keep_dim (bool, optional): Whether to reserve the reduced dimension in the
            output Tensor. The result tensor will have one fewer dimension
            than the :attr:`input` unless :attr:`keep_dim` is true, default
            value is False.
        name(str, optional): The default value is None.  Normally there is no need for
            user to set this property.  For more information, please refer to :ref:`api_guide_Name`

    Returns:
        Variable: Tensor, results of summation operation on the specified dim of input tensor,
        it's data type is the same as input's Tensor.

    Raises:
        TypeError, if out data type is different with the input data type.

    Examples:
        .. code-block:: python

            import paddle.fluid as fluid
            import paddle
            paddle.enable_static()
            # x is a Tensor variable with following elements:
            #    [[0.2, 0.3, 0.5, 0.9]
            #     [0.1, 0.2, 0.6, 0.7]]
            # Each example is followed by the corresponding output tensor.
            x = fluid.data(name='x', shape=[2, 4], dtype='float32')
            fluid.layers.reduce_sum(x)  # [3.5]
            fluid.layers.reduce_sum(x, dim=0)  # [0.3, 0.5, 1.1, 1.6]
            fluid.layers.reduce_sum(x, dim=-1)  # [1.9, 1.6]
            fluid.layers.reduce_sum(x, dim=1, keep_dim=True)  # [[1.9], [1.6]]

            # y is a Tensor variable with shape [2, 2, 2] and elements as below:
            #      [[[1, 2], [3, 4]],
            #      [[5, 6], [7, 8]]]
            # Each example is followed by the corresponding output tensor.
            y = fluid.data(name='y', shape=[2, 2, 2], dtype='float32')
            fluid.layers.reduce_sum(y, dim=[1, 2]) # [10, 26]
            fluid.layers.reduce_sum(y, dim=[0, 1]) # [16, 20]

    """
    reduce_all, dim = _get_reduce_dim(dim, input)

    if in_dygraph_mode():
        return _C_ops.sum(input, dim, None, keep_dim)
    elif _in_legacy_dygraph():
        return _legacy_C_ops.reduce_sum(
            input, 'dim', dim, 'keep_dim', keep_dim, 'reduce_all', reduce_all
        )
    attrs = {'dim': dim, 'keep_dim': keep_dim, 'reduce_all': reduce_all}
    check_variable_and_dtype(
        input,
        'input',
        ['float16', 'float32', 'float64', 'int32', 'int64'],
        'reduce_sum',
    )
    helper = LayerHelper('reduce_sum', **locals())
    out = helper.create_variable_for_type_inference(dtype=helper.input_dtype())
    helper.append_op(
        type='reduce_sum',
        inputs={'X': input},
        outputs={'Out': out},
        attrs=attrs,
    )
    return out


def split(input, num_or_sections, dim=-1, name=None):
    """
    Split the input tensor into multiple sub-Tensors.

    Args:
        input (Tensor): A N-D Tensor. The data type is bool, float16, float32, float64, int32 or int64.
        num_or_sections (int|list|tuple): If ``num_or_sections`` is int, then the ``num_or_sections``
            indicates the number of equal sized sub-Tensors that the ``input``
            will be divided into. If ``num_or_sections`` is a list or tuple, the length of it
            indicates the number of sub-Tensors and the elements in it indicate the sizes of sub-Tensors'
            dimension orderly. The length of the list mustn't be larger than the ``input`` 's size of specified dim.
        dim (int|Tensor, optional): The dimension along which to split, it can be a scalar with type ``int`` or
            a ``Tensor`` with shape [1] and data type ``int32`` or ``int64``. If :math:`dim < 0`,
            the dimension to split along is :math:`rank(input) + dim`. Default is -1.
        name (str, optional): The default value is None.  Normally there is no need for user to set this property.
            For more information, please refer to :ref:`api_guide_Name` .

    Returns:
        list(Tensor): The list of segmented Tensors.

    Example:
        .. code-block:: python

            import paddle.fluid as fluid

            # input is a Tensor which shape is [3, 9, 5]
            input = fluid.data(
                 name="input", shape=[3, 9, 5], dtype="float32")

            out0, out1, out2 = fluid.layers.split(input, num_or_sections=3, dim=1)
            # out0.shape [3, 3, 5]
            # out1.shape [3, 3, 5]
            # out2.shape [3, 3, 5]

            out0, out1, out2 = fluid.layers.split(input, num_or_sections=[2, 3, 4], dim=1)
            # out0.shape [3, 2, 5]
            # out1.shape [3, 3, 5]
            # out2.shape [3, 4, 5]

            out0, out1, out2 = fluid.layers.split(input, num_or_sections=[2, 3, -1], dim=1)
            # out0.shape [3, 2, 5]
            # out1.shape [3, 3, 5]
            # out2.shape [3, 4, 5]

            # dim is negative, the real dim is (rank(input) + axis) which real
            # value is 1.
            out0, out1, out2 = fluid.layers.split(input, num_or_sections=3, dim=-2)
            # out0.shape [3, 3, 5]
            # out1.shape [3, 3, 5]
            # out2.shape [3, 3, 5]

    """
    if _non_static_mode():
        num = None
        attrs = ()

        if isinstance(dim, Variable):
            dim = dim.numpy()
            dim = dim.item(0)
        assert len(input.shape) + dim >= 0, "(rank(x) + axis) must >= 0"
        dim = (len(input.shape) + dim) if dim < 0 else dim
        attrs += ('axis', dim)

        if isinstance(num_or_sections, int):
            num = num_or_sections
            attrs += ('num', num_or_sections)
        elif isinstance(num_or_sections, (list, tuple)):
            num = len(num_or_sections)
            if utils._contain_var(num_or_sections):
                for index, item in enumerate(num_or_sections):
                    if isinstance(item, Variable):
                        num_or_sections[index] = num_or_sections[index].numpy()[
                            0
                        ]
                attrs += ('sections', list(num_or_sections))
            else:
                attrs += ('sections', list(num_or_sections))
        else:
            raise TypeError(
                "The type of 'num_or_sections' in split must be int, list or tuple in imperative mode, but "
                "received %s." % (type(num_or_sections))
            )
        if in_dygraph_mode():
            if isinstance(num_or_sections, int):
                return _C_ops.split_with_num(input, num_or_sections, dim)
            else:
                return _C_ops.split(input, num_or_sections, dim)
        elif _in_legacy_dygraph():
            out = [_varbase_creator() for n in range(num)]
            _legacy_C_ops.split(input, out, *attrs)
            return out

    check_variable_and_dtype(
        input,
        'input',
        ['bool', 'float16', 'float32', 'float64', 'int32', 'int64'],
        'split',
    )
    check_type(num_or_sections, 'num_or_sections', (list, int, tuple), 'split')
    check_type(dim, 'dim', (int, Variable), 'split')
    if isinstance(dim, Variable):
        check_dtype(dim.dtype, 'dim', ['int32', 'int64'], 'split')

    helper = LayerHelper('split', **locals())

    input_shape = input.shape
    inputs = {'X': input}
    attrs = {'num': num_or_sections if isinstance(num_or_sections, int) else 0}

    def _get_SectionsTensorList(one_list):
        tensor_list = []
        unk_dim_idx = -1
        for idx, dim_size in enumerate(one_list):
            if isinstance(dim_size, Variable):
                dim_size.stop_gradient = True
                tensor_list.append(dim_size)
            else:
                assert isinstance(dim_size, int)
                if dim_size == -1:
                    assert unk_dim_idx == -1, (
                        "Only one value of 'num_or_section' in split can "
                        "be -1. But received num_or_section[%d] is also -1."
                        % idx
                    )
                    unk_dim_idx = idx
                temp_out = helper.create_variable_for_type_inference('int32')
                fill_constant(
                    [1], 'int32', dim_size, force_cpu=True, out=temp_out
                )
                tensor_list.append(temp_out)
        return tensor_list

    if isinstance(dim, Variable):
        dim.stop_gradient = True
        inputs['AxisTensor'] = dim
    else:
        assert len(input.shape) + dim >= 0, "(rank(x) + axis) must >= 0"
        dim = (len(input_shape) + dim) if dim < 0 else dim
        attrs['axis'] = dim

    if isinstance(num_or_sections, int):
        assert num_or_sections > 1, 'num_or_sections must be more than 1.'
        if isinstance(dim, int) and input_shape[dim] > 0:
            assert input_shape[dim] % num_or_sections == 0, (
                "The input's size along the split dimension "
                "must be evenly divisible by Attr(num_or_sections). "
                "But %d is not evenly divisible by %d. "
                % (num_or_sections, input_shape[dim])
            )
        num = num_or_sections
    else:
        if isinstance(dim, int) and input_shape[dim] > 0:
            assert (
                len(num_or_sections) <= input_shape[dim]
            ), 'len(num_or_sections) must not be more than input.shape[dim].'
        num = len(num_or_sections)
        attrs['sections'] = list(
            map(
                lambda ele: -1 if isinstance(ele, Variable) else ele,
                num_or_sections,
            )
        )
        if utils._contain_var(num_or_sections):
            inputs['SectionsTensorList'] = _get_SectionsTensorList(
                num_or_sections
            )

    outs = [
        helper.create_variable_for_type_inference(dtype=helper.input_dtype())
        for i in range(num)
    ]
    helper.append_op(
        type='split', inputs=inputs, outputs={'Out': outs}, attrs=attrs
    )
    return outs


def l2_normalize(x, axis, epsilon=1e-12, name=None):
    r"""

    This op normalizes `x` along dimension `axis` using an L2
    norm. For a 1-D tensor (`dim` is fixed to 0), this layer computes

    .. math::

        y = \\frac{x}{ \sqrt{\sum {x^2} + epsion }}

    For `x` with more dimensions, this layer independently normalizes each 1-D
    slice along dimension `axis`.

    Args:
        x(Variable|list): The input tensor could be N-D tensor, and the input data type could be float16, float32 or float64.
        axis(int): The axis on which to apply normalization. If `axis < 0`, \
            the dimension to normalization is rank(X) + axis. -1 is the
            last dimension.
        epsilon(float): The epsilon value is used to avoid division by zero, \
            the default value is 1e-12.
    name(str, optional): The default value is None.  Normally there is no need for user to set this property.  For more information, please refer to :ref:`api_guide_Name`

    Returns:
        Variable: The output has the same shape and data type with `x`.

    Examples:

    .. code-block:: python
        :name: code-example1

        import paddle

        X = paddle.randn(shape=[3, 5], dtype='float64')
        out = paddle.fluid.layers.l2_normalize(X, axis=-1)
        print(out)

        # [[ 0.21558504  0.56360189  0.47466096  0.46269539 -0.44326736]
        #  [-0.70602414 -0.52745777  0.37771788 -0.2804768  -0.04449922]
        #  [-0.33972208 -0.43014923  0.31772556  0.76617881 -0.10761525]]

    """
    if len(x.shape) == 1:
        axis = 0
    if _non_static_mode():
        if in_dygraph_mode():
            out, _ = _C_ops.norm(x, 1 if axis is None else axis, epsilon, False)
        elif _in_legacy_dygraph():
            _, out = _legacy_C_ops.norm(
                x, 'axis', 1 if axis is None else axis, 'epsilon', epsilon
            )
        return out

    check_variable_and_dtype(x, "X", ("float16", "float32", "float64"), "norm")

    helper = LayerHelper("l2_normalize", **locals())
    out = helper.create_variable_for_type_inference(dtype=x.dtype)
    norm = helper.create_variable_for_type_inference(dtype=x.dtype)
    helper.append_op(
        type="norm",
        inputs={"X": x},
        outputs={"Out": out, "Norm": norm},
        attrs={
            "axis": 1 if axis is None else axis,
            "epsilon": epsilon,
        },
    )
    return out


@deprecated(since="2.0.0", update_to="paddle.matmul")
def matmul(x, y, transpose_x=False, transpose_y=False, alpha=1.0, name=None):
    """
    Applies matrix multiplication to two tensors.

    Currently, the input tensors' rank can be any, but when the rank of any
    inputs is bigger than 3, this two inputs' rank should be equal.

    The actual behavior depends on the shapes of :math:`x`, :math:`y` and the
    flag values of :attr:`transpose_x`, :attr:`transpose_y`. Specifically:

    - If a transpose flag is specified, the last two dimensions of the tensor
      are transposed. If the tensor is rank-1 of shape :math:`[D]`, then for
      :math:`x` it is treated as :math:`[1, D]` in nontransposed form and as
      :math:`[D, 1]` in transposed form, whereas for :math:`y` it is the
      opposite: It is treated as :math:`[D, 1]` in nontransposed form and as
      :math:`[1, D]` in transposed form.

    - After transpose, the two tensors are 2-D or n-D and matrix multiplication
      performs in the following way.

      - If both are 2-D, they are multiplied like conventional matrices.
      - If either is n-D, it is treated as a stack of matrices residing in the
        last two dimensions and a batched matrix multiply supporting broadcast
        applies on the two tensors.

    Also note that if the raw tensor :math:`x` or :math:`y` is rank-1 and
    nontransposed, the prepended or appended dimension :math:`1` will be
    removed after matrix multiplication.

    Args:
        x (Variable): The input variable which is a Tensor or LoDTensor.
        y (Variable): The input variable which is a Tensor or LoDTensor.
        transpose_x (bool): Whether to transpose :math:`x` before multiplication.
        transpose_y (bool): Whether to transpose :math:`y` before multiplication.
        alpha (float): The scale of output. Default 1.0.
        name(str|None): A name for this layer(optional). If set None, the layer
            will be named automatically.

    Returns:
        Variable: The product Tensor (or LoDTensor) variable.

    Examples:
        .. code-block:: python

            # Examples to clarify shapes of the inputs and output
            # x: [B, ..., M, K], y: [B, ..., K, N]
            # fluid.layers.matmul(x, y)  # out: [B, ..., M, N]

            # x: [B, M, K], y: [B, K, N]
            # fluid.layers.matmul(x, y)  # out: [B, M, N]

            # x: [B, M, K], y: [K, N]
            # fluid.layers.matmul(x, y)  # out: [B, M, N]

            # x: [M, K], y: [K, N]
            # fluid.layers.matmul(x, y)  # out: [M, N]

            # x: [B, M, K], y: [K]
            # fluid.layers.matmul(x, y)  # out: [B, M]

            # x: [K], y: [K]
            # fluid.layers.matmul(x, y)  # out: [1]

            # x: [M], y: [N]
            # fluid.layers.matmul(x, y, True, True)  # out: [M, N]

            import paddle
            import paddle.fluid as fluid
            paddle.enable_static()

            x = fluid.layers.data(name='x', shape=[2, 3], dtype='float32')
            y = fluid.layers.data(name='y', shape=[3, 2], dtype='float32')
            out = fluid.layers.matmul(x, y, True, True)
    """
    if _non_static_mode():
        out = _varbase_creator(dtype=x.dtype)
        _legacy_C_ops.matmul(
            x,
            y,
            out,
            'transpose_X',
            transpose_x,
            'transpose_Y',
            transpose_y,
            'alpha',
            float(alpha),
        )
        return out

    def __check_input(x, y):
        var_names = {'x': x, 'y': y}
        for name, val in var_names.items():
            check_variable_and_dtype(
                val, name, ['float16', 'float32', 'float64'], 'matmul'
            )
        x_shape = list(x.shape)
        y_shape = list(y.shape)
        if len(x_shape) == 1:
            x_shape = [1] + x_shape
        if len(y_shape) == 1:
            y_shape = y_shape + [1]

        # check the inner 2 dimensions
        if transpose_x:
            x_shape[-2], x_shape[-1] = x_shape[-1], x_shape[-2]
        if transpose_y:
            y_shape[-2], y_shape[-1] = y_shape[-1], y_shape[-2]
        if x_shape[-1] != y_shape[-2]:
            assert (x_shape[-1] == -1) or (y_shape[-2] == -1), (
                "After performing an optional transpose, Input X's width should be "
                "equal to Y's width for multiplication "
                "prerequisites. But received X's shape: %s, Y's shape: %s\n"
                % (x_shape, y_shape)
            )

        if len(y_shape) > 2 and len(x_shape) > 2:
            for i, dim_x in enumerate(x_shape[:-2]):
                # don't check neg shape
                if dim_x < 0 or y_shape[i] < 0:
                    continue
                if dim_x != y_shape[i]:
                    raise ValueError(
                        "When the matrix is larger than 2 dimensions, the higher "
                        "dimensional values of the two matrices need to be equal. "
                        "But received x_shape[%d] != y_shape[%d]. X's shape: %s, "
                        "Y's shape: %s.\n" % (i, i, x_shape, y_shape)
                    )

    attrs = {
        'transpose_X': transpose_x,
        'transpose_Y': transpose_y,
        'alpha': float(alpha),
    }

    __check_input(x, y)

    helper = LayerHelper('matmul', **locals())
    out = helper.create_variable_for_type_inference(dtype=x.dtype)
    helper.append_op(
        type='matmul',
        inputs={'X': x, 'Y': y},
        outputs={'Out': out},
        attrs=attrs,
    )
    return out


@templatedoc()
def row_conv(input, future_context_size, param_attr=None, act=None):
    """
    :api_attr: Static Graph

    ${comment}

    Args:
        input (${x_type}): ${x_comment}.
        future_context_size (int): Future context size. Please note, the shape
            of convolution kernel is [future_context_size + 1, D].
        param_attr (ParamAttr): Attributes of parameters, including
            name, initializer etc.
        act (str): Non-linear activation to be applied to output variable.

    Returns:
        ${out_comment}.

    Examples:

      .. code-block:: python

        # for LodTensor inputs
        import paddle
        paddle.enable_static()
        x = paddle.static.data(name='x', shape=[9, 16],
                               dtype='float32', lod_level=1)
        out = paddle.static.nn.row_conv(input=x, future_context_size=2)
        # for Tensor inputs
        x = paddle.static.data(name='x', shape=[9, 4, 16], dtype='float32')
        out = paddle.static.nn.row_conv(input=x, future_context_size=2)
    """
    helper = LayerHelper('row_conv', **locals())
    check_variable_and_dtype(input, 'input', ['float32'], 'row_conv')
    dtype = helper.input_dtype()
    filter_shape = [future_context_size + 1, input.shape[-1]]
    filter_param = helper.create_parameter(
        attr=helper.param_attr, shape=filter_shape, dtype=dtype
    )
    out = helper.create_variable_for_type_inference(dtype)
    helper.append_op(
        type='row_conv',
        inputs={'X': [input], 'Filter': [filter_param]},
        outputs={'Out': [out]},
    )
    return helper.append_activation(out)


@deprecated(since='2.0.0', update_to='paddle.nn.functional.one_hot')
def one_hot(input, depth, allow_out_of_range=False):
    """

    **WARING:** This OP requires the last dimension of Tensor shape must be equal to 1.
    This OP will be deprecated in a future release. It is recommended to use fluid. :ref:`api_fluid_one_hot` .

    The operator converts each id in the input to an one-hot vector with a
    :attr:`depth` length. The value in the vector dimension corresponding to the id
    is 1, and the value in the remaining dimension is 0.

    The shape of output Tensor or LoDTensor is generated by adding :attr:`depth` dimension
    behind the last dimension of the input shape.

    .. code-block:: text

        Example 1 (allow_out_of_range=False):

        input:
            X.shape = [4, 1]
            X.data = [[1], [1], [3], [0]]
            depth = 4

        output:
            Out.shape = [4, 4]
            Out.data = [[0., 1., 0., 0.],
                        [0., 1., 0., 0.],
                        [0., 0., 0., 1.],
                        [1., 0., 0., 0.]]

        Example 2 (allow_out_of_range=True):

        input:
            X.shape = [4, 1]
            X.data = [[1], [1], [5], [0]]
            depth = 4
            allow_out_of_range = True

        output:
            Out.shape = [4, 4]
            Out.data = [[0., 1., 0., 0.],
                        [0., 1., 0., 0.],
                        [0., 0., 0., 0.], # This id is 5, which goes beyond depth, so set it all-zeros data.
                        [1., 0., 0., 0.]]

        Example 3 (allow_out_of_range=False):

        input:
            X.shape = [4, 1]
            X.data = [[1], [1], [5], [0]]
            depth = 4
            allow_out_of_range = False

        output: Throw an exception for Illegal value
            The second dimension in X is 5, which is greater than depth.
            Allow_out_of_range =False means that does not allow the word id to exceed depth,
            so it throws an exception.

    Args:
        input(Variable): Tensor or LoDTensor with shape :math:`[N_1, N_2, ..., N_k, 1]` ,
            which contains at least one dimension and the last dimension must be 1.
            The data type is int32 or int64.
        depth(scalar): An integer defining the :attr:`depth` of the one hot dimension. If input
            is word id, depth is generally the dictionary size.
        allow_out_of_range(bool): A bool value indicating whether the input
            indices could be out of range :math:`[0, depth)` . When input indices are
            out of range, exceptions :code:`Illegal value` is raised if :attr:`allow_out_of_range`
            is False, or zero-filling representations is created if it is set True.
            Default: False.

    Returns:
        Variable: The one-hot representations of input. A Tensor or LoDTensor with type float32.

    Examples:
        .. code-block:: python

            import paddle
            import paddle.fluid as fluid
            paddle.enable_static()

            # Correspond to the first example above, where label.shape is [4, 1] and one_hot_label.shape is [4, 4].
            label = fluid.data(name="label", shape=[4, 1], dtype="int64")
            one_hot_label = fluid.layers.one_hot(input=label, depth=4)
    """
    if _non_static_mode():
        if isinstance(depth, Variable):
            depth = depth.numpy()
            assert depth.shape == (
                1,
            ), "depth of type Variable should have shape [1]"
            depth = depth.item(0)
        out = _legacy_C_ops.one_hot(
            input, 'depth', depth, 'allow_out_of_range', allow_out_of_range
        )
        out.stop_gradient = True
        return out

    helper = LayerHelper("one_hot", **locals())
    check_variable_and_dtype(input, 'input', ['int32', 'int64'], 'one_hot')
    check_type(depth, 'depth', (int, Variable), 'one_hot')
    one_hot_out = helper.create_variable_for_type_inference(dtype='float32')

    if not isinstance(depth, Variable):
        # user attribute
        inputs = {'X': input}
        attrs = {'depth': depth, 'allow_out_of_range': allow_out_of_range}
    else:
        depth.stop_gradient = True
        inputs = {'X': input, 'depth_tensor': depth}
        attrs = {'allow_out_of_range': allow_out_of_range}
    helper.append_op(
        type="one_hot", inputs=inputs, attrs=attrs, outputs={'Out': one_hot_out}
    )
    one_hot_out.stop_gradient = True
    return one_hot_out


def autoincreased_step_counter(counter_name=None, begin=1, step=1):
    """
    :api_attr: Static Graph

    Create an auto-increase variable. which will be automatically increased
    by 1 in every iteration. By default, the first return of this counter is 1,
    and the step size is 1.

    Args:
        counter_name(str, optional): The counter name. Default '@STEP_COUNTER@'.
        begin(int, optional): The first return value of this counter. Default 1.
        step(int, optional): The step size. Default 1.

    Returns:
        Variable: The auto-increased Variable with data type int64.

    Examples:
        .. code-block:: python

           import paddle.fluid as fluid
           import paddle
           paddle.enable_static()
           global_step = fluid.layers.autoincreased_step_counter(
               counter_name='@LR_DECAY_COUNTER@', begin=0, step=1)
    """
    helper = LayerHelper('global_step_counter')
    if counter_name is None:
        counter_name = '@STEP_COUNTER@'
    counter, is_new_var = helper.create_or_get_global_variable(
        name=counter_name,
        dtype='int64',
        shape=[1],
        persistable=True,
        belong_to_optimizer=True,
    )
    if is_new_var:
        helper.set_variable_initializer(
            counter, initializer=Constant(value=begin - 1, force_cpu=True)
        )
        helper.main_program.global_block()._prepend_op(
            type='increment',
            inputs={'X': [counter]},
            outputs={'Out': [counter]},
            attrs={'step': float(step)},
        )
        counter.stop_gradient = True

    return counter


def unsqueeze(input, axes, name=None):
    """
    Insert single-dimensional entries to the shape of a Tensor. Takes one
    required argument axes, a list of dimensions that will be inserted.
    Dimension indices in axes are as seen in the output tensor.

    For example:

    .. code-block:: text

      Given a tensor such that tensor with shape [3, 4, 5],
      then Unsqueezed tensor with axes=[0, 4] has shape [1, 3, 4, 5, 1].

    Args:
        input (Variable): The input Tensor to be unsqueezed. Supported data type: float32, float64, bool, int8, int32, int64.
        axes (int|list|tuple|Variable): Indicates the dimensions to be inserted. The data type is ``int32`` . If ``axes`` is a list or tuple, the elements of it should be integers or Tensors with shape [1]. If ``axes`` is an Variable, it should be an 1-D Tensor .
        name (str|None): Name for this layer.

    Returns:
        Variable: Unsqueezed Tensor, with the same data type as input.

    Examples:
        .. code-block:: python

            import paddle.fluid as fluid
            x = fluid.layers.data(name='x', shape=[5, 10])
            y = fluid.layers.unsqueeze(input=x, axes=[1])

    """
    if _non_static_mode():
        if isinstance(axes, int):
            axes = [axes]
        elif isinstance(axes, Variable):
            axes = axes.numpy().tolist()
        elif isinstance(axes, (list, tuple)):
            axes = [
                item.numpy().item(0) if isinstance(item, Variable) else item
                for item in axes
            ]
        if _in_legacy_dygraph():
            out, _ = _legacy_C_ops.unsqueeze2(input, 'axes', axes)
            return out
        return _C_ops.unsqueeze(input, axes)

    check_type(axes, 'axis/axes', (int, list, tuple, Variable), 'unsqueeze')
    check_variable_and_dtype(
        input,
        'input',
        [
            'float16',
            'float32',
            'float64',
            'bool',
            'int8',
            'int16',
            'int32',
            'int64',
            'complex64',
            'complex128',
        ],
        'unsqueeze',
    )
    helper = LayerHelper("unsqueeze2", **locals())
    inputs = {"X": input}
    attrs = {}

    if isinstance(axes, int):
        axes = [axes]
    if isinstance(axes, Variable):
        axes.stop_gradient = True
        inputs["AxesTensor"] = axes
    elif isinstance(axes, (list, tuple)):
        if utils._contain_var(axes):
            inputs["AxesTensorList"] = utils._convert_to_tensor_list(axes)
        else:
            attrs["axes"] = axes

    out = helper.create_variable_for_type_inference(dtype=input.dtype)
    x_shape = helper.create_variable_for_type_inference(dtype=input.dtype)
    helper.append_op(
        type="unsqueeze2",
        inputs=inputs,
        attrs=attrs,
        outputs={"Out": out, "XShape": x_shape},
    )

    return out


def lod_reset(x, y=None, target_lod=None):
    """
    Set LoD of :attr:`x` to a new one specified by :attr:`y` or
    :attr:`target_lod`. When :attr:`y` provided, :attr:`y.lod` would be
    considered as target LoD first, otherwise :attr:`y.data` would be
    considered as target LoD. If :attr:`y` is not provided, target LoD should
    be specified by :attr:`target_lod`. If target LoD is specified by
    :attr:`y.data` or :attr:`target_lod`, only one level LoD is supported.

    .. code-block:: text

        * Example 1:

            Given a 1-level LoDTensor x:
                x.lod =  [[ 2,           3,                   1 ]]
                x.data = [[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]]
                x.dims = [6, 1]

            target_lod: [4, 2]

            then we get a 1-level LoDTensor:
                out.lod =  [[4,                          2]]
                out.data = [[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]]
                out.dims = [6, 1]

        * Example 2:

            Given a 1-level LoDTensor x:
                x.lod =  [[2,            3,                   1]]
                x.data = [[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]]
                x.dims = [6, 1]

            y is a Tensor:
                y.data = [[2, 4]]
                y.dims = [1, 3]

            then we get a 1-level LoDTensor:
                out.lod =  [[2,            4]]
                out.data = [[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]]
                out.dims = [6, 1]

        * Example 3:

            Given a 1-level LoDTensor x:
                x.lod =  [[2,            3,                   1]]
                x.data = [[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]]
                x.dims = [6, 1]

            y is a 2-level LoDTensor:
                y.lod =  [[2, 2], [2, 2, 1, 1]]
                y.data = [[1.1], [2.1], [3.1], [4.1], [5.1], [6.1]]
                y.dims = [6, 1]

            then we get a 2-level LoDTensor:
                out.lod =  [[2, 2], [2, 2, 1, 1]]
                out.data = [[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]]
                out.dims = [6, 1]

    Args:
        x (Variable): Input variable which could be a Tensor or LoDTensor.
                      The data type should be int32, int64, float32 or float64.
        y (Variable, optional): If provided, output's LoD would be derived from :attr:`y`.
                                If y's lod level>0, the data type can be any type.
                                If y's lod level=0, the data type should be int32.
        target_lod (list|tuple, optional): One level LoD which should be considered
                                      as target LoD when :attr:`y` not provided.

    Returns:
        Variable: Output variable with LoD specified by this layer.

    Raises:
        ValueError: If :attr:`y` and :attr:`target_lod` are both None.

    Examples:
        .. code-block:: python

            import paddle.fluid as fluid
            x = fluid.layers.data(name='x', shape=[10])
            y = fluid.layers.data(name='y', shape=[10, 20], lod_level=2)
            out = fluid.layers.lod_reset(x=x, y=y)
    """
    check_variable_and_dtype(
        x, 'x', ['float32', 'float64', 'int32', 'int64'], 'lod_reset'
    )
    helper = LayerHelper("lod_reset", **locals())
    out = helper.create_variable_for_type_inference(dtype=x.dtype)
    if y is not None:
        check_type(y, 'y', (Variable), 'lod_reset')
        # TODO: check y.lod_level = 0 dtype
        helper.append_op(
            type="lod_reset", inputs={'X': x, 'Y': y}, outputs={'Out': out}
        )
    elif target_lod is not None:
        helper.append_op(
            type="lod_reset",
            inputs={'X': x},
            attrs={'target_lod': target_lod},
            outputs={'Out': out},
        )
    else:
        raise ValueError("y and target_lod should not be both none.")
    return out


@deprecated(since="2.0.0", update_to="paddle.nn.functional.relu")
def relu(x, name=None):
    """
    ${comment}

    Args:
        x(Variable): ${x_comment}
        name(str, optional): The default value is None. Normally there is no
            need for user to set this property. For more information, please
            refer to :ref:`api_guide_Name`.

    Returns:
        Variable: ${out_comment}

    Examples:

        .. code-block:: python

            import paddle.fluid as fluid
            import numpy as np
            in1 = np.array([[-1,0],[1,2.6]])
            with fluid.dygraph.guard():
                x1 = fluid.dygraph.to_variable(in1)
                out1 = fluid.layers.relu(x1)
                print(out1.numpy())
                # [[0.  0. ]
                #  [1.  2.6]]"""

    if in_dygraph_mode():
        return _C_ops.relu(x)
    if _in_legacy_dygraph():
        return _legacy_C_ops.relu(x)

    check_variable_and_dtype(x, 'x', ['float16', 'float32', 'float64'], 'relu')

    inputs = {'X': [x]}
    helper = LayerHelper('relu', **locals())
    dtype = helper.input_dtype(input_param_name='x')
    out = helper.create_variable_for_type_inference(dtype)
    helper.append_op(
        type="relu", inputs={"X": helper.input('x')}, outputs={"Out": out}
    )
    return out


from paddle.fluid.framework import convert_np_dtype_to_dtype_


@deprecated(since="2.0.0", update_to="paddle.normal")
@templatedoc()
def gaussian_random(
    shape, mean=0.0, std=1.0, seed=0, dtype='float32', name=None
):
    """
    This OP returns a Tensor filled with random values sampled from a Gaussian
    distribution, with ``shape`` and ``dtype``.

    Args:
        shape(list|tuple|Tensor): The shape of the output Tensor. If ``shape``
            is a list or tuple, the elements of it should be integers or Tensors
            (with the shape [1], and the data type int32 or int64). If ``shape``
            is a Tensor, it should be a 1-D Tensor(with the data type int32 or
            int64).
        mean(float|int, optional): Mean of the output tensor, default is 0.0.
        std(float|int, optional): Standard deviation of the output tensor, default
            is 1.0.
        seed(int, optional): ${seed_comment}
        dtype(str|np.dtype|core.VarDesc.VarType, optional): The data type of
            the output Tensor. Supported data types: float32, float64.
            Default is float32.
        name(str, optional): The default value is None. Normally there is no
            need for user to set this property. For more information, please
            refer to :ref:`api_guide_Name`.

    Returns:
        Tensor: A Tensor filled with random values sampled from a Gaussian
        distribution, with ``shape`` and ``dtype``.

    Examples:
       .. code-block:: python

            import paddle
            import paddle.fluid as fluid
            paddle.enable_static()

            # example 1:
            # attr shape is a list which doesn't contain Tensor.
            result_1 = fluid.layers.gaussian_random(shape=[3, 4])
            # [[-0.31261674,  1.8736548,  -0.6274357,   0.96988016],
            #  [-0.12294637,  0.9554768,   1.5690808,  -1.2894802 ],
            #  [-0.60082096, -0.61138713,  1.5345167,  -0.21834975]]

            # example 2:
            # attr shape is a list which contains Tensor.
            dim_1 = fluid.layers.fill_constant([1], "int64", 2)
            dim_2 = fluid.layers.fill_constant([1], "int32", 3)
            result_2 = fluid.layers.gaussian_random(shape=[dim_1, dim_2])
            # [[ 0.51398206, -0.3389769,   0.23597084],
            #  [ 1.0388143,  -1.2015356,  -1.0499583 ]]

            # example 3:
            # attr shape is a Tensor, the data type must be int64 or int32.
            var_shape = fluid.data(name='var_shape', shape=[2], dtype="int64")
            result_3 = fluid.layers.gaussian_random(var_shape)
            # if var_shape's value is [2, 3]
            # result_3 is:
            # [[-0.12310527,  0.8187662,   1.923219  ]
            #  [ 0.70721835,  0.5210541,  -0.03214082]]

       .. code-block:: python

           # declarative mode
           # required: skiptest
           import numpy as np
           from paddle import fluid

           x = fluid.layers.gaussian_random((2, 3), std=2., seed=10)

           place = fluid.CPUPlace()
           exe = fluid.Executor(place)
           start = fluid.default_startup_program()
           main = fluid.default_main_program()

           exe.run(start)
           x_np, = exe.run(main, feed={}, fetch_list=[x])

           x_np
           # array([[2.3060477, 2.676496 , 3.9911983],
           #        [0.9990833, 2.8675377, 2.2279181]], dtype=float32)

       .. code-block:: python

           # imperative mode
           import numpy as np
           from paddle import fluid
           import paddle.fluid.dygraph as dg

           place = fluid.CPUPlace()
           with dg.guard(place) as g:
               x = fluid.layers.gaussian_random((2, 4), mean=2., dtype="float32", seed=10)
               x_np = x.numpy()
           x_np
           # array([[2.3060477 , 2.676496  , 3.9911983 , 0.9990833 ],
           #        [2.8675377 , 2.2279181 , 0.79029655, 2.8447366 ]], dtype=float32)
    """
    if not isinstance(dtype, core.VarDesc.VarType):
        dtype = convert_np_dtype_to_dtype_(dtype)

    if in_dygraph_mode():
        shape = utils.convert_shape_to_list(shape)
        place = _current_expected_place()
        return _C_ops.gaussian(
            shape, float(mean), float(std), seed, dtype, place
        )

    if _in_legacy_dygraph():
        shape = utils.convert_shape_to_list(shape)
        return _legacy_C_ops.gaussian_random(
            'shape',
            shape,
            'mean',
            float(mean),
            'std',
            float(std),
            'seed',
            seed,
            'dtype',
            dtype,
        )

    check_type(shape, 'shape', (list, tuple, Variable), 'gaussian_random/randn')
    check_dtype(dtype, 'dtype', ['float32', 'float64'], 'gaussian_random/randn')

    inputs = {}
    attrs = {
        'mean': mean,
        'std': std,
        'seed': seed,
        'dtype': dtype,
        'use_mkldnn': False,
    }
    utils.get_shape_tensor_inputs(
        inputs=inputs, attrs=attrs, shape=shape, op_type='gaussian_random/randn'
    )

    helper = LayerHelper('gaussian_random', **locals())
    out = helper.create_variable_for_type_inference(dtype)
    helper.append_op(
        type='gaussian_random', inputs=inputs, outputs={'Out': out}, attrs=attrs
    )

    return out


@templatedoc()
def sampling_id(x, min=0.0, max=1.0, seed=0, dtype='float32'):
    """
    This op is used for sampling id from multinomial distribution from the input, sampling one id for one sample.

    Parameters:
        x (Variable): 2-D tensor, [batch_size, input_feature_dimensions]
        min (Float): minimum , default 0.0.
        max (Float): maximum, default 1.0.
        seed (Float): Random seed, default 0. if seed is not 0, will generate same number every time.
        dtype(np.dtype|core.VarDesc.VarType|str): The type of output data : float32, float_16, int etc

    Returns:
        Variable: sampling tensor.

    Examples:
        .. code-block:: python

            import paddle.fluid as fluid
            x = fluid.data(
                name="X",
                shape=[13, 11],
                dtype='float32')

            out = fluid.layers.sampling_id(x)
    """

    helper = LayerHelper('sampling_id', **locals())
    out = helper.create_variable_for_type_inference(dtype)
    helper.append_op(
        type='sampling_id',
        inputs={'X': x},
        outputs={'Out': out},
        attrs={'min': min, 'max': max, 'seed': seed},
    )

    return out


def _elementwise_op(helper):
    op_type = helper.layer_type
    x = helper.kwargs.get('x', None)
    y = helper.kwargs.get('y', None)

    assert x is not None, 'x cannot be None in {}'.format(op_type)
    assert y is not None, 'y cannot be None in {}'.format(op_type)
    check_variable_and_dtype(
        x,
        'x',
        ['float16', 'uint16', 'float32', 'float64', 'int32', 'int64'],
        op_type,
    )
    check_variable_and_dtype(
        y,
        'y',
        ['float16', 'uint16', 'float32', 'float64', 'int32', 'int64'],
        op_type,
    )

    axis = helper.kwargs.get('axis', -1)
    use_mkldnn = helper.kwargs.get('use_mkldnn', False)
    name = helper.kwargs.get('name', None)
    out = helper.create_variable_for_type_inference(dtype=x.dtype)

    helper.append_op(
        type=op_type,
        inputs={'X': x, 'Y': y},
        outputs={'Out': out},
        attrs={'axis': axis, 'use_mkldnn': use_mkldnn},
    )
    return helper.append_activation(out)


def elementwise_add(x, y, axis=-1, act=None, name=None):
    """

    Examples:

        .. code-block:: python

            import paddle.fluid as fluid
            import numpy as np
            import paddle
            def gen_data():
                return {
                    "x": np.array([2, 3, 4]).astype('float32'),
                    "y": np.array([1, 5, 2]).astype('float32')
                }
            paddle.enable_static()
            x = fluid.data(name="x", shape=[3], dtype='float32')
            y = fluid.data(name="y", shape=[3], dtype='float32')
            z = fluid.layers.elementwise_add(x, y)
            # z = x + y

            place = fluid.CPUPlace()
            exe = fluid.Executor(place)
            z_value = exe.run(feed=gen_data(),
                                fetch_list=[z.name])

            print(z_value) # [3., 8., 6.]


        .. code-block:: python

            import paddle.fluid as fluid
            import numpy as np
            import paddle

            def gen_data():
                return {
                    "x": np.ones((2, 3, 4, 5)).astype('float32'),
                    "y": np.zeros((3, 4)).astype('float32')
                }
            paddle.enable_static()
            x = fluid.data(name="x", shape=[2,3,4,5], dtype='float32')
            y = fluid.data(name="y", shape=[3,4], dtype='float32')
            z = fluid.layers.elementwise_add(x, y, axis=1)
            # z = x + y

            place = fluid.CPUPlace()
            exe = fluid.Executor(place)

            z_value = exe.run(feed=gen_data(),
                                fetch_list=[z.name])

            print(z_value) # z.shape=[2,3,4,5]


        ..  code-block:: python

            import paddle.fluid as fluid
            import numpy as np
            import paddle

            def gen_data():
                return {
                    "x": np.random.randint(1, 5, size=[2, 3, 4, 5]).astype('float32'),
                    "y": np.random.randint(1, 5, size=[5]).astype('float32')
                }
            paddle.enable_static()
            x = fluid.data(name="x", shape=[2,3,4,5], dtype='float32')
            y = fluid.data(name="y", shape=[5], dtype='float32')
            z = fluid.layers.elementwise_add(x, y, axis=3)
            # z = x + y

            place = fluid.CPUPlace()
            exe = fluid.Executor(place)

            z_value = exe.run(feed=gen_data(),
                                fetch_list=[z.name])
            print(z_value) # z.shape=[2,3,4,5]

    """
    if _non_static_mode():
        return _elementwise_op_in_dygraph(
            x,
            y,
            axis=axis,
            act=act,
            op_name='elementwise_add',
            use_mkldnn=_global_flags()["FLAGS_use_mkldnn"],
        )

    return _elementwise_op(LayerHelper('elementwise_add', **locals()))


def elementwise_sub(x, y, axis=-1, act=None, name=None):
    """

    Examples:

        .. code-block:: python

            import paddle.fluid as fluid
            import numpy as np
            import paddle

            def gen_data():
                return {
                    "x": np.array([2, 3, 4]).astype('float32'),
                    "y": np.array([1, 5, 2]).astype('float32')
                }
            paddle.enable_static()
            x = fluid.data(name="x", shape=[3], dtype='float32')
            y = fluid.data(name="y", shape=[3], dtype='float32')
            z = fluid.layers.elementwise_sub(x, y)
            # z = x - y

            place = fluid.CPUPlace()
            exe = fluid.Executor(place)
            z_value = exe.run(feed=gen_data(),
                                fetch_list=[z.name])

            print(z_value) # [1., -2., 2.]


        .. code-block:: python

            import paddle.fluid as fluid
            import numpy as np
            import paddle

            def gen_data():
                return {
                    "x": np.ones((2, 3, 4, 5)).astype('float32'),
                    "y": np.zeros((3, 4)).astype('float32')
                }
            paddle.enable_static()
            x = fluid.data(name="x", shape=[2,3,4,5], dtype='float32')
            y = fluid.data(name="y", shape=[3,4], dtype='float32')
            z = fluid.layers.elementwise_sub(x, y, axis=1)
            # z = x - y

            place = fluid.CPUPlace()
            exe = fluid.Executor(place)

            z_value = exe.run(feed=gen_data(),
                                fetch_list=[z.name])

            print(z_value) # z.shape=[2,3,4,5]


        ..  code-block:: python

            import paddle.fluid as fluid
            import numpy as np
            import paddle

            def gen_data():
                return {
                    "x": np.random.randint(1, 5, size=[2, 3, 4, 5]).astype('float32'),
                    "y": np.random.randint(1, 5, size=[5]).astype('float32')
                }
            paddle.enable_static()
            x = fluid.data(name="x", shape=[2,3,4,5], dtype='float32')
            y = fluid.data(name="y", shape=[5], dtype='float32')
            z = fluid.layers.elementwise_sub(x, y, axis=3)
            # z = x - y

            place = fluid.CPUPlace()
            exe = fluid.Executor(place)

            z_value = exe.run(feed=gen_data(),
                                fetch_list=[z.name])
            print(z_value) # z.shape=[2,3,4,5]

    """
    if _non_static_mode():
        return _elementwise_op_in_dygraph(
            x, y, axis=axis, act=act, op_name='elementwise_sub'
        )

    return _elementwise_op(LayerHelper('elementwise_sub', **locals()))


for func in [
    elementwise_add,
    elementwise_sub,
]:
    op_proto = OpProtoHolder.instance().get_op_proto(func.__name__)

    # insert the c++ doc string on top of python doc string
    func.__doc__ = (
        _generate_doc_string_(
            op_proto,
            additional_args_lines=[
                "axis (int32, optional): If X.dimension != Y.dimension, \
            Y.dimension must be a subsequence of x.dimension. \
            And axis is the start dimension index for broadcasting Y onto X. ",
                "act (string, optional): Activation applied to the output. \
            Default is None. Details: :ref:`api_guide_activations_en` ",
                "name (string, optional): Name of the output. \
            Default is None. It's used to print debug info for developers. Details: \
            :ref:`api_guide_Name` ",
            ],
            skip_attrs_set={
                "x_data_format",
                "y_data_format",
                "axis",
                "use_quantizer",
                "mkldnn_data_type",
                "Scale_x",
                "Scale_y",
                "Scale_out",
            },
        )
        + """\n"""
        + str(func.__doc__)
    )

    doc_list = func.__doc__.splitlines()

    for idx, val in enumerate(doc_list):
        if (
            val.startswith("Warning: ")
            and val.endswith(" instead.")
            and "and will be removed in future versions." in val
        ):
            doc_list.insert(0, doc_list.pop(idx))
            func.__doc__ = "\n" + "\n".join(i for i in doc_list)
            break

for func in []:
    op_proto = OpProtoHolder.instance().get_op_proto(func.__name__)
    func.__doc__ = _generate_doc_string_(
        op_proto,
        additional_args_lines=[
            "act (basestring|None): Activation applied to the output.",
            "name (basestring|None): Name of the output.",
        ],
    )
    func.__doc__ = (
        func.__doc__
        + """

Examples:
  .. code-block:: python

    import paddle.fluid as fluid
    # example 1: shape(x) = (2, 3, 4, 5), shape(y) = (2, 3, 4, 5)
    x0 = fluid.layers.data(name="x0", shape=[2, 3, 4, 5], dtype='float32')
    y0 = fluid.layers.data(name="y0", shape=[2, 3, 4, 5], dtype='float32')
    z0 = fluid.layers.%s(x0, y0)

    # example 2: shape(X) = (2, 3, 4, 5), shape(Y) = (5)
    x1 = fluid.layers.data(name="x1", shape=[2, 3, 4, 5], dtype='float32')
    y1 = fluid.layers.data(name="y1", shape=[5], dtype='float32')
    z1 = fluid.layers.%s(x1, y1)

    # example 3: shape(X) = (2, 3, 4, 5), shape(Y) = (4, 5), with axis=-1(default) or axis=2
    x2 = fluid.layers.data(name="x2", shape=[2, 3, 4, 5], dtype='float32')
    y2 = fluid.layers.data(name="y2", shape=[4, 5], dtype='float32')
    z2 = fluid.layers.%s(x2, y2, axis=2)

    # example 4: shape(X) = (2, 3, 4, 5), shape(Y) = (3, 4), with axis=1
    x3 = fluid.layers.data(name="x3", shape=[2, 3, 4, 5], dtype='float32')
    y3 = fluid.layers.data(name="y3", shape=[3, 4], dtype='float32')
    z3 = fluid.layers.%s(x3, y3, axis=1)

    # example 5: shape(X) = (2, 3, 4, 5), shape(Y) = (2), with axis=0
    x4 = fluid.layers.data(name="x4", shape=[2, 3, 4, 5], dtype='float32')
    y4 = fluid.layers.data(name="y4", shape=[2], dtype='float32')
    z4 = fluid.layers.%s(x4, y4, axis=0)

    # example 6: shape(X) = (2, 3, 4, 5), shape(Y) = (2, 1), with axis=0
    x5 = fluid.layers.data(name="x5", shape=[2, 3, 4, 5], dtype='float32')
    y5 = fluid.layers.data(name="y5", shape=[2], dtype='float32')
    z5 = fluid.layers.%s(x5, y5, axis=0)
    """
        % (
            func.__name__,
            func.__name__,
            func.__name__,
            func.__name__,
            func.__name__,
            func.__name__,
        )
    )


def _logical_op(op_name, x, y, out=None, name=None, binary_op=True):
    if _non_static_mode():
        op = getattr(_legacy_C_ops, op_name)
        if binary_op:
            return op(x, y)
        else:
            return op(x)
    check_variable_and_dtype(
        x,
        "x",
        ["bool", "int8", "int16", "int32", "int64", "float32", "float64"],
        op_name,
    )
    if y is not None:
        check_variable_and_dtype(
            y,
            "y",
            ["bool", "int8", "int16", "int32", "int64", "float32", "float64"],
            op_name,
        )
    if out is not None:
        check_type(out, "out", Variable, op_name)

    helper = LayerHelper(op_name, **locals())

    if binary_op and x.dtype != y.dtype:
        raise ValueError(
            "(InvalidArgument) The DataType of %s Op's Variable must be consistent, but received %s and %s."
            % (op_name, x.dtype, y.dtype)
        )

    if out is None:
        out = helper.create_variable_for_type_inference(dtype=x.dtype)

    if binary_op:
        helper.append_op(
            type=op_name, inputs={"X": x, "Y": y}, outputs={"Out": out}
        )
    else:
        helper.append_op(type=op_name, inputs={"X": x}, outputs={"Out": out})

    return out


@templatedoc()
def clip(x, min, max, name=None):
    """
        :old_api: paddle.fluid.layers.clip

    ${comment}

    Args:
        x(${x_type}): ${x_comment}
        min(float): ${min_comment}
        max(float): ${max_comment}
        name(str, optional): The default value is None.
                             Normally there is no need for user to set this property.
                             For more information, please refer to :ref:`api_guide_Name`

    Returns:
        ${out_comment}

    Return Type:
        ${out_type}

    Examples:
        .. code-block:: python

            import paddle.fluid as fluid
            input = fluid.data(
                name='data', shape=[1], dtype='float32')
            reward = fluid.layers.clip(x=input, min=-1.0, max=1.0)
    """

    helper = LayerHelper("clip", **locals())
    check_variable_and_dtype(x, 'x', ['float16', 'float32', 'float64'], 'clip')

    if name is None:
        name = unique_name.generate_with_ignorable_key(
            ".".join([helper.name, 'tmp'])
        )

    out = helper.create_variable(
        type=x.type, name=name, dtype=x.dtype, persistable=False
    )

    helper.append_op(
        type="clip",
        inputs={"X": x},
        attrs={"min": min, "max": max},
        outputs={"Out": out},
    )

    return out


@templatedoc()
def clip_by_norm(x, max_norm, name=None):
    """
    ${comment}

    Args:
        x(${x_type}): ${x_comment}
        max_norm(${max_norm_type}): ${max_norm_comment}
        name(str, optional): For detailed information, please refer
            to :ref:`api_guide_Name`. Usually name is no need to set and
            None by default.

    Returns:
        Tensor:

        out(${out_type}): ${out_comment}


    Examples:
        .. code-block:: python

            import paddle
            import paddle.fluid as fluid

            input = paddle.to_tensor([[2.0, 2.0], [2.0, 2.0]], dtype='float32')
            reward = fluid.layers.clip_by_norm(x=input, max_norm=1.0)
            # [[0.5, 0.5], [0.5, 0.5]]
    """

    if in_dygraph_mode():
        return _C_ops.clip_by_norm(x, max_norm)
    if _non_static_mode():
        return _legacy_C_ops.clip_by_norm(x, 'max_norm', max_norm)

    helper = LayerHelper("clip_by_norm", **locals())
    check_variable_and_dtype(x, 'X', ['float32', 'float16'], 'clip_by_norm')
    check_type(max_norm, 'max_norm', (float), 'clip_by_norm')

    if name is None:
        name = unique_name.generate_with_ignorable_key(
            ".".join([helper.name, 'tmp'])
        )

    out = helper.create_variable(
        type=x.type, name=name, dtype=x.dtype, persistable=False
    )

    helper.append_op(
        type="clip_by_norm",
        inputs={"X": x},
        attrs={"max_norm": max_norm},
        outputs={"Out": out},
    )

    return out


@deprecated(since="2.0.0", update_to="paddle.mean")
@templatedoc()
def mean(x, name=None):
    """
    ${comment}

    Args:
        x(${x_type}): ${x_comment}
        name(basestring|None): Name of the output.

    Returns:
        out(${out_type}): ${out_comment}

    Examples:
        .. code-block:: python

            import paddle
            import paddle.fluid as fluid
            paddle.enable_static()

            input = fluid.layers.data(
                name='data', shape=[2, 3], dtype='float32')
            mean = paddle.mean(input)
    """

    if _in_legacy_dygraph():
        return _legacy_C_ops.mean(x)
    if in_dygraph_mode():
        return _C_ops.mean_all(x)

    helper = LayerHelper("mean", **locals())
    check_variable_and_dtype(x, 'x', ['float16', 'float32', 'float64'], 'mean')
    out = helper.create_variable_for_type_inference(dtype=x.dtype)

    helper.append_op(
        type="mean", inputs={"X": x}, attrs={}, outputs={"Out": out}
    )

    return out


@templatedoc()
def merge_selected_rows(x, name=None):
    """
    ${comment}

    Args:
        x(${x_type}): ${x_comment}
        name(basestring|None): Name of the output.

    Returns:
        out(${out_type}): ${out_comment}

    Examples:
        .. code-block:: python

            import paddle.fluid as fluid
            b = fluid.default_main_program().global_block()
            var = b.create_var(
                name="X", dtype="float32", persistable=True,
                type=fluid.core.VarDesc.VarType.SELECTED_ROWS)
            y = fluid.layers.merge_selected_rows(var)
    """
    if in_dygraph_mode():
        return _C_ops.merge_selected_rows(x)

    if _non_static_mode():
        return _legacy_C_ops.merge_selected_rows(x)

    helper = LayerHelper("merge_selected_rows", **locals())
    out = helper.create_variable_for_type_inference(dtype=x.dtype)
    helper.append_op(
        type="merge_selected_rows",
        inputs={"X": x},
        attrs={},
        outputs={"Out": out},
    )
    return out


def mul(x, y, x_num_col_dims=1, y_num_col_dims=1, name=None):
    """
    Mul Operator.
    This operator is used to perform matrix multiplication for input $x$ and $y$.
    The equation is:

    ..  math::
        Out = x * y

    Both the input $x$ and $y$ can carry the LoD (Level of Details) information, or not. But the output only shares the LoD information with input $x$.

    Args:
        x (Variable): The first input Tensor/LoDTensor of mul_op.
        y (Variable): The second input Tensor/LoDTensor of mul_op.
        x_num_col_dims (int, optional): The mul_op can take tensors with more than two dimensions as its inputs. If the input $x$ is a tensor with more than two dimensions, $x$ will be flattened into a two-dimensional matrix first. The flattening rule is: the first `num_col_dims` will be flattened to form the first dimension of the final matrix (the height of the matrix), and the rest `rank(x) - num_col_dims` dimensions are flattened to form the second dimension of the final matrix (the width of the matrix). As a result, height of the flattened matrix is equal to the product of $x$'s first `x_num_col_dims` dimensions' sizes, and width of the flattened matrix is equal to the product of $x$'s last `rank(x) - num_col_dims` dimensions' size. For example, suppose $x$ is a 6-dimensional tensor with the shape [2, 3, 4, 5, 6], and `x_num_col_dims` = 3. Thus, the flattened matrix will have a shape [2 x 3 x 4, 5 x 6] = [24, 30]. Default is 1.
        y_num_col_dims (int, optional): The mul_op can take tensors with more than two dimensions as its inputs. If the input $y$ is a tensor with more than two dimensions, $y$ will be flattened into a two-dimensional matrix first. The attribute `y_num_col_dims` determines how $y$ is flattened. See comments of `x_num_col_dims` for more details. Default is 1.
        name (str, optional): Name of the output. Normally there is no need for user to set this property. For more information, please refer to :ref:`api_guide_Name`. Default is None.

    Returns:
        Variable(Tensor/LoDTensor): The output Tensor/LoDTensor of mul op.

    Examples:
        ..  code-block:: python

            import paddle.fluid as fluid
            import paddle
            paddle.enable_static()
            dataX = fluid.layers.data(name="dataX", append_batch_size = False, shape=[2, 5], dtype="float32")
            dataY = fluid.layers.data(name="dataY", append_batch_size = False, shape=[5, 3], dtype="float32")
            output = fluid.layers.mul(dataX, dataY,
                                      x_num_col_dims = 1,
                                      y_num_col_dims = 1)


    """
    if _non_static_mode():
        return _legacy_C_ops.mul(
            x,
            y,
            'x_num_col_dims',
            x_num_col_dims,
            'y_num_col_dims',
            y_num_col_dims,
        )

    inputs = {"X": [x], "Y": [y]}
    attrs = {"x_num_col_dims": x_num_col_dims, "y_num_col_dims": y_num_col_dims}
    helper = LayerHelper("mul", **locals())
    check_variable_and_dtype(x, 'x', ['float16', 'float32', 'float64'], 'mul')
    check_variable_and_dtype(y, 'y', ['float16', 'float32', 'float64'], 'mul')
    out = helper.create_variable_for_type_inference(dtype=x.dtype)

    helper.append_op(
        type="mul", inputs={"X": x, "Y": y}, attrs=attrs, outputs={"Out": out}
    )
    return out


@templatedoc()
def get_tensor_from_selected_rows(x, name=None):
    """
    This operator gets tensor data from input with SelectedRows type, and outputs a LoDTensor.

    .. code-block:: text

        input x is SelectedRows:
           x.rows = [0, 5, 5, 4, 19]
           x.height = 20
           x.value = [[1, 1] [2, 2] [2, 2] [3, 3] [6, 6]]

        Output is LoDTensor:
           out.shape = [5, 2]
           out.data = [[1, 1],
                       [2, 2],
                       [2, 2],
                       [3, 3],
                       [6, 6]]

    Args:
        x(SelectedRows): Input with SelectedRows type. The data type is float32, float64, int32 or int64.
        name(str, optional): The default value is None.  Normally there is no need for user to set this property.
            For more information, please refer to :ref:`api_guide_Name` .

    Returns:
        Variable: LoDTensor transformed from SelectedRows. The data type is same with input.

    Examples:
        .. code-block:: python

            import paddle.fluid as fluid
            b = fluid.default_main_program().global_block()
            input = b.create_var(name="X", dtype="float32", persistable=True, type=fluid.core.VarDesc.VarType.SELECTED_ROWS)
            out = fluid.layers.get_tensor_from_selected_rows(input)
    """

    check_type(x, 'x', Variable, 'get_tensor_from_selected_rows')
    if x.type != core.VarDesc.VarType.SELECTED_ROWS:
        raise TypeError(
            "The type of 'x' in get_tensor_from_selected_rows must be SELECTED_ROWS."
        )
    helper = LayerHelper('get_tensor_from_selected_rows', **locals())
    out = helper.create_variable_for_type_inference(dtype=x.dtype)
    helper.append_op(
        type='get_tensor_from_selected_rows',
        inputs={'X': x},
        outputs={'Out': out},
        attrs={},
    )
    return out


def unfold(x, kernel_sizes, strides=1, paddings=0, dilations=1, name=None):
    r"""

    This op returns a col buffer of sliding local blocks of input x, also known
    as im2col for batched 2D image tensors. For each block under the convolution filter,
    all element will be rearranged as a column. While the convolution filter sliding over
    the input feature map, a series of such columns will be formed.

    For each input :math:`x` with shape [N, C, H, W], the output shape [N, Cout, Lout]
    can be calculated as following.

    .. math::

        dkernel[0] &= dilations[0] \times (kernel\_sizes[0] - 1) + 1

        dkernel[1] &= dilations[1] \times (kernel\_sizes[1] - 1) + 1

        hout &= \frac{H + paddings[0] + paddings[2] - dkernel[0]}{strides[0]} + 1

        wout &= \frac{W + paddings[1] + paddings[3] - dkernel[1]}{strides[1]} + 1

        Cout &= C \times kernel\_sizes[0] \times kernel\_sizes[1]

        Lout &= hout \times wout


    Parameters:
        x(Tensor):              4-D Tensor, input tensor of format [N, C, H, W],
                                  data type can be float32 or float64
        kernel_sizes(int|list):   The size of convolution kernel, should be [k_h, k_w]
                                  or an integer k treated as [k, k].
        strides(int|list):        The strides, should be [stride_h, stride_w]
                                  or an integer stride treated as [sride, stride].
                                  For default, strides will be [1, 1].
        paddings(int|list):       The paddings of each dimension, should be
                                  [padding_top, padding_left, padding_bottom, padding_right]
                                  or [padding_h, padding_w] or an integer padding.
                                  If [padding_h, padding_w] was given, it will expanded to
                                  [padding_h, padding_w, padding_h, padding_w]. If an integer
                                  padding was given, [padding, padding, padding, padding] will
                                  be used. For default, paddings will be [0, 0, 0, 0]
        dilations(int|list):      the dilations of convolution kernel, should be
                                  [dilation_h, dilation_w], or an integer dilation treated as
                                  [dilation, dilation]. For default, it will be [1, 1].
        name(str, optional): The default value is None.
                             Normally there is no need for user to set this property.
                             For more information, please refer to :ref:`api_guide_Name`


    Returns:
        The tensor corresponding to the sliding local blocks.
        The output shape is [N, Cout, Lout] as decriabled above.
        Cout is the  total number of values within each block,
        and Lout is the total number of such blocks.
        The data type of output is the same as the input :math:`x`

    Return Type:
        Tensor

    Examples:

        .. code-block:: python

            import paddle
            import paddle.nn.functional as F

            x = paddle.randn((100,3,224,224))
            y = F.unfold(x, [3, 3], 1, 1, 1)
    """

    return paddle.nn.functional.unfold(
        x, kernel_sizes, strides, paddings, dilations, name
    )


def deformable_roi_pooling(
    input,
    rois,
    trans,
    no_trans=False,
    spatial_scale=1.0,
    group_size=[1, 1],
    pooled_height=1,
    pooled_width=1,
    part_size=None,
    sample_per_part=1,
    trans_std=0.1,
    position_sensitive=False,
    name=None,
):
    r"""

    Deformable ROI Pooling Layer

    Performs deformable region-of-interest pooling on inputs. As described
    in `Deformable Convolutional Networks <https://arxiv.org/abs/1703.06211>`_, it will get offset for each bin after
    roi pooling so that pooling at correct region. Batch_size will change to the number of region bounding boxes after deformable_roi_pooling.

    The operation has three steps:

    1. Dividing each region proposal into equal-sized sections with the pooled_width and pooled_height.

    2. Add offset to pixel in ROI to get new location and the new value which are computed directly through
       bilinear interpolation with four nearest pixel.

    3. Sample several points in each bin to get average values as output.


    Args:
        input (Variable):The input of deformable roi pooling and it is tensor which value type is float32. The shape of input is
                         [N, C, H, W]. Where N is batch size, C is number of input channels,
                         H is height of the feature, and W is the width of the feature.
        rois (Variable): ROIs (Regions of Interest) with type float32 to pool over. It should be
                         a 2-D LoDTensor of shape (num_rois, 4), and the lod level
                         is 1. Given as [[x1, y1, x2, y2], ...], (x1, y1) is
                         the top left coordinates, and (x2, y2) is the bottom
                         right coordinates, which value type is float32.
        trans (Variable): Offset of features on ROIs while pooling which value type is float32. The format is [N, C, H, W], where
                          N is number of ROIs, C is number of channels, which indicate the offset distance
                          in the x and y directions, H is pooled height, and W is pooled width.
        no_trans (bool): Whether to add offset to get new value or not while roi pooling, which value with type bool is True or False.
                         If value is True, no offset will be added in operation. Default: False.
        spatial_scale (float): Ratio of input feature map height (or width) to raw image height (or width), which value type is float32.
                         Equals the reciprocal of total stride in convolutional layers, Default: 1.0.
        group_size (list|tuple): The number of groups which input channels are divided and the input is list or tuple, which value type is int32. (eg.number of input channels
                          is k1 * k2 * (C + 1), which k1 and k2 are group width and height and C+1 is number of output
                          channels.) eg.(4, 6), which 4 is height of group and 6 is width of group. Default: [1, 1].
        pooled_height (int): The pooled output height which value type is int32. Default: 1.
        pooled_width (int): The pooled output width which value type is int32. Default: 1.
        part_size (list|tuple): The height and width of offset which values in list or tuple is int32, eg.(4, 6), which height is 4 and width is 6, and values always equal to pooled_height \
                         and pooled_width. Default: if None, default value is [pooled_height, pooled_width].
        sample_per_part (int): The number of samples in each bin which value type is int32. If value is bigger, it will consume more performance. Default: 1.
        trans_std (float): Coefficient of offset which value type is float32. It controls weight of offset. Default: 0.1.
        position_sensitive (bool): Whether to choose deformable psroi pooling mode or not, and value type is bool(True or False). If value is False, input dimension equals to output dimension. \
                                   If value is True, input dimension should be output dimension * pooled_height * pooled_width. Default: False.
        name (str|None): Name of layer. Default: None.
    Returns:
        Variable: Output of deformable roi pooling is that, if position sensitive is False, input dimension equals to output dimension. If position sensitive is True,\
                  input dimension should be the result of output dimension divided by pooled height and pooled width.

    Examples:
      .. code-block:: python

        # position_sensitive=True
        import paddle.fluid as fluid
        input = fluid.data(name="input",
                           shape=[2, 192, 64, 64],
                           dtype='float32')
        rois = fluid.data(name="rois",
                          shape=[-1, 4],
                          dtype='float32',
                          lod_level=1)
        trans = fluid.data(name="trans",
                           shape=[2, 384, 64, 64],
                           dtype='float32')
        x = fluid.layers.deformable_roi_pooling(input=input,
                                                rois=rois,
                                                trans=trans,
                                                no_trans=False,
                                                spatial_scale=1.0,
                                                group_size=(1, 1),
                                                pooled_height=8,
                                                pooled_width=8,
                                                part_size=(8, 8),
                                                sample_per_part=4,
                                                trans_std=0.1,
                                                position_sensitive=True)

        # position_sensitive=False
        import paddle.fluid as fluid
        input = fluid.data(name="input",
                           shape=[2, 192, 64, 64],
                           dtype='float32')
        rois = fluid.data(name="rois",
                          shape=[-1, 4],
                          dtype='float32',
                          lod_level=1)
        trans = fluid.data(name="trans",
                           shape=[2, 384, 64, 64],
                           dtype='float32')
        x = fluid.layers.deformable_roi_pooling(input=input,
                                                rois=rois,
                                                trans=trans,
                                                no_trans=False,
                                                spatial_scale=1.0,
                                                group_size=(1, 1),
                                                pooled_height=8,
                                                pooled_width=8,
                                                part_size=(8, 8),
                                                sample_per_part=4,
                                                trans_std=0.1,
                                                position_sensitive=False)
    """

    check_variable_and_dtype(
        input, 'input', ['float32', 'float64'], 'deformable_roi_pooling'
    )
    check_variable_and_dtype(
        rois, 'rois', ['float32', 'float64'], 'deformable_roi_pooling'
    )
    check_variable_and_dtype(
        trans, 'trans', ['float32', 'float64'], 'deformable_roi_pooling'
    )
    check_type(
        group_size, 'group_size', (list, tuple), 'deformable_roi_pooling'
    )
    if part_size is not None:
        check_type(
            part_size, 'part_size', (list, tuple), 'deformable_roi_pooling'
        )

    input_channels = input.shape[1]
    if position_sensitive is False:
        output_channels = input_channels
    else:
        output_channels = input_channels / pooled_height / pooled_width

    if part_size is None:
        part_height = pooled_height
        part_width = pooled_width
        part_size = [part_height, part_width]
    part_size = utils.convert_to_list(part_size, 2, 'part_size')
    group_size = utils.convert_to_list(group_size, 2, 'group_size')
    helper = LayerHelper('deformable_psroi_pooling', **locals())
    dtype = helper.input_dtype()
    output = helper.create_variable_for_type_inference(dtype)
    top_count = helper.create_variable_for_type_inference(dtype='int32')
    helper.append_op(
        type="deformable_psroi_pooling",
        inputs={"Input": input, "ROIs": rois, "Trans": trans},
        outputs={"Output": output, "TopCount": top_count},
        attrs={
            "no_trans": no_trans,
            "spatial_scale": spatial_scale,
            "output_dim": output_channels,
            "group_size": group_size,
            "pooled_height": pooled_height,
            "pooled_width": pooled_width,
            "part_size": part_size,
            "sample_per_part": sample_per_part,
            "trans_std": trans_std,
        },
    )
    return output


@deprecated(since="2.0.0", update_to="paddle.shard_index")
def shard_index(input, index_num, nshards, shard_id, ignore_value=-1):
    """
    Reset the values of `input` according to the shard it beloning to.
    Every value in `input` must be a non-negative integer, and
    the parameter `index_num` represents the integer above the maximum
    value of `input`. Thus, all values in `input` must be in the range
    [0, index_num) and each value can be regarded as the offset to the beginning
    of the range. The range is further split into multiple shards. Specifically,
    we first compute the `shard_size` according to the following formula,
    which represents the number of integers each shard can hold. So for the
    i'th shard, it can hold values in the range [i*shard_size, (i+1)*shard_size).
    ::

        shard_size = (index_num + nshards - 1) // nshards

    For each value `v` in `input`, we reset it to a new value according to the
    following formula:
    ::

        v = v - shard_id * shard_size if shard_id * shard_size <= v < (shard_id+1) * shard_size else ignore_value

    That is, the value `v` is set to the new offset within the range represented by the shard `shard_id`
    if it in the range. Otherwise, we reset it to be `ignore_value`.

    Args:
        input (Tensor): Input tensor with data type int64 or int32. It's last dimension must be 1.
        index_num (int): An integer represents the integer above the maximum value of `input`.
        nshards (int): The number of shards.
        shard_id (int): The index of the current shard.
        ignore_value (int): An integer value out of sharded index range.

    Returns:
        Tensor.

    Examples:
        .. code-block:: python

            import paddle
            label = paddle.to_tensor([[16], [1]], "int64")
            shard_label = paddle.shard_index(input=label,
                                             index_num=20,
                                             nshards=2,
                                             shard_id=0)
            print(shard_label)
            # [[-1], [1]]
    """
    if in_dygraph_mode():
        return _C_ops.shard_index(
            input, index_num, nshards, shard_id, ignore_value
        )

    check_variable_and_dtype(input, 'input', ['int64', 'int32'], 'shard_index')
    op_type = 'shard_index'
    helper = LayerHelper(op_type, **locals())
    if shard_id < 0 or shard_id >= nshards:
        raise ValueError(
            'The shard_id(%d) should be in [0, %d)' % (shard_id, nshards)
        )

    out = helper.create_variable_for_type_inference(dtype=input.dtype)
    helper.append_op(
        type=op_type,
        inputs={'X': [input]},
        outputs={'Out': out},
        attrs={
            'index_num': index_num,
            'nshards': nshards,
            'shard_id': shard_id,
            'ignore_value': ignore_value,
        },
        stop_gradient=True,
    )
    return out


@templatedoc()
def hard_swish(x, threshold=6.0, scale=6.0, offset=3.0, name=None):
    r"""
    This operator implements the hard_swish activation function.
    Hard_swish is proposed in MobileNetV3, and performs better in computational stability and efficiency compared to swish function.
    For more details please refer to: https://arxiv.org/pdf/1905.02244.pdf

    The formula is as follows:

    .. math::

        out = \\frac{x * (min(max(0, x+offset), threshold))}{scale}

    In the above equation:

    ``threshold`` and ``scale`` should be positive, ``offset`` can be positive or negative. It is recommended to use default parameters.

    Args:
        x (Variable): Input feature, multi-dimensional Tensor. The data type should be float32 or float64.
        threshold (float, optional): The threshold in Relu function. Default: 6.0
        scale (float, optional): The scale factor. Default: 6.0
        offset (float, optional): The offset factor. Default: 3.0
        name (str, optional): The default value is None. Normally there is no need for user to set this property. For more information, please refer to :ref:`api_guide_Name`

    Returns:
        Variable: The output tensor with the same shape and data type as input.


    Examples:

    .. code-block:: python

        import paddle.fluid as fluid
        import paddle
        import numpy as np
        paddle.enable_static()

        DATATYPE='float32'

        x_data = np.array([i for i in range(1,5)]).reshape([1,1,4]).astype(DATATYPE)

        x = fluid.data(name="x", shape=[None,1,4], dtype=DATATYPE)
        y = fluid.layers.hard_swish(x)

        place = fluid.CPUPlace()
        #place = fluid.CUDAPlace(0)
        exe = fluid.Executor(place)
        out, = exe.run(feed={'x':x_data}, fetch_list=[y.name])
        print(out)  # [[0.66666667, 1.66666667,3., 4.]]
    """
    if _non_static_mode():
        return _legacy_C_ops.hard_swish(
            x, 'threshold', threshold, 'scale', scale, 'offset', offset
        )

    check_variable_and_dtype(
        x, 'x', ['float16', 'float32', 'float64'], 'hard_swish'
    )

    helper = LayerHelper('hard_swish', **locals())
    out = helper.create_variable_for_type_inference(dtype=x.dtype)
    helper.append_op(
        type='hard_swish',
        inputs={'X': x},
        outputs={'Out': out},
        attrs={'threshold': threshold, 'scale': scale, 'offset': offset},
    )
    return out


@templatedoc()
def mish(x, threshold=20, name=None):
    r"""
    This operator implements the mish activation function.
    Refer to `Mish: A Self Regularized Non-Monotonic Neural
    Activation Function <https://arxiv.org/abs/1908.08681>`_


    The formula is as follows if :attr:`threshold` is :code:`None` or negative:

    .. math::

        out = x * \\tanh(\\ln(1 + e^{x}))

    The formula is as follows if :attr:`threshold` is set as positive value:

    .. math::

    out = \\begin{cases}
        x \\ast \\tanh(x), \\text{if } x > \\text{threshold} \\\\
        x \\ast \\tanh(e^{x}), \\text{if } x < -\\text{threshold} \\\\
        x \\ast \\tanh(\\ln(1 + e^{x})),  \\text{otherwise}
          \\end{cases}

    Args:
        x (Variable): Input feature, multi-dimensional Tensor. The data type
                      should be float16, float32 or float64.
        threshold (float|None): threshold for softplus in Mish operator.
                Approximate value of softplus will be used if absolute value
                of input is greater than :attr:threshold and :attr:threshold
                is set as positive value. For none or negative threshold,
                approximate value is not used. Default 20.
        name (str, optional): The default value is None. Normally there is no
                need for user to set this property. For more information, please
                refer to :ref:`api_guide_Name`

    Returns:
        Variable: The output tensor with the same shape and data type as input.


    Examples:

    .. code-block:: python

        import paddle
        import paddle.fluid as fluid
        import numpy as np

        paddle.enable_static()
        DATATYPE='float32'

        x_data = np.array([i for i in range(1,5)]).reshape([1,1,4]).astype(DATATYPE)

        x = fluid.data(name="x", shape=[None,1,4], dtype=DATATYPE)
        y = fluid.layers.mish(x)

        place = fluid.CPUPlace()
        # place = fluid.CUDAPlace(0)
        exe = fluid.Executor(place)
        out, = exe.run(feed={'x':x_data}, fetch_list=[y.name])
        print(out)  # [[0.66666667, 1.66666667, 3., 4.]]
    """
    if in_dygraph_mode():
        return _C_ops.mish(x, threshold)
    if _in_legacy_dygraph():
        return _legacy_C_ops.mish(x, 'threshold', threshold)

    check_variable_and_dtype(x, 'x', ['float32', 'float64'], 'mish')
    check_type(threshold, 'threshold', (float, int), 'mish')
    assert (
        threshold > 0
    ), "threshold of mish should be greater than 0, " "but got {}".format(
        threshold
    )

    helper = LayerHelper('mish', **locals())
    out = helper.create_variable_for_type_inference(dtype=x.dtype)
    helper.append_op(
        type='mish',
        inputs={'X': x},
        outputs={'Out': out},
        attrs={'threshold': threshold},
    )
    return out


@deprecated(since="2.0.0", update_to="paddle.uniform")
@templatedoc()
def uniform_random(
    shape, dtype='float32', min=-1.0, max=1.0, seed=0, name=None
):
    """
    This OP returns a Tensor filled with random values sampled from a uniform
    distribution in the range [``min``, ``max``), with ``shape`` and ``dtype``.

    Examples:
    ::

        Input:
          shape = [1, 2]

        Output:
          result=[[0.8505902, 0.8397286]]

    Args:
        shape(list|tuple|Tensor): The shape of the output Tensor. If ``shape``
            is a list or tuple, the elements of it should be integers or Tensors
            (with the shape [1], and the data type int32 or int64). If ``shape``
            is a Tensor, it should be a 1-D Tensor(with the data type int32 or
            int64).
        dtype(str|np.dtype|core.VarDesc.VarType, optional): The data type of
            the output Tensor. Supported data types: float32, float64.
            Default is float32.
        min(float|int, optional): The lower bound on the range of random values
            to generate, ``min`` is included in the range. Default is -1.0.
        max(float|int, optional): The upper bound on the range of random values
            to generate, ``max`` is excluded in the range. Default is 1.0.
        seed(int, optional): Random seed used for generating samples. 0 means
            use a seed generated by the system. Note that if seed is not 0,
            this operator will always generate the same random numbers every
            time. Default is 0.
        name(str, optional): The default value is None. Normally there is no
            need for user to set this property. For more information, please
            refer to :ref:`api_guide_Name`.

    Returns:
        Tensor: A Tensor filled with random values sampled from a uniform
        distribution in the range [``min``, ``max``), with ``shape`` and ``dtype``.

    Raises:
        TypeError: If ``shape`` is not list, tuple, Tensor.
        TypeError: If ``dtype`` is not float32, float64.

    Examples:
        .. code-block:: python

            import paddle
            import paddle.fluid as fluid
            paddle.enable_static()

            # example 1:
            # attr shape is a list which doesn't contain Tensor.
            result_1 = fluid.layers.uniform_random(shape=[3, 4])
            # [[ 0.84524226,  0.6921872,   0.56528175,  0.71690357],
            #  [-0.34646994, -0.45116323, -0.09902662, -0.11397249],
            #  [ 0.433519,    0.39483607, -0.8660099,   0.83664286]]

            # example 2:
            # attr shape is a list which contains Tensor.
            dim_1 = fluid.layers.fill_constant([1], "int64", 2)
            dim_2 = fluid.layers.fill_constant([1], "int32", 3)
            result_2 = fluid.layers.uniform_random(shape=[dim_1, dim_2])
            # [[-0.9951253,   0.30757582, 0.9899647 ],
            #  [ 0.5864527,   0.6607096,  -0.8886161 ]]

            # example 3:
            # attr shape is a Tensor, the data type must be int64 or int32.
            var_shape = fluid.data(name='var_shape', shape=[2], dtype="int64")
            result_3 = fluid.layers.uniform_random(var_shape)
            # if var_shape's value is [2, 3]
            # result_3 is:
            # [[-0.8517412,  -0.4006908,   0.2551912 ],
            #  [ 0.3364414,   0.36278176, -0.16085452]]

    """
    if not isinstance(dtype, core.VarDesc.VarType):
        dtype = convert_np_dtype_to_dtype_(dtype)

    if in_dygraph_mode():
        shape = utils.convert_shape_to_list(shape)
        return _C_ops.uniform(
            shape,
            dtype,
            float(min),
            float(max),
            seed,
            _current_expected_place(),
        )
    elif _in_legacy_dygraph():
        shape = utils.convert_shape_to_list(shape)
        return _legacy_C_ops.uniform_random(
            'shape',
            shape,
            'min',
            float(min),
            'max',
            float(max),
            'seed',
            seed,
            'dtype',
            dtype,
        )

    check_type(shape, 'shape', (list, tuple, Variable), 'uniform_random/rand')
    check_dtype(
        dtype, 'dtype', ('float32', 'float64', 'uint16'), 'uniform_random/rand'
    )
    check_type(min, 'min', (float, int, Variable), 'uniform_random/rand')
    check_type(max, 'max', (float, int, Variable), 'uniform_random/rand')

    inputs = dict()
    attrs = {'seed': seed, 'min': min, 'max': max, 'dtype': dtype}
    utils.get_shape_tensor_inputs(
        inputs=inputs, attrs=attrs, shape=shape, op_type='uniform_random/rand'
    )

    helper = LayerHelper("uniform_random", **locals())
    out = helper.create_variable_for_type_inference(dtype)
    helper.append_op(
        type="uniform_random", inputs=inputs, attrs=attrs, outputs={"Out": out}
    )
    utils.try_set_static_shape_tensor(out, shape)
    return out


def unbind(input, axis=0):
    """
    Removes a tensor dimension, then split the input tensor into multiple sub-Tensors.
    Args:
        input (Variable): The input variable which is an N-D Tensor, data type being float32, float64, int32 or int64.

        axis (int32|int64, optional): A scalar with type ``int32|int64`` shape [1]. The dimension along which to unbind. If :math:`axis < 0`, the
            dimension to unbind along is :math:`rank(input) + axis`. Default is 0.
    Returns:
        list(Variable): The list of segmented Tensor variables.

    Example:
        .. code-block:: python
            import paddle
            # input is a variable which shape is [3, 4, 5]
            input = paddle.fluid.data(
                 name="input", shape=[3, 4, 5], dtype="float32")
            [x0, x1, x2] = paddle.tensor.unbind(input, axis=0)
            # x0.shape [4, 5]
            # x1.shape [4, 5]
            # x2.shape [4, 5]
            [x0, x1, x2, x3] = paddle.tensor.unbind(input, axis=1)
            # x0.shape [3, 5]
            # x1.shape [3, 5]
            # x2.shape [3, 5]
            # x3.shape [3, 5]

    """
    helper = LayerHelper("unbind", **locals())
    check_type(input, 'input', (Variable), 'unbind')
    dtype = helper.input_dtype()
    check_dtype(
        dtype, 'unbind', ['float32', 'float64', 'int32', 'int64'], 'unbind'
    )
    if not isinstance(axis, (int)):
        raise TypeError(
            "The type of 'axis'  must be int, but received %s." % (type(axis))
        )
    if isinstance(axis, np.generic):
        axis = np.asscalar(axis)
    input_shape = input.shape
    axis_ = axis if axis >= 0 else len(input_shape) + axis
    num = input_shape[axis_]
    outs = [
        helper.create_variable_for_type_inference(dtype=helper.input_dtype())
        for i in range(num)
    ]

    helper.append_op(
        type="unbind",
        inputs={"X": input},
        outputs={"Out": outs},
        attrs={"axis": axis},
    )
    return outs
