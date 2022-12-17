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
    'conv2d',
    'row_conv',
    'layer_norm',
    'spectral_norm',
    'one_hot',
    'autoincreased_step_counter',
    'lod_reset',
    'clip',
    'clip_by_norm',
    'mul',
    'merge_selected_rows',
    'get_tensor_from_selected_rows',
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
