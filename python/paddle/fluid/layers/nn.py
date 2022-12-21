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
from .tensor import concat, assign, fill_constant, zeros
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
    'autoincreased_step_counter',
    'clip',
    'clip_by_norm',
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
