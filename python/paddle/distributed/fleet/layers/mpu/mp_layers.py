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

import paddle
from paddle.autograd import PyLayer
from paddle.fluid import core
from paddle.nn import functional as F

from ...base import topology as tp
from . import mp_ops
from .random import get_rng_state_tracker

__all__ = []

# Follow this paper to achieve the file:
# Shoeybi M, Patwary M, Puri R, et al. Megatron-lm: Training multi-billion parameter
# language models using model parallelism[J]. arXiv preprint arXiv:1909.08053, 2019. (https://arxiv.org/abs/1909.08053)


def is_fused_matmul_bias_supported():
    return hasattr(core.eager.ops.legacy, 'fused_gemm_epilogue')


class VocabParallelEmbedding(paddle.nn.Layer):
    """Embedding mp parallelized in the vocabulary dimension.
    this class is used for splitting embedding in mp group.

    Args:
        num_embeddings(int): One element which indicate the size of the dictionary of embeddings.
        embedding_dim(int): One element which indicate the size of each embedding vector respectively.
        weight_attr(ParamAttr|None): To specify the weight parameter property. Default: None, which means the
            default weight parameter property is used. See usage for details in :ref:`api_ParamAttr` . In addition,
            user-defined or pre-trained word vectors can be loaded with the :attr:`param_attr` parameter.
            The local word vector needs to be transformed into numpy format, and the shape of local word
            vector should be consistent with :attr:`num_embeddings` . Then :ref:`api_initializer_NumpyArrayInitializer`
            is used to load custom or pre-trained word vectors. See code example for details.
        mp_group(Group): The tensor parallel group.
        name(str, optional): For detailed information, please refer
               to :ref:`api_guide_Name`. Usually name is no need to set and
               None by default.

    Examples:
        .. code-block:: python
        import paddle
        from paddle.distributed import fleet

        class SimpleMPNet(paddle.nn.Layer):
           def __init__(self, vocab_size, hidden_size, inner_size, output_size):
              super().__init__()
              self.linear1 = fleet.meta_parallel.ColumnParallelLinear(
                    hidden_size,
                    inner_size,
                    gather_output=False,
                    has_bias=True)

              self.linear2 = fleet.meta_parallel.RowParallelLinear(
                    inner_size,
                    hidden_size,
                    input_is_parallel=True,
                    has_bias=True)

              self.linear3 = paddle.nn.Linear(hidden_size, output_size)

              self.embedding = fleet.meta_parallel.VocabParallelEmbedding(
                                vocab_size,
                                hidden_size)

           def forward(self, x):
              x = self.embedding(x)
              x = self.linear1(x)
              x = self.linear2(x)
              x = self.linear3(x)
              return x
    """

    def __init__(
        self,
        num_embeddings,
        embedding_dim,
        weight_attr=None,
        mp_group=None,
        name=None,
    ):
        super().__init__()

        self.model_parallel_group = (
            tp._HYBRID_PARALLEL_GROUP.get_model_parallel_group()
            if mp_group is None
            else mp_group
        )
        self.world_size = (
            tp._HYBRID_PARALLEL_GROUP.get_model_parallel_world_size()
            if mp_group is None
            else mp_group.nranks
        )
        self.rank = (
            tp._HYBRID_PARALLEL_GROUP.get_model_parallel_rank()
            if mp_group is None
            else mp_group.rank
        )

        self.origin_num_embeddings = num_embeddings
        self.is_mp = self.world_size > 1

        assert (
            num_embeddings % self.world_size == 0
        ), "The length of the vocabulary must be divisible by the parallelism degree of MP"

        per_part_size = num_embeddings // self.world_size

        self.vocab_start_index = self.rank * per_part_size
        self._dtype = self._helper.get_default_dtype()
        self._size = [per_part_size, embedding_dim]
        self._weight_attr = weight_attr
        self._name = name

        if self.is_mp and paddle.in_dynamic_mode():
            with get_rng_state_tracker().rng_state():
                self.weight = self.create_parameter(
                    attr=self._weight_attr,
                    shape=self._size,
                    dtype=self._dtype,
                    is_bias=False,
                )
        else:
            self.weight = self.create_parameter(
                attr=self._weight_attr,
                shape=self._size,
                dtype=self._dtype,
                is_bias=False,
            )

        self.weight.is_distributed = True if self.is_mp else False
        if self.weight.is_distributed:
            self.weight.split_axis = 0

    def forward(self, x):
        if self.is_mp:
            output_parallel = mp_ops._c_lookup_table(
                self.weight,
                x,
                start_index=self.vocab_start_index,
                name=self._name,
            )
            output = mp_ops._mp_allreduce(
                output_parallel,
                group=self.model_parallel_group,
                use_calc_stream=True,
                use_model_parallel=True,
            )
        else:
            output = F.embedding(
                x,
                weight=self.weight,
                padding_idx=None,
                sparse=False,
                name=self._name,
            )
        return output


class ColumnParallelLinear(paddle.nn.Layer):
    """Linear layer with mp parallelized(column).
    this class is used for splitting Linear Layer in mp group, column split the weight of the Linear layer.

    Args:
        in_features(int): The number of input units.
        out_features(int): The number of output units.
        weight_attr(ParamAttr|None): The attribute for the learnable weight of this layer. The default value is None
            and the weight will be initialized to zero. For detailed information, please refer to paddle.ParamAttr.
        has_bias(bool): whether to add bias.
        gather_output(bool): whether to do allgahter for the output of each rank.
        fuse_matmul_bias(bool): whether to fuse matmul and bias.
        mp_group(Group): The tensor parallel group.
        name(str, optional): Normally there is no need for user to set this parameter.
            For detailed information, please refer to :ref:`api_guide_Name` .

    Examples:
        .. code-block:: python
        import paddle
        from paddle.distributed import fleet

        class SimpleMPNet(paddle.nn.Layer):
           def __init__(self, vocab_size, hidden_size, inner_size, output_size):
              super().__init__()
              self.linear1 = fleet.meta_parallel.ColumnParallelLinear(
                    hidden_size,
                    inner_size,
                    gather_output=False,
                    has_bias=True)

              self.linear2 = fleet.meta_parallel.RowParallelLinear(
                    inner_size,
                    hidden_size,
                    input_is_parallel=True,
                    has_bias=True)

              self.linear3 = paddle.nn.Linear(hidden_size, output_size)

              self.embedding = fleet.meta_parallel.VocabParallelEmbedding(
                                vocab_size,
                                hidden_size)

           def forward(self, x):
              x = self.embedding(x)
              x = self.linear1(x)
              x = self.linear2(x)
              x = self.linear3(x)
              return x
    """

    def __init__(
        self,
        in_features,
        out_features,
        weight_attr=None,
        has_bias=None,
        gather_output=True,
        fuse_matmul_bias=False,
        mp_group=None,
        name=None,
    ):
        super().__init__()

        self.model_parallel_group = (
            tp._HYBRID_PARALLEL_GROUP.get_model_parallel_group()
            if mp_group is None
            else mp_group
        )
        self.world_size = (
            tp._HYBRID_PARALLEL_GROUP.get_model_parallel_world_size()
            if mp_group is None
            else mp_group.nranks
        )
        self._name = name
        self.is_mp = self.world_size > 1

        self.gather_output = gather_output
        assert out_features % self.world_size == 0, (
            "Number of column of the weight for linear ({}) must be"
            " divisible by model parallel size ({})".format(
                out_features, self.world_size
            )
        )
        self.output_size_per_partition = out_features // self.world_size

        self._weight_attr = weight_attr
        self._dtype = self._helper.get_default_dtype()

        if self.is_mp and paddle.in_dynamic_mode():
            with get_rng_state_tracker().rng_state():
                self.weight = self.create_parameter(
                    shape=[in_features, self.output_size_per_partition],
                    attr=self._weight_attr,
                    dtype=self._dtype,
                    is_bias=False,
                )
        else:
            self.weight = self.create_parameter(
                shape=[in_features, self.output_size_per_partition],
                attr=self._weight_attr,
                dtype=self._dtype,
                is_bias=False,
            )

        self.weight.is_distributed = True if self.is_mp else False

        if self.weight.is_distributed:
            self.weight.split_axis = 1

        if has_bias:
            # initialize bias to zero like Megatron
            self.bias = self.create_parameter(
                shape=[self.output_size_per_partition],
                attr=paddle.nn.initializer.Constant(value=0.0),
                dtype=self._dtype,
                is_bias=True,
            )
            self.bias.is_distributed = True if self.is_mp else False
            if self.bias.is_distributed:
                self.bias.split_axis = 0
        else:
            self.bias = None

        self.linear = F.linear

        if fuse_matmul_bias:
            if not is_fused_matmul_bias_supported():
                raise NotImplementedError(
                    "You set fuse_matmul_bias=True in ColumnParallelLinear, "
                    "however, the paddle you are using not support this operation. "
                    "Please set fuse_matmul_bias=False or use paddle compiled "
                    "with cuda 11.6 or higher."
                )
            from paddle.incubate.nn.functional import fused_linear

            self.linear = fused_linear

    def forward(self, x):
        # use inner api to process identity
        if self.is_mp:
            input_parallel = mp_ops._c_identity(
                x, group=self.model_parallel_group
            )
        else:
            input_parallel = x

        output_parallel = self.linear(
            input_parallel, self.weight, self.bias, name=self._name
        )

        if self.gather_output and self.is_mp:
            output = mp_ops._c_concat(
                output_parallel, group=self.model_parallel_group
            )
        else:
            output = output_parallel
        return output


class MPScale(PyLayer):
    @staticmethod
    def forward(ctx, x, mp_degree):
        out = paddle.scale(x, 1.0 / mp_degree)
        return out

    @staticmethod
    def backward(ctx, dout):
        return dout


class RowParallelLinear(paddle.nn.Layer):
    """Linear layer with mp parallelized(row).
    this class is used for splitting Linear Layer in mp group, row split the weight of the Linear layer.

    Args:
        in_features(int): The number of input units.
        out_features(int): The number of output units.
        weight_attr(ParamAttr|None): The attribute for the learnable weight of this layer. The default value is None
            and the weight will be initialized to zero. For detailed information, please refer to paddle.ParamAttr.
        has_bias(bool): whether to add bias.
        input_is_parallel(bool): whether the input has alreadly been splitted across the mp group.
        fuse_matmul_bias(bool): whether to fuse matmul and bias.
        mp_group(Group): The tensor parallel group.
        name(str, optional): Normally there is no need for user to set this parameter.
            For detailed information, please refer to :ref:`api_guide_Name` .

    Examples:
        .. code-block:: python
        import paddle
        from paddle.distributed import fleet

        class SimpleMPNet(paddle.nn.Layer):
           def __init__(self, vocab_size, hidden_size, inner_size, output_size):
              super().__init__()
              self.linear1 = fleet.meta_parallel.ColumnParallelLinear(
                    hidden_size,
                    inner_size,
                    gather_output=False,
                    has_bias=True)

              self.linear2 = fleet.meta_parallel.RowParallelLinear(
                    inner_size,
                    hidden_size,
                    input_is_parallel=True,
                    has_bias=True)

              self.linear3 = paddle.nn.Linear(hidden_size, output_size)

              self.embedding = fleet.meta_parallel.VocabParallelEmbedding(
                                vocab_size,
                                hidden_size)

           def forward(self, x):
              x = self.embedding(x)
              x = self.linear1(x)
              x = self.linear2(x)
              x = self.linear3(x)
              return x
    """

    def __init__(
        self,
        in_features,
        out_features,
        weight_attr=None,
        has_bias=True,
        input_is_parallel=False,
        fuse_matmul_bias=False,
        mp_group=None,
        name=None,
    ):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.input_is_parallel = input_is_parallel
        self._weight_attr = weight_attr
        self._dtype = self._helper.get_default_dtype()
        self._name = name

        self.model_parallel_group = (
            tp._HYBRID_PARALLEL_GROUP.get_model_parallel_group()
            if mp_group is None
            else mp_group
        )
        self.world_size = (
            tp._HYBRID_PARALLEL_GROUP.get_model_parallel_world_size()
            if mp_group is None
            else mp_group.nranks
        )
        self.rank = (
            tp._HYBRID_PARALLEL_GROUP.get_model_parallel_rank()
            if mp_group is None
            else mp_group.rank
        )

        self.is_mp = self.world_size > 1
        assert in_features % self.world_size == 0, (
            "Number of row of the weight for linear ({}) must be"
            " divisible by model parallel size ({})".format(
                in_features, self.world_size
            )
        )

        self.input_size_per_partition = in_features // self.world_size

        if self.is_mp and paddle.in_dynamic_mode():
            with get_rng_state_tracker().rng_state():
                self.weight = self.create_parameter(
                    shape=[self.input_size_per_partition, self.out_features],
                    attr=self._weight_attr,
                    dtype=self._dtype,
                    is_bias=False,
                )
        else:
            self.weight = self.create_parameter(
                shape=[self.input_size_per_partition, self.out_features],
                attr=self._weight_attr,
                dtype=self._dtype,
                is_bias=False,
            )

        self.weight.is_distributed = True if self.is_mp else False
        if self.weight.is_distributed:
            self.weight.split_axis = 0

        if has_bias:
            self.bias = self.create_parameter(
                shape=[self.out_features],
                attr=paddle.nn.initializer.Constant(value=0.0),
                dtype=self._dtype,
                is_bias=True,
            )
        else:
            self.bias = None

        self.linear = F.linear

        if fuse_matmul_bias:
            if not is_fused_matmul_bias_supported():
                raise NotImplementedError(
                    "You set fuse_matmul_bias=True in RowParallelLinear, "
                    "however, the paddle you are using not support this operation. "
                    "Please set fuse_matmul_bias=False or use paddle compiled "
                    "with cuda 11.6 or higher."
                )
            from paddle.incubate.nn.functional import fused_linear

            self.linear = fused_linear
        self.fuse_matmul_bias = fuse_matmul_bias

    def forward(self, x):
        if self.input_is_parallel or (not self.is_mp):
            input_parallel = x
        else:
            # split last dim
            input_parallel = mp_ops._c_split(x, group=self.model_parallel_group)

        if self.is_mp:
            if self.fuse_matmul_bias:
                bias = MPScale.apply(self.bias, self.world_size)
                output_parallel = self.linear(
                    input_parallel, self.weight, bias, name=self._name
                )
                output = mp_ops._mp_allreduce(
                    output_parallel,
                    group=self.model_parallel_group,
                    use_calc_stream=True,
                    use_model_parallel=True,
                )
            else:
                output_parallel = self.linear(
                    input_parallel, self.weight, name=self._name
                )
                output_ = mp_ops._mp_allreduce(
                    output_parallel,
                    group=self.model_parallel_group,
                    use_calc_stream=True,
                    use_model_parallel=True,
                )
                output = (
                    output_ + self.bias if self.bias is not None else output_
                )
        else:
            output = self.linear(
                input_parallel, self.weight, self.bias, name=self._name
            )

        return output


class ParallelCrossEntropy(paddle.nn.Layer):
    """CrossEntropy with mp parallelized.
    this class is used for splitting softmax cross entropy in mp group.

    Args:
        mp_group(Group): The tensor parallel group.
        name(str, optional): Normally there is no need for user to set this parameter.
            For detailed information, please refer to :ref:`api_guide_Name` .
        ignore_index (long int, optional):  Specifies a target value that is ignored and
            does not contribute to the loss. A negative value means that no label value
            needs to be ignored. Default is -100 .

    Examples:
        .. code-block:: python
        loss_func = ParallelCrossEntropy()
        loss = loss_func(img, lable)
    """

    def __init__(self, mp_group=None, name=None, ignore_index=-100):
        super().__init__()
        self.name = name
        self.model_parallel_group = (
            tp._HYBRID_PARALLEL_GROUP.get_model_parallel_group()
            if mp_group is None
            else mp_group
        )
        self.world_size = (
            tp._HYBRID_PARALLEL_GROUP.get_model_parallel_world_size()
            if mp_group is None
            else mp_group.nranks
        )
        self.rank = (
            tp._HYBRID_PARALLEL_GROUP.get_model_parallel_rank()
            if mp_group is None
            else mp_group.rank
        )
        self.ignore_index = ignore_index

    def forward(self, input, label):
        loss = mp_ops._c_softmax_with_cross_entropy(
            input,
            label,
            group=self.model_parallel_group,
            ignore_index=self.ignore_index,
        )
        return loss
