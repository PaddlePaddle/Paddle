#   Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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
from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING

import paddle
import paddle.distributed as dist
from paddle.distributed.auto_parallel.ring_attention import (
    shard_seq_load_balance,
)

from .parallel_base import ParallelModel, ParallelOptimizer, is_tensor

if TYPE_CHECKING:
    from collections.abc import Callable

    from paddle import Tensor
    from paddle.distributed import ProcessMesh
    from paddle.nn import Layer


def c_split(x, process_mesh, need_transpose):
    mp_index = process_mesh.dim_names.index('mp')  # get the axis for the split
    dp_index = process_mesh.dim_names.index('dp')
    if isinstance(x, tuple):
        target_x = x[0]
    else:
        target_x = x
    assert is_tensor(target_x)
    assert len(target_x.shape) == 3
    if need_transpose:
        target_x = paddle.transpose(target_x, perm=[1, 0, 2])
    placements = target_x.placements
    if placements is None:
        placements = [dist.Replicate() for _ in range(len(process_mesh.shape))]
    if placements[dp_index] == dist.Shard(0):
        # NOTE(zhangwl):if shard(0) , input shape should be [b,s,h]
        split_dims = dist.Shard(1)
    elif placements[dp_index] == dist.Shard(1):
        # NOTE(zhangwl):if shard(1) , input shape should be [s,b,h]
        split_dims = dist.Shard(0)
    else:
        logging.warning(
            f"parallel api don't know {target_x.shape} which dimension is batch, default is to cut to the 0th dimension"
        )
        split_dims = dist.Shard(0)
    placements[mp_index] = split_dims
    target_x = dist.reshard(target_x, process_mesh, placements)
    if isinstance(x, tuple):
        x = list(x)
        x[0] = target_x
        x = tuple(x)
    else:
        x = target_x

    return x


def c_concat(x, process_mesh, need_transpose):
    index = process_mesh.dim_names.index('mp')  # get the axis for the split
    if isinstance(x, tuple):
        target_x = x[0]
    else:
        target_x = x
    assert is_tensor(target_x)
    assert len(target_x.shape) == 3
    placements = target_x.placements
    if placements is None:
        placements = [dist.Replicate() for _ in range(len(process_mesh.shape))]
    placements[index] = dist.Replicate()
    target_x = dist.reshard(target_x, process_mesh, placements)
    if need_transpose:
        target_x = paddle.transpose(target_x, perm=[1, 0, 2])
    if isinstance(x, tuple):
        x = list(x)
        x[0] = target_x
        x = tuple(x)
    else:
        x = target_x

    return x


class PlanBase:
    def __init__(self):
        self.share_param_list = {}

    def apply(self, layer, process_mesh, shard_param_list):
        raise NotImplementedError("Don't call the PlanBase directly.")


class ColWiseParallel(PlanBase):
    """
    Col wise parallel plan for mp config.
    Will try to split weight on the second dim and the bias on the first dim.
    This api is designed for paddle.nn.Linear or paddle.nn.Embedding.
    If any other instance of paddle.nn.Layer is passed,
    this plan will try to split `layer.weight` and `layer.bias` if it has.

    Note:
        1. `layer.weight` should have two dims.
        2. `layer.bias` should have one dim.

    Args:
        gather_output (bool): Whether gather the output to change it from a local tensor to a global tensor.
            If gather the local tensor to global, an extra communication will be called.
            The default value is `False`, which means keeping the output as a local tensor.

    Examples:
        .. code-block:: python

            >>> import paddle
            >>> import paddle.distributed as dist

            >>> class MLP(paddle.nn.Layer):
            ...     def __init__(self):
            ...         super().__init__()
            ...         self.fc1 = paddle.nn.Linear(8, 8)
            ...         self.fc2 = paddle.nn.Linear(8, 8)
            ...
            ...     def forward(self, input):
            ...         return self.fc2(self.fc1(input))

            >>> # doctest: +REQUIRES(env:DISTRIBUTED)
            >>> layer = MLP()
            >>> mp_config = {
            ...     'fc1': dist.ColWiseParallel()
            ... }

    """

    def __init__(self, gather_output: bool = False) -> None:
        super().__init__()
        self.gather_output = gather_output

    def gather_output_hook(self, process_mesh):
        def gather_hook(layer, input, output):
            assert output is not None
            return c_concat(output, process_mesh, False)

        return gather_hook

    def apply(self, layer, process_mesh, shard_param_list):
        index = process_mesh.dim_names.index('mp')  # get the axis for the split
        size = len(process_mesh.shape)
        placement = [dist.Replicate() for _ in range(size)]
        param_placements = {}
        assert isinstance(layer, paddle.nn.Layer)
        if not isinstance(layer, (paddle.nn.Linear, paddle.nn.Embedding)):
            logging.warning(
                f"ColWiseParallel is designed to handle Linear and Embedding. "
                f"But got {layer.__class__.__name__}. "
                f"Will try to shard weight and bias if the layer contains one."
            )
        shard_param_list = set(shard_param_list)
        if len(shard_param_list) == 0:
            shard_param_list.add("weight")
            shard_param_list.add("bias")

        def shard_param(param_name):
            if (
                hasattr(layer, param_name)
                and getattr(layer, param_name) is not None
            ):
                layer_param = getattr(layer, param_name)

                if layer_param.is_dist():
                    return

                if len(layer_param.shape) == 2:
                    placement[index] = dist.Shard(1)
                elif len(layer_param.shape) == 1:
                    placement[index] = dist.Shard(0)
                else:
                    raise ValueError(f"{layer_param} should have 1 or 2 dims.")
                # NOTE(zhangweilong):for share parameter, the parameter should be handled uniformly in the end
                if (
                    self.share_param_list is not None
                    and layer_param.name in self.share_param_list
                    and self.share_param_list[layer_param.name] > 1
                ):
                    param_placements.update({param_name: placement})
                else:
                    layer_param = dist.shard_tensor(
                        layer_param,
                        process_mesh,
                        placement,
                    )
                    setattr(layer, param_name, layer_param)

        for param_name in shard_param_list:
            shard_param(param_name)
        if self.gather_output:
            layer.register_forward_post_hook(
                self.gather_output_hook(process_mesh)
            )
        return param_placements


class RowWiseParallel(PlanBase):
    """
    Row wise parallel plan for mp config.
    Will try to split weight on the first dim.
    This api is designed for paddle.nn.Linear or paddle.nn.Embedding.
    If any other instance of paddle.nn.Layer is passed, this plan will try to split `layer.weight` if it has.

    Note:
        `layer.weight` should have two dims.

    Args:
        is_input_parallel (bool): Whether the input is a local tensor or a global tensor. If the input is a
            global tensor, an extra split will be called. The default value is `True`,
            which means the input is a local tensor.

    Examples:
        .. code-block:: python

            >>> import paddle
            >>> import paddle.distributed as dist

            >>> class MLP(paddle.nn.Layer):
            ...     def __init__(self):
            ...         super().__init__()
            ...         self.fc1 = paddle.nn.Linear(8, 8)
            ...         self.fc2 = paddle.nn.Linear(8, 8)
            ...
            ...     def forward(self, input):
            ...         return self.fc2(self.fc1(input))

            >>> # doctest: +REQUIRES(env:DISTRIBUTED)
            >>> layer = MLP()
            >>> mp_config = {
            ...     'fc1': dist.RowWiseParallel()
            ... }
    """

    def __init__(self, is_input_parallel: bool = True) -> None:
        super().__init__()
        self.is_input_parallel = is_input_parallel

    def split_input_hook(self, process_mesh):
        def split_hook(layer, input):
            return c_split(input, process_mesh, False)

        return split_hook

    def apply(self, layer, process_mesh, shard_param_list):
        index = process_mesh.dim_names.index('mp')  # get the axis for the split
        size = len(process_mesh.shape)
        placement = [dist.Replicate() for _ in range(size)]
        placement[index] = dist.Shard(0)
        param_placements = {}
        assert isinstance(layer, paddle.nn.Layer)
        if not isinstance(layer, (paddle.nn.Linear, paddle.nn.Embedding)):
            logging.warning(
                f"RowWiseParallel is designed to handle Linear and Embedding. "
                f"But got {layer.__class__.__name__}. "
                f"Will try to shard weight if the layer contains one."
            )
        shard_param_list = set(shard_param_list)
        shard_param_list.discard("bias")
        if len(shard_param_list) == 0:
            shard_param_list.add("weight")

        def shard_param(param_name):
            if (
                hasattr(layer, param_name)
                and getattr(layer, param_name) is not None
            ):
                layer_param = getattr(layer, param_name)
                if layer_param.is_dist():
                    return
                if len(layer_param.shape) != 2:
                    raise ValueError(f"{layer_param} should have 2 dims.")
                # NOTE(zhangweilong):for share parameter, the parameter should be handled uniformly in the end
                if (
                    self.share_param_list is not None
                    and layer_param.name in self.share_param_list
                    and self.share_param_list[layer_param.name] > 1
                ):
                    param_placements.update({param_name: placement})
                else:
                    layer_param = dist.shard_tensor(
                        layer_param,
                        process_mesh,
                        placement,
                    )
                    setattr(layer, param_name, layer_param)

        for param_name in shard_param_list:
            shard_param(param_name)
        if not self.is_input_parallel:
            layer.register_forward_pre_hook(self.split_input_hook(process_mesh))
        return param_placements


class PrepareLayerInput(PlanBase):
    """
    Prepare the input of specific layer. User should provide one callable function.

    Args:
        fn (callable): A function that prepare the layer input. The function should take exactly
            one parameter named `process_mesh` and return the pre hook.

    Examples:
        .. code-block:: python

            >>> import paddle
            >>> import paddle.distributed as dist

            >>> class MLP(paddle.nn.Layer):
            ...     def __init__(self):
            ...         super().__init__()
            ...         self.fc1 = paddle.nn.Linear(8, 8)
            ...         self.fc2 = paddle.nn.Linear(8, 8)
            ...
            ...     def forward(self, input):
            ...         return self.fc2(self.fc1(input))

            >>> def layer_input_hook(process_mesh):
            ...     def hook(layer, input, output):
            ...         return input
            ...     return hook

            >>> # doctest: +REQUIRES(env:DISTRIBUTED)
            >>> layer = MLP()
            >>> mp_config = {
            ...     'fc1': dist.PrepareLayerOutput(layer_input_hook)
            ... }
    """

    def __init__(
        self,
        fn: (
            Callable[
                [ProcessMesh],
                Callable[
                    [Layer, tuple[Tensor], tuple[Tensor]], [tuple[Tensor]]
                ],
            ]
            | None
        ) = None,
    ) -> None:
        super().__init__()
        assert callable(fn)
        self.fn = fn

    def apply(self, layer, process_mesh, shard_param_list):
        layer.register_forward_pre_hook(self.fn(process_mesh=process_mesh))


class PrepareLayerOutput(PlanBase):
    """
    Prepare the output of specific layer. User should provide one callable function.

    Args:
        fn (callable): A function that prepare the layer input. The function should take exactly
            one parameter named `process_mesh` and return the post hook.

    Examples:
        .. code-block:: python

            >>> import paddle
            >>> import paddle.distributed as dist

            >>> class MLP(paddle.nn.Layer):
            ...     def __init__(self):
            ...         super().__init__()
            ...         self.fc1 = paddle.nn.Linear(8, 8)
            ...         self.fc2 = paddle.nn.Linear(8, 8)
            ...
            ...     def forward(self, input):
            ...         return self.fc2(self.fc1(input))

            >>> def layer_output_hook(process_mesh):
            ...     def hook(layer, input, output):
            ...         return output
            ...     return hook

            >>> # doctest: +REQUIRES(env:DISTRIBUTED)
            >>> layer = MLP()
            >>> mp_config = {
            ...     'fc1': dist.PrepareLayerOutput(layer_output_hook)
            ... }
    """

    def __init__(
        self,
        fn: (
            Callable[
                [ProcessMesh],
                Callable[
                    [Layer, tuple[Tensor], tuple[Tensor]], [tuple[Tensor]]
                ],
            ]
            | None
        ) = None,
    ) -> None:
        super().__init__()
        assert callable(fn)
        self.fn = fn

    def apply(self, layer, process_mesh, shard_param_list):
        layer.register_forward_post_hook(self.fn(process_mesh=process_mesh))


class SequenceParallelBegin(PlanBase):
    """
    Sequence parallel plan for mp config.
    This plan marks the beginning of the sp and should be added to the LAST layer before the sp range.

    Note:
        DON'T mark any layer in the sp range.

    Args:
        need_transpose (bool): the default value is `True`. With `need_transpose=True`, this plan will transfer
            the output from [b, s, h] to [s/mp, b, h].  With `need_transpose=False`, this plan will transfer
            the output from [s, b, h] to [s/mp, b, h].

    Examples:
        .. code-block:: python

            >>> import paddle
            >>> import paddle.distributed as dist

            >>> class MLP(paddle.nn.Layer):
            ...     def __init__(self):
            ...         super().__init__()
            ...         self.fc1 = paddle.nn.Linear(8, 8)
            ...         self.fc2 = paddle.nn.Linear(8, 8)
            ...
            ...     def forward(self, input):
            ...         return self.fc2(self.fc1(input))

            >>> # doctest: +REQUIRES(env:DISTRIBUTED)
            >>> layer = MLP()
            >>> mp_config = {
            ...     'fc1': dist.SequenceParallelBegin()
            ... }
    """

    def __init__(self, need_transpose: bool = True) -> None:
        super().__init__()
        self.need_transpose = need_transpose

    def sequence_parallel_begin(self, process_mesh):
        def begin(layer, input, output):
            assert output is not None
            return c_split(output, process_mesh, self.need_transpose)

        return begin

    def apply(self, layer, process_mesh, shard_param_list):
        layer.register_forward_post_hook(
            self.sequence_parallel_begin(process_mesh)
        )


class SequenceParallelEnd(PlanBase):
    """
    Sequence parallel plan for mp config.
    This plan marks the ending of the sp and should be added to the FIRST layer after the sp range.

    Note:
        DON'T mark any layer in the sp range.

    Args:
        need_transpose (bool): the default value is `True`. With `need_transpose=True`, this plan will transfer
            the input from [s/mp, b, h] to [b, s, h]. With `need_transpose=False`, this plan will transfer the
            input from [s/mp, b, h] to [s, b, h].

    Examples:
        .. code-block:: python

            >>> import paddle
            >>> import paddle.distributed as dist

            >>> class MLP(paddle.nn.Layer):
            ...     def __init__(self):
            ...         super().__init__()
            ...         self.fc1 = paddle.nn.Linear(8, 8)
            ...         self.fc2 = paddle.nn.Linear(8, 8)
            ...
            ...     def forward(self, input):
            ...         return self.fc2(self.fc1(input))

            >>> # doctest: +REQUIRES(env:DISTRIBUTED)
            >>> layer = MLP()
            >>> mp_config = {
            ...     'fc1': dist.SequenceParallelEnd()
            ... }
    """

    def __init__(self, need_transpose: bool = True) -> None:
        super().__init__()
        self.need_transpose = need_transpose

    def sequence_parallel_end(self, process_mesh):
        def end(layer, input, output=None):
            assert input is not None
            return c_concat(input, process_mesh, self.need_transpose)

        return end

    def apply(self, layer, process_mesh, shard_param_list):
        layer.register_forward_pre_hook(
            self.sequence_parallel_end(process_mesh)
        )


class SequenceParallelEnable(PlanBase):
    """
    Sequence parallel plan for mp config.
    Do sequence parallel on the layer. Note the input should be in [b, s, h] format.

    Examples:
        .. code-block:: python

            >>> import paddle
            >>> import paddle.distributed as dist

            >>> class MLP(paddle.nn.Layer):
            ...     def __init__(self):
            ...         super().__init__()
            ...         self.fc1 = paddle.nn.Linear(8, 8)
            ...         self.fc2 = paddle.nn.Linear(8, 8)
            ...
            ...     def forward(self, input):
            ...         return self.fc2(self.fc1(input))

            >>> # doctest: +REQUIRES(env:DISTRIBUTED)
            >>> layer = MLP()
            >>> mp_config = {
            ...     'fc1': dist.SequenceParallelEnable()
            ... }
    """

    def __init__(self) -> None:
        super().__init__()

    def sequence_parallel_begin(self, process_mesh):
        def begin(layer, input, output=None):
            assert input is not None
            return c_split(input, process_mesh, True)

        return begin

    def sequence_parallel_end(self, process_mesh):
        def end(layer, input, output):
            assert output is not None
            return c_concat(output, process_mesh, True)

        return end

    def apply(self, layer, process_mesh, shard_param_list):
        logging.warning(
            "Sequence parallel with the usage of SequenceParallel may not reach the best throughput. "
            "Try to use SequenceParallelBegin/End to achieve better performance"
        )
        layer.register_forward_pre_hook(
            self.sequence_parallel_begin(process_mesh)
        )
        layer.register_forward_post_hook(
            self.sequence_parallel_end(process_mesh)
        )


class SequenceParallelDisable(PlanBase):
    """
    Sequence parallel plan for mp config.
    Disable sequence parallel on the layer.

    Args:
        need_transpose (bool): the default value is `True`. If the need_transpose is `True`: this plan will transfer
            the input from  [s/mp, b, h] to [b, s, h] and then transfer the output from [b, s, h] to [s/mp, b, h].
            If the need_transpose is `False`: this plan will transfer the input from  [s/mp, b, h] to [s, b, h] and
            then transfer the output from [s, b, h] to [s/mp, b, h].

    Examples:
        .. code-block:: python

            >>> import paddle
            >>> import paddle.distributed as dist

            >>> class MLP(paddle.nn.Layer):
            ...     def __init__(self):
            ...         super().__init__()
            ...         self.fc1 = paddle.nn.Linear(8, 8)
            ...         self.fc2 = paddle.nn.Linear(8, 8)
            ...
            ...     def forward(self, input):
            ...         return self.fc2(self.fc1(input))

            >>> # doctest: +REQUIRES(env:DISTRIBUTED)
            >>> layer = MLP()
            >>> mp_config = {
            ...     'fc1': dist.SequenceParallelDisable()
            ... }
    """

    def __init__(self, need_transpose: bool = True) -> None:
        super().__init__()
        self.need_transpose = need_transpose

    def sequence_parallel_begin(self, process_mesh):
        def begin(layer, input, output=None):
            return c_split(output, process_mesh, self.need_transpose)

        return begin

    def sequence_parallel_end(self, process_mesh):
        def end(layer, input, output=None):
            return c_concat(input, process_mesh, self.need_transpose)

        return end

    def apply(self, layer, process_mesh, shard_param_list):
        layer.register_forward_pre_hook(
            self.sequence_parallel_end(process_mesh)
        )

        layer.register_forward_post_hook(
            self.sequence_parallel_begin(process_mesh)
        )


class ConvParallel(PlanBase):
    """
    A strategy for enabling spatial parallelism on ``paddle.nn.Conv2D`` layers
    by sharding the input tensor along its Width (W) dimension.

    When this ``ConvParallel`` configuration is applied to a ``Conv2D`` layer,
    the layer's input tensor will have its width dimension split across devices
    in the model parallel group. This can help reduce memory usage from activations,
    especially when dealing with inputs that have a large width.

    To enable width-wise input sharding correctly, make sure your `Conv2D` layer
    satisfies the following conditions along the width dimension:

    - **Dilation** must be set to `1`.
    - **If no width padding is used:**
        - The input width must be evenly divisible by the stride width.
        - The stride width must be equal to the kernel width.
    - **If width padding is used:**
        - The stride width must be `1`.
        - The total input width must be at least half the kernel width.

    Examples:
        .. code-block:: python

            >>> import paddle
            >>> import paddle.nn as nn
            >>> import paddle.distributed as dist

            >>> class SimpleConvNet(nn.Layer):
            ...     def __init__(self, data_format="NCHW"):
            ...         super().__init__()
            ...         self.conv1 = nn.Conv2D(
            ...             3, 8, kernel_size=3, padding=1, data_format=data_format
            ...         )
            ...         self.relu = nn.ReLU()
            ...     def forward(self, x):
            ...         x = self.conv1(x)
            ...         return self.relu(x)
            ...
            >>> # doctest: +REQUIRES(env:DISTRIBUTED)
            >>> model = SimpleConvNet(data_format="NCHW")
            >>> mp_config = {
            ...    "parallelize_plan": {
            ...        "conv1": dist.ConvParallel()
            ...     }
            ... }
    """

    def __init__(self) -> None:
        super().__init__()

    @staticmethod
    def _is_supported(
        input_size,
        kernel_size,
        stride,
        padding,
        dilation,
        data_format,
        mp_group_size,
    ):
        idx_w_input = -1
        idx_w_kernel = -1

        if data_format == "NCHW":
            idx_w_input = 3
            idx_w_kernel = 3
        elif data_format == "NHWC":
            idx_w_input = 2
            idx_w_kernel = 3
        else:
            return False

        if input_size[idx_w_input] % mp_group_size != 0:
            return False

        dilation_w = dilation[1]
        padding_w = padding[1]
        stride_w = stride[1]

        input_w = input_size[idx_w_input]
        kernel_w = kernel_size[idx_w_kernel]

        if dilation_w != 1:
            # RingConv2d only supports dilation=1.
            # Larger dilation would require enlarged halo regions and more complex communication.
            return False

        if padding_w == 0:
            # To avoid halo exchange when padding=0, we require:
            # - input_w must be divisible by stride_w, so partitions align evenly across ranks.
            # - stride_w == kernel_w, so each kernel operates on disjoint local regions.
            if input_w % stride_w != 0:
                return False
            if stride_w != kernel_w:
                return False

        else:
            # When padding > 0, halo exchange is needed.
            # To simplify halo logic, we require:
            # - stride_w == 1: ensures each output element is computed from overlapping input,
            #   and no input region is skipped, simplifying halo construction.
            # - kernel_w // 2 <= input_w: prevents the kernel from exceeding local input.
            if stride_w != 1:
                return False
            if kernel_w // 2 > input_w:
                return False

        return True

    def conv_parallel_start(self, process_mesh, data_format):
        def start(layer, input, output=None):
            if data_format == "NCHW":
                shard_w_dim = 3
            elif data_format == "NHWC":
                shard_w_dim = 2
            else:
                raise ValueError(
                    f"Unsupported data_format: {data_format}. "
                    "Only NCHW and NHWC are supported."
                )

            if isinstance(input, tuple):
                x = input[0]
            else:
                x = input

            placements = x.placements
            mp_index = process_mesh.dim_names.index('mp')
            mp_group_size = process_mesh.get_dim_size('mp')

            # Note(luchang): for intermediate api, when this ConvLayer is
            # not supported, we just skip apply parallelization.
            if not ConvParallel._is_supported(
                x.shape,
                layer.weight.shape,
                layer._stride,
                layer._updated_padding,
                layer._dilation,
                data_format,
                mp_group_size,
            ):
                return input

            if placements is None:
                placements = [
                    dist.Replicate() for _ in range(len(process_mesh.shape))
                ]
            if placements[mp_index] == dist.Shard(shard_w_dim):
                return input

            placements[mp_index] = dist.Shard(shard_w_dim)

            if not x.is_dist():
                x = dist.shard_tensor(x, process_mesh, placements)
            else:
                x = dist.reshard(x, process_mesh, placements)

            if isinstance(input, tuple):
                input = list(input)
                input[0] = x
                input = tuple(input)
            else:
                input = x
            return input

        return start

    def apply(self, layer, process_mesh, shard_param_list):
        layer.register_forward_pre_hook(
            self.conv_parallel_start(process_mesh, layer._data_format)
        )


class TensorParallel(ParallelModel):
    def __init__(self, model, parallelize_plan=None):
        super().__init__(model)
        if parallelize_plan is not None:
            assert isinstance(parallelize_plan, dict)
            for key, plan in parallelize_plan.items():
                assert isinstance(
                    key, str
                ), "The key of the parallelize plan should be a string."
                if not isinstance(plan, list):
                    plan = [plan]
                for p in plan:
                    assert isinstance(
                        p, PlanBase
                    ), "The value the the parallelize plan should be a instance of PlanBase or a list of PlanBase."

            self.global_mesh = dist.auto_parallel.get_mesh()
            self.parallelize_plan = parallelize_plan
            self.tp_parallelizer = self.tensor_parallelizer_fn

    def match_layer(self, layer, name):
        # Match the layer to a plan.
        # Will return the plan if the layer hits one, otherwise return None.
        plans = []
        for key, plan in self.parallelize_plan.items():
            attr_name = key.split('.')[-1]
            shard_param_list = []
            # Find some plan for specific parameter, such as
            # "lm_head.weight": ColWiseParallel()
            # "qkv_proj.lora_A" ColWiseParallel()
            # if there is no plan for specific parameter, layer will be sharded by default: layer.weight and layer.bias
            if key.endswith(f".{attr_name}"):
                if hasattr(layer, attr_name) and is_tensor(
                    getattr(layer, attr_name)
                ):
                    key = key.replace(f".{attr_name}", "")
                    shard_param_list.append(attr_name)
            re_find = re.match(key, name)
            if key == name or (
                re_find is not None
                and int(re_find.end()) - int(re_find.start()) == len(name)
            ):
                if isinstance(plan, PlanBase):
                    plan = [plan]
                plans.append([plan, shard_param_list])
        return plans

    def tensor_parallelizer_fn(self, model):
        if self.parallelize_plan is None:
            return
        layer_param_placements = {}
        share_param_list = {}
        for name, layer in model.named_sublayers():
            for param_name in list(layer._parameters.keys()):
                param = getattr(layer, param_name)
                if param.name not in share_param_list:
                    share_param_list[param.name] = 1
                    continue
                share_param_list[param.name] += 1
        for name, layer in model.named_sublayers():
            plans = self.match_layer(layer, name)
            layer_param_placements[layer] = {}
            if len(plans) > 0:
                pp_idx = getattr(layer, "pipeline_stage_index", 0)
                for plan in plans:
                    real_plan, shard_param_list = plan
                    for p in real_plan:
                        p.share_param_list = share_param_list
                        param_placements = p.apply(
                            layer, self.get_mesh(pp_idx), shard_param_list
                        )
                        if param_placements is not None and param_placements:
                            layer_param_placements[layer].update(
                                param_placements
                            )
        return model, layer_param_placements


def tensor_parallel(model, optimizer=None, config=None):
    """
    Tensor parallel.
    Args:
        model (paddle.nn.Layer): the model to be shard into tensor parallel.
        optimizer (paddle.optimizer.Optimizer): the optimizer.
        config (dict): {
            "parallelize_plan": dict, the plan to shard the layer.
        }
    Returns:
        model: model after tp
        optimizer: optimizer after tp

    NOTE: the plan should be a dict maps layer name or parameter name to a split_plan,
    which will be used to split the layer or the parameter. The name can be written in regular format.

    An example for the plan is:
    ```
    plan = {
        "llama.embed_tokens": ColWiseParallel(),
        "llama.layers.*.self_attn.q_proj": ColWiseParallel(),
        "llama.layers.*.self_attn.k_proj": ColWiseParallel(),
        "llama.layers.*.self_attn.v_proj": ColWiseParallel(),
        "llama.layers.*.self_attn.o_proj": RowWiseParallel(),
        "llama.layers.*.mlp.gate_proj": ColWiseParallel(),
        "llama.layers.*.mlp.up_proj": ColWiseParallel(),
        "llama.layers.*.mlp.down_proj": RowWiseParallel(),
        "lm_head.weight": ColWiseParallel(),
    }
    ```
    """
    parallelize_plan = config.get("parallelize_plan")
    if parallelize_plan is None:
        # Do nothing if no plan.
        logging.warning(
            "No parallelize plan, tensor parallel won't do anything."
        )
        return model, optimizer

    global_mesh = dist.auto_parallel.get_mesh()

    assert (
        global_mesh is not None
    ), "global mesh must not be None, please call fleet.auto.set_mesh(global_mesh) firstly"
    assert (
        "mp" in global_mesh.dim_names
    ), "mp must in the mesh dim_names when use tensor_parallel"

    model = TensorParallel(model, parallelize_plan)
    if optimizer is not None:
        optimizer = ParallelOptimizer(optimizer)

    return model, optimizer


class ContextParallelPrefix(PlanBase):
    """

    Prepare Input for context parallel optimizations.

    This will work for Layer that calls like whole-llama Layer which is the first layer in the network.

    Users can set backend='p2p/all2all' for different context parallel strategys.

    backend='p2p' will use Ring FlashAttention strategy which segments input by balance in the sequence dimension before whole-llama Layer.
    backend='all2all' will use Deepspeed Ulysses strategy(Paddle SegmentParallel strategy) which segments input in the sequence dimension before whole-llama Layer.

    Note:


    Args:
        backend (string): select strategy for context parallel, now support 'p2p' and 'all2all'.
        period (string): select period for hook, now support 'begin' and 'end'.

    Examples:
        .. code-block:: python

        >>> import paddle
        >>> import paddle.distributed as dist

        >>> class SDPALayer(paddle.nn.Layer):
        ...     def __init__(self):
        ...         super().__init__()
        ...
        ...     def forward(self, q, k, v):
        ...         return paddle.nn.functional.scaled_dot_product_attention(q, k, v)
        >>>
        >>> class AttentionLayer(paddle.nn.Layer):
        ...     def __init__(self):
        ...         super().__init__()
        ...         self.hidden_size = 64
        ...         self.num_key_value_heads = 10
        ...         self.head_dim = 64
        ...         self.sdpa = SDPALayer()
        ...         self.q = paddle.nn.Linear(
        ...             self.hidden_size,
        ...             self.hidden_size,
        ...             bias_attr=False,
        ...         )
        ...         self.k = paddle.nn.Linear(
        ...             self.hidden_size,
        ...             self.num_key_value_heads * self.head_dim,
        ...             bias_attr=False,
        ...         )
        ...         self.v = paddle.nn.Linear(
        ...             self.hidden_size,
        ...             self.num_key_value_heads * self.head_dim,
        ...             bias_attr=False,
        ...         )
        ...
        ...     def forward(self, input):
        ...         q = self.q(input)
        ...         k = self.k(input)
        ...         v = self.v(input)
        ...         return self.sdpa(q, k, v)
        >>>
        >>> class LlamaLayer(paddle.nn.Layer):
        ...     def __init__(self):
        ...         super().__init__()
        ...         self.attention = AttentionLayer()
        ...
        ...     def forward(self, input):
        ...         return self.attention(input)
        >>>
        >>> class LlamaForCausalLayer(paddle.nn.Layer):
        ...     def __init__(self):
        ...         super().__init__()
        ...         self.llama = LlamaLayer()
        ...         self.weight = self.create_parameter(shape=[64, 1024])
        ...         self.loss_func = paddle.nn.CrossEntropyLoss()
        ...
        ...     def forward(self, input, label):
        ...         out = self.llama(input)
        ...         logits = paddle.matmul(out, self.weight)
        ...         loss = self.loss_func(logits, label)
        ...         return logits
        >>>
        >>> # doctest: +REQUIRES(env:DISTRIBUTED)
        >>> layer = LlamaForCausalLayer()
        >>> mp_config = {
        ...     'llama': dist.ContextParallelPrefix('p2p', 'begin'),
        ...     'sdpa': dist.ContextParallel('p2p'),
        ...     'loss_func': dist.ContextParallelPrefix('p2p', 'end'),
        ... }
    """

    def __init__(self, backend: str = 'p2p', period: str = 'begin') -> None:
        super().__init__()
        self.backend = backend
        self.period = period
        assert self.backend in [
            'p2p',
            'all2all',
        ], f"backend must be 'p2p' or 'all2all', but got {self.backend}"
        assert self.period in [
            'begin',
            'end',
        ], f"period must be 'begin' or 'end', but got {self.period}"

    def all2all_split_input_pre_hook(self, process_mesh):
        def shard_tensor(input_tensor, seq_dim):
            cp_index = process_mesh.dim_names.index('sep')
            placements = input_tensor.placements
            if placements is None:
                placements = [
                    dist.Replicate() for _ in range(len(process_mesh.shape))
                ]
            # split sequence dim
            placements[cp_index] = dist.Shard(seq_dim)
            reshard_input = dist.reshard(input_tensor, process_mesh, placements)
            return reshard_input

        def all2all_split_input(layer, args):
            cp_index = process_mesh.dim_names.index('sep')
            cp_degree = process_mesh.shape[cp_index]
            # check input_ids
            if isinstance(args, (list, tuple)):
                all_args = []
                for input_tensor in args:
                    print(
                        f'input:{type(input_tensor)}, is_dist: {input_tensor.is_dist()}'
                    )
                    assert (
                        input_tensor.is_dist()
                    ), "Input tensor must be a distributed tensor."
                    assert (
                        len(input_tensor.shape) == 2
                    ), f"input_ids should be [batch_size, seq_len], but got {input_tensor.shape}"
                    _, seq_len = input_tensor.shape
                    assert (
                        seq_len % cp_degree == 0
                    ), f"sequence length {seq_len} must be divisible by cp degree {cp_degree}"
                    reshard_input = shard_tensor(input_tensor, 1)
                    all_args.append(reshard_input)
                new_args = tuple(all_args)
                return new_args
            elif isinstance(args, paddle.Tensor):
                reshard_input = shard_tensor(args, 1)
                return reshard_input
            else:
                raise ValueError(
                    f"Unsupported argument type: {type(args)}. Expected list of tensors or single tensor."
                )

        return all2all_split_input

    def p2p_split_input_pre_hook(self, process_mesh):
        def p2p_split_input(layer, args):
            cp_index = process_mesh.dim_names.index('sep')
            cp_degree = process_mesh.shape[cp_index]
            if isinstance(args, (list, tuple)):
                all_args = []
                for input_tensor in args:
                    # check input_ids
                    assert (
                        input_tensor.is_dist()
                    ), "Input tensor must be a distributed tensor."
                    assert (
                        len(input_tensor.shape) == 2
                    ), f"input_ids should be [batch_size, seq_len], but got {input_tensor.shape}"
                    placements = input_tensor.placements
                    if placements is None:
                        placements = [
                            dist.Replicate()
                            for _ in range(len(process_mesh.shape))
                        ]
                    assert (
                        placements[cp_index] == dist.Replicate()
                    ), "Input tensor must be a replicated tensor in cp mesh."
                    reshard_input = shard_seq_load_balance(input_tensor, 1)
                    all_args.append(reshard_input)
                new_args = tuple(all_args)
                return new_args
            elif isinstance(args, paddle.Tensor):
                reshard_input = shard_seq_load_balance(input_tensor, 1)
                return reshard_input
            else:
                raise ValueError(
                    f"Unsupported argument type: {type(args)}. Expected list of tensors or single tensor."
                )

        return p2p_split_input

    def reshard_post_hook(self, process_mesh):
        def reshard_hook(layer, input, output):
            cp_index = process_mesh.dim_names.index('sep')
            cp_degree = process_mesh.shape[cp_index]
            placements = output.placements
            assert (
                output.is_dist()
            ), f"output {output} must be a distributed tensor."
            placements[cp_index] = dist.Replicate()
            target_output = dist.reshard(output, process_mesh, placements)
            return target_output

        return reshard_hook

    def apply(self, layer, process_mesh, shard_param_list):
        if self.backend == 'all2all':
            # Deepspeed Ulysses
            layer.register_forward_pre_hook(
                self.all2all_split_input_pre_hook(process_mesh)
            )
        elif self.backend == 'p2p':
            # Ring FlashAttention
            layer.register_forward_pre_hook(
                self.p2p_split_input_pre_hook(process_mesh)
            )
        else:
            logging.warning(
                f'{self.backend} is not supported backend for context parallel'
            )
        if self.period == 'end':
            layer.register_forward_post_hook(
                self.reshard_post_hook(process_mesh)
            )


class ContextParallel(PlanBase):
    """

    Applies context parallel optimizations to the attention layer.

    This will work for Layer that calls paddle.nn.functional.scaled_dot_product_attention).

    Users can set backend='p2p/all2all' for different context parallel strategys.

    backend='p2p' will use Ring FlashAttention strategy which segments q/k/v in the sequence dimension and communicates k/v between ranks.
    backend='all2all' will use Deepspeed Ulysses strategy(Paddle SegmentParallel strategy) which inserts all2all before and after sdpa compute.

    Note:


    Args:
        backend (string): select strategy for context parallel, now support 'p2p' and 'all2all'.

    Examples:
        .. code-block:: python

        >>> import paddle
        >>> import paddle.distributed as dist

        >>> class SDPALayer(paddle.nn.Layer):
        ...     def __init__(self):
        ...         super().__init__()
        ...
        ...     def forward(self, q, k, v):
        ...         return paddle.nn.functional.scaled_dot_product_attention(q, k, v)
        >>>
        >>> class AttentionLayer(paddle.nn.Layer):
        ...     def __init__(self):
        ...         super().__init__()
        ...         self.hidden_size = 64
        ...         self.num_key_value_heads = 10
        ...         self.head_dim = 64
        ...         self.sdpa = SDPALayer()
        ...         self.q = paddle.nn.Linear(
        ...             self.hidden_size,
        ...             self.hidden_size,
        ...             bias_attr=False,
        ...         )
        ...         self.k = paddle.nn.Linear(
        ...             self.hidden_size,
        ...             self.num_key_value_heads * self.head_dim,
        ...             bias_attr=False,
        ...         )
        ...         self.v = paddle.nn.Linear(
        ...             self.hidden_size,
        ...             self.num_key_value_heads * self.head_dim,
        ...             bias_attr=False,
        ...         )
        ...
        ...     def forward(self, input):
        ...         q = self.q(input)
        ...         k = self.k(input)
        ...         v = self.v(input)
        ...         return self.sdpa(q, k, v)
        >>>
        >>> class LlamaLayer(paddle.nn.Layer):
        ...     def __init__(self):
        ...         super().__init__()
        ...         self.attention = AttentionLayer()
        ...
        ...     def forward(self, input):
        ...         return self.attention(input)
        >>>
        >>> class LlamaForCausalLayer(paddle.nn.Layer):
        ...     def __init__(self):
        ...         super().__init__()
        ...         self.llama = LlamaLayer()
        ...         self.weight = self.create_parameter(shape=[64, 1024])
        ...         self.loss_func = paddle.nn.CrossEntropyLoss()
        ...
        ...     def forward(self, input, label):
        ...         out = self.llama(input)
        ...         logits = paddle.matmul(out, self.weight)
        ...         loss = self.loss_func(logits, label)
        ...         return logits
        >>>
        >>> # doctest: +REQUIRES(env:DISTRIBUTED)
        >>> layer = LlamaForCausalLayer()
        >>> mp_config = {
        ...     'llama': dist.ContextParallelPrefix('p2p', 'begin'),
        ...     'sdpa': dist.ContextParallel('p2p'),
        ...     'loss_func': dist.ContextParallelPrefix('p2p', 'end'),
        ... }
    """

    def __init__(self, backend: str = 'p2p') -> None:
        super().__init__()
        self.backend = backend

    def all2all_reshard_pre_hook(self, process_mesh):
        def all2all_reshard_hook(layer, args):
            cp_index = process_mesh.dim_names.index('sep')
            cp_degree = process_mesh.shape[cp_index]
            all_args = []
            for arg in args:
                # check q k v
                assert arg.is_dist(), f"arg {arg} must be a distributed tensor."
                assert len(arg.shape) == 3 or len(arg.shape) == 4
                placements = arg.placements
                assert placements[cp_index] == dist.Shard(
                    1
                ), f"arg {arg} must be sharded in sequence dimension."
                # reshard [batch_size，seq_len/sep，num_head，head_dim] -> [batch_size，seq_len，num_head/sep，head_dim]
                placements[cp_index] = dist.Shard(2)
                target_arg = dist.reshard(arg, process_mesh, placements)
                all_args.append(target_arg)
            new_args = tuple(all_args)
            return new_args

        return all2all_reshard_hook

    def all2all_reshard_post_hook(self, process_mesh):
        def all2all_reshard_hook(layer, input, output):
            cp_index = process_mesh.dim_names.index('sep')
            cp_degree = process_mesh.shape[cp_index]
            placements = output.placements
            assert (
                output.is_dist()
            ), f"output {output} must be a distributed tensor."
            assert len(output.shape) == 4 or len(output.shape) == 3
            assert placements[cp_index] == dist.Shard(
                2
            ), f"output {output} must be Shard(2) in sequence dimension."
            # reshard [batch_size，seq_len，num_head/seq，head_dim]  ->  [batch_size，seq_len/sep，num_head，head_dim]
            placements[cp_index] = dist.Shard(1)
            target_output = dist.reshard(output, process_mesh, placements)
            return target_output

        return all2all_reshard_hook

    def p2p_reshard_pre_hook(self, process_mesh):
        def input_hook(layer, args, kwargs):
            cp_index = process_mesh.dim_names.index('sep')
            cp_degree = process_mesh.shape[cp_index]
            for arg in args:
                # check q k v
                assert (
                    arg.is_dist()
                ), "Input tensor must be a distributed tensor."
                assert len(arg.shape) == 3 or len(arg.shape) == 4
                placements = arg.placements
                assert placements[cp_index] == dist.Shard(
                    1
                ), f"arg {arg} must be Shard(1) in sequence dimension."
            # edit kwarg backend to 'p2p'
            new_kwargs = kwargs
            new_kwargs['backend'] = 'p2p'
            return args, new_kwargs

        return input_hook

    def apply(self, layer, process_mesh, shard_param_list):
        if self.backend == 'all2all':
            # Deepspeed Ulysses
            layer.register_forward_pre_hook(
                self.all2all_reshard_pre_hook(process_mesh)
            )
            layer.register_forward_post_hook(
                self.all2all_reshard_post_hook(process_mesh)
            )
        elif self.backend == 'p2p':
            # Ring FlashAttention
            layer.register_forward_pre_hook(
                self.p2p_reshard_pre_hook(process_mesh), with_kwargs=True
            )
        else:
            logging.warning(
                f'{self.backend} is not supported backend for context parallel'
            )
