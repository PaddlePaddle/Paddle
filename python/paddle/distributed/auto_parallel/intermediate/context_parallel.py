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

import paddle
import paddle.distributed as dist
from paddle.distributed.auto_parallel.ring_attention import (
    shard_seq_load_balance,
)

from .tensor_parallel import PlanBase


class PrepareContextParallel(PlanBase):
    """

    Prepare Input for context parallel optimizations.

    This will work for Layer that calls like whole-llama Layer which is the first layer in the network.

    Users can set backend='p2p/all2all' for different context parallel strategys.

    backend='p2p' will use Ring FlashAttention strategy which segments input with balance in the sequence dimension before whole-llama Layer.
    backend='all2all' will use Deepspeed Ulysses strategy(Paddle SegmentParallel strategy) which segments input in the sequence dimension before whole-llama Layer.

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
        ...     def forward(self, input, label):
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
        ...         out = self.llama(input, label)
        ...         logits = paddle.matmul(out, self.weight)
        ...         loss = self.loss_func(logits, label)
        ...         return logits
        >>>
        >>> # doctest: +REQUIRES(env:DISTRIBUTED)
        >>> layer = LlamaForCausalLayer()
        >>> mp_config = {
        ...     'llama': dist.PrepareContextParallel('p2p'),
        ...     'sdpa': dist.ContextParallel('p2p'),
        ... }
    """

    def __init__(self, backend: str = 'p2p') -> None:
        super().__init__()
        self.backend = backend
        assert self.backend in [
            'p2p',
            'all2all',
        ], f"backend must be 'p2p' or 'all2all', but got {self.backend}"

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
        ...     def forward(self, input, label):
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
        ...         out = self.llama(input, label)
        ...         logits = paddle.matmul(out, self.weight)
        ...         loss = self.loss_func(logits, label)
        ...         return logits
        >>>
        >>> # doctest: +REQUIRES(env:DISTRIBUTED)
        >>> layer = LlamaForCausalLayer()
        >>> mp_config = {
        ...     'llama': dist.PrepareContextParallel('p2p'),
        ...     'sdpa': dist.ContextParallel('p2p'),
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
