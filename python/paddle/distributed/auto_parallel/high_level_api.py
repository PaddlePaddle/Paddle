# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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
import math
import warnings

import numpy as np

import paddle
import paddle.distributed as dist
from paddle.base.framework import in_dygraph_mode

logger = logging.getLogger(__name__)


class ToDistributedConfig:
    def __init__(self):
        self.input_spec = None
        self.sequence_parallel = False


def cost_model(matched_programs, device_num, node_num):
    # TODO(jeff41404): multi-node will be supported later
    assert (
        node_num == 1
    ), "we only support single node now, multi-node will be supported later"

    # TODO(jeff41404): will evaluate the best combination of parallel strategies
    # based on cost_model and return global_mesh, currently using pre-defined parallel strategy
    if device_num % 2 == 0:
        if device_num == 8:
            return dist.ProcessMesh(
                np.arange(device_num).reshape(2, 2, 2).tolist(),
                dim_names=["pp", "dp", "mp"],
            )
        elif device_num == 6:
            return dist.ProcessMesh(
                np.arange(device_num).reshape(3, 2).tolist(),
                dim_names=["dp", "mp"],
            )
        elif device_num == 4:
            return dist.ProcessMesh(
                np.arange(device_num).reshape(2, 2).tolist(),
                dim_names=["dp", "mp"],
            )
        elif device_num == 2:
            return dist.ProcessMesh(list(range(device_num)), dim_names=["dp"])
        else:
            raise ValueError(
                f"device_num must be an even number to be able to use at least 2 parallel strategies, but got: {device_num}"
            )
    else:
        logger.debug(
            f'device_num must be an even number to be able to use at least 2 parallel strategies, but got: {device_num}, only use data parallel.'
        )
        return dist.ProcessMesh(list(range(device_num)), dim_names=["dp"])


def record_program_ops_pre_hook(layer, inputs):
    """
    A pre-hook to mark op numbers before enter layer.forward.
    """
    if not in_dygraph_mode():
        # Because ir_guard._switch_to_pir() will change default_main_program in python/paddle/__init__.py.
        # In order to avoid errors, we import default_main_program until this hook running.
        # After fully switching to pir, can move this import to the beginning of the file.
        from paddle.base import default_main_program

        if layer._op_recorder.start < 0:
            layer._op_recorder.start = len(
                default_main_program().global_block().ops
            )
            layer._op_recorder.is_valid = True
        else:
            layer._op_recorder.is_valid = False
            warnings.warn(
                f"{layer._full_name} has recorded the op information before. Please check whether you call this layer twice."
            )


def transpose_reshard_embedding_layer_output(layer, inputs, outputs):
    if hasattr(layer, "current_mesh"):
        current_mesh = layer.__getattr__("current_mesh")
        new_output = paddle.transpose(outputs, [1, 0, 2])
        new_output = dist.reshard(
            new_output, current_mesh, [dist.Shard(1), dist.Shard(0)]
        )
        return new_output


def reshard_transpose_attention_layer_input(layer, inputs):
    new_inputs = list(inputs)
    x = new_inputs[0]
    if hasattr(layer, "current_mesh"):
        current_mesh = layer.__getattr__("current_mesh")
        new_x = dist.reshard(x, current_mesh, [dist.Shard(1), dist.Replicate()])
        new_x = paddle.transpose(new_x, [1, 0, 2])
        new_inputs[0] = new_x
        return tuple(new_inputs)


def transpose_reshard_attention_layer_output(layer, inputs, outputs):
    attn_out = outputs
    if hasattr(layer, "current_mesh"):
        current_mesh = layer.__getattr__("current_mesh")
        new_attn_out = paddle.transpose(attn_out, [1, 0, 2])
        new_attn_out = dist.reshard(
            new_attn_out, current_mesh, [dist.Shard(1), dist.Shard(0)]
        )
        return new_attn_out


def reshard_mlp_layer_input(layer, inputs):
    new_inputs = list(inputs)
    mlp_input = new_inputs[0]
    if hasattr(layer, "current_mesh"):
        current_mesh = layer.__getattr__("current_mesh")
        new_mlp_input = dist.reshard(
            mlp_input, current_mesh, [dist.Shard(1), dist.Replicate()]
        )
        new_inputs[0] = new_mlp_input
        return tuple(new_inputs)


def reshard_mlp_layer_output(layer, inputs, outputs):
    mlp_out = outputs
    if hasattr(layer, "current_mesh"):
        current_mesh = layer.__getattr__("current_mesh")
        new_mlp_out = dist.reshard(
            mlp_out, current_mesh, [dist.Shard(1), dist.Shard(0)]
        )
        return new_mlp_out


def reshard_transpose_rms_norm_layer_output(layer, inputs, outputs):
    if hasattr(layer, "current_mesh"):
        current_mesh = layer.__getattr__("current_mesh")
        new_output = dist.reshard(
            outputs, current_mesh, [dist.Shard(1), dist.Replicate()]
        )
        new_output = paddle.transpose(new_output, [1, 0, 2])
        return new_output


def reshard_all_inputs(layer, inputs):
    if hasattr(layer, "current_mesh"):
        current_mesh = layer.__getattr__("current_mesh")
        if type(inputs) is tuple:
            new_inputs = []
            for input in inputs:
                if paddle.is_tensor(input):
                    if input.is_dist():
                        new_input = dist.reshard(
                            input,
                            current_mesh,
                            input.placements,
                        )
                    else:
                        new_input = dist.shard_tensor(
                            input,
                            current_mesh,
                            [dist.Shard(0), dist.Replicate()],
                        )
                    new_inputs.append(new_input)
                else:
                    new_inputs.append(input)
            return tuple(new_inputs)
        else:
            if input.is_dist():
                new_input = dist.reshard(
                    input, current_mesh, [dist.Shard(0), dist.Replicate()]
                )
            else:
                new_input = dist.shard_tensor(
                    input, current_mesh, [dist.Shard(0), dist.Replicate()]
                )
            return new_input


def reshard_all_outputs(layer, inputs, outputs):
    if hasattr(layer, "next_mesh"):
        next_mesh = layer.__getattr__("next_mesh")
        if type(outputs) is tuple:
            new_outputs = []
            for output in outputs:
                if paddle.is_tensor(output):
                    new_output = dist.reshard(
                        output, next_mesh, [dist.Shard(0), dist.Replicate()]
                    )
                    new_outputs.append(new_output)
                else:
                    new_outputs.append(output)
            return new_outputs
        else:
            new_output = dist.reshard(
                outputs, next_mesh, [dist.Shard(0), dist.Replicate()]
            )
            return new_output


def record_program_ops_post_hook(layer, inputs, outputs):
    """
    A post-hook to mark op numbers after enter layer.forward, and record corresponding ops of the layer.
    """
    if not in_dygraph_mode():
        # Because ir_guard._switch_to_pir() will change default_main_program in python/paddle/__init__.py.
        # In order to avoid errors, we import default_main_program until this hook running.
        # After fully switching to pir, can move this import to the beginning of the file.
        from paddle.base import default_main_program

        assert (
            layer._op_recorder.start >= 0
            and layer._op_recorder.is_valid is True
        ), f"{layer._full_name} has not recorded the start of the corresponding ops before"
        end = len(default_main_program().global_block().ops)
        # some layers, such as rotary_embedding, will not add new ops to program
        # assert end > layer._op_recorder.start, f"{layer._full_name} has not added new ops to the program"
        ops = []
        if end > layer._op_recorder.start:
            layer._op_recorder.end = end
            ops = (
                default_main_program()
                .global_block()
                .ops[layer._op_recorder.start : layer._op_recorder.end]
            )
        logger.debug(
            f'start: {layer._op_recorder.start}, end: {layer._op_recorder.end}, ops: {ops}'
        )
        layer._op_recorder.ops = ops


def get_layer_pp_info(mesh, num_hidden_layers, layer_index):
    if "pp" in mesh.dim_names:
        pp_degree = mesh.get_dim_size("pp")
        layer_per_stage = math.ceil(num_hidden_layers / pp_degree)
        return layer_index // layer_per_stage
    else:
        # return None, False
        return None


def to_distributed(
    model: paddle.nn.Layer,
    optimizer: paddle.optimizer.Optimizer,
    dataloader: paddle.io.DataLoader,
    device_num: int,
    node_num: int | None = 1,
    config: ToDistributedConfig | None = None,
) -> tuple[
    paddle.nn.Layer,
    paddle.optimizer.Optimizer,
    paddle.distributed.auto_parallel.ShardDataloader,
]:
    """
    `to_distributed` can automatically convert neural networks, optimizer, and dataloader
    that do not contain any distributed code into neural networks, optimizers, and dataloader
    that are suitable for distributed training and ensure their correctness.
    At the same time, during the transformation process, the optimal distributed strategy
    will be automatically selected based on `node_num` and `device_num` to maximize performance.

    Args:
        model(paddle.nn.Layer): The model in dygraph mode, whose parameters
            are ordinary tensors, do not contain any distributed code.
            If one device has sufficient memory, it can train directly.
        optimizer(paddle.optimizer.Optimizer): The optimizer for training.
            one instance of a regular optimizer, e.g. `paddle.optimizer.Adam` etc.
        dataloader(paddle.io.DataLoader): The dataloader used in dygraph mode,
            It is instantiated through regular `paddle.io.Dataset` and `paddle.io.Sampler`,
            not `paddle.io.DistributedBatchSampler`.
        device_num(int): the number of devices on each node or machine.
        node_num(int|None, optional): the number of nodes or machines.
        config(ToDistributedConfig| None = None): Configs for input_spec and sequence_parallel.
            The custom input specs specify the most likely shape, dtype, and name information
            of each model inputs. If it is not None, the input specs and
            will be inferred from the custom input specs. If it is None, will use default with
            shape of [BATCH_SIZE=4, SEQ_LENGTH=1024], The custom
            input specs should be a list of `paddle.static.InputSpec`. Default: None.
            sequence_parallel indicates whether to use sequence parallel. Default: False.

    Returns:
        model. The model in dygraph mode but contain distributed attributes.

        optimizer. The optimizer for training and may be sharded states.

        dataloader. The dataloader can be used in distributed training.

    Examples:
        .. code-block:: python

            >>> # doctest: +SKIP('run in distributed env')
            >>> import math
            >>> import numpy as np
            >>> import paddle
            >>> import paddle.nn.functional as F
            >>> from paddle import nn
            >>> from paddle.distributed import to_distributed
            >>> from paddle.distributed.auto_parallel.high_level_api import ToDistributedConfig

            >>> EPOCHS = 1
            >>> VOCAB_SIZE = 8000
            >>> BATCH_NUM = 2
            >>> BATCH_SIZE = 4
            >>> HIDDEN_SIZE = 2048
            >>> INTERMEDIATE_SIZE = 4096
            >>> SEQ_LENGTH = 1024
            >>> N_HEAD = 32
            >>> NUM_HIDDEN_LAYERS = 4
            >>> class RandomDataset(paddle.io.Dataset): # type: ignore[type-arg]
            ...     def __init__(self, inputs, labels, num_samples):
            ...         self.inputs = inputs
            ...         self.labels = labels
            ...         self.num_samples = num_samples
            ...     def __getitem__(self, idx):
            ...         return self.inputs[idx], self.labels[idx]
            ...     def __len__(self):
            ...         return self.num_samples

            >>> class RotaryEmbedding(nn.Layer):
            ...     def __init__(self, dim, max_position_embeddings=2048, base=10000):
            ...         super().__init__()
            ...         self.dim = dim
            ...         self.max_position_embeddings = max_position_embeddings
            ...         self.base = base
            ...         self.inv_freq = 1.0 / (
            ...             self.base ** (
            ...                 paddle.cast(paddle.arange(0, self.dim, 2), dtype="float32")
            ...                 / self.dim
            ...             )
            ...         )
            ...         self._set_cos_sin_cache(seq_len=max_position_embeddings)

            ...     def _set_cos_sin_cache(self, seq_len):
            ...         self.max_seq_len_cached = seq_len
            ...         t = paddle.arange(seq_len, dtype="float32")
            ...         freqs = paddle.einsum("i,j->ij", t, self.inv_freq)
            ...         emb = paddle.concat([freqs, freqs], axis=-1)
            ...         self.cos_cached = emb.cos()[None, :, None, :]
            ...         self.sin_cached = emb.sin()[None, :, None, :]

            ...     def forward(self, x, seq_len=None):
            ...         cos = self.cos_cached[:, :seq_len, :, :]
            ...         sin = self.sin_cached[:, :seq_len, :, :]
            ...         return (
            ...             cos.cast(x.dtype) if cos.dtype != x.dtype else cos,
            ...             sin.cast(x.dtype) if sin.dtype != x.dtype else sin,
            ...         )

            >>> def rotate_half(x):
            ...     x1 = x[..., : x.shape[-1] // 2]
            ...     x2 = x[..., x.shape[-1] // 2 :]
            ...     return paddle.concat([-x2, x1], axis=-1)

            >>> def apply_rotary_pos_emb(q, k, cos, sin, position_ids):
            ...     if position_ids is None:
            ...         cos = cos[:, : q.shape[1], :, :]
            ...         sin = sin[:, : q.shape[1], :, :]
            ...     else:
            ...         cos = cos.squeeze(axis=[0, 2])
            ...         sin = sin.squeeze(axis=[0, 2])
            ...         cos = cos[position_ids].unsqueeze(2)
            ...         sin = sin[position_ids].unsqueeze(2)
            ...     q_embed = (q * cos) + (rotate_half(q) * sin)
            ...     k_embed = (k * cos) + (rotate_half(k) * sin)
            ...     return q_embed, k_embed

            >>> def scaled_dot_product_attention(
            ...     query_states,
            ...     key_states,
            ...     value_states,
            ...     attention_mask,
            ... ):
            ...     bsz, q_len, num_heads, head_dim = query_states.shape
            ...     _, kv_seq_len, _, _ = value_states.shape
            ...     query_states = paddle.transpose(query_states, [0, 2, 1, 3])
            ...     key_states = paddle.transpose(key_states, [0, 2, 1, 3])
            ...     value_states = paddle.transpose(value_states, [0, 2, 1, 3])
            ...     attn_weights = paddle.matmul(
            ...         query_states / math.sqrt(head_dim), key_states.transpose([0, 1, 3, 2])
            ...     )
            ...     attention_mask = attention_mask.reshape([bsz, 1, q_len, kv_seq_len])
            ...     attn_weights = attn_weights + attention_mask
            ...     if not paddle.in_dynamic_mode():
            ...         attn_weights = F.softmax(attn_weights, axis=-1, dtype="float32").astype(
            ...             query_states.dtype
            ...         )
            ...     else:
            ...         with paddle.amp.auto_cast(False):
            ...             attn_weights = F.softmax(
            ...                 attn_weights, axis=-1, dtype="float32"
            ...             ).astype(query_states.dtype)
            ...     attn_output = paddle.matmul(attn_weights, value_states)
            ...     attn_output = attn_output.transpose([0, 2, 1, 3])
            ...     attn_output = attn_output.reshape([bsz, q_len, head_dim * num_heads])
            ...     return attn_output

            >>> class Attention(nn.Layer):
            ...     def __init__(self, hidden_size=HIDDEN_SIZE, n_head=N_HEAD):
            ...         super().__init__()
            ...         self.hidden_size = hidden_size
            ...         self.num_heads = n_head
            ...         self.head_dim = hidden_size // n_head
            ...         self.q_proj = nn.Linear(
            ...             hidden_size, hidden_size, bias_attr=False
            ...         )
            ...         self.k_proj = nn.Linear(
            ...             hidden_size, hidden_size, bias_attr=False
            ...         )
            ...         self.v_proj = nn.Linear(
            ...             hidden_size, hidden_size, bias_attr=False
            ...         )
            ...         self.o_proj = nn.Linear(
            ...             hidden_size, hidden_size, bias_attr=False
            ...         )
            ...         self.rotary_emb = RotaryEmbedding(
            ...             self.head_dim, max_position_embeddings=SEQ_LENGTH, base=10000
            ...         )

            ...     def forward(
            ...         self,
            ...         hidden_states,
            ...         position_ids=None,
            ...         attention_mask=None,
            ...     ):
            ...         query_states = self.q_proj(hidden_states)
            ...         key_states = self.k_proj(hidden_states)
            ...         value_states = self.v_proj(hidden_states)
            ...         target_query_shape = [0, 0, self.num_heads, self.head_dim]
            ...         target_key_value_shape = [0, 0, self.num_heads, self.head_dim]
            ...         query_states = query_states.reshape(shape=target_query_shape)
            ...         key_states = key_states.reshape(shape=target_key_value_shape)
            ...         value_states = value_states.reshape(shape=target_key_value_shape)
            ...         kv_seq_len = key_states.shape[-3]
            ...         cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
            ...         query_states, key_states = apply_rotary_pos_emb(
            ...             query_states, key_states, cos, sin, position_ids
            ...         )
            ...         output = scaled_dot_product_attention(
            ...             query_states,
            ...             key_states,
            ...             value_states,
            ...             attention_mask,
            ...         )
            ...         attn_output = output
            ...         attn_output = self.o_proj(attn_output)
            ...         return attn_output

            >>> class Mlp(nn.Layer):
            ...     def __init__(
            ...         self,
            ...         hidden_size=HIDDEN_SIZE,
            ...         intermediate_size=INTERMEDIATE_SIZE,
            ...     ):
            ...         super().__init__()
            ...         self.hidden_size = hidden_size
            ...         self.intermediate_size = intermediate_size
            ...         self.gate_proj = nn.Linear(
            ...             hidden_size, intermediate_size, bias_attr=False
            ...         )
            ...         self.up_proj = nn.Linear(
            ...             hidden_size, intermediate_size, bias_attr=False
            ...         )
            ...         self.down_proj = nn.Linear(
            ...             intermediate_size, hidden_size, bias_attr=False
            ...         )

            ...     def forward(self, x):
            ...         x = paddle.incubate.nn.functional.swiglu(
            ...             self.gate_proj(x), self.up_proj(x)
            ...         )
            ...         out = self.down_proj(x)
            ...         return out

            >>> class RMSNorm(nn.Layer):
            ...     def __init__(self, hidden_size=HIDDEN_SIZE):
            ...         super().__init__()
            ...         self.hidden_size = hidden_size
            ...         self.weight = paddle.create_parameter(
            ...             shape=[self.hidden_size],
            ...             dtype=paddle.get_default_dtype(),
            ...             default_initializer=nn.initializer.Constant(1.0),
            ...         )
            ...         self.variance_epsilon = 1.0

            ...     def forward(self, hidden_states):
            ...         with paddle.amp.auto_cast(False):
            ...             hidden_states = hidden_states.astype("float32")
            ...             variance = hidden_states.pow(2).mean(-1, keepdim=True)
            ...             hidden_states = (
            ...                 paddle.rsqrt(variance + self.variance_epsilon) * hidden_states
            ...             )
            ...         if self.weight.dtype in [paddle.float16, paddle.bfloat16]:
            ...             hidden_states = paddle.cast(hidden_states, self.weight.dtype)
            ...         return hidden_states * self.weight

            >>> class DecoderLayer(nn.Layer):
            ...     def __init__(
            ...         self,
            ...         hidden_size=HIDDEN_SIZE,
            ...         intermediate_size=INTERMEDIATE_SIZE,
            ...     ):
            ...         super().__init__()
            ...         self.hidden_size = hidden_size
            ...         self.intermediate_size = intermediate_size
            ...         self.self_attn = Attention(hidden_size)
            ...         self.mlp = Mlp()
            ...         self.input_layernorm = RMSNorm(hidden_size)
            ...         self.post_attn_layernorm = RMSNorm(hidden_size)

            ...     def forward(
            ...         self,
            ...         hidden_states,
            ...         position_ids=None,
            ...         attention_mask=None,
            ...     ):
            ...         residual = hidden_states
            ...         hidden_states = self.input_layernorm(hidden_states)
            ...         hidden_states = self.self_attn(
            ...             hidden_states, position_ids, attention_mask
            ...         )
            ...         hidden_states = residual + hidden_states
            ...         residual = hidden_states
            ...         hidden_states = self.post_attn_layernorm(hidden_states)
            ...         hidden_states = self.mlp(hidden_states)
            ...         hidden_states = residual + hidden_states
            ...         return hidden_states

            >>> def _prepare_decoder_attention_mask(
            ...     attention_mask, input_shape, dtype
            ... ):
            ...     batch_size, src_length = attention_mask.shape[0], attention_mask.shape[-1]
            ...     batch_size, target_length = input_shape
            ...     attention_mask = attention_mask[:, None, None, :].astype("bool")
            ...     attention_mask.stop_gradient = True
            ...     expanded_attn_mask = attention_mask.expand([batch_size, 1, target_length, src_length])
            ...     mask = paddle.tril(paddle.ones((target_length, target_length), dtype="bool"))
            ...     combined_attention_mask = mask[None, None, :, :].expand(
            ...         [batch_size, 1, target_length, target_length]
            ...     )
            ...     expanded_attn_mask = (expanded_attn_mask & combined_attention_mask)
            ...     expanded_attn_mask = paddle.where(
            ...         expanded_attn_mask, 0.0, paddle.finfo(dtype).min
            ...     ).astype(dtype)
            ...     return expanded_attn_mask

            >>> class Model(nn.Layer):
            ...     def __init__(
            ...         self,
            ...         vocab_size=VOCAB_SIZE,
            ...         hidden_size=HIDDEN_SIZE,
            ...         intermediate_size=INTERMEDIATE_SIZE,
            ...     ):
            ...         super().__init__()
            ...         self.vocab_size = vocab_size
            ...         self.hidden_size = hidden_size
            ...         self.intermediate_size = intermediate_size
            ...         self.embed_tokens = nn.Embedding(
            ...             vocab_size,
            ...             hidden_size,
            ...         )
            ...         self.layers = nn.LayerList(
            ...             [
            ...                 DecoderLayer()
            ...                 for i in range(NUM_HIDDEN_LAYERS)
            ...             ]
            ...         )
            ...         self.norm = RMSNorm(hidden_size)
            ...         self.weight = self.create_parameter(
            ...             shape=[hidden_size, vocab_size],
            ...             dtype=paddle.get_default_dtype(),
            ...         )
            ...         self.ignore_index = -100
            ...         self.loss_func = paddle.nn.CrossEntropyLoss(
            ...             reduction="none", ignore_index=self.ignore_index
            ...         )

            ...     def forward(
            ...         self,
            ...         input_ids=None,
            ...         position_ids=None,
            ...         attention_mask=None,
            ...         labels=None,
            ...     ):
            ...         batch_size, seq_length = input_ids.shape
            ...         inputs_embeds = self.embed_tokens(input_ids)
            ...         attention_mask = paddle.ones(
            ...             (batch_size, seq_length), dtype=paddle.bool
            ...         )
            ...         if position_ids is None:
            ...             position_ids = paddle.arange(seq_length, dtype="int64").expand(
            ...                 (batch_size, seq_length)
            ...             )
            ...         attention_mask = _prepare_decoder_attention_mask(
            ...             attention_mask,
            ...             (batch_size, seq_length),
            ...             inputs_embeds.dtype,
            ...         )
            ...         hidden_states = inputs_embeds
            ...         for idx, (decoder_layer) in enumerate(self.layers):
            ...             layer_outputs = decoder_layer(
            ...                 hidden_states,
            ...                 position_ids,
            ...                 attention_mask,
            ...             )
            ...             hidden_states = layer_outputs
            ...         hidden_states = self.norm(hidden_states)
            ...         logits = paddle.matmul(hidden_states, self.weight)
            ...         loss = None
            ...         if labels is not None:
            ...             masked_lm_loss = self.loss_func(
            ...                 logits.astype("float32"),
            ...                 labels.unsqueeze(2),
            ...             )
            ...             binary_sequence = paddle.where(
            ...                 masked_lm_loss > 0,
            ...                 paddle.ones_like(masked_lm_loss),
            ...                 paddle.zeros_like(masked_lm_loss),
            ...             )
            ...             count = paddle.sum(binary_sequence)
            ...             if count == 0:
            ...                 loss = paddle.sum(masked_lm_loss * binary_sequence)
            ...             else:
            ...                 loss = paddle.sum(masked_lm_loss * binary_sequence) / count
            ...         return (loss, logits)

            >>> model = Model() # There is no distributed code or markup in Model
            >>> input_seqs = np.random.randint(
            ...     low=0, high=1024, size=(BATCH_SIZE * BATCH_NUM, SEQ_LENGTH)
            ... ).astype("int64")
            >>> labels = np.random.randint(
            ...     low=0, high=1024, size=(BATCH_SIZE * BATCH_NUM, SEQ_LENGTH)
            ... ).astype("int64")
            >>> dataset = RandomDataset(
            ...     input_seqs, labels, BATCH_SIZE * BATCH_NUM
            ... )
            >>> sampler = paddle.io.BatchSampler(
            ...     dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=True
            ... )
            >>> loader = paddle.io.DataLoader(
            ...     dataset, batch_sampler=sampler
            ... )
            >>> opt = paddle.optimizer.SGD(
            ...     learning_rate=0.1, parameters=model.parameters()
            ... )
            >>> input_seq_spec = paddle.static.InputSpec(
            ...     [BATCH_SIZE, SEQ_LENGTH], 'float32', 'input_seq', True
            ... )
            >>> dist_config = ToDistributedConfig()
            >>> dist_config.sequence_parallel = True

            >>> # wrap model, opt, dataloader by using **to_distributed**
            >>> dist_model, dist_opt, dist_loader = to_distributed(
            ...     model,
            ...     opt,
            ...     loader,
            ...     device_num=8,
            ...     node_num=1,
            ...     config=dist_config,
            ... )

            >>> for epoch in range(EPOCHS):
            ...     dist_model.train()
            ...     for i, data in enumerate(dist_loader()):
            ...         inputs, labels = data
            ...         loss, _ = dist_model(inputs, labels=labels)
            ...         print(f"epoch {epoch}, step {i}: loss {loss}")
            ...         loss.backward()
            ...         dist_opt.step()
            ...         dist_opt.clear_grad()
            >>> # This case need to be executed in multi-card environment
            >>> # python -m paddle.distributed.launch --gpus=0,1,2,3,4,5,6,7 {test_case}.py
    """
    # Because some API(`paddle.randn` etc.) will be used when building pattern,
    # In order to avoid circle import, we import get_pattern until function running.
    from .static.tuner.to_distributed_api_patterns import (
        clear_used_patterns,
        get_pattern,
        match_all_patterns,
        register_used_patterns,
    )

    logger.debug(f'input model: {model}')
    # paddle.distributed.init_parallel_env()

    # step 1: identifying network structure and pattern recogincation
    # step 1.1: register pre-hooks and post-hooks, thus recording corresponding static ops in following paddle.jit.to_static
    for layer in model.sublayers():
        pre_hook_helper = layer.register_forward_pre_hook(
            record_program_ops_pre_hook
        )
        post_hook_helper = layer.register_forward_post_hook(
            record_program_ops_post_hook
        )
        layer._op_recorder.hooks.append(pre_hook_helper)
        layer._op_recorder.hooks.append(post_hook_helper)

    # step 1.2: call @to_static, get program, and corresponding static ops of each layer
    custom_input_spec = (
        config.input_spec
        if config.input_spec
        else [paddle.static.InputSpec([4, 1024], 'float32', 'input_seq', True)]
    )
    static_func = paddle.jit.to_static(
        model.forward, input_spec=custom_input_spec, full_graph=True
    )
    program = static_func.concrete_program.main_program
    # currently, paddle.jit.to_static has side effects that will affect model.
    # After fixing it, one line of code below can be dropped
    static_func.rollback()
    logger.debug(
        f'Converted model to pir program: {program}, for pattern matching'
    )

    # step 1.3: get the mapping [dynamic-layers : static ops]
    op_to_id = {}
    for idx, op in enumerate(program.global_block().ops):
        op_to_id[op] = idx

    ops_id_to_layer = {}
    op_id_to_layer = {}
    for layer in model.sublayers():
        layer_ops = layer._op_recorder.ops
        logger.debug(
            f'layer name: {layer.__class__.__name__}, layer_ops: {layer_ops}'
        )
        ops_id = []
        for op in layer_ops:
            assert op in op_to_id.keys(), f"{op.name} is not in program"
            op_id = op_to_id[op]
            op_id_to_layer[op_id] = layer
            ops_id.append(op_id)
        ops_id_to_layer[tuple(ops_id)] = layer
    logger.debug(f'ops_id_to_layer is: {ops_id_to_layer}')

    # step 1.4: pattern recogincation
    DECODER_LAYER_NAME = 'decoder_layer'
    register_used_patterns(DECODER_LAYER_NAME)
    results = match_all_patterns(program)
    logger.debug(f'Matched decoder layer patterns are: {results}')

    matched_programs = {}
    for pattern_name, matched_patterns in results.items():
        # process one pattern
        pattern_ops_dist_infos = get_pattern(pattern_name).ops_dist_infos
        assert (
            pattern_ops_dist_infos is not None
        ), f"{pattern_name} does not contain ops_dist_infos, cannot reshard, please check"
        processed_patterns = []
        for matched_pattern in matched_patterns:
            # convert pattern_ops_dist_infos to program_ops_dist_infos
            program_ops_dist_infos = {}
            for pattern_ops_id, op_dist_info in pattern_ops_dist_infos.items():
                program_ops_id = []
                for pattern_op_id in pattern_ops_id:
                    assert (
                        pattern_op_id in matched_pattern.keys()
                    ), f"please check ops_dist_infos of {pattern_name}, {pattern_op_id} not in matched_pattern: {matched_pattern.keys()}"
                    program_op_id = matched_pattern[pattern_op_id]
                    program_ops_id.append(program_op_id)
                program_ops_dist_infos[tuple(program_ops_id)] = op_dist_info
            processed_patterns.append(program_ops_dist_infos)
        matched_programs[pattern_name] = processed_patterns
    logger.debug(f'Matched decoder layer patterns are: {matched_programs}')

    # step 2: calculate the optimal parallel strategies based on the network structure
    mesh = cost_model(matched_programs, device_num, node_num)
    logger.debug(f'mesh: {mesh}')

    with_pp = True if "pp" in mesh.dim_names else False
    with_mp = True if "mp" in mesh.dim_names else False
    with_dp = True if "dp" in mesh.dim_names else False
    with_sp = (
        True if "mp" in mesh.dim_names and config.sequence_parallel else False
    )

    # step 3: processing tensor parallel if necessary, according to the optimal parallel strategies shard weight tensors in decoder blocks
    if with_mp:
        num_hidden_layers = len(matched_programs[DECODER_LAYER_NAME])
        for pattern_name, processed_patterns in matched_programs.items():
            assert (
                len(processed_patterns) == num_hidden_layers
            ), "transformer patterns matched are incomplete"
            for idx, processed_pattern in enumerate(processed_patterns):
                local_mesh = mesh
                if with_pp:
                    pp_stage_id = get_layer_pp_info(
                        mesh, num_hidden_layers, idx
                    )
                    local_mesh = mesh.get_mesh_with_dim("pp", pp_stage_id)

                for program_ops_id, dist_infos in processed_pattern.items():
                    assert (
                        program_ops_id in ops_id_to_layer.keys()
                    ), f"program_ops: {program_ops_id} is not corresponding to a dynamic layer"
                    dynamic_layer = ops_id_to_layer[program_ops_id]
                    mesh_num_dims = len(local_mesh.shape)
                    sharding_info = dist_infos.get_dist_info(mesh_num_dims)
                    dynamic_layer.weight = dist.shard_tensor(
                        dynamic_layer.weight, local_mesh, sharding_info[0]
                    )
                    if dynamic_layer.bias is not None:
                        dynamic_layer.bias = dist.shard_tensor(
                            dynamic_layer.bias, local_mesh, sharding_info[1]
                        )
    logger.debug(f'after tensor parallel, model: {model}')

    # step 4: processing pipeline parallel if necessary, reshard inputs of decoder blocks to next pp mesh b when switching from pp stage a to pp stage b
    if with_pp:
        decoder_layers = []
        for pattern_name, matched_all_patterns in results.items():
            if pattern_name == DECODER_LAYER_NAME:
                for matched_pattern in matched_all_patterns:
                    program_ops_id = []
                    for a, b in matched_pattern.items():
                        program_ops_id.append(b)
                    if tuple(sorted(program_ops_id)) in ops_id_to_layer.keys():
                        decoder_layers.append(
                            ops_id_to_layer[tuple(sorted(program_ops_id))]
                        )

        if decoder_layers is not None:
            num_decoder_blocks = len(decoder_layers)
            assert (
                num_decoder_blocks == num_hidden_layers
            ), f"decoder pattern layers matched are incomplete, num_decoder_blocks: {num_decoder_blocks} should be equal to num_hidden_layers: {num_hidden_layers}"

            pp_degree = mesh.get_dim_size("pp")
            num_blocks_per_stage = num_decoder_blocks // pp_degree
            for i in range(num_decoder_blocks):
                pp_stage_id = get_layer_pp_info(mesh, num_decoder_blocks, i)
                current_mesh = mesh.get_mesh_with_dim("pp", pp_stage_id)
                decoder_layer = decoder_layers[i]
                decoder_layer.__setattr__("current_mesh", current_mesh)
                pre_hook_helper = decoder_layer.register_forward_pre_hook(
                    reshard_all_inputs
                )
    logger.debug(f'after pipeline parallel, model: {model}')

    # step 5: processing sequence parallel if necessary, reshard or transpose sequence dims for inputs of attention/mlp inputs
    if with_sp:
        clear_used_patterns()
        EMBEDDING_LAYER_NAME = "embedding"
        ATTENTION_LAYER_NAME = "attention"
        MLP_LAYER_NAME = "mlp_3_with_swiglu"
        RMS_NORM_LAYER_NAME = "rmsnorm"
        used_patterns = [
            EMBEDDING_LAYER_NAME,
            ATTENTION_LAYER_NAME,
            MLP_LAYER_NAME,
            RMS_NORM_LAYER_NAME,
        ]
        register_used_patterns(used_patterns)
        results = match_all_patterns(program)

        matched_layers = {}
        for pattern_name, matched_all_patterns in results.items():
            if pattern_name in used_patterns:
                for matched_pattern in matched_all_patterns:
                    program_ops_id = []
                    for a, b in matched_pattern.items():
                        program_ops_id.append(b)
                    if tuple(sorted(program_ops_id)) in ops_id_to_layer.keys():
                        if pattern_name in matched_layers.keys():
                            matched_layers[pattern_name].append(
                                ops_id_to_layer[tuple(sorted(program_ops_id))]
                            )
                        else:
                            matched_layers[pattern_name] = [
                                ops_id_to_layer[tuple(sorted(program_ops_id))]
                            ]

        logger.debug(f'Matched attention/mlp layers are: {matched_layers}')
        # init mesh
        GLOBAL_MESH = []
        if with_pp:
            pp_degree = mesh.get_dim_size("pp")
            for i in range(pp_degree):
                local_mesh = mesh.get_mesh_with_dim("pp", i)
                GLOBAL_MESH.append(local_mesh)
        else:
            GLOBAL_MESH.append(mesh)

        # embedding: from [b/dp_degree, s, h] reshard+transpose to [s/mp_degree, b/dp_degree, h]
        embedding_layer = matched_layers[EMBEDDING_LAYER_NAME][0]
        embedding_layer_mesh = GLOBAL_MESH[0]
        embedding_layer.__setattr__("current_mesh", embedding_layer_mesh)
        post_hook_helper = embedding_layer.register_forward_post_hook(
            transpose_reshard_embedding_layer_output
        )

        # attention: input from [s/mp_degree, b/dp_degree, h] to [b/dp_degree, s, h], output from [b/dp_degree, s, h] to [s/mp_degree, b/dp_degree, h]
        attention_layers = matched_layers[ATTENTION_LAYER_NAME]
        num_attention_layers = len(attention_layers)
        if attention_layers is not None:
            for i in range(num_attention_layers):
                current_mesh = GLOBAL_MESH[0]
                if with_pp:
                    pp_stage_id = get_layer_pp_info(
                        mesh, num_attention_layers, i
                    )
                    current_mesh = GLOBAL_MESH[pp_stage_id]
                attention_layer = attention_layers[i]
                attention_layer.__setattr__("current_mesh", current_mesh)
                pre_hook_helper = attention_layer.register_forward_pre_hook(
                    reshard_transpose_attention_layer_input
                )
                post_hook_helper = attention_layer.register_forward_post_hook(
                    transpose_reshard_attention_layer_output
                )

        # mlp: input from [s/mp_degree, b/dp_degree, h] to [s, b/dp_degree, h], output from [s, b/dp_degree, h] to [s/mp_degree, b/dp_degree, h]
        mlp_layers = matched_layers[MLP_LAYER_NAME]
        num_mlp_layers = len(mlp_layers)
        if mlp_layers is not None:
            for i in range(num_mlp_layers):
                current_mesh = GLOBAL_MESH[0]
                if with_pp:
                    pp_stage_id = get_layer_pp_info(
                        mesh, num_attention_layers, i
                    )
                    current_mesh = GLOBAL_MESH[pp_stage_id]
                mlp_layer = mlp_layers[i]
                mlp_layer.__setattr__("current_mesh", current_mesh)
                pre_hook_helper = mlp_layer.register_forward_pre_hook(
                    reshard_mlp_layer_input
                )
                post_hook_helper = mlp_layer.register_forward_post_hook(
                    reshard_mlp_layer_output
                )

        # rms norm: for the last rms norm (after decoder blocks), input from [s/mp_degree, b/dp_degree, h] to [b, s, h]
        rms_norm_layers = matched_layers[RMS_NORM_LAYER_NAME]
        if rms_norm_layers is not None:
            last_rms_norm_layer = rms_norm_layers[-1]
            current_mesh = GLOBAL_MESH[-1]
            last_rms_norm_layer.__setattr__("current_mesh", current_mesh)
            post_hook_helper = last_rms_norm_layer.register_forward_post_hook(
                reshard_transpose_rms_norm_layer_output
            )

    # step 6: processing data parallel if necessary, shard dataloader
    # TODO(jeff41404): shard optimizer
    if with_dp:
        if with_pp:
            first_stage_mesh = mesh.get_mesh_with_dim("pp", 0)
            last_stage_mesh = mesh.get_mesh_with_dim("pp", 1)
            loader = dist.shard_dataloader(
                dataloader,
                meshes=[first_stage_mesh, last_stage_mesh],
                shard_dims="dp",
            )
        else:
            loader = dist.shard_dataloader(
                dataloader, meshes=[mesh], shard_dims="dp"
            )
    else:
        loader = dist.shard_dataloader(
            dataloader, meshes=[mesh], shard_dims=None
        )

    # step 7: clean layer_op recorder hooks
    for layer in model.sublayers():
        for hook_helper in layer._op_recorder.hooks:
            hook_helper.remove()

    return model, optimizer, loader
