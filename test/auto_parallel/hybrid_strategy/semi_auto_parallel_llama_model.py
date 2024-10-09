# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

import math

import paddle
import paddle.distributed as dist
import paddle.nn.functional as F
from paddle import nn
from paddle.distributed.fleet.utils import recompute

try:
    from paddle.nn.functional.flash_attention import flash_attention
except:
    flash_attention = None


def is_pp_enable():
    global_mesh = dist.auto_parallel.get_mesh()
    return "pp" in global_mesh.dim_names


def get_mesh(pp_idx=None):
    global_mesh = dist.auto_parallel.get_mesh()
    assert global_mesh is not None, "global_mesh is not initialized!"
    if pp_idx is None:
        return global_mesh
    if is_pp_enable():
        mesh = global_mesh.get_mesh_with_dim("pp")[pp_idx]
        return mesh
    else:
        return global_mesh


def global_mesh_starts_with_pp():
    global_mesh = dist.auto_parallel.get_mesh()
    if is_pp_enable():
        return global_mesh.get_mesh_with_dim("pp")
    else:
        return global_mesh


class LlamaRotaryEmbedding(nn.Layer):
    def __init__(self, dim, max_position_embeddings=2048, base=10000):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        # [dim / 2]
        self.inv_freq = 1.0 / (
            self.base
            ** (
                paddle.cast(paddle.arange(0, self.dim, 2), dtype="float32")
                / self.dim
            )
        )
        self._set_cos_sin_cache(seq_len=max_position_embeddings)

    def _set_cos_sin_cache(self, seq_len):
        self.max_seq_len_cached = seq_len
        # [seq_len]
        t = paddle.arange(seq_len, dtype="float32")
        # [seq_len, dim/2]
        freqs = paddle.einsum("i,j->ij", t, self.inv_freq)
        # [seq_len, dim]
        emb = paddle.concat([freqs, freqs], axis=-1)
        # [1, seqlen, 1, dim]
        self.cos_cached = emb.cos()[None, :, None, :]
        self.sin_cached = emb.sin()[None, :, None, :]

    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        cos = self.cos_cached[:, :, :seq_len, ...]
        sin = self.sin_cached[:, :, :seq_len, ...]
        return (
            cos.cast(x.dtype) if cos.dtype != x.dtype else cos,
            sin.cast(x.dtype) if sin.dtype != x.dtype else sin,
        )


class LlamaAttentionAuto(nn.Layer):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config, layerwise_recompute=False, ipp=None):
        super().__init__()

        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads

        self.head_dim = self.hidden_size // config.num_attention_heads

        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = (
            config.num_attention_heads // config.num_key_value_heads
        )

        self.max_position_embeddings = config.max_position_embeddings
        self.seq_length = config.seq_length

        self.kv_indices = None
        self.ipp = ipp

        self.q_proj = nn.Linear(
            self.hidden_size,
            self.hidden_size,
            bias_attr=False,
        )
        self.q_proj.weight = dist.shard_tensor(
            self.q_proj.weight,
            get_mesh(self.ipp),
            [dist.Replicate(), dist.Shard(1)],
        )

        self.k_proj = nn.Linear(
            self.hidden_size,
            self.config.num_key_value_heads * self.head_dim,
            bias_attr=False,
        )
        self.k_proj.weight = dist.shard_tensor(
            self.k_proj.weight,
            get_mesh(self.ipp),
            [dist.Replicate(), dist.Shard(1)],
        )

        self.v_proj = nn.Linear(
            self.hidden_size,
            self.config.num_key_value_heads * self.head_dim,
            bias_attr=False,
        )
        self.v_proj.weight = dist.shard_tensor(
            self.v_proj.weight,
            get_mesh(self.ipp),
            [dist.Replicate(), dist.Shard(1)],
        )

        self.o_proj = nn.Linear(
            self.hidden_size,
            self.hidden_size,
            bias_attr=False,
        )
        self.o_proj.weight = dist.shard_tensor(
            self.o_proj.weight,
            get_mesh(self.ipp),
            [dist.Replicate(), dist.Shard(0)],
        )

        if config.rope:
            self._init_rope()

        self.config = config

    def _init_rope(self):
        self.rotary_emb = LlamaRotaryEmbedding(
            self.head_dim,
            max_position_embeddings=self.max_position_embeddings,
        )

    def forward(
        self,
        hidden_states,
        position_ids: tuple[paddle.Tensor] | None = None,
        past_key_value: tuple[paddle.Tensor] | None = None,
        attention_mask: paddle.Tensor | None = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        alibi: paddle.Tensor | None = None,
    ) -> tuple[
        paddle.Tensor, paddle.Tensor | None, tuple[paddle.Tensor] | None
    ]:
        """Input shape: Batch x Time x Channel"""
        # [bs, seq_len, num_head * head_dim] -> [seq_len / n, bs, num_head * head_dim] (n is model parallelism)

        target_query_shape = [0, 0, self.num_heads, self.head_dim]
        target_key_value_shape = [
            0,
            0,
            self.num_key_value_heads,
            self.head_dim,
        ]

        if self.config.sequence_parallel:
            hidden_states = dist.reshard(
                hidden_states,
                get_mesh(self.ipp),
                [dist.Shard(1), dist.Replicate()],
            )

        query_states = self.q_proj(hidden_states).reshape(
            shape=target_query_shape
        )
        key_states = self.k_proj(hidden_states).reshape(
            shape=target_key_value_shape
        )
        value_states = self.v_proj(hidden_states).reshape(
            shape=target_key_value_shape
        )

        if self.config.sequence_parallel:
            query_states = paddle.transpose(query_states, [1, 0, 2, 3])
            key_states = paddle.transpose(key_states, [1, 0, 2, 3])
            value_states = paddle.transpose(value_states, [1, 0, 2, 3])

        kv_seq_len = key_states.shape[-3]

        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-3]

        if self.config.rope:
            cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
            query_states, key_states = apply_rotary_pos_emb(
                query_states, key_states, cos, sin, position_ids
            )

        # [bs, seq_len, num_head, head_dim]
        if past_key_value is not None:
            # reuse k, v, self_attention
            key_states = paddle.concat([past_key_value[0], key_states], axis=1)
            value_states = paddle.concat(
                [past_key_value[1], value_states], axis=1
            )

        past_key_value = (key_states, value_states) if use_cache else None
        if self.kv_indices is not None:
            key_states = paddle.index_select(
                key_states, self.kv_indices, axis=2
            )
            value_states = paddle.index_select(
                value_states, self.kv_indices, axis=2
            )

        # TODO(wj-Mcat): use broadcast strategy when n_kv_heads = 1
        # repeat k/v heads if n_kv_heads < n_heads
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        if (
            self.config.recompute
            and self.config.recompute_granularity == "core_attn"
        ):
            outputs = recompute(
                scaled_dot_product_attention,
                query_states,
                self.config,
                key_states,
                value_states,
                attention_mask,
                output_attentions,
                None,
                False,
            )
        else:
            outputs = scaled_dot_product_attention(
                query_states,
                self.config,
                key_states,
                value_states,
                attention_mask,
                output_attentions,
            )

        if output_attentions:
            attn_output, attn_weights = outputs
        else:
            attn_output = outputs

        attn_output = self.o_proj(attn_output)
        # TODO add should be in SP region
        if self.config.sequence_parallel:
            attn_output = paddle.transpose(attn_output, [1, 0, 2])
            attn_output = dist.reshard(
                attn_output, get_mesh(self.ipp), [dist.Shard(1), dist.Shard(0)]
            )

        if not output_attentions:
            attn_weights = None

        outputs = (attn_output,)

        if output_attentions:
            outputs += (attn_weights,)

        if use_cache:
            outputs += (past_key_value,)

        if type(outputs) is tuple and len(outputs) == 1:
            outputs = outputs[0]

        return outputs


class LlamaMLPAuto(nn.Layer):
    def __init__(self, config, ipp: int | None = None):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.ipp = ipp
        self.config = config

        self.gate_proj = nn.Linear(
            self.hidden_size, self.intermediate_size, bias_attr=False
        )
        self.gate_proj.weight = dist.shard_tensor(
            self.gate_proj.weight,
            get_mesh(self.ipp),
            [dist.Replicate(), dist.Shard(1)],
        )

        self.up_proj = nn.Linear(
            self.hidden_size, self.intermediate_size, bias_attr=False
        )
        self.up_proj.weight = dist.shard_tensor(
            self.up_proj.weight,
            get_mesh(self.ipp),
            [dist.Replicate(), dist.Shard(1)],
        )

        self.down_proj = nn.Linear(
            self.intermediate_size, self.hidden_size, bias_attr=False
        )
        self.down_proj.weight = dist.shard_tensor(
            self.down_proj.weight,
            get_mesh(self.ipp),
            [dist.Replicate(), dist.Shard(0)],
        )

    def forward(self, x):
        out = self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))
        return out


class LlamaRMSNormAuto(nn.Layer):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.weight = paddle.create_parameter(
            shape=[self.hidden_size],
            dtype=paddle.get_default_dtype(),
            default_initializer=nn.initializer.Constant(1.0),
        )
        self.variance_epsilon = config.rms_norm_eps
        self.config = config

    def forward(self, hidden_states):
        variance = hidden_states.astype("float32").pow(2).mean(-1, keepdim=True)
        hidden_states = (
            paddle.rsqrt(variance + self.variance_epsilon) * hidden_states
        )

        if self.weight.dtype in [paddle.float16, paddle.bfloat16]:
            hidden_states = paddle.cast(hidden_states, self.weight.dtype)

        return hidden_states * self.weight


class LlamaDecoderLayerAuto(nn.Layer):
    def __init__(
        self,
        config,
        layerwise_recompute: bool = False,
        ipp: int | None = None,
    ):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.self_attn = LlamaAttentionAuto(config, layerwise_recompute, ipp)
        self.mlp = LlamaMLPAuto(config, ipp)
        self.input_layernorm = LlamaRMSNormAuto(config)
        self.post_attention_layernorm = LlamaRMSNormAuto(config)
        self.ipp = ipp

    def forward(
        self,
        hidden_states: paddle.Tensor,
        position_ids: tuple[paddle.Tensor] | None = None,
        attention_mask: paddle.Tensor | None = None,
        output_attentions: bool = False,
        past_key_value: tuple[paddle.Tensor] | None = None,
        use_cache: bool = False,
        alibi: paddle.Tensor | None = None,
    ):
        # [bs * seq_len, embed_dim] -> [seq_len * bs / n, embed_dim] (sequence_parallel)
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        if (
            self.config.recompute
            and self.config.recompute_granularity == "full_attn"
        ):
            outputs = recompute(
                self.self_attn,
                hidden_states,
                position_ids,
                past_key_value,
                attention_mask,
                output_attentions,
                use_cache,
                None,
            )
        else:
            outputs = self.self_attn(
                hidden_states,
                position_ids,
                past_key_value,
                attention_mask,
                output_attentions,
                use_cache,
            )
        if type(outputs) is tuple:
            hidden_states = outputs[0]
        else:
            hidden_states = outputs

        if output_attentions:
            self_attn_weights = outputs[1]

        if use_cache:
            present_key_value = outputs[2 if output_attentions else 1]

        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)

        if self.config.sequence_parallel:
            hidden_states = dist.reshard(
                hidden_states,
                get_mesh(self.ipp),
                [dist.Shard(1), dist.Replicate()],
            )
        hidden_states = self.mlp(hidden_states)

        if self.config.sequence_parallel:
            hidden_states = dist.reshard(
                hidden_states,
                get_mesh(self.ipp),
                [dist.Shard(1), dist.Shard(0)],
            )

        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        # remove empty tuple for pipeline parallel
        if type(outputs) is tuple and len(outputs) == 1:
            outputs = outputs[0]

        return outputs


class LlamaModelAuto(nn.Layer):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size
        self.hidden_size = config.hidden_size

        self.embed_tokens = nn.Embedding(
            self.vocab_size,
            self.hidden_size,
        )
        self.embed_tokens.weight = dist.shard_tensor(
            self.embed_tokens.weight,
            get_mesh(0),
            [dist.Replicate(), dist.Shard(1)],
        )

        def get_layer_pp_info(layer_index):
            if is_pp_enable() is False:
                return None, False
            else:
                global_mesh = dist.auto_parallel.get_mesh()
                pp_degree = global_mesh.get_dim_size("pp")
                layer_per_stage = math.ceil(
                    config.num_hidden_layers / pp_degree
                )
                input_need_reshard = layer_index % layer_per_stage == 0
                return layer_index // layer_per_stage, input_need_reshard

        decoder_layers = []
        self.next_pp_stage_indexes = []
        for i in range(config.num_hidden_layers):
            pp_stage_id, input_need_reshard = get_layer_pp_info(i)
            decoder_layers.append(
                LlamaDecoderLayerAuto(config, False, pp_stage_id)
            )
            if input_need_reshard:
                self.next_pp_stage_indexes.append(i)

        self.layers = nn.LayerList(decoder_layers)

        self.norm = LlamaRMSNormAuto(config)

        self.gradient_checkpointing = False

        self.placements = (
            [dist.Shard(1), dist.Shard(0)]
            if self.config.sequence_parallel
            else [dist.Shard(0), dist.Replicate()]
        )

    @staticmethod
    def _prepare_decoder_attention_mask(
        attention_mask, input_shape, past_key_values_length, dtype, mesh
    ):
        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            if len(attention_mask.shape) == 2:
                expanded_attn_mask = _expand_2d_mask(
                    attention_mask, dtype, tgt_length=input_shape[-1]
                )
                # For decoding phase in generation, seq_length = 1, we don't need to add causal mask
                if input_shape[-1] > 1:
                    combined_attention_mask = _make_causal_mask(
                        input_shape,
                        past_key_values_length=past_key_values_length,
                    )
                    expanded_attn_mask = (
                        expanded_attn_mask & combined_attention_mask
                    )
            # [bsz, seq_len, seq_len] -> [bsz, 1, seq_len, seq_len]
            elif len(attention_mask.shape) == 3:
                expanded_attn_mask = attention_mask.unsqueeze(1).astype("bool")
            # if attention_mask is already 4-D, do nothing
            else:
                expanded_attn_mask = attention_mask
        else:
            expanded_attn_mask = _make_causal_mask(
                input_shape, past_key_values_length=past_key_values_length
            )
        # Convert bool attention_mask to float attention mask, which will be added to attention_scores later
        expanded_attn_mask = paddle.where(
            expanded_attn_mask, 0.0, paddle.finfo(dtype).min
        ).astype(dtype)
        return expanded_attn_mask

    def forward(
        self,
        input_ids=None,
        position_ids=None,
        attention_mask=None,
        inputs_embeds=None,
        use_cache=None,
        past_key_values=None,
        output_attentions=False,
        output_hidden_states=None,
        **kwargs,
    ):
        output_attentions = (
            output_attentions if output_attentions is not None else False
        )
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else False
        )
        use_cache = (
            use_cache if use_cache is not None else self.config.use_cache
        )

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time"
            )
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError(
                "You have to specify either decoder_input_ids or decoder_inputs_embeds"
            )

        if past_key_values is None:
            past_key_values = tuple([None] * len(self.layers))

        seq_length_with_past = seq_length
        cache_length = 0
        if past_key_values[0] is not None:
            cache_length = paddle.shape(past_key_values[0][0])[1]
            seq_length_with_past += cache_length

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if self.config.sequence_parallel:
            # [B, S, H] -> [S, B, H]
            inputs_embeds = paddle.transpose(inputs_embeds, [1, 0, 2])

        mesh = global_mesh_starts_with_pp()
        # embed positions
        if attention_mask is None:
            # [bs, seq_len]
            attention_mask = paddle.ones(
                (batch_size, seq_length_with_past), dtype=paddle.bool
            )
        attention_mask = self._prepare_decoder_attention_mask(
            attention_mask,
            (batch_size, seq_length),
            cache_length,
            inputs_embeds.dtype,
            mesh,
        )  # [bs, 1, seq_len, seq_len]
        attention_mask = dist.shard_tensor(
            attention_mask,
            mesh,
            [dist.Replicate() for _ in range(len(mesh._shape))],
        )
        if not hasattr(self.config, "sep_parallel_degree"):
            self.config.sep_parallel_degree = -1
        if position_ids is None and self.config.sep_parallel_degree > 1:
            position_ids = paddle.arange(seq_length, dtype="int64").expand(
                (batch_size, seq_length)
            )
        if position_ids is not None:
            position_ids = dist.shard_tensor(
                position_ids,
                mesh,
                [dist.Replicate() for _ in range(len(mesh._shape))],
            )

        if self.config.use_flash_attention:
            is_casual = is_casual_mask(attention_mask)
            if is_casual:
                attention_mask = None
        hidden_states = inputs_embeds
        hidden_states = dist.reshard(
            hidden_states, get_mesh(0), self.placements
        )

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = () if use_cache else None

        for idx, (decoder_layer) in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            past_key_value = (
                past_key_values[idx] if past_key_values is not None else None
            )

            if not is_pp_enable():
                position_ids_input = position_ids
                attention_mask_input = attention_mask
            else:
                ipp = decoder_layer.ipp
                if position_ids is not None:
                    position_ids_input = dist.reshard(
                        position_ids,
                        get_mesh(ipp),
                        [dist.Replicate(), dist.Replicate()],
                    )
                else:
                    position_ids_input = position_ids
                attention_mask_input = (
                    dist.reshard(
                        attention_mask,
                        get_mesh(ipp),
                        [dist.Replicate(), dist.Replicate()],
                    )
                    if attention_mask is not None
                    else None
                )

            if idx in self.next_pp_stage_indexes:
                hidden_states = dist.reshard(
                    hidden_states,
                    get_mesh(ipp),
                    self.placements,
                )

            if (
                self.config.recompute
                and self.config.recompute_granularity == "full"
            ):
                layer_outputs = recompute(
                    decoder_layer,
                    hidden_states,
                    position_ids_input,
                    attention_mask_input,
                    output_attentions,
                    past_key_value,
                    use_cache,
                    None,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    position_ids_input,
                    attention_mask_input,
                    output_attentions,
                    past_key_value,
                    use_cache,
                )

            if type(layer_outputs) is tuple:
                hidden_states = layer_outputs[0]
            else:
                hidden_states = layer_outputs

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

            if use_cache:
                next_decoder_cache += (
                    layer_outputs[2 if output_attentions else 1],
                )

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None

        return tuple(
            v
            for v in [
                hidden_states,
                next_cache,
                all_hidden_states,
                all_self_attns,
            ]
            if v is not None
        )


class LlamaLMHeadAuto(nn.Layer):
    def __init__(self, config):
        super().__init__()
        self.config = config
        vocab_size = config.vocab_size

        self.weight = dist.shard_tensor(
            self.create_parameter(
                shape=[config.hidden_size, vocab_size],
                dtype=paddle.get_default_dtype(),
            ),
            get_mesh(-1),
            [dist.Replicate(), dist.Shard(1)],
        )

    def forward(self, hidden_states, tensor_parallel_output=None):
        logits = paddle.matmul(hidden_states, self.weight, transpose_y=False)
        return logits


class LlamaPretrainingCriterionAuto(paddle.nn.Layer):
    def __init__(self, config):
        super().__init__()
        self.ignore_index = getattr(config, "ignore_index", -100)
        self.config = config
        self.loss_func = paddle.nn.CrossEntropyLoss(
            reduction="none", ignore_index=self.ignore_index
        )

    def forward(self, prediction_scores, masked_lm_labels):
        # Force Replicated to match dy & st
        prediction_scores1 = dist.reshard(
            prediction_scores,
            get_mesh(-1),
            [dist.Replicate(), dist.Replicate()],
        )
        masked_lm_labels1 = dist.reshard(
            masked_lm_labels, get_mesh(-1), [dist.Replicate(), dist.Replicate()]
        )

        # Force entropy same kernel
        if isinstance(prediction_scores1, paddle.Tensor):
            masked_lm_loss = self.loss_func(
                prediction_scores1.astype("float32")._use_gpudnn(False),
                masked_lm_labels1.unsqueeze(2),
            )
        else:
            masked_lm_loss = self.loss_func(
                prediction_scores1.astype("float32"),
                masked_lm_labels1.unsqueeze(2),
            )

        masked_lm_loss = paddle.masked_select(
            masked_lm_loss, masked_lm_loss > 0
        ).astype("float32")
        loss = paddle.mean(masked_lm_loss)
        return loss


class LlamaForCausalLMAuto(nn.Layer):
    enable_to_static_method = True

    def __init__(self, config):
        super().__init__()
        self.config = config

        # with paddle.LazyGuard():
        #    self.llama = LlamaModelAuto(config)
        #    self.lm_head = LlamaLMHeadAuto(config)
        #    self.criterion = LlamaPretrainingCriterionAuto(config)

        self.llama = LlamaModelAuto(config)
        self.lm_head = LlamaLMHeadAuto(config)
        # self.criterion = LlamaPretrainingCriterionAuto(config)

    def forward(
        self,
        input_ids=None,
        labels=None,
        position_ids=None,
        attention_mask=None,
        inputs_embeds=None,
        use_cache=False,
        past_key_values=None,
        output_attentions=None,
        output_hidden_states=None,
    ):
        input_ids.stop_gradient = True

        output_attentions = (
            output_attentions if output_attentions is not None else False
        )
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else False
        )
        outputs = self.llama(
            input_ids,  # [bs, seq_len]
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            past_key_values=past_key_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        hidden_states = outputs[0]  # [bs, seq_len, dim]

        # if labels is None，means we need full output, instead of tensor_parallel_output
        if self.config.sequence_parallel:
            hidden_states = dist.reshard(
                hidden_states, get_mesh(-1), [dist.Shard(1), dist.Replicate()]
            )
            # [S, B, H] -> [B, S, H]
            hidden_states = paddle.transpose(hidden_states, [1, 0, 2])

        logits = self.lm_head(hidden_states)

        # loss = None
        # if labels is not None:
        #     labels.stop_gradient = True
        #     labels = dist.shard_tensor(
        #         labels, get_mesh(-1), [dist.Shard(0), dist.Replicate()]
        #     )
        #     loss = self.criterion(logits, labels)

        # output = (logits,) + outputs[1:]
        # return (loss,) + output if loss is not None else output
        return logits


def _expand_2d_mask(mask, dtype, tgt_length):
    """
    Expands attention_mask from `[batch_size, src_length]` to `[batch_size, 1, tgt_length, src_length]`.
    """
    batch_size, src_length = mask.shape[0], mask.shape[-1]
    tgt_length = tgt_length if tgt_length is not None else src_length

    mask = mask[:, None, None, :].astype("bool")
    mask.stop_gradient = True
    expanded_mask = mask.expand([batch_size, 1, tgt_length, src_length])

    return expanded_mask


def apply_rotary_pos_emb(q, k, cos, sin, position_ids):
    if position_ids is None:
        # Note: Only for LlamaForCausalLMPipe model pretraining
        cos = cos[:, : q.shape[1], :, :]  # [bs, seq_len, 1, dim]
        sin = sin[:, : q.shape[1], :, :]  # [bs, seq_len, 1, dim]
    else:
        cos = cos.squeeze(axis=[0, 2])  # [seq_len, dim]
        sin = sin.squeeze(axis=[0, 2])  # [seq_len, dim]
        cos = cos[position_ids].unsqueeze(2)  # [bs, seq_len, 1, dim]
        sin = sin[position_ids].unsqueeze(2)  # [bs, seq_len, 1, dim]
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def get_triangle_upper_mask(x, mask=None):
    if mask is not None:
        return mask
    # [bsz, n_head, q_len, kv_seq_len]
    shape = x.shape
    #  [bsz, 1, q_len, kv_seq_len]
    shape[1] = 1
    mask = paddle.full(shape, paddle.finfo(x.dtype).min, dtype=x.dtype)
    mask = paddle.triu(mask, diagonal=1)
    mask.stop_gradient = True
    return mask


def is_casual_mask(attention_mask):
    """
    Upper triangular of attention_mask equals to attention_mask is casual
    """
    return (paddle.triu(attention_mask) == attention_mask).all().item()


def repeat_kv(hidden_states: paddle.Tensor, n_rep: int) -> paddle.Tensor:
    """
    This is the equivalent of paddle.repeat_interleave(hidden_states, n_rep, axis=1). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, slen, num_key_value_heads, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states

    hidden_states = hidden_states.unsqueeze(-2).tile([1, 1, 1, n_rep, 1])
    return hidden_states.reshape(
        [batch, slen, num_key_value_heads * n_rep, head_dim]
    )


def _make_causal_mask(input_ids_shape, past_key_values_length):
    """
    Make causal mask used for self-attention
    """
    batch_size, target_length = input_ids_shape  # target_length: seq_len

    mask = paddle.tril(
        paddle.ones((target_length, target_length), dtype="bool")
    )

    if past_key_values_length > 0:
        # [tgt_len, tgt_len + past_len]
        mask = paddle.concat(
            [
                paddle.ones(
                    [target_length, past_key_values_length], dtype="bool"
                ),
                mask,
            ],
            axis=-1,
        )

    # [bs, 1, tgt_len, tgt_len + past_len]
    return mask[None, None, :, :].expand(
        [batch_size, 1, target_length, target_length + past_key_values_length]
    )


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return paddle.concat([-x2, x1], axis=-1)  # shape is the same as x


def scaled_dot_product_attention(
    query_states,
    config,
    key_states,
    value_states,
    attention_mask,
    output_attentions,
    alibi=None,
    sequence_parallel=False,
):
    bsz, q_len, num_heads, head_dim = query_states.shape
    _, kv_seq_len, _, _ = value_states.shape

    if config.use_flash_attention and flash_attention:
        # Paddle Flash Attention input [ bz, seqlen, nhead, head_dim]
        # Torch Flash Attention input [ bz, nhead, seqlen, head_dim]

        version = paddle.version.full_version
        if version != "0.0.0" and version <= "2.5.2":
            if alibi is not None:
                raise ValueError("Flash Attention doesn't support alibi")
            attn_output, attn_weights = flash_attention(
                query_states,
                key_states,
                value_states,
                causal=True,
                return_softmax=output_attentions,
            )
        else:
            if alibi is not None:
                alibi = alibi.reshape([bsz, num_heads, 1, -1])
                attention_mask = attention_mask.cast(alibi.dtype) + alibi
            attn_output = F.scaled_dot_product_attention(
                query_states,
                key_states,
                value_states,
                attn_mask=attention_mask,
                is_causal=attention_mask is None,
            )
            attn_weights = None

        if sequence_parallel:
            attn_output = attn_output.reshape(
                [bsz * q_len, head_dim * num_heads]
            )
        else:
            attn_output = attn_output.reshape(
                [bsz, q_len, head_dim * num_heads]
            )
        return (attn_output, attn_weights) if output_attentions else attn_output
    else:
        #  [ bz, seqlen, nhead, head_dim] -> [bs, nhead, seq_len, head_dim]
        query_states = paddle.transpose(query_states, [0, 2, 1, 3])
        # merge with the next transpose
        key_states = paddle.transpose(key_states, [0, 2, 1, 3])
        value_states = paddle.transpose(value_states, [0, 2, 1, 3])

        # matmul and divide by sqrt(head_dim)
        attn_weights = paddle.matmul(
            query_states / math.sqrt(head_dim),
            key_states.transpose([0, 1, 3, 2]),
        )
        # then add alibi bias
        if alibi is not None:
            alibi = alibi.reshape([bsz, num_heads, 1, -1])
            attn_weights = attn_weights + alibi

        if list(attn_weights.shape) != [bsz, num_heads, q_len, kv_seq_len]:
            raise ValueError(
                f"Attention weights should be of shape {(bsz, num_heads, q_len, kv_seq_len)}, but is"
                f" {attn_weights.shape}"
            )

        # NOTE: we only call get_triangle_upper_mask under PP setup
        # FIXME ZHUI when we use pipeline parallel, the attention_mask can be None
        # we just make it triangle_upper_mask
        if attention_mask is None:
            attention_mask = get_triangle_upper_mask(attn_weights)

        attention_mask = attention_mask.reshape([bsz, 1, q_len, kv_seq_len])
        if list(attention_mask.shape) != [bsz, 1, q_len, kv_seq_len]:
            raise ValueError(
                f"Attention mask should be of shape {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.shape}"
            )

        attn_weights = attn_weights + attention_mask
        if not paddle.in_dynamic_mode():
            attn_weights = F.softmax(
                attn_weights, axis=-1, dtype="float32"
            ).astype(query_states.dtype)
        else:
            with paddle.amp.auto_cast(False):
                attn_weights = F.softmax(
                    attn_weights, axis=-1, dtype="float32"
                ).astype(query_states.dtype)

        attn_output = paddle.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose([0, 2, 1, 3])
        if sequence_parallel:
            attn_output = attn_output.reshape(
                [bsz * q_len, head_dim * num_heads]
            )
        else:
            attn_output = attn_output.reshape(
                [bsz, q_len, head_dim * num_heads]
            )
        return (attn_output, attn_weights) if output_attentions else attn_output
