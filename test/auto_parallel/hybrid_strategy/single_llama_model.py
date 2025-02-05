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

import paddle
import paddle.nn.functional as F
from paddle import nn
from paddle.nn.functional.flash_attention import _math_attention


class LlamaAttention(nn.Layer):
    def __init__(self, config):
        super().__init__()

        self.config = config
        self.hidden_size = self.config.hidden_size
        self.num_heads = self.config.num_attention_heads
        self.head_dim = self.hidden_size // self.config.num_attention_heads

        self.q_proj = nn.Linear(
            self.hidden_size,
            self.hidden_size,
            bias_attr=True,
        )

        self.k_proj = nn.Linear(
            self.hidden_size,
            self.hidden_size,
            bias_attr=True,
        )

        self.v_proj = nn.Linear(
            self.hidden_size,
            self.hidden_size,
            bias_attr=True,
        )

        self.o_proj = nn.Linear(
            self.hidden_size,
            self.hidden_size,
            bias_attr=True,
        )

    def forward(self, hidden_states):
        query_states = self.q_proj(hidden_states).reshape(
            shape=[0, 0, self.num_heads, self.head_dim]
        )
        key_states = self.k_proj(hidden_states).reshape(
            shape=[0, 0, self.num_heads, self.head_dim]
        )
        value_states = self.v_proj(hidden_states).reshape(
            shape=[0, 0, self.num_heads, self.head_dim]
        )

        bsz, q_len, _, _ = query_states.shape

        outputs, _ = _math_attention(
            query_states,
            key_states,
            value_states,
            causal=True,
        )

        attn_output = outputs.reshape(
            [bsz, q_len, self.head_dim * self.num_heads]
        )
        attn_output = self.o_proj(attn_output)

        return attn_output


class LlamaMLP(nn.Layer):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = self.config.hidden_size
        self.intermediate_size = self.config.intermediate_size

        self.gate_proj = nn.Linear(
            self.hidden_size, self.intermediate_size, bias_attr=False
        )

        self.up_proj = nn.Linear(
            self.hidden_size, self.intermediate_size, bias_attr=False
        )

        self.down_proj = nn.Linear(
            self.intermediate_size, self.hidden_size, bias_attr=False
        )

    def forward(self, x, test_for_list_input_output):
        out = self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))
        return out, test_for_list_input_output


class LlamaRMSNorm(nn.Layer):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = self.config.hidden_size
        self.weight = paddle.create_parameter(
            shape=[self.hidden_size],
            dtype=paddle.get_default_dtype(),
            default_initializer=nn.initializer.Constant(1.0),
        )
        self.variance_epsilon = self.config.rms_norm_eps

    def forward(self, hidden_states):
        variance = hidden_states.astype("float32").pow(2).mean(-1, keepdim=True)
        hidden_states = (
            paddle.rsqrt(variance + self.variance_epsilon) * hidden_states
        )

        if self.weight.dtype in [paddle.float16, paddle.bfloat16]:
            hidden_states = paddle.cast(hidden_states, self.weight.dtype)

        return hidden_states * self.weight


class LlamaDecoderLayer(nn.Layer):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.self_attn = LlamaAttention(self.config)
        self.mlp = LlamaMLP(self.config)
        self.input_layernorm = LlamaRMSNorm(self.config)
        self.post_attention_layernorm = LlamaRMSNorm(self.config)

    def forward(self, hidden_states, global_tensor):
        residual = hidden_states + global_tensor
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states, _ = self.mlp(hidden_states, "ONLY_FOR_TEST")
        hidden_states = residual + hidden_states

        return hidden_states


class GlobalOutputNet(nn.Layer):
    def __init__(self, config) -> None:
        super().__init__()
        self.config = config

    def forward(self, input):
        return (
            input
            if input is not None
            else paddle.rand([self.config.hidden_size], dtype="float32")
        )


class LlamaModel(nn.Layer):
    def __init__(self, config, position_embedding=False):
        super().__init__()
        self.config = config
        self.vocab_size = self.config.vocab_size
        self.hidden_size = self.config.hidden_size

        self.embed_tokens = nn.Embedding(
            self.vocab_size,
            self.hidden_size,
        )

        self.position_embedding = (
            nn.Embedding(
                self.vocab_size,
                self.hidden_size,
            )
            if position_embedding
            else None
        )

        self.global_layer = GlobalOutputNet(self.config)

        decoder_layers = []
        for i in range(self.config.num_hidden_layers):
            decoder_layers.append(LlamaDecoderLayer(self.config))

        self.layers = nn.LayerList(decoder_layers)
        self.norm = LlamaRMSNorm(self.config)

    def forward(self, input_ids):
        hidden_states = self.embed_tokens(input_ids)
        if self.position_embedding is not None:
            ones = paddle.ones(input_ids.shape, dtype="int64")
            seq_length = paddle.cumsum(ones, axis=-1)
            position_ids = seq_length - ones
            position_embeddings = self.position_embedding(position_ids)
            hidden_states = hidden_states + position_embeddings

        global_tensor = self.global_layer(None)

        for idx, (decoder_layer) in enumerate(self.layers):
            hidden_states = decoder_layer(hidden_states, global_tensor)

        hidden_states = self.norm(hidden_states)

        return hidden_states


class LlamaLMHead(nn.Layer):
    def __init__(self, config, weight=None):
        super().__init__()
        self.config = config
        self.transpose_y = False
        if weight is not None:
            self.weight = weight
            self.transpose_y = True
        else:
            self.weight = self.create_parameter(
                shape=[self.config.hidden_size, self.config.vocab_size],
                dtype=paddle.get_default_dtype(),
            )

    def forward(self, hidden_states):
        logits = paddle.matmul(
            hidden_states, self.weight, transpose_y=self.transpose_y
        )
        return logits


class LlamaPretrainingCriterion(paddle.nn.Layer):
    def __init__(self, config):
        super().__init__()
        self.ignore_index = getattr(config, "ignore_index", -100)
        self.config = config
        self.loss_func = paddle.nn.CrossEntropyLoss(
            reduction="none", ignore_index=self.ignore_index
        )

    def forward(self, prediction_scores, masked_lm_labels):
        if isinstance(prediction_scores, paddle.Tensor):
            masked_lm_loss = self.loss_func(
                prediction_scores.astype("float32")._use_gpudnn(False),
                masked_lm_labels.unsqueeze(2),
            )
        else:
            masked_lm_loss = self.loss_func(
                prediction_scores.astype("float32"),
                masked_lm_labels.unsqueeze(2),
            )

        masked_lm_loss = paddle.masked_select(
            masked_lm_loss, masked_lm_loss > 0
        ).astype("float32")
        loss = paddle.mean(masked_lm_loss)
        return loss


class LlamaForCausalLM(nn.Layer):
    enable_to_static_method = True

    def __init__(self, config, share_embedding=False, position_embedding=False):
        super().__init__()
        self.config = config

        self.llama = LlamaModel(self.config, position_embedding)
        if share_embedding:
            self.lm_head = LlamaLMHead(
                self.config, self.llama.embed_tokens.weight
            )
        else:
            self.lm_head = LlamaLMHead(self.config)

    def forward(self, input_ids=None):
        input_ids.stop_gradient = True

        hidden_states = self.llama(input_ids)
        logits = self.lm_head(hidden_states)

        return logits
