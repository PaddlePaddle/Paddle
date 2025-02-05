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

import functools
import warnings

import numpy as np

import paddle
import paddle.nn.functional as F
from paddle import Tensor, nn
from paddle.base.data_feeder import convert_dtype
from paddle.distributed.fleet.utils import recompute
from paddle.io import DataLoader, Dataset
from paddle.nn import MultiHeadAttention

try:
    from paddle.incubate.nn import FusedTransformerEncoderLayer
except ImportError:
    FusedTransformerEncoderLayer = None


VOCAB_SIZE = 30522


class Stack:
    def __init__(self, axis=0, dtype=None):
        self._axis = axis
        self._dtype = dtype

    def __call__(self, data):
        data = (
            np.stack(data, axis=self._axis).astype(self._dtype)
            if self._dtype
            else np.stack(data, axis=self._axis)
        )
        return data


def is_tensor(x):
    if isinstance(x, paddle.Tensor):
        return True

    return isinstance(x, np.ndarray)


class BertConfig:
    def __init__(self):
        self.attention_probs_dropout_prob = 0.1
        self.fuse = False
        self.hidden_act = 'gelu'
        self.hidden_dropout_prob = 0.1
        # Decrease config to speed up unittest
        # self.hidden_size = 768
        self.hidden_size = 60
        self.initializer_range = 0.02
        self.intermediate_size = 3072
        self.layer_norm_eps = 1e-12
        self.max_position_embeddings = 512
        self.model_type = 'bert'
        # self.num_attention_heads = 12
        self.num_attention_heads = 6
        # self.num_hidden_layers = 12
        self.num_hidden_layers = 6
        self.pad_token_id = 0
        self.paddlenlp_version = None
        self.pool_act = 'tanh'
        self.type_vocab_size = 2
        self.vocab_size = VOCAB_SIZE
        self.use_return_dict = False
        self.output_hidden_states = False
        self.output_attentions = False
        self.use_cache = False


class BertLMPredictionHead(nn.Layer):
    def __init__(self, config: BertConfig, embedding_weights=None):
        super().__init__()

        self.transform = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = getattr(nn.functional, config.hidden_act)
        self.layer_norm = nn.LayerNorm(config.hidden_size)
        self.decoder_weight = (
            self.create_parameter(
                shape=[config.vocab_size, config.hidden_size],
                dtype=self.transform.weight.dtype,
                is_bias=False,
            )
            if embedding_weights is None
            else embedding_weights
        )

        self.decoder_bias = self.create_parameter(
            shape=[config.vocab_size],
            dtype=self.decoder_weight.dtype,
            is_bias=True,
        )

    def forward(self, hidden_states, masked_positions=None):
        if masked_positions is not None:
            hidden_states = paddle.reshape(
                hidden_states, [-1, hidden_states.shape[-1]]
            )
            hidden_states = paddle.tensor.gather(
                hidden_states, masked_positions
            )
        # gather masked tokens might be more quick
        hidden_states = self.transform(hidden_states)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.layer_norm(hidden_states)
        hidden_states = (
            paddle.tensor.matmul(
                hidden_states, self.decoder_weight, transpose_y=True
            )
            + self.decoder_bias
        )
        return hidden_states


class BertPretrainingHeads(nn.Layer):
    def __init__(self, config: BertConfig, embedding_weights=None):
        super().__init__()
        self.predictions = BertLMPredictionHead(config, embedding_weights)
        self.seq_relationship = nn.Linear(config.hidden_size, 2)

    def forward(self, sequence_output, pooled_output, masked_positions=None):
        prediction_scores = self.predictions(sequence_output, masked_positions)
        seq_relationship_score = self.seq_relationship(pooled_output)
        return prediction_scores, seq_relationship_score


class BertEmbeddings(nn.Layer):
    def __init__(self, config: BertConfig):
        super().__init__()

        self.word_embeddings = nn.Embedding(
            config.vocab_size, config.hidden_size
        )
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, config.hidden_size
        )
        self.token_type_embeddings = nn.Embedding(
            config.type_vocab_size, config.hidden_size
        )
        self.layer_norm = nn.LayerNorm(config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(
        self,
        input_ids: Tensor,
        token_type_ids: Tensor | None = None,
        position_ids: Tensor | None = None,
        past_key_values_length: int | None = None,
    ):
        if position_ids is None:
            ones = paddle.ones_like(input_ids, dtype="int64")
            seq_length = paddle.cumsum(ones, axis=-1)

            position_ids = seq_length - ones
            if past_key_values_length is not None:
                position_ids += past_key_values_length
            position_ids.stop_gradient = True
        if token_type_ids is None:
            token_type_ids = paddle.zeros_like(input_ids, dtype="int64")

        input_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = (
            input_embeddings + position_embeddings + token_type_embeddings
        )
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class BertPooler(nn.Layer):
    def __init__(self, config: BertConfig):
        super().__init__()

        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()
        self.pool_act = config.pool_act

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        if self.pool_act == "tanh":
            pooled_output = self.activation(pooled_output)
        return pooled_output


class BertModel(nn.Layer):
    def __init__(self, config: BertConfig, to_static, enable_cinn):
        super().__init__()
        self.config = config
        self.pad_token_id = config.pad_token_id
        self.initializer_range = config.initializer_range
        self.embeddings = BertEmbeddings(config)
        if config.fuse and FusedTransformerEncoderLayer is None:
            warnings.warn(
                "FusedTransformerEncoderLayer is not supported by the running Paddle. "
                "The flag fuse_transformer will be ignored. Try Paddle >= 2.3.0"
            )
        self.fuse = config.fuse and FusedTransformerEncoderLayer is not None
        if self.fuse:
            self.encoder = nn.LayerList(
                [
                    FusedTransformerEncoderLayer(
                        config.hidden_size,
                        config.num_attention_heads,
                        config.intermediate_size,
                        dropout_rate=config.hidden_dropout_prob,
                        activation=config.hidden_act,
                        attn_dropout_rate=config.attention_probs_dropout_prob,
                        act_dropout_rate=0.0,
                    )
                    for _ in range(config.num_hidden_layers)
                ]
            )
        else:
            encoder_layer = nn.TransformerEncoderLayer(
                config.hidden_size,
                config.num_attention_heads,
                config.intermediate_size,
                dropout=config.hidden_dropout_prob,
                activation=config.hidden_act,
                attn_dropout=config.attention_probs_dropout_prob,
                act_dropout=0,
            )
            self.encoder = nn.TransformerEncoder(
                encoder_layer, config.num_hidden_layers
            )
            if to_static:
                build_strategy = paddle.static.BuildStrategy()
                if enable_cinn:
                    build_strategy.build_cinn_pass = True
                self.encoder = paddle.jit.to_static(
                    self.encoder, None, build_strategy, full_graph=True
                )
        self.pooler = BertPooler(config)
        # self.apply(self.init_weights)

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def forward(
        self,
        input_ids: Tensor,
        token_type_ids: Tensor | None = None,
        position_ids: Tensor | None = None,
        attention_mask: Tensor | None = None,
        past_key_values: tuple[tuple[Tensor]] | None = None,
        use_cache: bool | None = None,
        output_hidden_states: bool | None = None,
        output_attentions: bool | None = None,
        return_dict: bool | None = None,
    ):
        return_dict = (
            return_dict
            if return_dict is not None
            else self.config.use_return_dict
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        use_cache = (
            use_cache if use_cache is not None else self.config.use_cache
        )

        past_key_values_length = None
        if past_key_values is not None:
            past_key_values_length = past_key_values[0][0].shape[2]
        if attention_mask is None:
            attention_mask = paddle.unsqueeze(
                (input_ids == self.pad_token_id).astype(
                    self.pooler.dense.weight.dtype
                )
                * -1e4,
                axis=[1, 2],
            )
            if past_key_values is not None:
                batch_size = past_key_values[0][0].shape[0]
                past_mask = paddle.zeros(
                    [batch_size, 1, 1, past_key_values_length],
                    dtype=attention_mask.dtype,
                )
                attention_mask = paddle.concat(
                    [past_mask, attention_mask], axis=-1
                )

        else:
            if attention_mask.ndim == 2:
                # attention_mask [batch_size, sequence_length] -> [batch_size, 1, 1, sequence_length]
                attention_mask = attention_mask.unsqueeze(axis=[1, 2]).astype(
                    paddle.get_default_dtype()
                )
                attention_mask = (1.0 - attention_mask) * -1e4

        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            past_key_values_length=past_key_values_length,
        )
        if self.fuse:
            assert (
                not output_attentions
            ), "Not support attentions output currently."
            assert (
                past_key_values is None
            ), "Not support past_key_values currently."
            hidden_states = embedding_output
            all_hidden_states = [] if output_hidden_states else None
            for layer in self.encoder:
                hidden_states = layer(hidden_states, attention_mask)
                if output_hidden_states:
                    all_hidden_states.append(hidden_states)
            pooled_output = self.pooler(hidden_states)

            return (
                (hidden_states, pooled_output, all_hidden_states)
                if output_hidden_states
                else (hidden_states, pooled_output)
            )
        else:
            self.encoder._use_cache = use_cache  # To be consistent with HF
            encoder_outputs = self.encoder(
                embedding_output,
                src_mask=attention_mask,
                cache=past_key_values,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            if isinstance(encoder_outputs, type(embedding_output)):
                sequence_output = encoder_outputs
                pooled_output = self.pooler(sequence_output)
                return (sequence_output, pooled_output)
            else:
                sequence_output = encoder_outputs[0]
                pooled_output = self.pooler(sequence_output)
                return (sequence_output, pooled_output) + encoder_outputs[1:]


class Bert(nn.Layer):
    def __init__(self, to_static, enable_cinn):
        super().__init__()
        config = BertConfig()
        self.bert = BertModel(config, to_static, enable_cinn)
        self.cls = BertPretrainingHeads(
            config,
            embedding_weights=self.bert.embeddings.word_embeddings.weight,
        )

        # self.apply(self.init_weights)

    def forward(
        self,
        input_ids: Tensor,
        token_type_ids: Tensor | None = None,
        position_ids: Tensor | None = None,
        attention_mask: Tensor | None = None,
        masked_positions: Tensor | None = None,
        labels: Tensor | None = None,
        next_sentence_label: Tensor | None = None,
        output_hidden_states: bool | None = None,
        output_attentions: bool | None = None,
        return_dict: bool | None = None,
    ):
        with paddle.static.amp.fp16_guard():
            outputs = self.bert(
                input_ids,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                attention_mask=attention_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

            sequence_output, pooled_output = outputs[:2]
            prediction_scores, seq_relationship_score = self.cls(
                sequence_output, pooled_output, masked_positions
            )

            total_loss = None
            if labels is not None and next_sentence_label is not None:
                loss_fct = paddle.nn.CrossEntropyLoss()
                masked_lm_loss = loss_fct(
                    prediction_scores.reshape(
                        (-1, prediction_scores.shape[-1])
                    ),
                    labels.reshape((-1,)),
                )
                next_sentence_loss = loss_fct(
                    seq_relationship_score.reshape((-1, 2)),
                    next_sentence_label.reshape((-1,)),
                )
                total_loss = masked_lm_loss + next_sentence_loss

            output = (prediction_scores, seq_relationship_score) + outputs[2:]
            return ((total_loss, *output)) if total_loss is not None else output


class BertPretrainingCriterion(paddle.nn.Layer):
    def __init__(self, vocab_size=VOCAB_SIZE):
        super().__init__()
        # CrossEntropyLoss is expensive since the inner reshape (copy)
        self.loss_fn = paddle.nn.loss.CrossEntropyLoss(ignore_index=-1)
        self.vocab_size = vocab_size

    def forward(
        self,
        prediction_scores,
        seq_relationship_score,
        masked_lm_labels,
        next_sentence_labels,
        masked_lm_scale,
    ):
        with paddle.static.amp.fp16_guard():
            masked_lm_loss = F.cross_entropy(
                prediction_scores,
                masked_lm_labels,
                reduction="none",
                ignore_index=-1,
            )
            masked_lm_loss = masked_lm_loss / masked_lm_scale
            next_sentence_loss = F.cross_entropy(
                seq_relationship_score, next_sentence_labels, reduction="none"
            )
        return paddle.sum(masked_lm_loss) + paddle.mean(next_sentence_loss)


def layer_init_wrapper(func):
    @functools.wraps(func)
    def _impl(self, *args, **kwargs):
        enable_recompute = kwargs.pop("enable_recompute", False)
        func(self, *args, **kwargs)
        if paddle.in_dynamic_mode():
            self.enable_recompute = enable_recompute
        else:
            self.enable_recompute = False

    return _impl


def _convert_attention_mask(attn_mask, dtype):
    if attn_mask is not None and attn_mask.dtype != dtype:
        attn_mask_dtype = convert_dtype(attn_mask.dtype)
        if attn_mask_dtype == 'bool' or 'int' in attn_mask_dtype:
            attn_mask = (paddle.cast(attn_mask, dtype) - 1.0) * 1e9
        else:
            attn_mask = paddle.cast(attn_mask, dtype)
    return attn_mask


def _transformer_encoder_layer_fwd(
    self, src, src_mask=None, cache=None, output_attentions=False
):
    self.self_attn.need_weights = output_attentions
    src_mask = _convert_attention_mask(src_mask, src.dtype)

    residual = src
    if self.normalize_before:
        src = self.norm1(src)

    attn_outputs = self.self_attn(src, src, src, src_mask, cache)
    if isinstance(attn_outputs, tuple):
        src = attn_outputs[0]
        outputs = attn_outputs[1:]
    else:
        src = attn_outputs
        outputs = None

    src = residual + self.dropout1(src)
    if not self.normalize_before:
        src = self.norm1(src)

    residual = src
    if self.normalize_before:
        src = self.norm2(src)
    src = self.linear2(self.dropout(self.activation(self.linear1(src))))
    src = residual + self.dropout2(src)
    if not self.normalize_before:
        src = self.norm2(src)

    return (
        src if outputs is None else ((src,) + outputs[::-1])
    )  # hidden_states, cache, attentions


def _transformer_decoder_layer_fwd(
    self,
    tgt,
    memory,
    tgt_mask=None,
    memory_mask=None,
    cache=None,
    output_attentions=False,
):
    residual = tgt

    # self attention
    self.self_attn.need_weights = output_attentions
    tgt_mask = _convert_attention_mask(tgt_mask, tgt.dtype)

    if self.normalize_before:
        tgt = self.norm1(tgt)

    self_attn_outputs = self.self_attn(
        tgt, tgt, tgt, tgt_mask, cache[0] if cache else None
    )
    # self_attn_outputs = (tgt, attn_weights, incremental_cache) or only tgt
    if isinstance(self_attn_outputs, type(tgt)):
        tgt = self_attn_outputs
    else:
        tgt = self_attn_outputs[0]
        if output_attentions:
            self_attn_weights = self_attn_outputs[1]
        if cache:
            incremental_cache = self_attn_outputs[-1]

    tgt = residual + self.dropout1(tgt)
    if not self.normalize_before:
        tgt = self.norm1(tgt)

    residual = tgt

    # cross attention
    if memory is not None:
        self.cross_attn.need_weights = output_attentions
        memory_mask = _convert_attention_mask(memory_mask, memory.dtype)

        if self.normalize_before:
            tgt = self.norm2(tgt)

        cross_attn_outputs = self.cross_attn(
            tgt, memory, memory, memory_mask, cache[1] if cache else None
        )
        if isinstance(cross_attn_outputs, type(tgt)):
            tgt = cross_attn_outputs
        else:
            tgt = cross_attn_outputs[0]
            if output_attentions:
                cross_attn_weights = cross_attn_outputs[1]
            if cache:
                static_cache = cross_attn_outputs[-1]

        tgt = residual + self.dropout2(tgt)
        if not self.normalize_before:
            tgt = self.norm2(tgt)

        residual = tgt

    if self.normalize_before:
        tgt = self.norm3(tgt)
    tgt = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
    tgt = residual + self.dropout3(tgt)
    if not self.normalize_before:
        tgt = self.norm3(tgt)

    if not output_attentions and cache is None:
        return tgt
    else:
        outputs = (tgt,)
        if output_attentions:
            outputs += (
                self_attn_weights,
                cross_attn_weights if memory is not None else None,
            )
        if cache:
            outputs += (
                (
                    incremental_cache,
                    static_cache if memory is not None else None,
                ),
            )
        return outputs


def _transformer_encoder_fwd(
    self,
    src,
    src_mask=None,
    cache=None,
    output_attentions=False,
    output_hidden_states=False,
    return_dict=False,
):
    src_mask = _convert_attention_mask(src_mask, src.dtype)

    output = src
    # To get cache from None when use_cache is True, which is compatible with HF
    # while HF requires decoder. The implementation here uses cache update in the
    # MultiHeadAttention not so efficiently, and maybe optimize it later.
    if cache is None and getattr(self, "_use_cache", False):
        cache = [tuple(self.layers[0].gen_cache(src))] * len(self.layers)
    # To be compatible with `TransformerEncoder.forward`, `_use_cache` defaults
    # to True when cache is not None.
    new_caches = (
        [] if cache is not None and getattr(self, "_use_cache", True) else None
    )
    all_attentions = [] if output_attentions else None
    # NOTE: Also includes embedding output which is same as HF.
    all_hidden_states = [output] if output_hidden_states else None
    for i, mod in enumerate(self.layers):
        if self.enable_recompute:
            # Note: recompute do not support pass as **kwargs yet.
            layer_outputs = recompute(
                mod,
                output,
                src_mask,
                (
                    None
                    if cache is None
                    else (
                        cache[i]
                        if isinstance(cache[i], MultiHeadAttention.Cache)
                        else MultiHeadAttention.Cache(*cache[i])
                    )
                ),
                output_attentions,
            )
        else:
            layer_outputs = mod(
                output,
                src_mask=src_mask,
                cache=(
                    None
                    if cache is None
                    else (
                        cache[i]
                        if isinstance(cache[i], MultiHeadAttention.Cache)
                        else MultiHeadAttention.Cache(*cache[i])
                    )
                ),
                output_attentions=output_attentions,
            )

        if isinstance(layer_outputs, tuple):
            output = layer_outputs[0]
            outputs = layer_outputs[1:]
        else:
            output = layer_outputs
            outputs = None

        if output_hidden_states:
            all_hidden_states.append(output)
        if output_attentions:
            all_attentions.append(outputs[-1])
        if new_caches is not None:
            new_caches.append(
                outputs[0]
                if isinstance(cache[i], MultiHeadAttention.Cache)
                else (tuple(outputs[0]))
            )

    if self.norm is not None:
        output = self.norm(output)

        if output_hidden_states:
            all_hidden_states[-1] = output

    outputs = tuple(
        tuple(v) if isinstance(v, list) else v
        for v in [
            output,
            new_caches,
            all_hidden_states,
            all_attentions,
        ]
        if v is not None
    )
    if len(outputs) == 1:
        return output
    else:
        return outputs


def _transformer_decoder_fwd(
    self,
    tgt,
    memory=None,
    tgt_mask=None,
    memory_mask=None,
    cache=None,
    output_attentions=False,
    output_hidden_states=False,
    return_dict=False,
):
    tgt_mask = _convert_attention_mask(tgt_mask, tgt.dtype)
    if memory is not None:
        memory_mask = _convert_attention_mask(memory_mask, memory.dtype)

    new_caches = [] if cache else None
    all_hidden_states = [tgt] if output_hidden_states else None
    all_self_attns = [] if output_attentions else None
    all_cross_attns = [] if output_attentions else None

    for i, mod in enumerate(self.layers):
        if cache is None:
            if self.enable_recompute:
                outputs = recompute(
                    mod,
                    tgt,
                    memory,
                    tgt_mask,
                    memory_mask,
                    None,
                    output_attentions,
                )
            else:
                outputs = mod(
                    tgt,
                    memory,
                    tgt_mask=tgt_mask,
                    memory_mask=memory_mask,
                    cache=None,
                    output_attentions=output_attentions,
                )
        else:
            outputs = mod(
                tgt,
                memory,
                tgt_mask=tgt_mask,
                memory_mask=memory_mask,
                cache=cache[i] if cache else None,
                output_attentions=output_attentions,
            )
        if isinstance(outputs, type(tgt)):
            tgt = outputs
        else:
            tgt = outputs[0]
        if cache:
            new_caches.append(outputs[-1])
        if output_attentions:
            all_self_attns.append(outputs[1])
            all_cross_attns.append(outputs[2])
        if output_hidden_states:
            all_hidden_states.append(tgt)

    if self.norm is not None:
        tgt = self.norm(tgt)
        if output_hidden_states:
            all_hidden_states[-1] = tgt

    if isinstance(outputs, type(tgt)):
        return tgt

    temp_list = [
        tgt,
        new_caches if cache else None,
        all_hidden_states,
        all_self_attns,
        all_cross_attns,
    ]
    return tuple(v for v in temp_list if v is not None)


# patches of paddle.nn.Transformer to get all hidden_states and attentions
paddle.nn.TransformerEncoderLayer.forward = _transformer_encoder_layer_fwd
paddle.nn.TransformerDecoderLayer.forward = _transformer_decoder_layer_fwd
paddle.nn.TransformerEncoder.forward = _transformer_encoder_fwd
paddle.nn.TransformerDecoder.forward = _transformer_decoder_fwd

_encoder_init = paddle.nn.TransformerEncoder.__init__
_decoder_init = paddle.nn.TransformerDecoder.__init__
paddle.nn.TransformerEncoder.__init__ = layer_init_wrapper(_encoder_init)
paddle.nn.TransformerDecoder.__init__ = layer_init_wrapper(_decoder_init)


class PretrainingDataset(Dataset):
    def __init__(self, input_file, max_pred_length):
        self.input_file = input_file
        self.max_pred_length = max_pred_length
        keys = [
            "input_ids",
            "input_mask",
            "segment_ids",
            "masked_lm_positions",
            "masked_lm_ids",
            "next_sentence_labels",
        ]
        self.inputs = np.load(input_file)
        self.inputs = [self.inputs[key] for key in keys]

    def __len__(self):
        "Denotes the total number of samples"
        return len(self.inputs[0])

    def __getitem__(self, index):
        [
            input_ids,
            input_mask,
            segment_ids,
            masked_lm_positions,
            masked_lm_ids,
            next_sentence_labels,
        ] = [
            (
                input[index].astype(np.int64)
                if indice < 5
                else np.asarray(input[index].astype(np.int64))
            )
            for indice, input in enumerate(self.inputs)
        ]
        # TODO: whether to use reversed mask by changing 1s and 0s to be
        # consistent with nv bert
        input_mask = (
            1
            - np.reshape(
                input_mask.astype(np.float32), [1, 1, input_mask.shape[0]]
            )
        ) * -1e9

        index = self.max_pred_length
        # store number of  masked tokens in index
        # outputs of torch.nonzero diff with that of numpy.nonzero by zip
        padded_mask_indices = (masked_lm_positions == 0).nonzero()[0]
        if len(padded_mask_indices) != 0:
            index = padded_mask_indices[0].item()
        else:
            index = self.max_pred_length
        # masked_lm_labels = np.full(input_ids.shape, -1, dtype=np.int64)
        # masked_lm_labels[masked_lm_positions[:index]] = masked_lm_ids[:index]
        masked_lm_labels = masked_lm_ids[:index]
        masked_lm_positions = masked_lm_positions[:index]
        # softmax_with_cross_entropy enforce last dim size equal 1
        masked_lm_labels = np.expand_dims(masked_lm_labels, axis=-1)
        next_sentence_labels = np.expand_dims(next_sentence_labels, axis=-1)
        return [
            input_ids,
            segment_ids,
            input_mask,
            masked_lm_positions,
            masked_lm_labels,
            next_sentence_labels,
        ]


def create_pretraining_dataset(
    input_file, max_pred_length, shared_list, batch_size, worker_init
):
    train_data = PretrainingDataset(
        input_file=input_file, max_pred_length=max_pred_length
    )
    # files have been sharded, no need to dispatch again
    train_batch_sampler = paddle.io.BatchSampler(
        train_data, batch_size=batch_size, shuffle=True
    )

    # DataLoader cannot be pickled because of its place.
    # If it can be pickled, use global function instead of lambda and use
    # ProcessPoolExecutor instead of ThreadPoolExecutor to prefetch.
    def _collate_data(data, stack_fn=Stack()):
        num_fields = len(data[0])
        out = [None] * num_fields
        # input_ids, segment_ids, input_mask, masked_lm_positions,
        # masked_lm_labels, next_sentence_labels, mask_token_num
        for i in (0, 1, 2, 5):
            out[i] = stack_fn([x[i] for x in data])
        _, seq_length = out[0].shape
        size = sum(len(x[3]) for x in data)
        # Padding for divisibility by 8 for fp16 or int8 usage
        if size % 8 != 0:
            size += 8 - (size % 8)
        # masked_lm_positions
        # Organize as a 1D tensor for gather or use gather_nd
        out[3] = np.full(size, 0, dtype=np.int32)
        # masked_lm_labels
        out[4] = np.full([size, 1], -1, dtype=np.int64)
        mask_token_num = 0
        for i, x in enumerate(data):
            for j, pos in enumerate(x[3]):
                out[3][mask_token_num] = i * seq_length + pos
                out[4][mask_token_num] = x[4][j]
                mask_token_num += 1
        # mask_token_num
        out.append(np.asarray([mask_token_num], dtype=np.float32))
        return out

    train_data_loader = DataLoader(
        dataset=train_data,
        batch_sampler=train_batch_sampler,
        collate_fn=_collate_data,
        num_workers=0,
        worker_init_fn=worker_init,
        return_list=True,
    )
    return train_data_loader


if __name__ == '__main__':
    bert = Bert()
