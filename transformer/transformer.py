# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

from __future__ import print_function

import numpy as np

import paddle.fluid as fluid
import paddle.fluid.layers as layers
from paddle.fluid.dygraph import Embedding, LayerNorm, Linear, Layer, to_variable
from paddle.fluid.dygraph.learning_rate_scheduler import LearningRateDecay
from model import Model, shape_hints, CrossEntropy, Loss


def position_encoding_init(n_position, d_pos_vec):
    """
    Generate the initial values for the sinusoid position encoding table.
    """
    channels = d_pos_vec
    position = np.arange(n_position)
    num_timescales = channels // 2
    log_timescale_increment = (np.log(float(1e4) / float(1)) /
                               (num_timescales - 1))
    inv_timescales = np.exp(
        np.arange(num_timescales)) * -log_timescale_increment
    scaled_time = np.expand_dims(position, 1) * np.expand_dims(
        inv_timescales, 0)
    signal = np.concatenate([np.sin(scaled_time), np.cos(scaled_time)], axis=1)
    signal = np.pad(signal, [[0, 0], [0, np.mod(channels, 2)]], 'constant')
    position_enc = signal
    return position_enc.astype("float32")


class NoamDecay(LearningRateDecay):
    """
    learning rate scheduler
    """
    def __init__(self,
                 d_model,
                 warmup_steps,
                 static_lr=2.0,
                 begin=1,
                 step=1,
                 dtype='float32'):
        super(NoamDecay, self).__init__(begin, step, dtype)
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.static_lr = static_lr

    def step(self):
        a = self.create_lr_var(self.step_num**-0.5)
        b = self.create_lr_var((self.warmup_steps**-1.5) * self.step_num)
        lr_value = (self.d_model**-0.5) * layers.elementwise_min(
            a, b) * self.static_lr
        return lr_value


class PrePostProcessLayer(Layer):
    """
    PrePostProcessLayer
    """
    def __init__(self, process_cmd, d_model, dropout_rate):
        super(PrePostProcessLayer, self).__init__()
        self.process_cmd = process_cmd
        self.functors = []
        for cmd in self.process_cmd:
            if cmd == "a":  # add residual connection
                self.functors.append(lambda x, y: x + y if y else x)
            elif cmd == "n":  # add layer normalization
                self.functors.append(
                    self.add_sublayer(
                        "layer_norm_%d" %
                        len(self.sublayers(include_sublayers=False)),
                        LayerNorm(
                            normalized_shape=d_model,
                            param_attr=fluid.ParamAttr(
                                initializer=fluid.initializer.Constant(1.)),
                            bias_attr=fluid.ParamAttr(
                                initializer=fluid.initializer.Constant(0.)))))
            elif cmd == "d":  # add dropout
                self.functors.append(lambda x: layers.dropout(
                    x, dropout_prob=dropout_rate, is_test=False)
                                     if dropout_rate else x)

    def forward(self, x, residual=None):
        for i, cmd in enumerate(self.process_cmd):
            if cmd == "a":
                x = self.functors[i](x, residual)
            else:
                x = self.functors[i](x)
        return x


class MultiHeadAttention(Layer):
    """
    Multi-Head Attention
    """
    def __init__(self, d_key, d_value, d_model, n_head=1, dropout_rate=0.):
        super(MultiHeadAttention, self).__init__()
        self.n_head = n_head
        self.d_key = d_key
        self.d_value = d_value
        self.d_model = d_model
        self.dropout_rate = dropout_rate
        self.q_fc = Linear(input_dim=d_model,
                           output_dim=d_key * n_head,
                           bias_attr=False)
        self.k_fc = Linear(input_dim=d_model,
                           output_dim=d_key * n_head,
                           bias_attr=False)
        self.v_fc = Linear(input_dim=d_model,
                           output_dim=d_value * n_head,
                           bias_attr=False)
        self.proj_fc = Linear(input_dim=d_value * n_head,
                              output_dim=d_model,
                              bias_attr=False)

    def _prepare_qkv(self, queries, keys, values, cache=None):
        if keys is None:  # self-attention
            keys, values = queries, queries
            static_kv = False
        else:  # cross-attention
            static_kv = True

        q = self.q_fc(queries)
        q = layers.reshape(x=q, shape=[0, 0, self.n_head, self.d_key])
        q = layers.transpose(x=q, perm=[0, 2, 1, 3])

        if cache is not None and static_kv and "static_k" in cache:
            # for encoder-decoder attention in inference and has cached
            k = cache["static_k"]
            v = cache["static_v"]
        else:
            k = self.k_fc(keys)
            v = self.v_fc(values)
            k = layers.reshape(x=k, shape=[0, 0, self.n_head, self.d_key])
            k = layers.transpose(x=k, perm=[0, 2, 1, 3])
            v = layers.reshape(x=v, shape=[0, 0, self.n_head, self.d_value])
            v = layers.transpose(x=v, perm=[0, 2, 1, 3])

        if cache is not None:
            if static_kv and not "static_k" in cache:
                # for encoder-decoder attention in inference and has not cached
                cache["static_k"], cache["static_v"] = k, v
            elif not static_kv:
                # for decoder self-attention in inference
                cache_k, cache_v = cache["k"], cache["v"]
                k = layers.concat([cache_k, k], axis=2)
                v = layers.concat([cache_v, v], axis=2)
                cache["k"], cache["v"] = k, v

        return q, k, v

    def forward(self, queries, keys, values, attn_bias, cache=None):
        # compute q ,k ,v
        q, k, v = self._prepare_qkv(queries, keys, values, cache)

        # scale dot product attention
        product = layers.matmul(x=q,
                                y=k,
                                transpose_y=True,
                                alpha=self.d_model**-0.5)
        if attn_bias:
            product += attn_bias
        weights = layers.softmax(product)
        if self.dropout_rate:
            weights = layers.dropout(weights,
                                     dropout_prob=self.dropout_rate,
                                     is_test=False)

        out = layers.matmul(weights, v)

        # combine heads
        out = layers.transpose(out, perm=[0, 2, 1, 3])
        out = layers.reshape(x=out, shape=[0, 0, out.shape[2] * out.shape[3]])

        # project to output
        out = self.proj_fc(out)
        return out


class FFN(Layer):
    """
    Feed-Forward Network
    """
    def __init__(self, d_inner_hid, d_model, dropout_rate):
        super(FFN, self).__init__()
        self.dropout_rate = dropout_rate
        self.fc1 = Linear(input_dim=d_model, output_dim=d_inner_hid, act="relu")
        self.fc2 = Linear(input_dim=d_inner_hid, output_dim=d_model)

    def forward(self, x):
        hidden = self.fc1(x)
        if self.dropout_rate:
            hidden = layers.dropout(hidden,
                                    dropout_prob=self.dropout_rate,
                                    is_test=False)
        out = self.fc2(hidden)
        return out


class EncoderLayer(Layer):
    """
    EncoderLayer
    """
    def __init__(self,
                 n_head,
                 d_key,
                 d_value,
                 d_model,
                 d_inner_hid,
                 prepostprocess_dropout,
                 attention_dropout,
                 relu_dropout,
                 preprocess_cmd="n",
                 postprocess_cmd="da"):

        super(EncoderLayer, self).__init__()

        self.preprocesser1 = PrePostProcessLayer(preprocess_cmd, d_model,
                                                 prepostprocess_dropout)
        self.self_attn = MultiHeadAttention(d_key, d_value, d_model, n_head,
                                            attention_dropout)
        self.postprocesser1 = PrePostProcessLayer(postprocess_cmd, d_model,
                                                  prepostprocess_dropout)

        self.preprocesser2 = PrePostProcessLayer(preprocess_cmd, d_model,
                                                 prepostprocess_dropout)
        self.ffn = FFN(d_inner_hid, d_model, relu_dropout)
        self.postprocesser2 = PrePostProcessLayer(postprocess_cmd, d_model,
                                                  prepostprocess_dropout)

    def forward(self, enc_input, attn_bias):
        attn_output = self.self_attn(self.preprocesser1(enc_input), None, None,
                                     attn_bias)
        attn_output = self.postprocesser1(attn_output, enc_input)

        ffn_output = self.ffn(self.preprocesser2(attn_output))
        ffn_output = self.postprocesser2(ffn_output, attn_output)
        return ffn_output


class Encoder(Layer):
    """
    encoder
    """
    def __init__(self,
                 n_layer,
                 n_head,
                 d_key,
                 d_value,
                 d_model,
                 d_inner_hid,
                 prepostprocess_dropout,
                 attention_dropout,
                 relu_dropout,
                 preprocess_cmd="n",
                 postprocess_cmd="da"):

        super(Encoder, self).__init__()

        self.encoder_layers = list()
        for i in range(n_layer):
            self.encoder_layers.append(
                self.add_sublayer(
                    "layer_%d" % i,
                    EncoderLayer(n_head, d_key, d_value, d_model, d_inner_hid,
                                 prepostprocess_dropout, attention_dropout,
                                 relu_dropout, preprocess_cmd,
                                 postprocess_cmd)))
        self.processer = PrePostProcessLayer(preprocess_cmd, d_model,
                                             prepostprocess_dropout)

    def forward(self, enc_input, attn_bias):
        for encoder_layer in self.encoder_layers:
            enc_output = encoder_layer(enc_input, attn_bias)
            enc_input = enc_output

        return self.processer(enc_output)


class Embedder(Layer):
    """
    Word Embedding + Position Encoding
    """
    def __init__(self, vocab_size, emb_dim, bos_idx=0):
        super(Embedder, self).__init__()

        self.word_embedder = Embedding(
            size=[vocab_size, emb_dim],
            padding_idx=bos_idx,
            param_attr=fluid.ParamAttr(
                initializer=fluid.initializer.Normal(0., emb_dim**-0.5)))

    def forward(self, word):
        word_emb = self.word_embedder(word)
        return word_emb


class WrapEncoder(Layer):
    """
    embedder + encoder
    """
    def __init__(self, src_vocab_size, max_length, n_layer, n_head, d_key,
                 d_value, d_model, d_inner_hid, prepostprocess_dropout,
                 attention_dropout, relu_dropout, preprocess_cmd,
                 postprocess_cmd, word_embedder):
        super(WrapEncoder, self).__init__()

        self.emb_dropout = prepostprocess_dropout
        self.emb_dim = d_model
        self.word_embedder = word_embedder
        self.pos_encoder = Embedding(
            size=[max_length, self.emb_dim],
            param_attr=fluid.ParamAttr(
                initializer=fluid.initializer.NumpyArrayInitializer(
                    position_encoding_init(max_length, self.emb_dim)),
                trainable=False))

        self.encoder = Encoder(n_layer, n_head, d_key, d_value, d_model,
                               d_inner_hid, prepostprocess_dropout,
                               attention_dropout, relu_dropout, preprocess_cmd,
                               postprocess_cmd)

    def forward(self, src_word, src_pos, src_slf_attn_bias):
        word_emb = self.word_embedder(src_word)
        word_emb = layers.scale(x=word_emb, scale=self.emb_dim**0.5)
        pos_enc = self.pos_encoder(src_pos)
        pos_enc.stop_gradient = True
        emb = word_emb + pos_enc
        enc_input = layers.dropout(emb,
                                   dropout_prob=self.emb_dropout,
                                   is_test=False) if self.emb_dropout else emb

        enc_output = self.encoder(enc_input, src_slf_attn_bias)
        return enc_output


class DecoderLayer(Layer):
    """
    decoder
    """
    def __init__(self,
                 n_head,
                 d_key,
                 d_value,
                 d_model,
                 d_inner_hid,
                 prepostprocess_dropout,
                 attention_dropout,
                 relu_dropout,
                 preprocess_cmd="n",
                 postprocess_cmd="da"):
        super(DecoderLayer, self).__init__()

        self.preprocesser1 = PrePostProcessLayer(preprocess_cmd, d_model,
                                                 prepostprocess_dropout)
        self.self_attn = MultiHeadAttention(d_key, d_value, d_model, n_head,
                                            attention_dropout)
        self.postprocesser1 = PrePostProcessLayer(postprocess_cmd, d_model,
                                                  prepostprocess_dropout)

        self.preprocesser2 = PrePostProcessLayer(preprocess_cmd, d_model,
                                                 prepostprocess_dropout)
        self.cross_attn = MultiHeadAttention(d_key, d_value, d_model, n_head,
                                             attention_dropout)
        self.postprocesser2 = PrePostProcessLayer(postprocess_cmd, d_model,
                                                  prepostprocess_dropout)

        self.preprocesser3 = PrePostProcessLayer(preprocess_cmd, d_model,
                                                 prepostprocess_dropout)
        self.ffn = FFN(d_inner_hid, d_model, relu_dropout)
        self.postprocesser3 = PrePostProcessLayer(postprocess_cmd, d_model,
                                                  prepostprocess_dropout)

    def forward(self,
                dec_input,
                enc_output,
                self_attn_bias,
                cross_attn_bias,
                cache=None):
        self_attn_output = self.self_attn(self.preprocesser1(dec_input), None,
                                          None, self_attn_bias, cache)
        self_attn_output = self.postprocesser1(self_attn_output, dec_input)

        cross_attn_output = self.cross_attn(
            self.preprocesser2(self_attn_output), enc_output, enc_output,
            cross_attn_bias, cache)
        cross_attn_output = self.postprocesser2(cross_attn_output,
                                                self_attn_output)

        ffn_output = self.ffn(self.preprocesser3(cross_attn_output))
        ffn_output = self.postprocesser3(ffn_output, cross_attn_output)

        return ffn_output


class Decoder(Layer):
    """
    decoder
    """
    def __init__(self, n_layer, n_head, d_key, d_value, d_model, d_inner_hid,
                 prepostprocess_dropout, attention_dropout, relu_dropout,
                 preprocess_cmd, postprocess_cmd):
        super(Decoder, self).__init__()

        self.decoder_layers = list()
        for i in range(n_layer):
            self.decoder_layers.append(
                self.add_sublayer(
                    "layer_%d" % i,
                    DecoderLayer(n_head, d_key, d_value, d_model, d_inner_hid,
                                 prepostprocess_dropout, attention_dropout,
                                 relu_dropout, preprocess_cmd,
                                 postprocess_cmd)))
        self.processer = PrePostProcessLayer(preprocess_cmd, d_model,
                                             prepostprocess_dropout)

    def forward(self,
                dec_input,
                enc_output,
                self_attn_bias,
                cross_attn_bias,
                caches=None):
        for i, decoder_layer in enumerate(self.decoder_layers):
            dec_output = decoder_layer(dec_input, enc_output, self_attn_bias,
                                       cross_attn_bias,
                                       None if caches is None else caches[i])
            dec_input = dec_output

        return self.processer(dec_output)


class WrapDecoder(Layer):
    """
    embedder + decoder
    """
    def __init__(self, trg_vocab_size, max_length, n_layer, n_head, d_key,
                 d_value, d_model, d_inner_hid, prepostprocess_dropout,
                 attention_dropout, relu_dropout, preprocess_cmd,
                 postprocess_cmd, share_input_output_embed, word_embedder):
        super(WrapDecoder, self).__init__()

        self.emb_dropout = prepostprocess_dropout
        self.emb_dim = d_model
        self.word_embedder = word_embedder
        self.pos_encoder = Embedding(
            size=[max_length, self.emb_dim],
            param_attr=fluid.ParamAttr(
                initializer=fluid.initializer.NumpyArrayInitializer(
                    position_encoding_init(max_length, self.emb_dim)),
                trainable=False))

        self.decoder = Decoder(n_layer, n_head, d_key, d_value, d_model,
                               d_inner_hid, prepostprocess_dropout,
                               attention_dropout, relu_dropout, preprocess_cmd,
                               postprocess_cmd)

        if share_input_output_embed:
            self.linear = lambda x: layers.matmul(x=x,
                                                  y=self.word_embedder.
                                                  word_embedder.weight,
                                                  transpose_y=True)
        else:
            self.linear = Linear(input_dim=d_model,
                                 output_dim=trg_vocab_size,
                                 bias_attr=False)

    def forward(self,
                trg_word,
                trg_pos,
                trg_slf_attn_bias,
                trg_src_attn_bias,
                enc_output,
                caches=None):
        word_emb = self.word_embedder(trg_word)
        word_emb = layers.scale(x=word_emb, scale=self.emb_dim**0.5)
        pos_enc = self.pos_encoder(trg_pos)
        pos_enc.stop_gradient = True
        emb = word_emb + pos_enc
        dec_input = layers.dropout(emb,
                                   dropout_prob=self.emb_dropout,
                                   is_test=False) if self.emb_dropout else emb
        dec_output = self.decoder(dec_input, enc_output, trg_slf_attn_bias,
                                  trg_src_attn_bias, caches)
        dec_output = layers.reshape(
            dec_output,
            shape=[-1, dec_output.shape[-1]],
        )
        logits = self.linear(dec_output)
        return logits


# class CrossEntropyCriterion(object):
#     def __init__(self, label_smooth_eps):
#         self.label_smooth_eps = label_smooth_eps

#     def __call__(self, predict, label, weights):
#         if self.label_smooth_eps:
#             label_out = layers.label_smooth(label=layers.one_hot(
#                 input=label, depth=predict.shape[-1]),
#                                             epsilon=self.label_smooth_eps)

#         cost = layers.softmax_with_cross_entropy(
#             logits=predict,
#             label=label_out,
#             soft_label=True if self.label_smooth_eps else False)
#         weighted_cost = cost * weights
#         sum_cost = layers.reduce_sum(weighted_cost)
#         token_num = layers.reduce_sum(weights)
#         token_num.stop_gradient = True
#         avg_cost = sum_cost / token_num
#         return sum_cost, avg_cost, token_num


class CrossEntropyCriterion(Loss):
    def __init__(self, label_smooth_eps):
        super(CrossEntropyCriterion, self).__init__()
        self.label_smooth_eps = label_smooth_eps

    def forward(self, outputs, labels):
        predict = outputs[0]
        label, weights = labels
        if self.label_smooth_eps:
            label = layers.label_smooth(label=layers.one_hot(
                input=label, depth=predict.shape[-1]),
                                            epsilon=self.label_smooth_eps)

        cost = layers.softmax_with_cross_entropy(
            logits=predict,
            label=label,
            soft_label=True if self.label_smooth_eps else False)
        weighted_cost = cost * weights
        sum_cost = layers.reduce_sum(weighted_cost)
        token_num = layers.reduce_sum(weights)
        token_num.stop_gradient = True
        avg_cost = sum_cost / token_num
        return avg_cost

    def infer_shape(self, _):
        return [[None, 1], [None, 1]]

    def infer_dtype(self, _):
        return ["int64", "float32"]


class Transformer(Model):
    """
    model
    """
    def __init__(self,
                 src_vocab_size,
                 trg_vocab_size,
                 max_length,
                 n_layer,
                 n_head,
                 d_key,
                 d_value,
                 d_model,
                 d_inner_hid,
                 prepostprocess_dropout,
                 attention_dropout,
                 relu_dropout,
                 preprocess_cmd,
                 postprocess_cmd,
                 weight_sharing,
                 bos_id=0,
                 eos_id=1):
        super(Transformer, self).__init__()
        src_word_embedder = Embedder(vocab_size=src_vocab_size,
                                     emb_dim=d_model,
                                     bos_idx=bos_id)
        self.encoder = WrapEncoder(src_vocab_size, max_length, n_layer, n_head,
                                   d_key, d_value, d_model, d_inner_hid,
                                   prepostprocess_dropout, attention_dropout,
                                   relu_dropout, preprocess_cmd,
                                   postprocess_cmd, src_word_embedder)
        if weight_sharing:
            assert src_vocab_size == trg_vocab_size, (
                "Vocabularies in source and target should be same for weight sharing."
            )
            trg_word_embedder = src_word_embedder
        else:
            trg_word_embedder = Embedder(vocab_size=trg_vocab_size,
                                         emb_dim=d_model,
                                         bos_idx=bos_id)
        self.decoder = WrapDecoder(trg_vocab_size, max_length, n_layer, n_head,
                                   d_key, d_value, d_model, d_inner_hid,
                                   prepostprocess_dropout, attention_dropout,
                                   relu_dropout, preprocess_cmd,
                                   postprocess_cmd, weight_sharing,
                                   trg_word_embedder)

        self.trg_vocab_size = trg_vocab_size
        self.n_layer = n_layer
        self.n_head = n_head
        self.d_key = d_key
        self.d_value = d_value

    @shape_hints(src_word=[None, None],
                 src_pos=[None, None],
                 src_slf_attn_bias=[None, 8, None, None],
                 trg_word=[None, None],
                 trg_pos=[None, None],
                 trg_slf_attn_bias=[None, 8, None, None],
                 trg_src_attn_bias=[None, 8, None, None])
    def forward(self, src_word, src_pos, src_slf_attn_bias, trg_word, trg_pos,
                trg_slf_attn_bias, trg_src_attn_bias):
        enc_output = self.encoder(src_word, src_pos, src_slf_attn_bias)
        predict = self.decoder(trg_word, trg_pos, trg_slf_attn_bias,
                               trg_src_attn_bias, enc_output)
        return predict

    def beam_search_v2(self,
                       src_word,
                       src_pos,
                       src_slf_attn_bias,
                       trg_word,
                       trg_src_attn_bias,
                       bos_id=0,
                       eos_id=1,
                       beam_size=4,
                       max_len=None,
                       alpha=0.6):
        """
        Beam search with the alive and finished two queues, both have a beam size
        capicity separately. It includes `grow_topk` `grow_alive` `grow_finish` as
        steps.

        1. `grow_topk` selects the top `2*beam_size` candidates to avoid all getting
        EOS.

        2. `grow_alive` selects the top `beam_size` non-EOS candidates as the inputs
        of next decoding step.

        3. `grow_finish` compares the already finished candidates in the finished queue
        and newly added finished candidates from `grow_topk`, and selects the top
        `beam_size` finished candidates.
        """
        def expand_to_beam_size(tensor, beam_size):
            tensor = layers.reshape(tensor,
                                    [tensor.shape[0], 1] + tensor.shape[1:])
            tile_dims = [1] * len(tensor.shape)
            tile_dims[1] = beam_size
            return layers.expand(tensor, tile_dims)

        def merge_beam_dim(tensor):
            return layers.reshape(tensor, [-1] + tensor.shape[2:])

        # run encoder
        enc_output = self.encoder(src_word, src_pos, src_slf_attn_bias)

        # constant number
        inf = float(1. * 1e7)
        batch_size = enc_output.shape[0]
        max_len = (enc_output.shape[1] + 20) if max_len is None else max_len

        ### initialize states of beam search ###
        ## init for the alive ##
        initial_log_probs = to_variable(
            np.array([[0.] + [-inf] * (beam_size - 1)], dtype="float32"))
        alive_log_probs = layers.expand(initial_log_probs, [batch_size, 1])
        alive_seq = to_variable(
            np.tile(np.array([[[bos_id]]], dtype="int64"),
                    (batch_size, beam_size, 1)))

        ## init for the finished ##
        finished_scores = to_variable(
            np.array([[-inf] * beam_size], dtype="float32"))
        finished_scores = layers.expand(finished_scores, [batch_size, 1])
        finished_seq = to_variable(
            np.tile(np.array([[[bos_id]]], dtype="int64"),
                    (batch_size, beam_size, 1)))
        finished_flags = layers.zeros_like(finished_scores)

        ### initialize inputs and states of transformer decoder ###
        ## init inputs for decoder, shaped `[batch_size*beam_size, ...]`
        trg_word = layers.reshape(alive_seq[:, :, -1],
                                  [batch_size * beam_size, 1])
        trg_src_attn_bias = merge_beam_dim(
            expand_to_beam_size(trg_src_attn_bias, beam_size))
        enc_output = merge_beam_dim(expand_to_beam_size(enc_output, beam_size))
        ## init states (caches) for transformer, need to be updated according to selected beam
        caches = [{
            "k":
            layers.fill_constant(
                shape=[batch_size * beam_size, self.n_head, 0, self.d_key],
                dtype=enc_output.dtype,
                value=0),
            "v":
            layers.fill_constant(
                shape=[batch_size * beam_size, self.n_head, 0, self.d_value],
                dtype=enc_output.dtype,
                value=0),
        } for i in range(self.n_layer)]

        def update_states(caches, beam_idx, beam_size):
            for cache in caches:
                cache["k"] = gather_2d_by_gather(cache["k"], beam_idx,
                                                 beam_size, batch_size, False)
                cache["v"] = gather_2d_by_gather(cache["v"], beam_idx,
                                                 beam_size, batch_size, False)
            return caches

        def gather_2d_by_gather(tensor_nd,
                                beam_idx,
                                beam_size,
                                batch_size,
                                need_flat=True):
            batch_idx = layers.range(0, batch_size, 1,
                                     dtype="int64") * beam_size
            flat_tensor = merge_beam_dim(tensor_nd) if need_flat else tensor_nd
            idx = layers.reshape(layers.elementwise_add(beam_idx, batch_idx, 0),
                                 [-1])
            new_flat_tensor = layers.gather(flat_tensor, idx)
            new_tensor_nd = layers.reshape(
                new_flat_tensor,
                shape=[batch_size, beam_idx.shape[1]] +
                tensor_nd.shape[2:]) if need_flat else new_flat_tensor
            return new_tensor_nd

        def early_finish(alive_log_probs, finished_scores,
                         finished_in_finished):
            max_length_penalty = np.power(((5. + max_len) / 6.), alpha)
            # The best possible score of the most likely alive sequence
            lower_bound_alive_scores = alive_log_probs[:, 0] / max_length_penalty

            # Now to compute the lowest score of a finished sequence in finished
            # If the sequence isn't finished, we multiply it's score by 0. since
            # scores are all -ve, taking the min will give us the score of the lowest
            # finished item.
            lowest_score_of_fininshed_in_finished = layers.reduce_min(
                finished_scores * finished_in_finished, 1)
            # If none of the sequences have finished, then the min will be 0 and
            # we have to replace it by -ve INF if it is. The score of any seq in alive
            # will be much higher than -ve INF and the termination condition will not
            # be met.
            lowest_score_of_fininshed_in_finished += (
                1. - layers.reduce_max(finished_in_finished, 1)) * -inf
            bound_is_met = layers.reduce_all(
                layers.greater_than(lowest_score_of_fininshed_in_finished,
                                    lower_bound_alive_scores))

            return bound_is_met

        def grow_topk(i, logits, alive_seq, alive_log_probs, states):
            logits = layers.reshape(logits, [batch_size, beam_size, -1])
            candidate_log_probs = layers.log(layers.softmax(logits, axis=2))
            log_probs = layers.elementwise_add(candidate_log_probs,
                                               alive_log_probs, 0)

            length_penalty = np.power(5.0 + (i + 1.0) / 6.0, alpha)
            curr_scores = log_probs / length_penalty
            flat_curr_scores = layers.reshape(curr_scores, [batch_size, -1])

            topk_scores, topk_ids = layers.topk(flat_curr_scores,
                                                k=beam_size * 2)

            topk_log_probs = topk_scores * length_penalty

            topk_beam_index = topk_ids // self.trg_vocab_size
            topk_ids = topk_ids % self.trg_vocab_size

            # use gather as gather_nd, TODO: use gather_nd
            topk_seq = gather_2d_by_gather(alive_seq, topk_beam_index,
                                           beam_size, batch_size)
            topk_seq = layers.concat(
                [topk_seq,
                 layers.reshape(topk_ids, topk_ids.shape + [1])],
                axis=2)
            states = update_states(states, topk_beam_index, beam_size)
            eos = layers.fill_constant(shape=topk_ids.shape,
                                       dtype="int64",
                                       value=eos_id)
            topk_finished = layers.cast(layers.equal(topk_ids, eos), "float32")

            #topk_seq: [batch_size, 2*beam_size, i+1]
            #topk_log_probs, topk_scores, topk_finished: [batch_size, 2*beam_size]
            return topk_seq, topk_log_probs, topk_scores, topk_finished, states

        def grow_alive(curr_seq, curr_scores, curr_log_probs, curr_finished,
                       states):
            curr_scores += curr_finished * -inf
            _, topk_indexes = layers.topk(curr_scores, k=beam_size)
            alive_seq = gather_2d_by_gather(curr_seq, topk_indexes,
                                            beam_size * 2, batch_size)
            alive_log_probs = gather_2d_by_gather(curr_log_probs, topk_indexes,
                                                  beam_size * 2, batch_size)
            states = update_states(states, topk_indexes, beam_size * 2)

            return alive_seq, alive_log_probs, states

        def grow_finished(finished_seq, finished_scores, finished_flags,
                          curr_seq, curr_scores, curr_finished):
            # finished scores
            finished_seq = layers.concat([
                finished_seq,
                layers.fill_constant(shape=[batch_size, beam_size, 1],
                                     dtype="int64",
                                     value=eos_id)
            ],
                                         axis=2)
            # Set the scores of the unfinished seq in curr_seq to large negative
            # values
            curr_scores += (1. - curr_finished) * -inf
            # concatenating the sequences and scores along beam axis
            curr_finished_seq = layers.concat([finished_seq, curr_seq], axis=1)
            curr_finished_scores = layers.concat([finished_scores, curr_scores],
                                                 axis=1)
            curr_finished_flags = layers.concat([finished_flags, curr_finished],
                                                axis=1)
            _, topk_indexes = layers.topk(curr_finished_scores, k=beam_size)
            finished_seq = gather_2d_by_gather(curr_finished_seq, topk_indexes,
                                               beam_size * 3, batch_size)
            finished_scores = gather_2d_by_gather(curr_finished_scores,
                                                  topk_indexes, beam_size * 3,
                                                  batch_size)
            finished_flags = gather_2d_by_gather(curr_finished_flags,
                                                 topk_indexes, beam_size * 3,
                                                 batch_size)
            return finished_seq, finished_scores, finished_flags

        for i in range(max_len):
            trg_pos = layers.fill_constant(shape=trg_word.shape,
                                           dtype="int64",
                                           value=i)
            logits = self.decoder(trg_word, trg_pos, None, trg_src_attn_bias,
                                  enc_output, caches)
            topk_seq, topk_log_probs, topk_scores, topk_finished, states = grow_topk(
                i, logits, alive_seq, alive_log_probs, caches)
            alive_seq, alive_log_probs, states = grow_alive(
                topk_seq, topk_scores, topk_log_probs, topk_finished, states)
            finished_seq, finished_scores, finished_flags = grow_finished(
                finished_seq, finished_scores, finished_flags, topk_seq,
                topk_scores, topk_finished)
            trg_word = layers.reshape(alive_seq[:, :, -1],
                                      [batch_size * beam_size, 1])

            if early_finish(alive_log_probs, finished_scores,
                            finished_flags).numpy():
                break

        return finished_seq, finished_scores

    def beam_search(self,
                    src_word,
                    src_pos,
                    src_slf_attn_bias,
                    trg_word,
                    trg_src_attn_bias,
                    bos_id=0,
                    eos_id=1,
                    beam_size=4,
                    max_len=256):
        if beam_size == 1:
            return self._greedy_search(src_word,
                                       src_pos,
                                       src_slf_attn_bias,
                                       trg_word,
                                       trg_src_attn_bias,
                                       bos_id=bos_id,
                                       eos_id=eos_id,
                                       max_len=max_len)
        else:
            return self._beam_search(src_word,
                                     src_pos,
                                     src_slf_attn_bias,
                                     trg_word,
                                     trg_src_attn_bias,
                                     bos_id=bos_id,
                                     eos_id=eos_id,
                                     beam_size=beam_size,
                                     max_len=max_len)

    def _beam_search(self,
                     src_word,
                     src_pos,
                     src_slf_attn_bias,
                     trg_word,
                     trg_src_attn_bias,
                     bos_id=0,
                     eos_id=1,
                     beam_size=4,
                     max_len=256):
        def expand_to_beam_size(tensor, beam_size):
            tensor = layers.reshape(tensor,
                                    [tensor.shape[0], 1] + tensor.shape[1:])
            tile_dims = [1] * len(tensor.shape)
            tile_dims[1] = beam_size
            return layers.expand(tensor, tile_dims)

        def merge_batch_beams(tensor):
            return layers.reshape(tensor, [tensor.shape[0] * tensor.shape[1]] +
                                  tensor.shape[2:])

        def split_batch_beams(tensor):
            return layers.reshape(tensor,
                                  shape=[-1, beam_size] +
                                  list(tensor.shape[1:]))

        def mask_probs(probs, finished, noend_mask_tensor):
            # TODO: use where_op
            finished = layers.cast(finished, dtype=probs.dtype)
            probs = layers.elementwise_mul(layers.expand(
                layers.unsqueeze(finished, [2]), [1, 1, self.trg_vocab_size]),
                                           noend_mask_tensor,
                                           axis=-1) - layers.elementwise_mul(
                                               probs, (finished - 1), axis=0)
            return probs

        def gather(x, indices, batch_pos):
            topk_coordinates = layers.stack([batch_pos, indices], axis=2)
            return layers.gather_nd(x, topk_coordinates)

        def update_states(func, caches):
            for cache in caches:  # no need to update static_kv
                cache["k"] = func(cache["k"])
                cache["v"] = func(cache["v"])
            return caches

        # run encoder
        enc_output = self.encoder(src_word, src_pos, src_slf_attn_bias)

        # constant number
        inf = float(1. * 1e7)
        batch_size = enc_output.shape[0]
        max_len = (enc_output.shape[1] + 20) if max_len is None else max_len
        vocab_size_tensor = layers.fill_constant(shape=[1],
                                                 dtype="int64",
                                                 value=self.trg_vocab_size)
        end_token_tensor = to_variable(
            np.full([batch_size, beam_size], eos_id, dtype="int64"))
        noend_array = [-inf] * self.trg_vocab_size
        noend_array[eos_id] = 0
        noend_mask_tensor = to_variable(np.array(noend_array,dtype="float32"))
        batch_pos = layers.expand(
            layers.unsqueeze(
                to_variable(np.arange(0, batch_size, 1, dtype="int64")), [1]),
            [1, beam_size])

        predict_ids = []
        parent_ids = []
        ### initialize states of beam search ###
        log_probs = to_variable(
            np.array([[0.] + [-inf] * (beam_size - 1)] * batch_size,
                     dtype="float32"))
        finished = to_variable(np.full([batch_size, beam_size], 0,
                                       dtype="bool"))
        ### initialize inputs and states of transformer decoder ###
        ## init inputs for decoder, shaped `[batch_size*beam_size, ...]`
        trg_word = layers.fill_constant(shape=[batch_size * beam_size, 1],
                                        dtype="int64",
                                        value=bos_id)
        trg_pos = layers.zeros_like(trg_word)
        trg_src_attn_bias = merge_batch_beams(
            expand_to_beam_size(trg_src_attn_bias, beam_size))
        enc_output = merge_batch_beams(expand_to_beam_size(enc_output, beam_size))
        ## init states (caches) for transformer, need to be updated according to selected beam
        caches = [{
            "k":
            layers.fill_constant(
                shape=[batch_size * beam_size, self.n_head, 0, self.d_key],
                dtype=enc_output.dtype,
                value=0),
            "v":
            layers.fill_constant(
                shape=[batch_size * beam_size, self.n_head, 0, self.d_value],
                dtype=enc_output.dtype,
                value=0),
        } for i in range(self.n_layer)]

        for i in range(max_len):
            trg_pos = layers.fill_constant(shape=trg_word.shape,
                                           dtype="int64",
                                           value=i)
            caches = update_states(  # can not be reshaped since the 0 size
                lambda x: x if i == 0 else merge_batch_beams(x), caches)
            logits = self.decoder(trg_word, trg_pos, None, trg_src_attn_bias,
                                  enc_output, caches)
            caches = update_states(split_batch_beams, caches)
            step_log_probs = split_batch_beams(
                layers.log(layers.softmax(logits)))
            step_log_probs = mask_probs(step_log_probs, finished,
                                        noend_mask_tensor)
            log_probs = layers.elementwise_add(x=step_log_probs,
                                               y=log_probs,
                                               axis=0)
            log_probs = layers.reshape(log_probs,
                                       [-1, beam_size * self.trg_vocab_size])
            scores = log_probs
            topk_scores, topk_indices = layers.topk(input=scores, k=beam_size)
            beam_indices = layers.elementwise_floordiv(
                topk_indices, vocab_size_tensor)
            token_indices = layers.elementwise_mod(
                topk_indices, vocab_size_tensor)

            # update states
            caches = update_states(lambda x: gather(x, beam_indices, batch_pos),
                                   caches)
            log_probs = gather(log_probs, topk_indices, batch_pos)
            finished = gather(finished, beam_indices, batch_pos)
            finished = layers.logical_or(
                finished, layers.equal(token_indices, end_token_tensor))
            trg_word = layers.reshape(token_indices, [-1, 1])

            predict_ids.append(token_indices)
            parent_ids.append(beam_indices)

            if layers.reduce_all(finished).numpy():
                break

        predict_ids = layers.stack(predict_ids, axis=0)
        parent_ids = layers.stack(parent_ids, axis=0)
        finished_seq = layers.transpose(
            layers.gather_tree(predict_ids, parent_ids), [1, 2, 0])
        finished_scores = topk_scores

        return finished_seq, finished_scores

    def _greedy_search(self,
                       src_word,
                       src_pos,
                       src_slf_attn_bias,
                       trg_word,
                       trg_src_attn_bias,
                       bos_id=0,
                       eos_id=1,
                       max_len=256):
        # run encoder
        enc_output = self.encoder(src_word, src_pos, src_slf_attn_bias)

        # constant number
        batch_size = enc_output.shape[0]
        max_len = (enc_output.shape[1] + 20) if max_len is None else max_len
        end_token_tensor = layers.fill_constant(shape=[batch_size, 1],
                                                dtype="int64",
                                                value=eos_id)

        predict_ids = []
        log_probs = layers.fill_constant(shape=[batch_size, 1],
                                         dtype="float32",
                                         value=0)
        trg_word = layers.fill_constant(shape=[batch_size, 1],
                                        dtype="int64",
                                        value=bos_id)
        finished = layers.fill_constant(shape=[batch_size, 1],
                                        dtype="bool",
                                        value=0)

        ## init states (caches) for transformer
        caches = [{
            "k":
            layers.fill_constant(
                shape=[batch_size, self.n_head, 0, self.d_key],
                dtype=enc_output.dtype,
                value=0),
            "v":
            layers.fill_constant(
                shape=[batch_size, self.n_head, 0, self.d_value],
                dtype=enc_output.dtype,
                value=0),
        } for i in range(self.n_layer)]

        for i in range(max_len):
            trg_pos = layers.fill_constant(shape=trg_word.shape,
                                           dtype="int64",
                                           value=i)
            logits = self.decoder(trg_word, trg_pos, None, trg_src_attn_bias,
                                  enc_output, caches)
            step_log_probs = layers.log(layers.softmax(logits))
            log_probs = layers.elementwise_add(x=step_log_probs,
                                               y=log_probs,
                                               axis=0)
            scores = log_probs
            topk_scores, topk_indices = layers.topk(input=scores, k=1)

            finished = layers.logical_or(
                finished, layers.equal(topk_indices, end_token_tensor))
            trg_word = topk_indices
            log_probs = topk_scores

            predict_ids.append(topk_indices)

            if layers.reduce_all(finished).numpy():
                break

        predict_ids = layers.stack(predict_ids, axis=0)
        finished_seq = layers.transpose(predict_ids, [1, 2, 0])
        finished_scores = topk_scores

        return finished_seq, finished_scores
