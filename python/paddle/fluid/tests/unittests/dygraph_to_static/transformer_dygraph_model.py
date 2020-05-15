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
from paddle.fluid.dygraph import Embedding, Layer, LayerNorm, Linear, to_variable
from paddle.fluid.dygraph.jit import dygraph_to_static_func
from paddle.fluid.layers.utils import map_structure


def position_encoding_init(n_position, d_pos_vec):
    """
    Generate the initial values for the sinusoid position encoding table.
    """
    channels = d_pos_vec
    position = np.arange(n_position)
    num_timescales = channels // 2
    log_timescale_increment = (np.log(float(1e4) / float(1)) /
                               (num_timescales - 1))
    inv_timescales = np.exp(np.arange(
        num_timescales)) * -log_timescale_increment
    scaled_time = np.expand_dims(position, 1) * np.expand_dims(inv_timescales,
                                                               0)
    signal = np.concatenate([np.sin(scaled_time), np.cos(scaled_time)], axis=1)
    signal = np.pad(signal, [[0, 0], [0, np.mod(channels, 2)]], 'constant')
    position_enc = signal
    return position_enc.astype("float32")


class PrePostProcessLayer(Layer):
    def __init__(self, process_cmd, d_model, dropout_rate):
        super(PrePostProcessLayer, self).__init__()
        self.process_cmd = process_cmd
        self.functors = []
        for cmd in self.process_cmd:
            if cmd == "a":  # add residual connection
                self.functors.append(lambda x, y: x + y if y is not None else x)
            elif cmd == "n":  # add layer normalization
                self.functors.append(
                    self.add_sublayer(
                        "layer_norm_%d" % len(
                            self.sublayers(include_sublayers=False)),
                        LayerNorm(
                            normalized_shape=d_model,
                            param_attr=fluid.ParamAttr(
                                initializer=fluid.initializer.Constant(1.)),
                            bias_attr=fluid.ParamAttr(
                                initializer=fluid.initializer.Constant(0.)))))
            elif cmd == "d":  # add dropout
                if dropout_rate:
                    self.functors.append(lambda x: layers.dropout(
                        x, dropout_prob=dropout_rate, is_test=False))

    def forward(self, x, residual=None):
        for i, cmd in enumerate(self.process_cmd):
            if cmd == "a":
                x = self.functors[i](x, residual)
            else:
                x = self.functors[i](x)
        return x


class MultiHeadAttention(Layer):
    def __init__(self,
                 d_key,
                 d_value,
                 d_model,
                 n_head=1,
                 dropout_rate=0.,
                 param_initializer=None):
        super(MultiHeadAttention, self).__init__()
        self.n_head = n_head
        self.d_key = d_key
        self.d_value = d_value
        self.d_model = d_model
        self.dropout_rate = dropout_rate
        self.q_fc = Linear(
            input_dim=d_model,
            output_dim=d_key * n_head,
            bias_attr=False,
            param_attr=fluid.ParamAttr(initializer=param_initializer))
        self.k_fc = Linear(
            input_dim=d_model,
            output_dim=d_key * n_head,
            bias_attr=False,
            param_attr=fluid.ParamAttr(initializer=param_initializer))
        self.v_fc = Linear(
            input_dim=d_model,
            output_dim=d_value * n_head,
            bias_attr=False,
            param_attr=fluid.ParamAttr(initializer=param_initializer))
        self.proj_fc = Linear(
            input_dim=d_value * n_head,
            output_dim=d_model,
            bias_attr=False,
            param_attr=fluid.ParamAttr(initializer=param_initializer))

    def forward(self, queries, keys, values, attn_bias, cache=None):
        # compute q ,k ,v
        keys = queries if keys is None else keys
        values = keys if values is None else values
        q = self.q_fc(queries)
        k = self.k_fc(keys)
        v = self.v_fc(values)
        # split head
        q = layers.reshape(x=q, shape=[0, 0, self.n_head, self.d_key])
        q = layers.transpose(x=q, perm=[0, 2, 1, 3])
        k = layers.reshape(x=k, shape=[0, 0, self.n_head, self.d_key])
        k = layers.transpose(x=k, perm=[0, 2, 1, 3])
        v = layers.reshape(x=v, shape=[0, 0, self.n_head, self.d_value])
        v = layers.transpose(x=v, perm=[0, 2, 1, 3])

        if cache is not None:
            cache_k, cache_v = cache["k"], cache["v"]
            k = layers.concat([cache_k, k], axis=2)
            v = layers.concat([cache_v, v], axis=2)
            cache["k"], cache["v"] = k, v
        # scale dot product attention
        product = layers.matmul(
            x=q, y=k, transpose_y=True, alpha=self.d_model**-0.5)
        if attn_bias is not None:
            product += attn_bias
        weights = layers.softmax(product)
        if self.dropout_rate:
            weights = layers.dropout(
                weights, dropout_prob=self.dropout_rate, is_test=False)
            out = layers.matmul(weights, v)
        out = layers.transpose(out, perm=[0, 2, 1, 3])
        out = layers.reshape(x=out, shape=[0, 0, out.shape[2] * out.shape[3]])
        out = self.proj_fc(out)
        return out


class FFN(Layer):
    def __init__(self, d_inner_hid, d_model, dropout_rate):
        super(FFN, self).__init__()
        self.dropout_rate = dropout_rate
        self.fc1 = Linear(input_dim=d_model, output_dim=d_inner_hid, act="relu")
        self.fc2 = Linear(input_dim=d_inner_hid, output_dim=d_model)

    def forward(self, x):
        hidden = self.fc1(x)
        if self.dropout_rate:
            hidden = layers.dropout(
                hidden, dropout_prob=self.dropout_rate, is_test=False)
        out = self.fc2(hidden)
        return out


class EncoderLayer(Layer):
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
        attn_output = self.self_attn(
            self.preprocesser1(enc_input), None, None, attn_bias)
        attn_output = self.postprocesser1(attn_output, enc_input)
        ffn_output = self.ffn(self.preprocesser2(attn_output))
        ffn_output = self.postprocesser2(ffn_output, attn_output)
        return ffn_output


class Encoder(Layer):
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
        enc_input = layers.dropout(
            emb, dropout_prob=self.emb_dropout,
            is_test=False) if self.emb_dropout else emb
        enc_output = self.encoder(enc_input, src_slf_attn_bias)
        return enc_output


class DecoderLayer(Layer):
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
        self_attn_output = self.self_attn(
            self.preprocesser1(dec_input), None, None, self_attn_bias, cache)
        self_attn_output = self.postprocesser1(self_attn_output, dec_input)
        cross_attn_output = self.cross_attn(
            self.preprocesser2(self_attn_output), enc_output, enc_output,
            cross_attn_bias)
        cross_attn_output = self.postprocesser2(cross_attn_output,
                                                self_attn_output)
        ffn_output = self.ffn(self.preprocesser3(cross_attn_output))
        ffn_output = self.postprocesser3(ffn_output, cross_attn_output)
        return ffn_output


class Decoder(Layer):
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
                                       cross_attn_bias, None
                                       if caches is None else caches[i])
            dec_input = dec_output
        return self.processer(dec_output)


class WrapDecoder(Layer):
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
            self.linear = Linear(
                input_dim=d_model, output_dim=trg_vocab_size, bias_attr=False)

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
        dec_input = layers.dropout(
            emb, dropout_prob=self.emb_dropout,
            is_test=False) if self.emb_dropout else emb
        dec_output = self.decoder(dec_input, enc_output, trg_slf_attn_bias,
                                  trg_src_attn_bias, caches)
        dec_output = layers.reshape(
            dec_output,
            shape=[-1, dec_output.shape[-1]], )
        logits = self.linear(dec_output)
        return logits


class CrossEntropyCriterion(object):
    def __init__(self, label_smooth_eps):
        self.label_smooth_eps = label_smooth_eps

    def __call__(self, predict, label, weights):
        if self.label_smooth_eps:
            label_out = layers.label_smooth(
                label=layers.one_hot(
                    input=label, depth=predict.shape[-1]),
                epsilon=self.label_smooth_eps)

        cost = layers.softmax_with_cross_entropy(
            logits=predict,
            label=label_out,
            soft_label=True if self.label_smooth_eps else False)
        weighted_cost = cost * weights
        sum_cost = layers.reduce_sum(weighted_cost)
        token_num = layers.reduce_sum(weights)
        token_num.stop_gradient = True
        avg_cost = sum_cost / token_num
        return sum_cost, avg_cost, token_num


class Transformer(Layer):
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
        src_word_embedder = Embedder(
            vocab_size=src_vocab_size, emb_dim=d_model, bos_idx=bos_id)
        self.encoder = WrapEncoder(
            src_vocab_size, max_length, n_layer, n_head, d_key, d_value,
            d_model, d_inner_hid, prepostprocess_dropout, attention_dropout,
            relu_dropout, preprocess_cmd, postprocess_cmd, src_word_embedder)
        if weight_sharing:
            assert src_vocab_size == trg_vocab_size, (
                "Vocabularies in source and target should be same for weight sharing."
            )
            trg_word_embedder = src_word_embedder
        else:
            trg_word_embedder = Embedder(
                vocab_size=trg_vocab_size, emb_dim=d_model, bos_idx=bos_id)
        self.decoder = WrapDecoder(
            trg_vocab_size, max_length, n_layer, n_head, d_key, d_value,
            d_model, d_inner_hid, prepostprocess_dropout, attention_dropout,
            relu_dropout, preprocess_cmd, postprocess_cmd, weight_sharing,
            trg_word_embedder)

        self.trg_vocab_size = trg_vocab_size
        self.n_layer = n_layer
        self.n_head = n_head
        self.d_key = d_key
        self.d_value = d_value

    @dygraph_to_static_func
    def forward(self, src_word, src_pos, src_slf_attn_bias, trg_word, trg_pos,
                trg_slf_attn_bias, trg_src_attn_bias):
        enc_output = self.encoder(src_word, src_pos, src_slf_attn_bias)
        predict = self.decoder(trg_word, trg_pos, trg_slf_attn_bias,
                               trg_src_attn_bias, enc_output)
        return predict

    @dygraph_to_static_func
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
        def expand_to_beam_size(tensor, beam_size):
            tensor = layers.reshape(
                tensor, [tensor.shape[0], 1] + list(tensor.shape[1:]))
            tile_dims = [1] * len(tensor.shape)
            tile_dims[1] = beam_size
            return layers.expand(tensor, tile_dims)

        def merge_batch_beams(tensor):
            var_dim_in_state = 2  # count in beam dim
            tensor = layers.transpose(
                tensor,
                list(range(var_dim_in_state, len(tensor.shape))) +
                list(range(0, var_dim_in_state)))

            tensor = layers.reshape(tensor,
                                    [0] * (len(tensor.shape) - var_dim_in_state
                                           ) + [batch_size * beam_size])
            res = layers.transpose(
                tensor,
                list(
                    range((len(tensor.shape) + 1 - var_dim_in_state),
                          len(tensor.shape))) +
                list(range(0, (len(tensor.shape) + 1 - var_dim_in_state))))
            return res

        def split_batch_beams(tensor):
            var_dim_in_state = 1
            tensor = layers.transpose(
                tensor,
                list(range(var_dim_in_state, len(tensor.shape))) +
                list(range(0, var_dim_in_state)))
            tensor = layers.reshape(tensor,
                                    [0] * (len(tensor.shape) - var_dim_in_state
                                           ) + [batch_size, beam_size])
            res = layers.transpose(
                tensor,
                list(
                    range((len(tensor.shape) - 1 - var_dim_in_state),
                          len(tensor.shape))) +
                list(range(0, (len(tensor.shape) - 1 - var_dim_in_state))))
            return res

        def mask_probs(probs, finished, noend_mask_tensor):
            finished = layers.cast(finished, dtype=probs.dtype)
            probs = layers.elementwise_mul(
                layers.expand(
                    layers.unsqueeze(finished, [2]),
                    [1, 1, self.trg_vocab_size]),
                noend_mask_tensor,
                axis=-1) - layers.elementwise_mul(
                    probs, (finished - 1), axis=0)
            return probs

        def gather(input, indices, batch_pos):
            topk_coordinates = fluid.layers.stack([batch_pos, indices], axis=2)
            return layers.gather_nd(input, topk_coordinates)

        # run encoder
        enc_output = self.encoder(src_word, src_pos, src_slf_attn_bias)
        batch_size = enc_output.shape[0]

        # constant number
        inf = float(1. * 1e7)
        max_len = (enc_output.shape[1] + 20) if max_len is None else max_len
        vocab_size_tensor = layers.fill_constant(
            shape=[1], dtype="int64", value=self.trg_vocab_size)
        end_token_tensor = to_variable(
            np.full(
                [batch_size, beam_size], eos_id, dtype="int64"))
        noend_array = [-inf] * self.trg_vocab_size
        noend_array[eos_id] = 0
        noend_mask_tensor = to_variable(np.array(noend_array, dtype="float32"))
        batch_pos = layers.expand(
            layers.unsqueeze(
                to_variable(np.arange(
                    0, batch_size, 1, dtype="int64")), [1]), [1, beam_size])
        predict_ids = []
        parent_ids = []
        ### initialize states of beam search ###
        log_probs = to_variable(
            np.array(
                [[0.] + [-inf] * (beam_size - 1)] * batch_size,
                dtype="float32"))

        finished = to_variable(
            np.full(
                [batch_size, beam_size], 0, dtype="bool"))

        trg_word = layers.fill_constant(
            shape=[batch_size * beam_size, 1], dtype="int64", value=bos_id)

        trg_src_attn_bias = merge_batch_beams(
            expand_to_beam_size(trg_src_attn_bias, beam_size))
        enc_output = merge_batch_beams(
            expand_to_beam_size(enc_output, beam_size))

        # init states (caches) for transformer, need to be updated according to selected beam
        caches = [{
            "k": layers.fill_constant(
                shape=[batch_size, beam_size, self.n_head, 0, self.d_key],
                dtype=enc_output.dtype,
                value=0),
            "v": layers.fill_constant(
                shape=[batch_size, beam_size, self.n_head, 0, self.d_value],
                dtype=enc_output.dtype,
                value=0),
        } for i in range(self.n_layer)]

        for i in range(max_len):
            trg_pos = layers.fill_constant(
                shape=trg_word.shape, dtype="int64", value=i)
            caches = map_structure(merge_batch_beams,
                                   caches)  # TODO: modified for dygraph2static
            logits = self.decoder(trg_word, trg_pos, None, trg_src_attn_bias,
                                  enc_output, caches)
            caches = map_structure(split_batch_beams, caches)
            step_log_probs = split_batch_beams(
                fluid.layers.log(fluid.layers.softmax(logits)))

            step_log_probs = mask_probs(step_log_probs, finished,
                                        noend_mask_tensor)
            log_probs = layers.elementwise_add(
                x=step_log_probs, y=log_probs, axis=0)
            log_probs = layers.reshape(log_probs,
                                       [-1, beam_size * self.trg_vocab_size])
            scores = log_probs
            topk_scores, topk_indices = fluid.layers.topk(
                input=scores, k=beam_size)
            beam_indices = fluid.layers.elementwise_floordiv(topk_indices,
                                                             vocab_size_tensor)
            token_indices = fluid.layers.elementwise_mod(topk_indices,
                                                         vocab_size_tensor)

            # update states
            caches = map_structure(lambda x: gather(x, beam_indices, batch_pos),
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
