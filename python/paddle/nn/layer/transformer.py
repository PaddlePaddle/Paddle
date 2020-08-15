#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

# TODO: define the classes of Transformer neural network
# __all__ = [ ]

import numpy as np

from ...fluid import layers
from ...fluid.dygraph import Layer, Linear
from ...fluid.initializer import Normal
from .. import functional as F
from ...fluid.layers import utils
from ...fluid.layers.utils import map_structure


class MultiHeadAttention(Layer):
    """
    Attention mapps queries and a set of key-value pairs to outputs, and
    Multi-Head Attention performs multiple parallel attention to jointly attending
    to information from different representation subspaces.

    Please refer to `Attention Is All You Need <https://arxiv.org/pdf/1706.03762.pdf>`_
    for more details.

    Parameters:
        embed_dim (int): The expected feature size in the input and output.
        num_heads (int): The number of heads in multi-head attention.
        dropout (float, optional): The dropout probability used on attention
            weights to drop some attention targets. 0 for no dropout. Default 0
        kdim (int, optional): The feature size in key. If None, assumed equal to
            `embed_dim`. Default None.
        vdim (int, optional): The feature size in key. If None, assumed equal to
            `embed_dim`. Default None.
        need_weights (bool, optional): Indicate whether to return the attention
            weights. Default False.
        param_attr(ParamAttr, optional):  To specify the weight parameter property.
            Default: None, which means the default weight parameter property is used.
            See usage for details in :ref:`api_fluid_ParamAttr` .
        bias_attr (ParamAttr, optional): To specify the bias parameter property.
            Default: None, which means the default bias parameter property is used.
            See usage for details in :ref:`api_fluid_ParamAttr` .
         
    Examples:

        .. code-block:: python

            import paddle
            from paddle.incubate.hapi.text import MultiHeadAttention

            # encoder input: [batch_size, sequence_length, d_model]
            query = paddle.rand((2, 4, 128))
            # self attention bias: [batch_size, n_head, src_len, src_len]
            attn_bias = paddle.rand((2, 2, 4, 4))
            multi_head_attn = MultiHeadAttention(64, 64, 128, n_head=2)
            output = multi_head_attn(query, attn_bias=attn_bias)  # [2, 4, 128]
    """

    def __init__(self,
                 embed_dim,
                 num_heads,
                 dropout=0.,
                 kdim=None,
                 vdim=None,
                 need_weights=True,
                 param_attr=None,
                 bias_attr=None):
        super(MultiHeadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.need_weights = need_weights

        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        self.q_proj = Linear(
            input_dim=embed_dim,
            output_dim=embed_dim,
            param_attr=param_attr,
            bias_attr=bias_attr)
        self.k_proj = Linear(
            input_dim=self.kdim,
            output_dim=embed_dim,
            param_attr=param_attr,
            bias_attr=bias_attr)
        self.v_proj = Linear(
            input_dim=self.vdim,
            output_dim=embed_dim,
            param_attr=param_attr,
            bias_attr=bias_attr)
        self.out_proj = Linear(
            input_dim=embed_dim,
            output_dim=embed_dim,
            param_attr=param_attr,
            bias_attr=bias_attr)

    def _prepare_qkv(self, query, key, value, cache=None):
        """
        Prapares linear projected queries, keys and values for usage of subsequnt
        multiple parallel attention. If `cache` is not None, using cached results
        to reduce redundant calculations.

        Parameters:
            query (Variable): The queries for multi-head attention. It is a
                tensor with shape `[batch_size, sequence_length, embed_dim]`. The
                data type should be float32 or float64.
            key (Variable): The keys for multi-head attention. It is
                a tensor with shape `[batch_size, sequence_length, kdim]`. The
                data type should be float32 or float64. If None, use `query` as
                `key`.
            value (Variable): The values for multi-head attention. It
                is a tensor with shape `[batch_size, sequence_length, vdim]`.
                The data type should be float32 or float64. If None, use `query` as
                `value`.
            cache (dict, optional): It is a dict with `k` and `v` as keys or
                `static_k` and `static_v` as keys, and values are tensors shaped
                `[batch_size, num_heads, length, embed_dim]` which are results of
                linear projection, reshape and transpose calculations. If keys are
                `k` and `v`, the values reserve intermediate results of previous
                positions, and would be updated by new tensors concatanating raw
                tensors with results of current position, which mostly used for
                decoder self attention. If keys are `static_k` and `static_v`,
                `key` and `value` args would be ignored, and the values in dict
                would be used as calculated results on `key` and `value`, which
                mostly used for decoder-encoder cross attention. It is only used
                for inference and should be None for training. Default None.

        Returns:
            tuple: A tuple including linear projected keys and values. These two \
                tensors have shapes `[batch_size, n_head, sequence_length, d_key]` \
                and `[batch_size, n_head, sequence_length, d_value]` separately, \
                and their data types are same as inputs.
        """
        q = self.q_proj(query)
        q = layers.reshape(x=q, shape=[0, 0, self.n_head, self.d_key])
        q = layers.transpose(x=q, perm=[0, 2, 1, 3])

        if cache is not None and "static_k" in cache:
            # for encoder-decoder attention in inference and has cached
            k, v = cache["static_k"], cache["static_v"]
        else:
            k, v = self.cal_kv(key, value)

        if cache is not None and "static_k" not in cache:
            # for decoder self-attention in inference
            cache_k, cache_v = cache["k"], cache["v"]
            k = layers.concat([cache_k, k], axis=2)
            v = layers.concat([cache_v, v], axis=2)
            cache["k"], cache["v"] = k, v

        return q, k, v

    def cal_kv(self, key, value):
        """
        Applies linear projection on input keys and values, then splits heads
        (reshape and transpose) to get keys and values from different representation
        subspaces. The results are used as key-values pairs for subsequent multiple
        parallel attention.
        
        It is part of calculations in multi-head attention, and is provided as
        a method to prefetch these results, by which we can use them as cache.

        Parameters:
            key (Variable, optional): The keys for multi-head attention. It is
                a tensor with shape `[batch_size, sequence_length, kdim]`. The
                data type should be float32 or float64.
            value (Variable, optional): The values for multi-head attention. It
                is a tensor with shape `[batch_size, sequence_length, vdim]`.
                The data type should be float32 or float64.

        Returns:
            tuple: A tuple including linear projected keys and values. Their shapes \
                both are `[batch_size, num_heads, sequence_length, embed_dim // num_heads]`. \
                and their data types are same as inputs.
        """
        k = self.k_proj(key)
        v = self.v_proj(value)
        k = layers.reshape(x=k, shape=[0, 0, self.num_heads, self.head_dim])
        k = layers.transpose(x=k, perm=[0, 2, 1, 3])
        v = layers.reshape(x=v, shape=[0, 0, self.num_heads, self.head_dim])
        v = layers.transpose(x=v, perm=[0, 2, 1, 3])
        return k, v

    def forward(self, query, key, value, attn_mask=None, cache=None):
        """
        Applies multi-head attention to map queries and a set of key-value pairs
        to outputs.

        Parameters:
            query (Variable): The queries for multi-head attention. It is a
                tensor with shape `[batch_size, sequence_length, embed_dim]`. The
                data type should be float32 or float64.
            key (Variable, optional): The keys for multi-head attention. It is
                a tensor with shape `[batch_size, sequence_length, kdim]`. The
                data type should be float32 or float64. If None, use `query` as
                `key`. Default None.
            value (Variable, optional): The values for multi-head attention. It
                is a tensor with shape `[batch_size, sequence_length, vdim]`.
                The data type should be float32 or float64. If None, use `query` as
                `value`. Default None.
            attn_mask (Variable, optional): A tensor used in multi-head attention
                to prevents attention to some unwanted positions, usually the
                paddings or the subsequent positions. It is a tensor with shape
                broadcasted to `[batch_size, n_head, sequence_length, sequence_length]`,
                where the unwanted positions have `-INF` values and the others
                have 0 values. The data type should be float32 or float64. It can
                be None when nothing wanted or needed to be prevented attention to.
                Default None
            cache (dict, optional): It is a dict with `k` and `v` as keys or
                `static_k` and `static_v` as keys, and values are tensors shaped
                `[batch_size, num_heads, length, embed_dim]` which are results of
                linear projection, reshape and transpose calculations. If keys are
                `k` and `v`, the values reserve intermediate results of previous
                positions, and would be updated by new tensors concatanating raw
                tensors with results of current position, which mostly used for
                decoder self attention. If keys are `static_k` and `static_v`,
                `key` and `value` args would be ignored, and the values in dict
                would be used as calculated results on `key` and `value`, which
                mostly used for decoder-encoder cross attention. It is only used
                for inference and should be None for training. Default None.

        Returns:
            Variable: The output of multi-head attention. It is a tensor \
                that has the same shape and data type as `queries`.
        """
        key = query if key is None else key
        value = query if value is None else value
        # compute q ,k ,v
        q, k, v = self._prepare_qkv(query, key, value, cache)

        # scale dot product attention
        product = layers.matmul(
            x=q, y=k, transpose_y=True, alpha=self.head_dim**-0.5)
        if attn_mask is not None:
            # TODO(guosheng): support bool mask
            product = product + attn_mask
        weights = layers.softmax(product)
        if self.dropout:
            weights = layers.dropout(
                weights,
                dropout_prob=self.dropout,
                dropout_implementation="upscale_in_train",
                is_test=False)

        out = layers.matmul(weights, v)

        # combine heads
        out = layers.transpose(out, perm=[0, 2, 1, 3])
        out = layers.reshape(x=out, shape=[0, 0, out.shape[2] * out.shape[3]])

        # project to output
        out = self.out_proj(out)
        return (out, weights) if self.need_weights else out


class TransformerEncoderLayer(Layer):
    """
    TransformerEncoderLayer is composed of two sub-layers which are self (multi-head)
    attention and feedforward network. Before and after each sub-layer, pre-process
    and post-precess would be applied on the input and output.

    Parameters:
        n_head (int): The number of heads in multi-head attention(MHA).
        d_key (int): The feature size to transformer queries and keys as in
            multi-head attention. Mostly it equals to `d_model // n_head`.
        d_value (int): The feature size to transformer values as in multi-head
            attention. Mostly it equals to `d_model // n_head`.
        d_model (int): The expected feature size in the input and output.
        d_inner_hid (int): The hidden layer size in the feedforward network(FFN).
        prepostprocess_dropout (float, optional): The dropout probability used
            in pre-process and post-precess of MHA and FFN sub-layer. Default 0.1
        attention_dropout (float, optional): The dropout probability used
            in MHA to drop some attention target. Default 0.1
        relu_dropout (float, optional): The dropout probability used after FFN
            activition. Default 0.1
        preprocess_cmd (str, optional): The process applied before each MHA and
            FFN sub-layer, and it also would be applied on output of the last
            stacked layer. It should be a string composed of `d`, `a`, `n`,
            where `d` for dropout, `a` for add residual connection, `n` for
            layer normalization. Default `n`.
        postprocess_cmd (str, optional): The process applied after each MHA and
            FFN sub-layer. Same as `preprocess_cmd`. It should be a string
            composed of `d`, `a`, `n`, where `d` for dropout, `a` for add
            residual connection, `n` for layer normalization. Default `da`.
        ffn_fc1_act (str, optional): The activation function in the feedforward
            network. Default relu.
         
    Examples:

        .. code-block:: python

            import paddle
            import paddle.fluid as fluid
            from paddle.incubate.hapi.text import TransformerEncoderLayer

            # encoder input: [batch_size, src_len, d_model]
            enc_input = paddle.rand((2, 4, 128))
            # self attention bias: [batch_size, n_head, src_len, src_len]
            attn_bias = paddle.rand((2, 2, 4, 4))
            encoder_layer = TransformerEncoderLayer(2, 64, 64, 128, 512)
            enc_output = encoder_layer(enc_input, attn_bias)  # [2, 4, 128]
    """

    def __init__(self,
                 d_model,
                 nhead,
                 dim_feedforward,
                 dropout=0.1,
                 activation="relu",
                 attn_dropout=0.1,
                 act_dropout=0.1,
                 norm=True):

        super(TransformerEncoderLayer, self).__init__()

        self.preprocesser1 = PrePostProcessLayer(preprocess_cmd, d_model,
                                                 prepostprocess_dropout)
        self.self_attn = MultiHeadAttention(d_key, d_value, d_model, n_head,
                                            attention_dropout)
        self.postprocesser1 = PrePostProcessLayer(postprocess_cmd, d_model,
                                                  prepostprocess_dropout)

        self.preprocesser2 = PrePostProcessLayer(preprocess_cmd, d_model,
                                                 prepostprocess_dropout)
        self.ffn = FFN(d_inner_hid, d_model, relu_dropout, fc1_act=ffn_fc1_act)
        self.postprocesser2 = PrePostProcessLayer(postprocess_cmd, d_model,
                                                  prepostprocess_dropout)

    def forward(self, src, src_mask=None):
        """
        Applies a Transformer encoder layer on the input.

        Parameters:
            enc_input (Variable): The input of Transformer encoder layer. It is
                a tensor with shape `[batch_size, sequence_length, d_model]`.
                The data type should be float32 or float64.
            attn_bias(Variable, optional): A tensor used in encoder self attention
                to mask out attention on unwanted positions, usually the paddings. It
                is a tensor with shape `[batch_size, n_head, sequence_length, sequence_length]`,
                where the unwanted positions have `-INF` values and the others
                have 0 values. The data type should be float32 or float64. It can
                be None when nothing wanted or needed to be masked out. Default None

        Returns:
            Variable: The output of Transformer encoder layer. It is a tensor that \
                has the same shape and data type as `enc_input`.
        """
        attn_output = self.self_attn(
            self.preprocesser1(enc_input), None, None, attn_bias)
        attn_output = self.postprocesser1(attn_output, enc_input)

        ffn_output = self.ffn(self.preprocesser2(attn_output))
        ffn_output = self.postprocesser2(ffn_output, attn_output)
        return ffn_output


class TransformerCell(Layer):
    """
    TransformerCell wraps a Transformer decoder producing logits from `inputs`
    composed by ids and position.

    Parameters:
        decoder(callable): A TransformerDecoder instance. Or a wrapper of it that
            includes a embedding layer accepting ids and positions instead of embeddings
            and includes a output layer transforming decoder output features to logits.
        embedding_fn(function, optional): A callable that accepts ids and position
            as arguments and return embeddings as input of `decoder`. It can be
            None if `decoder` includes a embedding layer. Default None.
        output_fn(callable, optional): A callable applid on `decoder` output to
            transform decoder output features to get logits. Mostly it is a Linear
            layer with vocabulary size. It can be None if `decoder` includes a
            output layer. Default None.

    Examples:

        .. code-block:: python

            import paddle
            import paddle.fluid as fluid
            from paddle.fluid.dygraph import Embedding, Linear
            from paddle.incubate.hapi.text import TransformerDecoder
            from paddle.incubate.hapi.text import TransformerCell
            from paddle.incubate.hapi.text import TransformerBeamSearchDecoder
            from paddle.incubate.hapi.text import DynamicDecode

            paddle.disable_static()

            class Embedder(fluid.dygraph.Layer):
                def __init__(self):
                    super(Embedder, self).__init__()
                    self.word_embedder = Embedding(size=[1000, 128])
                    self.pos_embedder = Embedding(size=[500, 128])

                def forward(self, word, position):
                    return self.word_embedder(word) + self.pos_embedder(position)

            embedder = Embedder()
            output_layer = Linear(128, 1000)
            decoder = TransformerDecoder(2, 2, 64, 64, 128, 512)
            transformer_cell = TransformerCell(decoder, embedder, output_layer)
            dynamic_decoder = DynamicDecode(
                TransformerBeamSearchDecoder(
                    transformer_cell,
                    start_token=0,
                    end_token=1,
                    beam_size=4,
                    var_dim_in_state=2),
                max_step_num=10,
                is_test=True)
            
            enc_output = paddle.rand((2, 4, 128))
            # cross attention bias: [batch_size, n_head, trg_len, src_len]
            trg_src_attn_bias = paddle.rand((2, 2, 1, 4))
            # inputs for beam search on Transformer
            caches = transformer_cell.get_initial_states(enc_output)
            enc_output = TransformerBeamSearchDecoder.tile_beam_merge_with_batch(
                enc_output, beam_size=4)
            trg_src_attn_bias = TransformerBeamSearchDecoder.tile_beam_merge_with_batch(
                trg_src_attn_bias, beam_size=4)
            static_caches = decoder.prepare_static_cache(enc_output)
            outputs = dynamic_decoder(
                inits=caches,
                enc_output=enc_output,
                trg_src_attn_bias=trg_src_attn_bias,
                static_caches=static_caches)
    """

    def __init__(self, decoder, embed_layer=None, output_layer=None):
        super(TransformerCell, self).__init__()
        self.decoder = decoder
        self.embedding_fn = embedding_fn
        self.output_fn = output_fn

    def forward(self,
                inputs,
                states=None,
                enc_output=None,
                trg_slf_attn_bias=None,
                trg_src_attn_bias=None,
                static_caches=[]):
        """
        Produces logits from `inputs` composed by ids and positions.

        Parameters:
            inputs(tuple): A tuple includes target ids and positions. The two
                tensors both have int64 data type and with 2D shape 
                `[batch_size, sequence_length]` where `sequence_length` is 1
                for inference.
            states(list): It caches the multi-head attention intermediate results
                of history decoding steps. It is a list of dict where the length
                of list is decoder layer number, and each dict has `k` and `v` as
                keys and values are cached results. Default None
            enc_output(Variable): The output of Transformer encoder. It is a tensor
                with shape `[batch_size, sequence_length, d_model]`. The data type
                should be float32 or float64.
            trg_slf_attn_bias(Variable, optional): A tensor used in decoder self
                attention to mask out attention on unwanted target positions. It
                is a tensor with shape `[batch_size, n_head, target_length, target_length]`,
                where the unwanted positions have `-INF` values and the others
                have 0 values. It can be None when nothing wanted or needed to
                be masked out. It can be None for inference. The data type should
                be float32 or float64. Default None
            trg_src_attn_bias(Variable, optional): A tensor used in decoder-encoder
                cross attention to mask out unwanted attention on source (encoder output).
                It is a tensor with shape `[batch_size, n_head, target_length, source_length]`,
                where the unwanted positions have `-INF` values and the others
                have 0 values. It can be None when nothing wanted or needed to
                be masked out. The data type should be float32 or float64. Default None
            static_caches(list): It stores projected results of encoder output
                to be used as keys and values in decoder-encoder cross attention
                It is a list of dict where the length of list is decoder layer
                number, and each dict has `static_k` and `static_v` as keys and
                values are stored results. Default empty list

        Returns:
            tuple: A tuple( :code:`(outputs, new_states)` ), where `outputs` \
                is a float32 or float64 3D tensor representing logits shaped \
                `[batch_size, sequence_length, vocab_size]`. `new_states has \
                the same structure and data type with `states` while the length \
                is one larger since the intermediate results of current step are \
                concatenated into it.
        """
        trg_word, trg_pos = inputs
        if states and static_caches:
            for cache, static_cache in zip(states, static_caches):
                cache.update(static_cache)
        if self.embedding_fn is not None:
            dec_input = self.embedding_fn(trg_word, trg_pos)
            outputs = self.decoder(dec_input, enc_output, None,
                                   trg_src_attn_bias, states)
        else:
            outputs = self.decoder(trg_word, trg_pos, enc_output, None,
                                   trg_src_attn_bias, states)
        if self.output_fn is not None:
            outputs = self.output_fn(outputs)

        new_states = [{
            "k": cache["k"],
            "v": cache["v"]
        } for cache in states] if states else states
        return outputs, new_states

    @property
    def state_shape(self):
        """
        States of TransformerCell cache the multi-head attention intermediate
        results of history decoding steps, and have a increasing length as
        decoding continued.
        
        `state_shape` of TransformerCell is used to initialize states. It is a
        list of dict where the length of list is decoder layer, and each dict
        has `k` and `v` as keys and values are `[n_head, 0, d_key]`, `[n_head, 0, d_value]`
        separately. (-1 for batch size would be automatically inserted into shape).

        Returns:
            list: It is a list of dict where the length of list is decoder layer \
                number, and each dict has `k` and `v` as keys and values are cached \
                results.
        """
        return [{
            "k": [self.decoder.n_head, 0, self.decoder.d_key],
            "v": [self.decoder.n_head, 0, self.decoder.d_value],
        } for i in range(self.decoder.n_layer)]


class TransformerBeamSearchDecoder(layers.BeamSearchDecoder):
    """
    Compared with a RNN step :code:`outputs, new_states = cell(inputs, states)`,
    Transformer decoder's `inputs` uses 2D tensor shaped `[batch_size * beam_size, 1]`
    and includes extra position data. And its `states` (caches) has increasing
    length. These are not consistent with `BeamSearchDecoder`, thus subclass
    `BeamSearchDecoder` to make beam search adapt to Transformer decoder.

    Parameters:
        cell(TransformerCell): An instance of `TransformerCell`.
        start_token(int): The start token id.
        end_token(int): The end token id.
        beam_size(int): The beam width used in beam search.
        var_dim_in_state(int): Indicate which dimension of states is variant.

    Examples:

        .. code-block:: python

            import paddle
            import paddle.fluid as fluid
            from paddle.fluid.dygraph import Embedding, Linear
            from paddle.incubate.hapi.text import TransformerDecoder
            from paddle.incubate.hapi.text import TransformerCell
            from paddle.incubate.hapi.text import TransformerBeamSearchDecoder
            from paddle.incubate.hapi.text import DynamicDecode

            paddle.disable_static()

            class Embedder(fluid.dygraph.Layer):
                def __init__(self):
                    super(Embedder, self).__init__()
                    self.word_embedder = Embedding(size=[1000, 128])
                    self.pos_embedder = Embedding(size=[500, 128])

                def forward(self, word, position):
                    return self.word_embedder(word) + self.pos_embedder(position)

            embedder = Embedder()
            output_layer = Linear(128, 1000)
            decoder = TransformerDecoder(2, 2, 64, 64, 128, 512)
            transformer_cell = TransformerCell(decoder, embedder, output_layer)
            dynamic_decoder = DynamicDecode(
                TransformerBeamSearchDecoder(
                    transformer_cell,
                    start_token=0,
                    end_token=1,
                    beam_size=4,
                    var_dim_in_state=2),
                max_step_num=10,
                is_test=True)
            
            enc_output = paddle.rand((2, 4, 128))
            # cross attention bias: [batch_size, n_head, trg_len, src_len]
            trg_src_attn_bias = paddle.rand((2, 2, 1, 4))
            # inputs for beam search on Transformer
            caches = transformer_cell.get_initial_states(enc_output)
            enc_output = TransformerBeamSearchDecoder.tile_beam_merge_with_batch(
                enc_output, beam_size=4)
            trg_src_attn_bias = TransformerBeamSearchDecoder.tile_beam_merge_with_batch(
                trg_src_attn_bias, beam_size=4)
            static_caches = decoder.prepare_static_cache(enc_output)
            outputs = dynamic_decoder(
                inits=caches,
                enc_output=enc_output,
                trg_src_attn_bias=trg_src_attn_bias,
                static_caches=static_caches)
    """

    def __init__(self, cell, start_token, end_token, beam_size,
                 var_dim_in_state):
        super(TransformerBeamSearchDecoder,
              self).__init__(cell, start_token, end_token, beam_size)
        self.cell = cell
        self.var_dim_in_state = var_dim_in_state

    def _merge_batch_beams_with_var_dim(self, x):
        """
        Reshape a tensor with shape `[batch_size, beam_size, ...]` to a new
        tensor with shape `[batch_size * beam_size, ...]`. 

        Parameters:
            x(Variable): A tensor with shape `[batch_size, beam_size, ...]`. The
                data type should be float32, float64, int32, int64 or bool.

        Returns:
            Variable: A tensor with shape `[batch_size * beam_size, ...]`, whose \
                data type is same as `x`.
        """
        # init length of cache is 0, and it increases with decoding carrying on,
        # thus need to reshape elaborately
        var_dim_in_state = self.var_dim_in_state + 1  # count in beam dim
        x = layers.transpose(x,
                             list(range(var_dim_in_state, len(x.shape))) +
                             list(range(0, var_dim_in_state)))
        x = layers.reshape(
            x, [0] * (len(x.shape) - var_dim_in_state
                      ) + [self.batch_size * self.beam_size] +
            [int(size) for size in x.shape[-var_dim_in_state + 2:]])
        x = layers.transpose(
            x,
            list(range((len(x.shape) + 1 - var_dim_in_state), len(x.shape))) +
            list(range(0, (len(x.shape) + 1 - var_dim_in_state))))
        return x

    def _split_batch_beams_with_var_dim(self, x):
        """
        Reshape a tensor with shape `[batch_size * beam_size, ...]` to a new
        tensor with shape `[batch_size, beam_size, ...]`. 

        Parameters:
            x(Variable): A tensor with shape `[batch_size * beam_size, ...]`. The
                data type should be float32, float64, int32, int64 or bool.

        Returns:
            Variable: A tensor with shape `[batch_size, beam_size, ...]`, whose \
                data type is same as `x`.     
        """
        var_dim_size = layers.shape(x)[self.var_dim_in_state]
        x = layers.reshape(
            x, [-1, self.beam_size] +
            [int(size)
             for size in x.shape[1:self.var_dim_in_state]] + [var_dim_size] +
            [int(size) for size in x.shape[self.var_dim_in_state + 1:]])
        return x

    def step(self, time, inputs, states, **kwargs):
        """
        Perform a beam search decoding step, which uses `cell` to get probabilities,
        and follows a beam search step to calculate scores and select candidate
        token ids.

        Note: compared with `BeamSearchDecoder.step`, it feed 2D id tensor shaped
        `[batch_size * beam_size, 1]` rather than `[batch_size * beam_size]` combined
        position data as inputs to `cell`.

        Parameters:
            time(Variable): An `int64` tensor with shape `[1]` provided by the caller,
                representing the current time step number of decoding.
            inputs(Variable): A tensor variable. It is same as `initial_inputs`
                returned by `initialize()` for the first decoding step and
                `next_inputs` returned by `step()` for the others. It is a int64
                id tensor with shape `[batch_size * beam_size]`
            states(Variable): A structure of tensor variables.
                It is same as the `initial_states` returned by `initialize()` for
                the first decoding step and `beam_search_state` returned by
                `step()` for the others.
            **kwargs: Additional keyword arguments, provided by the caller. 
        
        Returns:
            tuple: A tuple( :code:`(beam_search_output, beam_search_state, next_inputs, finished)` ). \
                `beam_search_state` and `next_inputs` have the same structure, \
                shape and data type as the input arguments `states` and `inputs` separately. \
                `beam_search_output` is a namedtuple(including scores, predicted_ids, \
                parent_ids as fields) of tensor variables, where \
                `scores, predicted_ids, parent_ids` all has a tensor value shaped \
                `[batch_size, beam_size]` with data type `float32, int64, int64`. \
                `finished` is a `bool` tensor with shape `[batch_size, beam_size]`.
        """
        # compared to RNN, Transformer has 3D data at every decoding step
        inputs = layers.reshape(inputs, [-1, 1])  # token
        pos = layers.ones_like(inputs) * time  # pos
        cell_states = map_structure(self._merge_batch_beams_with_var_dim,
                                    states.cell_states)

        cell_outputs, next_cell_states = self.cell((inputs, pos), cell_states,
                                                   **kwargs)

        # squeeze to adapt to BeamSearchDecoder which use 2D logits
        cell_outputs = map_structure(
            lambda x: layers.squeeze(x, [1]) if len(x.shape) == 3 else x,
            cell_outputs)
        cell_outputs = map_structure(self._split_batch_beams, cell_outputs)
        next_cell_states = map_structure(self._split_batch_beams_with_var_dim,
                                         next_cell_states)

        beam_search_output, beam_search_state = self._beam_search_step(
            time=time,
            logits=cell_outputs,
            next_cell_states=next_cell_states,
            beam_state=states)
        next_inputs, finished = (beam_search_output.predicted_ids,
                                 beam_search_state.finished)

        return (beam_search_output, beam_search_state, next_inputs, finished)
