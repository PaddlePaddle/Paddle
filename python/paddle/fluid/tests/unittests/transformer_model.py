#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

from functools import partial

import numpy as np

import paddle
import paddle.fluid as fluid
import paddle.fluid.layers as layers

pos_enc_param_names = (
    "src_pos_enc_table",
    "trg_pos_enc_table",
)

batch_size = 2


def position_encoding_init(n_position, d_pos_vec):
    """
    Generate the initial values for the sinusoid position encoding table.
    """
    position_enc = np.array(
        [
            [
                pos / np.power(10000, 2 * (j // 2) / d_pos_vec)
                for j in range(d_pos_vec)
            ]
            if pos != 0
            else np.zeros(d_pos_vec)
            for pos in range(n_position)
        ]
    )
    position_enc[1:, 0::2] = np.sin(position_enc[1:, 0::2])  # dim 2i
    position_enc[1:, 1::2] = np.cos(position_enc[1:, 1::2])  # dim 2i+1
    return position_enc.astype("float32")


def multi_head_attention(
    queries,
    keys,
    values,
    attn_bias,
    d_key,
    d_value,
    d_model,
    n_head=1,
    dropout_rate=0.0,
):
    """
    Multi-Head Attention. Note that attn_bias is added to the logit before
    computing softmax activiation to mask certain selected positions so that
    they will not considered in attention weights.
    """
    if not (len(queries.shape) == len(keys.shape) == len(values.shape) == 3):
        raise ValueError(
            "Inputs: queries, keys and values should all be 3-D tensors."
        )

    def __compute_qkv(queries, keys, values, n_head, d_key, d_value):
        """
        Add linear projection to queries, keys, and values.
        """
        q = layers.fc(
            input=queries,
            size=d_key * n_head,
            param_attr=fluid.initializer.Xavier(
                uniform=False, fan_in=d_model * d_key, fan_out=n_head * d_key
            ),
            bias_attr=False,
            num_flatten_dims=2,
        )
        k = layers.fc(
            input=keys,
            size=d_key * n_head,
            param_attr=fluid.initializer.Xavier(
                uniform=False, fan_in=d_model * d_key, fan_out=n_head * d_key
            ),
            bias_attr=False,
            num_flatten_dims=2,
        )
        v = layers.fc(
            input=values,
            size=d_value * n_head,
            param_attr=fluid.initializer.Xavier(
                uniform=False,
                fan_in=d_model * d_value,
                fan_out=n_head * d_value,
            ),
            bias_attr=False,
            num_flatten_dims=2,
        )
        return q, k, v

    def __split_heads(x, n_head):
        """
        Reshape the last dimension of input tensor x so that it becomes two
        dimensions and then transpose. Specifically, input a tensor with shape
        [bs, max_sequence_length, n_head * hidden_dim] then output a tensor
        with shape [bs, n_head, max_sequence_length, hidden_dim].
        """
        if n_head == 1:
            return x

        hidden_size = x.shape[-1]
        # FIXME(guosheng): Decouple the program desc with batch_size.
        reshaped = paddle.reshape(
            x=x, shape=[batch_size, -1, n_head, hidden_size // n_head]
        )

        # permute the dimensions into:
        # [batch_size, n_head, max_sequence_len, hidden_size_per_head]
        return paddle.transpose(x=reshaped, perm=[0, 2, 1, 3])

    def __combine_heads(x):
        """
        Transpose and then reshape the last two dimensions of input tensor x
        so that it becomes one dimension, which is reverse to __split_heads.
        """
        if len(x.shape) == 3:
            return x
        if len(x.shape) != 4:
            raise ValueError("Input(x) should be a 4-D Tensor.")

        trans_x = paddle.transpose(x, perm=[0, 2, 1, 3])
        # FIXME(guosheng): Decouple the program desc with batch_size.
        return paddle.reshape(
            x=trans_x,
            shape=list(
                map(int, [batch_size, -1, trans_x.shape[2] * trans_x.shape[3]])
            ),
        )

    def scaled_dot_product_attention(q, k, v, attn_bias, d_model, dropout_rate):
        """
        Scaled Dot-Product Attention
        """

        # FIXME(guosheng): Optimize the shape in reshape_op or softmax_op.

        # The current implementation of softmax_op only supports 2D tensor,
        # consequently it cannot be directly used here.
        # If to use the reshape_op, Besides, the shape of product inferred in
        # compile-time is not the actual shape in run-time. It can't be used
        # to set the attribute of reshape_op.
        # So, here define the softmax for temporary solution.

        def __softmax(x, eps=1e-9):
            exp_out = paddle.exp(x=x)
            sum_out = paddle.sum(exp_out, axis=-1, keepdim=True)
            return paddle.divide(x=exp_out, y=sum_out)

        scaled_q = paddle.scale(x=q, scale=d_model**-0.5)
        product = paddle.matmul(x=scaled_q, y=k, transpose_y=True)
        weights = __softmax(paddle.add(x=product, y=attn_bias))
        if dropout_rate:
            weights = paddle.nn.functional.dropout(weights, p=dropout_rate)
        out = paddle.matmul(weights, v)

        return out

    q, k, v = __compute_qkv(queries, keys, values, n_head, d_key, d_value)

    q = __split_heads(q, n_head)
    k = __split_heads(k, n_head)
    v = __split_heads(v, n_head)

    ctx_multiheads = scaled_dot_product_attention(
        q, k, v, attn_bias, d_model, dropout_rate
    )

    out = __combine_heads(ctx_multiheads)

    # Project back to the model size.
    proj_out = layers.fc(
        input=out,
        size=d_model,
        param_attr=fluid.initializer.Xavier(uniform=False),
        bias_attr=False,
        num_flatten_dims=2,
    )
    return proj_out


def positionwise_feed_forward(x, d_inner_hid, d_hid):
    """
    Position-wise Feed-Forward Networks.
    This module consists of two linear transformations with a ReLU activation
    in between, which is applied to each position separately and identically.
    """
    hidden = layers.fc(
        input=x,
        size=d_inner_hid,
        num_flatten_dims=2,
        param_attr=fluid.initializer.Uniform(
            low=-(d_hid**-0.5), high=(d_hid**-0.5)
        ),
        act="relu",
    )
    out = layers.fc(
        input=hidden,
        size=d_hid,
        num_flatten_dims=2,
        param_attr=fluid.initializer.Uniform(
            low=-(d_inner_hid**-0.5), high=(d_inner_hid**-0.5)
        ),
    )
    return out


def pre_post_process_layer(prev_out, out, process_cmd, dropout=0.0):
    """
    Add residual connection, layer normalization and droput to the out tensor
    optionally according to the value of process_cmd.

    This will be used before or after multi-head attention and position-wise
    feed-forward networks.
    """
    for cmd in process_cmd:
        if cmd == "a":  # add residual connection
            out = out + prev_out if prev_out else out
        elif cmd == "n":  # add layer normalization
            out = layers.layer_norm(
                out,
                begin_norm_axis=len(out.shape) - 1,
                param_attr=fluid.initializer.Constant(1.0),
                bias_attr=fluid.initializer.Constant(0.0),
            )
        elif cmd == "d":  # add dropout
            if dropout:
                out = paddle.nn.functional.dropout(out, p=dropout)
    return out


pre_process_layer = partial(pre_post_process_layer, None)
post_process_layer = pre_post_process_layer


def prepare_encoder(
    src_word,
    src_pos,
    src_vocab_size,
    src_emb_dim,
    src_pad_idx,
    src_max_len,
    dropout=0.0,
    pos_pad_idx=0,
    pos_enc_param_name=None,
):
    """Add word embeddings and position encodings.
    The output tensor has a shape of:
    [batch_size, max_src_length_in_batch, d_model].

    This module is used at the bottom of the encoder stacks.
    """
    src_word_emb = layers.embedding(
        src_word,
        size=[src_vocab_size, src_emb_dim],
        padding_idx=src_pad_idx,
        param_attr=fluid.initializer.Normal(0.0, 1.0),
    )
    src_pos_enc = layers.embedding(
        src_pos,
        size=[src_max_len, src_emb_dim],
        padding_idx=pos_pad_idx,
        param_attr=fluid.ParamAttr(name=pos_enc_param_name, trainable=False),
    )
    src_pos_enc.stop_gradient = True
    enc_input = src_word_emb + src_pos_enc

    # FIXME(guosheng): Decouple the program desc with batch_size.
    enc_input = paddle.reshape(x=enc_input, shape=[batch_size, -1, src_emb_dim])
    return (
        paddle.nn.functional.dropout(enc_input, p=dropout)
        if dropout
        else enc_input
    )


prepare_encoder = partial(
    prepare_encoder, pos_enc_param_name=pos_enc_param_names[0]
)
prepare_decoder = partial(
    prepare_encoder, pos_enc_param_name=pos_enc_param_names[1]
)


def encoder_layer(
    enc_input,
    attn_bias,
    n_head,
    d_key,
    d_value,
    d_model,
    d_inner_hid,
    dropout_rate=0.0,
):
    """The encoder layers that can be stacked to form a deep encoder.

    This module consits of a multi-head (self) attention followed by
    position-wise feed-forward networks and both the two components companied
    with the post_process_layer to add residual connection, layer normalization
    and droput.
    """
    attn_output = multi_head_attention(
        enc_input,
        enc_input,
        enc_input,
        attn_bias,
        d_key,
        d_value,
        d_model,
        n_head,
        dropout_rate,
    )
    attn_output = post_process_layer(
        enc_input, attn_output, "dan", dropout_rate
    )
    ffd_output = positionwise_feed_forward(attn_output, d_inner_hid, d_model)
    return post_process_layer(attn_output, ffd_output, "dan", dropout_rate)


def encoder(
    enc_input,
    attn_bias,
    n_layer,
    n_head,
    d_key,
    d_value,
    d_model,
    d_inner_hid,
    dropout_rate=0.0,
):
    """
    The encoder is composed of a stack of identical layers returned by calling
    encoder_layer.
    """
    for i in range(n_layer):
        enc_output = encoder_layer(
            enc_input,
            attn_bias,
            n_head,
            d_key,
            d_value,
            d_model,
            d_inner_hid,
            dropout_rate,
        )
        enc_input = enc_output
    return enc_output


def decoder_layer(
    dec_input,
    enc_output,
    slf_attn_bias,
    dec_enc_attn_bias,
    n_head,
    d_key,
    d_value,
    d_model,
    d_inner_hid,
    dropout_rate=0.0,
):
    """The layer to be stacked in decoder part.

    The structure of this module is similar to that in the encoder part except
    a multi-head attention is added to implement encoder-decoder attention.
    """
    slf_attn_output = multi_head_attention(
        dec_input,
        dec_input,
        dec_input,
        slf_attn_bias,
        d_key,
        d_value,
        d_model,
        n_head,
        dropout_rate,
    )
    slf_attn_output = post_process_layer(
        dec_input,
        slf_attn_output,
        "dan",  # residual connection + dropout + layer normalization
        dropout_rate,
    )
    enc_attn_output = multi_head_attention(
        slf_attn_output,
        enc_output,
        enc_output,
        dec_enc_attn_bias,
        d_key,
        d_value,
        d_model,
        n_head,
        dropout_rate,
    )
    enc_attn_output = post_process_layer(
        slf_attn_output,
        enc_attn_output,
        "dan",  # residual connection + dropout + layer normalization
        dropout_rate,
    )
    ffd_output = positionwise_feed_forward(
        enc_attn_output,
        d_inner_hid,
        d_model,
    )
    dec_output = post_process_layer(
        enc_attn_output,
        ffd_output,
        "dan",  # residual connection + dropout + layer normalization
        dropout_rate,
    )
    return dec_output


def decoder(
    dec_input,
    enc_output,
    dec_slf_attn_bias,
    dec_enc_attn_bias,
    n_layer,
    n_head,
    d_key,
    d_value,
    d_model,
    d_inner_hid,
    dropout_rate=0.0,
):
    """
    The decoder is composed of a stack of identical decoder_layer layers.
    """
    for i in range(n_layer):
        dec_output = decoder_layer(
            dec_input,
            enc_output,
            dec_slf_attn_bias,
            dec_enc_attn_bias,
            n_head,
            d_key,
            d_value,
            d_model,
            d_inner_hid,
            dropout_rate,
        )
        dec_input = dec_output
    return dec_output


def build_inputs(max_length, n_head):
    names = [
        'src_word',
        'src_pos',
        'trg_word',
        'trg_pos',
        'src_slf_attn_bias',
        'trg_slf_attn_bias',
        'trg_src_attn_bias',
        'gold',
        'weights',
    ]

    shapes = [
        [batch_size * max_length, 1],
        [batch_size * max_length, 1],
        [batch_size * max_length, 1],
        [batch_size * max_length, 1],
        [batch_size, n_head, max_length, max_length],
        [batch_size, n_head, max_length, max_length],
        [batch_size, n_head, max_length, max_length],
        [batch_size * max_length, 1],
        [batch_size * max_length, 1],
    ]

    dtypes = [
        'int64',
        'int64',
        'int64',
        'int64',
        'float32',
        'float32',
        'float32',
        'int64',
        'float32',
    ]

    all_inputs = []
    for name, shape, dtype in zip(names, shapes, dtypes):
        all_inputs.append(
            fluid.layers.data(
                name=name, shape=shape, dtype=dtype, append_batch_size=False
            )
        )
    return all_inputs


def transformer(
    src_vocab_size,
    trg_vocab_size,
    max_length,
    n_layer,
    n_head,
    d_key,
    d_value,
    d_model,
    d_inner_hid,
    dropout_rate,
    src_pad_idx,
    trg_pad_idx,
    pos_pad_idx,
):

    (
        src_word,
        src_pos,
        trg_word,
        trg_pos,
        src_slf_attn_bias,
        trg_slf_attn_bias,
        trg_src_attn_bias,
        gold,
        weights,
    ) = build_inputs(max_length, n_head)

    enc_input = prepare_encoder(
        src_word,
        src_pos,
        src_vocab_size,
        d_model,
        src_pad_idx,
        max_length,
        dropout_rate,
    )
    enc_output = encoder(
        enc_input,
        src_slf_attn_bias,
        n_layer,
        n_head,
        d_key,
        d_value,
        d_model,
        d_inner_hid,
        dropout_rate,
    )

    dec_input = prepare_decoder(
        trg_word,
        trg_pos,
        trg_vocab_size,
        d_model,
        trg_pad_idx,
        max_length,
        dropout_rate,
    )
    dec_output = decoder(
        dec_input,
        enc_output,
        trg_slf_attn_bias,
        trg_src_attn_bias,
        n_layer,
        n_head,
        d_key,
        d_value,
        d_model,
        d_inner_hid,
        dropout_rate,
    )

    # TODO(guosheng): Share the weight matrix between the embedding layers and
    # the pre-softmax linear transformation.
    predict = paddle.reshape(
        x=layers.fc(
            input=dec_output,
            size=trg_vocab_size,
            param_attr=fluid.initializer.Xavier(uniform=False),
            bias_attr=False,
            num_flatten_dims=2,
        ),
        shape=[-1, trg_vocab_size],
    )
    predict = paddle.nn.functional.softmax(predict)

    cost = paddle.nn.functional.cross_entropy(
        input=predict, label=gold, reduction='none', use_softmax=False
    )
    weighted_cost = cost * weights
    return paddle.sum(weighted_cost)
