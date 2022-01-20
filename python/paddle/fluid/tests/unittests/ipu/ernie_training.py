# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

# refrenece : https://github.com/PaddlePaddle/PaddleNLP/tree/develop/examples/language_model/ernie

import os
import copy
import argparse
from contextlib import contextmanager
from functools import partial

import numpy as np
import paddle
import paddle.static
import paddle.fluid as fluid
import paddle.fluid.layers as layers
import paddle.fluid.compiler as compiler
paddle.enable_static()

SEED = 2021
INT_DTYPE = None

# ernie related block 
ernie_config = {
    "emb_size": 128,
    "emb_mapping_in": False,
    "hidden_size": 192,
    "num_hidden_layers": 2,
    "n_layer_per_block": 2,
    "num_attention_heads": 12,
    "vocab_size": 300,
    "max_position_embeddings": 512,
    "sent_type_vocab_size": 4,
    "task_type_vocab_size": 16,
    "hidden_act": "gelu",
    "hidden_dropout_prob": 0.0,
    "attention_probs_dropout_prob": 0.0,
    "preln": False,
    "pre_encoder_cmd": "n",
    "preprocess_cmd": "",
    "postprocess_cmd": "an",
    "epsilon": 1e-12,
    "initializer_range": 0.02,
    "seq_len": 32
}


def gelu(x):
    """Gaussian Error Linear Unit.

    This is a smoother version of the RELU.
    Original paper: https://arxiv.org/abs/1606.08415
    Args:
      x: float Tensor to perform activation.

    Returns:
      `x` with the GELU activation applied.
    """
    cdf = 0.5 * (1.0 + fluid.layers.tanh(
        (np.sqrt(2.0 / np.pi) * (x + 0.044715 * fluid.layers.pow(x, 3.0)))))
    return x * cdf


def pre_post_process_layer(prev_out,
                           out,
                           process_cmd,
                           dropout_rate=0.,
                           epsilon=1e-12,
                           name=''):
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
                param_attr=fluid.ParamAttr(
                    name=name + '_layer_norm_scale',
                    initializer=fluid.initializer.Constant(1.)),
                bias_attr=fluid.ParamAttr(
                    name=name + '_layer_norm_bias',
                    initializer=fluid.initializer.Constant(0.)),
                epsilon=epsilon)
        elif cmd == "d":  # add dropout
            if dropout_rate:
                out = layers.dropout(
                    out,
                    dropout_prob=dropout_rate,
                    dropout_implementation="upscale_in_train",
                    is_test=False)
    return out


pre_process_layer = partial(pre_post_process_layer, None)
post_process_layer = pre_post_process_layer


def positionwise_feed_forward(x,
                              d_inner_hid,
                              d_hid,
                              dropout_rate,
                              hidden_act,
                              param_initializer=None,
                              name='ffn'):
    """
    Position-wise Feed-Forward Networks.
    This module consists of two linear transformations with a ReLU activation
    in between, which is applied to each position separately and identically.
    """

    #assert hidden_act == 'gelu.approximate'
    hidden = layers.fc(input=x,
                       size=d_inner_hid,
                       num_flatten_dims=2,
                       act=None,
                       param_attr=fluid.ParamAttr(
                           name=name + '_fc_0.w_0',
                           initializer=param_initializer),
                       bias_attr=name + '_fc_0.b_0')
    hidden = gelu(hidden)

    if dropout_rate:
        hidden = layers.dropout(
            hidden,
            dropout_prob=dropout_rate,
            dropout_implementation="upscale_in_train",
            is_test=False)

    out = layers.fc(input=hidden,
                    size=d_hid,
                    num_flatten_dims=2,
                    param_attr=fluid.ParamAttr(
                        name=name + '_fc_1.w_0', initializer=param_initializer),
                    bias_attr=name + '_fc_1.b_0')

    return out


def multi_head_attention(queries,
                         keys,
                         values,
                         attn_bias,
                         d_key,
                         d_value,
                         d_model,
                         n_head=1,
                         dropout_rate=0.,
                         cache=None,
                         param_initializer=None,
                         name='multi_head_att'):
    """
    Multi-Head Attention. Note that attn_bias is added to the logit before
    computing softmax activiation to mask certain selected positions so that
    they will not considered in attention weights.
    """
    keys = queries if keys is None else keys
    values = keys if values is None else values

    def __compute_qkv(queries, keys, values, n_head, d_key, d_value):
        """
        Add linear projection to queries, keys, and values.
        """
        q = layers.fc(input=queries,
                      size=d_key * n_head,
                      num_flatten_dims=2,
                      param_attr=fluid.ParamAttr(
                          name=name + '_query_fc.w_0',
                          initializer=param_initializer),
                      bias_attr=name + '_query_fc.b_0')
        k = layers.fc(input=keys,
                      size=d_key * n_head,
                      num_flatten_dims=2,
                      param_attr=fluid.ParamAttr(
                          name=name + '_key_fc.w_0',
                          initializer=param_initializer),
                      bias_attr=name + '_key_fc.b_0')
        v = layers.fc(input=values,
                      size=d_value * n_head,
                      num_flatten_dims=2,
                      param_attr=fluid.ParamAttr(
                          name=name + '_value_fc.w_0',
                          initializer=param_initializer),
                      bias_attr=name + '_value_fc.b_0')

        return q, k, v

    def __split_heads(x, n_head):
        """
        Reshape the last dimension of inpunt tensor x so that it becomes two
        dimensions and then transpose. Specifically, input a tensor with shape
        [bs, max_sequence_length, n_head * hidden_dim] then output a tensor
        with shape [bs, n_head, max_sequence_length, hidden_dim].
        """
        hidden_size = x.shape[-1]
        # The value 0 in shape attr means copying the corresponding dimension
        # size of the input as the output dimension size.
        reshaped = layers.reshape(
            x=x, shape=[0, 0, n_head, hidden_size // n_head], inplace=False)

        # permuate the dimensions into:
        # [batch_size, n_head, max_sequence_len, hidden_size_per_head]
        return layers.transpose(x=reshaped, perm=[0, 2, 1, 3])

    def __combine_heads(x):
        """
        Transpose and then reshape the last two dimensions of inpunt tensor x
        so that it becomes one dimension, which is reverse to __split_heads.
        """
        if len(x.shape) == 3: return x
        if len(x.shape) != 4:
            raise ValueError("Input(x) should be a 4-D Tensor.")

        trans_x = layers.transpose(x, perm=[0, 2, 1, 3])
        # The value 0 in shape attr means copying the corresponding dimension
        # size of the input as the output dimension size.
        return layers.reshape(
            x=trans_x,
            shape=[0, 0, trans_x.shape[2] * trans_x.shape[3]],
            inplace=False)

    def scaled_dot_product_attention(q, k, v, attn_bias, d_key, dropout_rate):
        """
        Scaled Dot-Product Attention
        """
        scaled_q = layers.scale(x=q, scale=d_key**-0.5)
        product = layers.matmul(x=scaled_q, y=k, transpose_y=True)

        if attn_bias:
            product += attn_bias
        weights = layers.softmax(product)
        if dropout_rate:
            weights = layers.dropout(
                weights,
                dropout_prob=dropout_rate,
                dropout_implementation="upscale_in_train",
                is_test=False)
        out = layers.matmul(weights, v)
        return out

    q, k, v = __compute_qkv(queries, keys, values, n_head, d_key, d_value)

    if cache is not None:  # use cache and concat time steps
        # Since the inplace reshape in __split_heads changes the shape of k and
        # v, which is the cache input for next time step, reshape the cache
        # input from the previous time step first.
        k = cache["k"] = layers.concat(
            [layers.reshape(
                cache["k"], shape=[0, 0, d_model]), k], axis=1)
        v = cache["v"] = layers.concat(
            [layers.reshape(
                cache["v"], shape=[0, 0, d_model]), v], axis=1)

    q = __split_heads(q, n_head)
    k = __split_heads(k, n_head)
    v = __split_heads(v, n_head)

    ctx_multiheads = scaled_dot_product_attention(q, k, v, attn_bias, d_key,
                                                  dropout_rate)

    out = __combine_heads(ctx_multiheads)

    # Project back to the model size.
    proj_out = layers.fc(input=out,
                         size=d_model,
                         num_flatten_dims=2,
                         param_attr=fluid.ParamAttr(
                             name=name + '_output_fc.w_0',
                             initializer=param_initializer),
                         bias_attr=name + '_output_fc.b_0')

    return proj_out


def encoder_layer(enc_input,
                  attn_bias,
                  n_head,
                  d_key,
                  d_value,
                  d_model,
                  d_inner_hid,
                  prepostprocess_dropout,
                  attention_dropout,
                  relu_dropout,
                  hidden_act,
                  preprocess_cmd="n",
                  postprocess_cmd="da",
                  param_initializer=None,
                  name='',
                  epsilon=1e-12):
    """The encoder layers that can be stacked to form a deep encoder.
    This module consits of a multi-head (self) attention followed by
    position-wise feed-forward networks and both the two components companied
    with the post_process_layer to add residual connection, layer normalization
    and droput.
    """

    attn_output = multi_head_attention(
        enc_input,
        None,
        None,
        attn_bias,
        d_key,
        d_value,
        d_model,
        n_head,
        attention_dropout,
        param_initializer=param_initializer,
        name=name + '_multi_head_att')

    attn_output = post_process_layer(
        enc_input,
        attn_output,
        'an',
        prepostprocess_dropout,
        name=name + '_post_att',
        epsilon=epsilon)

    ffd_output = positionwise_feed_forward(
        attn_output,
        d_inner_hid,
        d_model,
        relu_dropout,
        hidden_act,
        param_initializer=param_initializer,
        name=name + '_ffn')

    post_output = post_process_layer(
        attn_output,
        ffd_output,
        'an',
        prepostprocess_dropout,
        name=name + '_post_ffn',
        epsilon=epsilon)

    return post_output


def encoder_inner_share(enc_input,
                        attn_bias,
                        n_head,
                        d_key,
                        d_value,
                        d_model,
                        d_inner_hid,
                        prepostprocess_dropout,
                        attention_dropout,
                        relu_dropout,
                        hidden_act,
                        preprocess_cmd,
                        postprocess_cmd,
                        epsilon,
                        param_initializer=None,
                        name='',
                        n_layer_per_block=1):
    """
       The encoder_inner_share is composed of n_layer_per_block layers returned by calling
       encoder_layer.
    """

    for i in range(n_layer_per_block):
        enc_output = encoder_layer(
            enc_input,
            attn_bias,
            n_head,
            d_key,
            d_value,
            d_model,
            d_inner_hid,
            prepostprocess_dropout,
            attention_dropout,
            relu_dropout,
            hidden_act,
            preprocess_cmd,
            postprocess_cmd,
            param_initializer=param_initializer,
            name=name + '_layer_' + str(i),
            epsilon=epsilon)

        enc_input = enc_output

    return enc_output


def encoder(enc_input,
            attn_bias,
            n_layer,
            n_head,
            d_key,
            d_value,
            d_model,
            d_inner_hid,
            prepostprocess_dropout,
            attention_dropout,
            relu_dropout,
            hidden_act,
            preprocess_cmd,
            postprocess_cmd,
            epsilon,
            n_layer_per_block,
            param_initializer=None,
            name='',
            preln=False):
    """
    The encoder is composed of a stack of identical layers returned by calling
    encoder_layer .
    """

    for _ in range(n_layer // n_layer_per_block):
        attn_bias.stop_gradient = True
        attn_bias.persistable = False
        enc_output = encoder_inner_share(
            enc_input,
            attn_bias,
            n_head,
            d_key,
            d_value,
            d_model,
            d_inner_hid,
            prepostprocess_dropout,
            attention_dropout,
            relu_dropout,
            hidden_act,
            preprocess_cmd,
            postprocess_cmd,
            epsilon,
            param_initializer=param_initializer,
            name=name,
            n_layer_per_block=n_layer_per_block)

        enc_input = enc_output

    if preln:
        enc_output = post_process_layer(
            None,
            enc_output,
            'n',
            prepostprocess_dropout,
            name='post_encoder',
            epsilon=epsilon)

    enc_output = pre_process_layer(
        enc_output,
        preprocess_cmd,
        prepostprocess_dropout,
        name="post_encoder",
        epsilon=epsilon)

    return enc_output


class ErnieModel(object):
    def __init__(self, src_ids, sent_ids, pos_ids, input_mask, config):

        self._emb_size = config['emb_size'] if config[
            'emb_mapping_in'] else config['hidden_size']
        self._hidden_size = config['hidden_size']
        self._n_layer = config['num_hidden_layers']
        self._n_head = config['num_attention_heads']
        self._voc_size = config['vocab_size']
        self._max_position_seq_len = config['max_position_embeddings']
        self._sent_types = config['sent_type_vocab_size']
        self._task_types = config['task_type_vocab_size']
        self._hidden_act = config['hidden_act']
        self._prepostprocess_dropout = config['hidden_dropout_prob']
        self._attention_dropout = config['attention_probs_dropout_prob']
        self.config = config
        self.preln = config['preln'] if 'preln' in config.keys() else False
        self.pre_encoder_cmd = "" if self.preln else self.config[
            'pre_encoder_cmd']

        self._word_emb_name = "word_embedding"
        self._pos_emb_name = "pos_embedding"
        self._sent_emb_name = "sent_embedding"
        self._task_emb_name = "task_embedding"
        self._dtype = "float32"
        self._emb_dtype = "float32"

        # Initialize all weigths by truncated normal initializer, and all biases
        # will be initialized by constant zero by default.
        self._param_initializer = fluid.initializer.TruncatedNormal(
            scale=config['initializer_range'])

        self.src_ids = src_ids
        self.sent_ids = sent_ids
        self.pos_ids = pos_ids
        self.input_mask = input_mask
        '''
        _build_position_ids: range op doesn't support
        _build_input_mask: logic_not op doesn't support
        '''

        self._build_model()

    def _build_model(self, emb=None):
        with fluid.ipu_shard(ipu_index=0, ipu_stage=0):
            # padding id in vocabulary must be set to 0
            self.emb_out = fluid.layers.embedding(
                input=self.src_ids,
                size=[self._voc_size, self._emb_size],
                dtype=self._emb_dtype,
                param_attr=fluid.ParamAttr(
                    name=self._word_emb_name,
                    initializer=self._param_initializer),
                is_sparse=False)

            self.position_emb_out = fluid.layers.embedding(
                input=self.pos_ids,
                size=[self._max_position_seq_len, self._emb_size],
                dtype=self._emb_dtype,
                param_attr=fluid.ParamAttr(
                    name=self._pos_emb_name,
                    initializer=self._param_initializer))

            self.sent_emb_out = fluid.layers.embedding(
                self.sent_ids,
                size=[self._sent_types, self._emb_size],
                dtype=self._emb_dtype,
                param_attr=fluid.ParamAttr(
                    name=self._sent_emb_name,
                    initializer=self._param_initializer))

            sum_emb = self.emb_out + self.position_emb_out + self.sent_emb_out

            sum_emb = pre_process_layer(
                sum_emb,
                self.config['pre_encoder_cmd'],
                self._prepostprocess_dropout,
                name='pre_encoder',
                epsilon=self.config['epsilon'])

            if self.config['emb_mapping_in']:
                sum_emb = fluid.layers.fc(
                    input=sum_emb,
                    num_flatten_dims=2,
                    size=self._hidden_size,
                    param_attr=fluid.ParamAttr(
                        name='emb_hidden_mapping',
                        initializer=self._param_initializer),
                    bias_attr='emb_hidden_mapping_bias')

            self_attn_mask = fluid.layers.matmul(
                x=self.input_mask, y=self.input_mask, transpose_y=True)

            self_attn_mask = fluid.layers.scale(
                x=self_attn_mask,
                scale=10000.0,
                bias=-1.0,
                bias_after_scale=False)

        with fluid.ipu_shard(ipu_index=1, ipu_stage=1):
            n_head_self_attn_mask = fluid.layers.stack(
                x=[self_attn_mask] * self._n_head,
                axis=1)  # [bs, _n_head, seqlen, seq_len]
            n_head_self_attn_mask.stop_gradient = True

            self._enc_out = encoder(
                enc_input=sum_emb,
                attn_bias=n_head_self_attn_mask,
                n_layer=self._n_layer,
                n_head=self._n_head,
                d_key=self._hidden_size // self._n_head,
                d_value=self._hidden_size // self._n_head,
                d_model=self._hidden_size,
                d_inner_hid=self._hidden_size * 4,
                prepostprocess_dropout=self._prepostprocess_dropout,
                attention_dropout=self._attention_dropout,
                relu_dropout=0,
                hidden_act=self._hidden_act,
                preprocess_cmd=self.config['preprocess_cmd'],
                postprocess_cmd=self.config['postprocess_cmd'],
                param_initializer=self._param_initializer,
                name='encoder',
                epsilon=self.config['epsilon'],
                n_layer_per_block=self.config['n_layer_per_block'],
                preln=self.preln)

    def _build_position_ids(self):
        d_shape = fluid.layers.shape(self.src_ids)
        d_seqlen = d_shape[1]
        d_batch = d_shape[0]
        position_ids = fluid.layers.reshape(
            fluid.layers.range(
                0, d_seqlen, 1, dtype='int32'), [1, d_seqlen, 1],
            inplace=False)
        position_ids = fluid.layers.expand(position_ids, [d_batch, 1, 1])
        position_ids = fluid.layers.cast(position_ids, INT_DTYPE)
        position_ids.stop_gradient = True
        return position_ids

    def _build_input_mask(self):
        zero = fluid.layers.fill_constant([1], dtype=INT_DTYPE, value=0)
        input_mask = fluid.layers.logical_not(
            fluid.layers.equal(self.src_ids, zero))  # assume pad id == 0
        input_mask = fluid.layers.cast(input_mask, 'float32')
        input_mask.stop_gradient = True
        return input_mask

    def get_sequence_output(self):
        return self._enc_out

    def get_pooled_output(self):
        """Get the first feature of each sequence for classification"""
        next_sent_feat = fluid.layers.slice(
            input=self._enc_out, axes=[1], starts=[0], ends=[1])

        next_sent_feat = fluid.layers.fc(
            input=next_sent_feat,
            size=self._hidden_size,
            act="tanh",
            param_attr=fluid.ParamAttr(
                name="pooled_fc.w_0", initializer=self._param_initializer),
            bias_attr="pooled_fc.b_0")
        return next_sent_feat

    def get_next_sentence_output(self, labels):
        next_sent_feat = self.get_pooled_output()
        next_sent_fc_out = fluid.layers.fc(
            input=next_sent_feat,
            num_flatten_dims=1,
            size=33,
            param_attr=fluid.ParamAttr(
                name="next_sent_fc.w_0", initializer=self._param_initializer),
            bias_attr="next_sent_fc.b_0")
        next_sent_fc_out = fluid.layers.reshape(
            next_sent_fc_out, [-1, 33], inplace=False)
        #next_sent_loss, next_sent_softmax = fluid.layers.softmax_with_cross_entropy(
        #    logits=next_sent_fc_out, label=labels, return_softmax=True)
        next_sent_softmax = fluid.layers.softmax(next_sent_fc_out)
        next_sent_loss = fluid.layers.cross_entropy(next_sent_softmax, labels)
        next_sent_acc = fluid.layers.accuracy(
            input=next_sent_softmax, label=labels)
        mean_next_sent_loss = fluid.layers.mean(next_sent_loss,
                                                "mean_next_sent_loss")
        return next_sent_acc, mean_next_sent_loss

    def get_lm_output(self, mask_label, mask_pos):
        """Get the loss & accuracy for pretraining"""
        mask_pos = fluid.layers.cast(x=mask_pos, dtype='int32')

        # extract the first token feature in each sentence
        reshaped_emb_out = fluid.layers.reshape(
            x=self._enc_out, shape=[-1, self._hidden_size])

        # extract masked tokens' feature
        mask_feat = fluid.layers.gather(input=reshaped_emb_out, index=mask_pos)
        if self._dtype == "float16":
            mask_feat = fluid.layers.cast(x=mask_feat, dtype=self._emb_dtype)

        # transform: fc
        if self._hidden_act == 'gelu' or self._hidden_act == 'gelu.precise':
            _hidden_act = 'gelu'
        else:
            _hidden_act = None

        mask_trans_feat = fluid.layers.fc(
            input=mask_feat,
            size=self._emb_size,
            act=_hidden_act,
            param_attr=fluid.ParamAttr(
                name='mask_lm_trans_fc.w_0',
                initializer=self._param_initializer),
            bias_attr=fluid.ParamAttr(name='mask_lm_trans_fc.b_0'))

        if self._hidden_act == 'gelu' or self._hidden_act == 'gelu.precise':
            pass
        else:
            mask_trans_feat = gelu(mask_trans_feat)

        # transform: layer norm
        mask_trans_feat = fluid.layers.layer_norm(
            mask_trans_feat,
            begin_norm_axis=len(mask_trans_feat.shape) - 1,
            param_attr=fluid.ParamAttr(
                name='mask_lm_trans_layer_norm_scale',
                initializer=fluid.initializer.Constant(1.)),
            bias_attr=fluid.ParamAttr(
                name='mask_lm_trans_layer_norm_bias',
                initializer=fluid.initializer.Constant(0.)),
            epsilon=self.config['epsilon'])

        mask_lm_out_bias_attr = fluid.ParamAttr(
            name="mask_lm_out_fc.b_0",
            initializer=fluid.initializer.Constant(value=0.0))

        fc_out = fluid.layers.fc(input=mask_trans_feat,
                                 size=self._voc_size,
                                 param_attr=fluid.ParamAttr(
                                     name="mask_lm_out_fc.w_0",
                                     initializer=self._param_initializer),
                                 bias_attr=mask_lm_out_bias_attr)
        #mask_lm_loss = fluid.layers.softmax_with_cross_entropy(
        #    logits=fc_out, label=mask_label)
        mask_lm_softmax = fluid.layers.softmax(fc_out)
        mask_lm_loss = fluid.layers.cross_entropy(mask_lm_softmax, mask_label)
        mean_mask_lm_loss = fluid.layers.mean(
            mask_lm_loss, name="mean_mask_lm_loss")

        return mask_lm_loss, mean_mask_lm_loss

    def get_task_output(self, task, task_labels):
        task_fc_out = fluid.layers.fc(input=self.next_sent_feat,
                                      size=task["num_labels"],
                                      param_attr=fluid.ParamAttr(
                                          name=task["task_name"] + "_fc.w_0",
                                          initializer=self._param_initializer),
                                      bias_attr=task["task_name"] + "_fc.b_0")
        #task_loss, task_softmax = fluid.layers.softmax_with_cross_entropy(
        #    logits=task_fc_out, label=task_labels, return_softmax=True)
        task_softmax = fluid.layers.softmax(task_fc_out)
        task_loss = fluid.layers.cross_entropy(task_softmax, task_labels)
        task_acc = fluid.layers.accuracy(input=task_softmax, label=task_labels)
        mean_task_loss = fluid.layers.mean(task_loss)
        return mean_task_loss, task_acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument(
        "--run_on_ipu", type=bool, default=True, help="Run model with IPU")
    parser.add_argument(
        "--is_training", type=bool, default=True, help="Train of inference")
    parser.add_argument(
        "--num_ipus", type=int, default=2, help="Number of ipus")
    parser.add_argument(
        "--enable_pipelining", type=bool, default=False, help="Pipelining")
    parser.add_argument(
        "--save_model", type=bool, default=False, help="Save model or not")
    parser.add_argument(
        "--model_path", type=str, default="ernie", help="Save model to where")
    parser.add_argument(
        "--model_name", type=str, default="ernie", help="Save model name")
    parser.add_argument(
        "--ipu_run_steps", type=int, default=10, help="Number steps exe.run()")
    parser.add_argument(
        "--export_ops", type=bool, default=False, help="Export ops to ops.txt")
    parser.add_argument(
        "--export_ipu_idx", type=bool, default=False, help="Export op-idx pair")
    args = parser.parse_args()

    # set random seed
    np.random.seed(SEED)
    paddle.static.default_startup_program().random_seed = SEED
    paddle.static.default_main_program().random_seed = SEED

    # IPU doesn't support int64, so we change here
    INT_DTYPE = "int32" if args.run_on_ipu else "int64"

    # paddle input placeholder, batch_size = 1
    micro_bs = 1
    seq_len = ernie_config["seq_len"]
    input_shape = [micro_bs, seq_len, 1]
    input_fields = {
        'names': [
            'src_ids', 'sent_ids', 'pos_ids', 'input_mask', 'mask_label',
            'mask_pos'
        ],
        'shapes': [
            input_shape, input_shape, input_shape, input_shape, [micro_bs, 1],
            [micro_bs, 1]
        ],
        'dtypes':
        [INT_DTYPE, INT_DTYPE, INT_DTYPE, 'float32', INT_DTYPE, INT_DTYPE],
        'range': [[0, seq_len], [0, 4], [0, seq_len], None, [0, seq_len],
                  [0, seq_len]],
        'lod_levels': [0, 0, 0, 0, 0, 0],
    }

    inputs = [
        fluid.data(
            name=input_fields['names'][i],
            shape=input_fields['shapes'][i],
            dtype=input_fields['dtypes'][i],
            lod_level=input_fields['lod_levels'][i])
        for i in range(len(input_fields['names']))
    ]

    # total_samples: assum disable pipelining
    batches_per_step = 1
    if args.enable_pipelining:
        batches_per_step = \
            ((args.num_ipus+1) if args.is_training else args.num_ipus)
    total_samples = args.ipu_run_steps * batches_per_step

    total_steps = args.ipu_run_steps
    if not args.run_on_ipu:  # run on cpu
        total_steps = total_samples // micro_bs

    # synthetic data
    np_inputs = []
    for i in range(len(input_fields['names'])):
        field_name = input_fields['names'][i]
        if field_name == 'input_mask':
            src_ids = np_inputs[0]
            dtype = input_fields['dtypes'][i]
            data = np.where(src_ids > 0,
                            np.ones_like(src_ids),
                            np.zeros_like(src_ids)).astype(dtype)
        else:
            shape = copy.copy(input_fields['shapes'][i])
            shape[0] = total_samples
            min_val, max_val = input_fields['range'][i]
            data = np.random.randint(
                min_val, max_val, shape, dtype=input_fields['dtypes'][i])
        np_inputs.append(data)

    # paddle input placeholder
    (src_ids, sent_ids, pos_ids, input_mask, mask_label, mask_pos) = inputs

    # ernie model
    ernie = ErnieModel(src_ids, sent_ids, pos_ids, input_mask, ernie_config)
    fetch_node = ernie.get_sequence_output()
    if args.is_training:
        with fluid.ipu_shard(ipu_index=1, ipu_stage=1):
            _, mean_mask_lm_loss = ernie.get_lm_output(mask_label, mask_pos)
            fetch_node = mean_mask_lm_loss
            adam = paddle.optimizer.Adam(learning_rate=1e-2)
            adam.minimize(mean_mask_lm_loss)

    # place = paddle.CPUPlace()
    if args.run_on_ipu:
        place = paddle.IPUPlace()
    else:
        place = paddle.CPUPlace()
    executor = paddle.static.Executor(place)

    # feed & fetch list
    if args.is_training:
        feed_list = input_fields['names']
    else:
        feed_list = input_fields['names'][:4]
    fetch_list = [fetch_node.name]

    # program
    startup_prog = paddle.static.default_startup_program()
    executor.run(startup_prog)

    main_prog = paddle.static.default_main_program()
    paddle.static.save(main_prog, "model/ernie")
    paddle.static.load(main_prog, "model/ernie")

    if args.run_on_ipu:
        ipu_strategy = paddle.static.IpuStrategy()
        ipu_strategy.SetGraphConfig(
            num_ipus=args.num_ipus,
            is_training=args.is_training,
            enable_manual_shard=args.num_ipus > 1)
        ipu_strategy.SetPipeliningConfig(
            enable_pipelining=args.enable_pipelining,
            batches_per_step=args.num_ipus + 1)

        ipu_compiler = compiler.IPUCompiledProgram(
            main_prog, ipu_strategy=ipu_strategy)
        program = ipu_compiler.compile(feed_list, fetch_list)
    else:
        program = main_prog

    # executor run
    results = []
    for i in range(total_steps):
        start = i * (batches_per_step if args.run_on_ipu else 1)
        end = start + (batches_per_step if args.run_on_ipu else 1)
        feed_dict = {
            src_ids.name: np_inputs[0][start:end],
            sent_ids.name: np_inputs[1][start:end],
            pos_ids.name: np_inputs[2][start:end],
            input_mask.name: np_inputs[3][start:end]
        }
        if args.is_training:
            feed_dict[mask_label.name] = np_inputs[4][start:end]
            feed_dict[mask_pos.name] = np_inputs[5][start:end]

        res = executor.run(program, feed=feed_dict, fetch_list=[fetch_node])
        results.append(res)

    paddle.static.save(main_prog, "model/ernie")

    results = np.asarray(results).flatten()
    if results.size > 32:
        results = results[-32:]
    print(results)

    if args.save_model:
        full_name = args.model_path + '/' + args.model_name
        if args.is_training:
            fluid.save(program=main_prog, model_path=full_name)
        else:
            with fluid.ipu_shard(ipu_index=1, ipu_stage=1):
                paddle.static.save_inference_model(
                    full_name, [src_ids, sent_ids, pos_ids, input_mask],
                    [fetch_node], executor)

    if args.export_ops:
        op_type_list = []
        for op in main_prog.global_block().ops:
            op_type_list.append(op.desc.type())

        with open("ops.txt", "w") as fp:
            for op_type in set(op_type_list):
                fp.write(op_type + os.linesep)

    if args.export_ipu_idx:
        op_ipu_idx_list = []
        for op in main_prog.global_block().ops:
            if op._is_backward_op():
                continue

            op_ipu_idx_pair = [op.desc.type()]
            if op.desc.has_attr("ipu_index"):
                op_ipu_idx_pair.append(op.desc.attr("ipu_index"))
            else:
                op_ipu_idx_pair.append(-1)  # not assign ipu_index
            op_ipu_idx_list.append(op_ipu_idx_pair)
        op_ipu_idx_list.sort(key=lambda item: item[-1])

        with open("ops_ipu_idx.txt", "w") as fp:
            for op_ipu_idx_pair in op_ipu_idx_list:
                fp.write(str(op_ipu_idx_pair) + os.linesep)
