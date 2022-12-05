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

import collections

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import paddle.tensor as tensor
from paddle.distributed.fleet import auto
from paddle.fluid import layers
from paddle.nn.layer.transformer import _convert_param_attr_to_list

paddle.enable_static()


def init_global():
    global _global_parallel_strategy
    _global_parallel_strategy = None
    global _global_process_mesh
    global PP_MESH_LIST
    global DPPP_MESH_LIST
    global MPPP_MESH_LIST
    global DPMPPP_MESH_LIST


class MultiHeadAttention(nn.Layer):
    """
    Attention mapps queries and a set of key-value pairs to outputs, and
    Multi-Head Attention performs multiple parallel attention to jointly attending
    to information from different representation subspaces.
    """

    Cache = collections.namedtuple("Cache", ["k", "v"])
    StaticCache = collections.namedtuple("StaticCache", ["k", "v"])

    def __init__(
        self,
        embed_dim,
        num_heads,
        dropout=0.0,
        kdim=None,
        vdim=None,
        need_weights=False,
        weight_attr=None,
        bias_attr=None,
        fuse=False,
        mesh_idx=None,
        use_new_recompute=False,
        recompute_granularity="full",
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.need_weights = need_weights
        self.fuse = fuse
        self.mesh_idx = mesh_idx
        self.use_new_recompute = use_new_recompute
        self.recompute_granularity = recompute_granularity

        self.head_dim = embed_dim // num_heads
        assert (
            self.head_dim * num_heads == self.embed_dim
        ), "embed_dim must be divisible by num_heads"
        if self.fuse:
            assert self.kdim == embed_dim
            assert self.vdim == embed_dim
            self.qkv_proj = nn.Linear(
                embed_dim, 3 * embed_dim, weight_attr, bias_attr=bias_attr
            )
        else:
            self.q_proj = nn.Linear(
                embed_dim,
                embed_dim,
                weight_attr=weight_attr,
                bias_attr=bias_attr,
            )
            self.k_proj = nn.Linear(
                self.kdim,
                embed_dim,
                weight_attr=weight_attr,
                bias_attr=bias_attr,
            )
            self.v_proj = nn.Linear(
                self.vdim,
                embed_dim,
                weight_attr=weight_attr,
                bias_attr=bias_attr,
            )
        self.out_proj = nn.Linear(
            embed_dim, embed_dim, weight_attr=weight_attr, bias_attr=bias_attr
        )

    def _fuse_prepare_qkv(self, query):
        mix_layer = self.qkv_proj(query)
        mix_layer = paddle.reshape_(
            mix_layer, [0, 0, self.num_heads, 3 * self.head_dim]
        )
        mix_layer = paddle.transpose(mix_layer, [0, 2, 1, 3])
        q, k, v = paddle.split(mix_layer, num_or_sections=3, axis=-1)
        return q, k, v

    def _prepare_qkv(self, query, key, value, use_cache=False, cache=None):
        """
        Prapares linear projected queries, keys and values for usage of subsequnt
        multiple parallel attention. If `cache` is not None, using cached results
        to reduce redundant calculations.
        """
        q = self.q_proj(query)
        if _global_parallel_strategy == "mp":
            auto.shard_tensor(
                self.q_proj.weight, _global_process_mesh, [None, "x"]
            )
        elif _global_parallel_strategy == "dp_mp":
            auto.shard_tensor(
                self.q_proj.weight, _global_process_mesh, [None, "y"]
            )
        elif _global_parallel_strategy == "mp_pp":
            auto.shard_tensor(
                self.q_proj.weight, MPPP_MESH_LIST[self.mesh_idx], [None, "x"]
            )
        elif _global_parallel_strategy == "dp_mp_pp":
            auto.shard_tensor(
                self.q_proj.weight, DPMPPP_MESH_LIST[self.mesh_idx], [None, "y"]
            )

        q = tensor.reshape(x=q, shape=[0, 0, self.num_heads, self.head_dim])
        q = tensor.transpose(x=q, perm=[0, 2, 1, 3])
        if isinstance(cache, self.StaticCache):
            # for encoder-decoder attention in inference and has cached
            k, v = cache.k, cache.v
        else:
            k, v = self.compute_kv(key, value)
        if isinstance(cache, self.Cache):
            # for decoder self-attention in inference
            k = tensor.concat([cache.k, k], axis=2)
            v = tensor.concat([cache.v, v], axis=2)
        if use_cache is True:
            cache = self.Cache(k, v)
        return (q, k, v) if use_cache is False else (q, k, v, cache)

    def compute_kv(self, key, value):
        """
        Applies linear projection on input keys and values, then splits heads
        (reshape and transpose) to get keys and values from different representation
        subspaces. The results are used as key-values pairs for subsequent multiple
        parallel attention.
        It is part of calculations in multi-head attention, and is provided as
        a method to pre-compute and prefetch these results, thus we can use them
        to construct cache for inference.
        """
        k = self.k_proj(key)
        if _global_parallel_strategy == "mp":
            auto.shard_tensor(
                self.k_proj.weight, _global_process_mesh, [None, "x"]
            )
        elif _global_parallel_strategy == "dp_mp":
            auto.shard_tensor(
                self.k_proj.weight, _global_process_mesh, [None, "y"]
            )
        elif _global_parallel_strategy == "mp_pp":
            auto.shard_tensor(
                self.k_proj.weight, MPPP_MESH_LIST[self.mesh_idx], [None, "x"]
            )
        elif _global_parallel_strategy == "dp_mp_pp":
            auto.shard_tensor(
                self.k_proj.weight, DPMPPP_MESH_LIST[self.mesh_idx], [None, "y"]
            )
        v = self.v_proj(value)
        if _global_parallel_strategy == "mp":
            auto.shard_tensor(
                self.v_proj.weight, _global_process_mesh, [None, "x"]
            )
        elif _global_parallel_strategy == "dp_mp":
            auto.shard_tensor(
                self.v_proj.weight, _global_process_mesh, [None, "y"]
            )
        elif _global_parallel_strategy == "mp_pp":
            auto.shard_tensor(
                self.v_proj.weight, MPPP_MESH_LIST[self.mesh_idx], [None, "x"]
            )
        elif _global_parallel_strategy == "dp_mp_pp":
            auto.shard_tensor(
                self.v_proj.weight, DPMPPP_MESH_LIST[self.mesh_idx], [None, "y"]
            )
        k = tensor.reshape(x=k, shape=[0, 0, self.num_heads, self.head_dim])
        k = tensor.transpose(x=k, perm=[0, 2, 1, 3])
        v = tensor.reshape(x=v, shape=[0, 0, self.num_heads, self.head_dim])
        v = tensor.transpose(x=v, perm=[0, 2, 1, 3])
        return k, v

    def gen_cache(self, key, value=None, type=Cache):
        """
        Generates cache for `forward` usage in inference accroding to arguments.
        The generated cache is an instance of `MultiHeadAttention.Cache` or an
        instance of `MultiHeadAttention.StaticCache`.
        """
        if type == MultiHeadAttention.StaticCache:  # static_kv
            k, v = self.compute_kv(key, value)
            return self.StaticCache(k, v)
        elif value is None:  # incremental_state
            k = layers.fill_constant_batch_size_like(
                input=key,
                shape=[-1, self.num_heads, 0, self.head_dim],
                dtype=key.dtype,
                value=0,
            )
            v = layers.fill_constant_batch_size_like(
                input=key,
                shape=[-1, self.num_heads, 0, self.head_dim],
                dtype=key.dtype,
                value=0,
            )
            return self.Cache(k, v)
        else:
            # incremental_state with initial value, mainly for usage like UniLM
            return self.Cache(key, value)

    def core_attn(self, q, k, v, attn_mask):
        product = paddle.matmul(x=q, y=k, transpose_y=True)
        product = paddle.multiply(
            product,
            paddle.to_tensor(self.head_dim**-0.5, dtype=product.dtype),
        )
        if attn_mask is not None:
            product = product + attn_mask
        weights = F.softmax(product)
        if self.dropout:
            weights = F.dropout(
                weights,
                self.dropout,
                training=self.training,
                mode="upscale_in_train",
            )
        out = tensor.matmul(weights, v)
        # combine heads
        out = tensor.transpose(out, perm=[0, 2, 1, 3])
        out = tensor.reshape(x=out, shape=[0, 0, out.shape[2] * out.shape[3]])

        return out, weights

    def forward(
        self, query, key, value, attn_mask=None, use_cache=False, cache=None
    ):
        """
        Applies multi-head attention to map queries and a set of key-value pairs
        to outputs.
        """
        key = query if key is None else key
        value = query if value is None else value
        # compute q ,k ,v
        if use_cache is False:
            if self.fuse:
                q, k, v = self._fuse_prepare_qkv(query)
            else:
                q, k, v = self._prepare_qkv(query, key, value, use_cache, cache)
        else:
            q, k, v, cache = self._prepare_qkv(
                query, key, value, use_cache, cache
            )

        if self.use_new_recompute and self.recompute_granularity == "core_attn":
            out, weights = auto.recompute(self.core_attn)(q, k, v, attn_mask)
        else:
            out, weights = self.core_attn(q, k, v, attn_mask)

        # project to output
        out = self.out_proj(out)
        if _global_parallel_strategy == "mp":
            auto.shard_tensor(
                self.out_proj.weight, _global_process_mesh, ["x", None]
            )
        elif _global_parallel_strategy == "dp_mp":
            auto.shard_tensor(
                self.out_proj.weight, _global_process_mesh, ["y", None]
            )
        elif _global_parallel_strategy == "mp_pp":
            auto.shard_tensor(
                self.out_proj.weight, MPPP_MESH_LIST[self.mesh_idx], ["x", None]
            )
        elif _global_parallel_strategy == "dp_mp_pp":
            auto.shard_tensor(
                self.out_proj.weight,
                DPMPPP_MESH_LIST[self.mesh_idx],
                ["y", None],
            )

        outs = [out]
        if self.need_weights:
            outs.append(weights)
        if use_cache:
            outs.append(cache)
        return out if len(outs) == 1 else tuple(outs)


class TransformerDecoder(nn.Layer):
    """
    TransformerDecoder is a stack of N decoder layers.
    """

    def __init__(
        self,
        decoder_layers,
        num_layers,
        norm=None,
        hidden_size=None,
        use_new_recompute=False,
        recompute_granularity="full",
    ):
        super().__init__()

        self.num_layers = num_layers
        self.layers = decoder_layers
        self.norm = norm
        self.use_new_recompute = use_new_recompute
        self.recompute_granularity = recompute_granularity
        if norm == "LayerNorm":
            self.norm = nn.LayerNorm(hidden_size)
        elif norm is not None:
            raise ValueError("Only support LayerNorm")
        self.checkpoints = []

    def forward(
        self,
        tgt,
        memory,
        tgt_mask=None,
        memory_mask=None,
        use_cache=False,
        cache=None,
    ):
        """
        Applies a stack of N Transformer decoder layers on inputs. If `norm` is
        provided, also applies layer normalization on the output of last decoder
        layer.
        """
        output = tgt
        new_caches = []
        self.checkpoints = []
        if _global_parallel_strategy == "pp":
            auto.shard_tensor(
                output,
                PP_MESH_LIST[0],
                [None for i in range(len(output.shape))],
            )
        if _global_parallel_strategy == "dp_pp":
            auto.shard_tensor(
                output,
                DPPP_MESH_LIST[0],
                ["x"] + [None for i in range(len(output.shape) - 1)],
            )
        if _global_parallel_strategy == "mp_pp":
            auto.shard_tensor(
                output,
                MPPP_MESH_LIST[0],
                [None for i in range(len(output.shape))],
            )
        if _global_parallel_strategy == "dp_mp_pp":
            auto.shard_tensor(
                output,
                DPMPPP_MESH_LIST[0],
                ["x"] + [None for i in range(len(output.shape) - 1)],
            )

        for i, mod in enumerate(self.layers):
            if self.use_new_recompute and self.recompute_granularity == "full":
                mod = auto.recompute(mod)

            if cache is None:
                if use_cache:
                    output, new_cache = mod(
                        output,
                        memory,
                        tgt_mask=tgt_mask,
                        use_cache=use_cache,
                        cache=cache,
                    )
                    new_caches.append(new_cache)
                else:
                    output = mod(output, memory, tgt_mask, use_cache, cache)
            else:
                output, new_cache = mod(
                    output,
                    memory,
                    tgt_mask=tgt_mask,
                    use_cache=use_cache,
                    cache=cache[i],
                )
                new_caches.append(new_cache)

            if not self.use_new_recompute:
                self.checkpoints.append(output.name)

        if self.norm is not None:
            output = self.norm(output)
        return output if use_cache is False else (output, new_caches)

    def gen_cache(self, memory, do_zip=False):
        """
        Generates cache for `forward` usage. The generated cache is a list, and
        each element in it is a tuple( :code:`(incremental_cache, static_cache)` )
        produced by `TransformerDecoderLayer.gen_cache`. See `TransformerDecoderLayer.gen_cache`
        for more details. If `do_zip` is True, apply `zip` on these tuples to get
        a list with two elements.
        """
        cache = [layer.gen_cache(memory) for layer in self.layers]
        if do_zip:
            cache = list(zip(*cache))
        return cache


class TransformerDecoderLayer(nn.Layer):
    """
    The transformer decoder layer.
    It contains multiheadattention and some linear layers.
    """

    def __init__(
        self,
        d_model,
        nhead,
        dim_feedforward,
        dropout=0.1,
        activation="gelu",
        attn_dropout=None,
        act_dropout=None,
        normalize_before=True,
        weight_attr=None,
        bias_attr=None,
        mesh_idx=None,
        use_new_recompute=False,
        recompute_granularity="full",
    ):
        self._config = locals()
        self._config.pop("self")
        self._config.pop("__class__", None)  # py3
        self.mesh_idx = mesh_idx
        super().__init__()
        attn_dropout = dropout if attn_dropout is None else attn_dropout
        act_dropout = dropout if act_dropout is None else act_dropout
        self.normalize_before = normalize_before
        self.use_new_recompute = use_new_recompute
        self.recompute_granularity = recompute_granularity

        weight_attrs = _convert_param_attr_to_list(weight_attr, 3)
        bias_attrs = _convert_param_attr_to_list(bias_attr, 3)

        self.self_attn = MultiHeadAttention(
            d_model,
            nhead,
            dropout=attn_dropout,
            weight_attr=weight_attrs[0],
            bias_attr=bias_attrs[0],
            mesh_idx=self.mesh_idx,
            use_new_recompute=self.use_new_recompute,
            recompute_granularity=self.recompute_granularity,
        )
        self.linear1 = nn.Linear(
            d_model, dim_feedforward, weight_attrs[2], bias_attr=bias_attrs[2]
        )
        self.linear2 = nn.Linear(
            dim_feedforward, d_model, weight_attrs[2], bias_attr=bias_attrs[2]
        )
        self.norm1 = nn.LayerNorm(d_model, epsilon=1e-5)
        self.norm2 = nn.LayerNorm(d_model, epsilon=1e-5)
        self.dropout1 = nn.Dropout(dropout, mode="upscale_in_train")
        self.dropout2 = nn.Dropout(act_dropout, mode="upscale_in_train")
        self.activation = getattr(F, activation)

    def forward(self, tgt, memory, tgt_mask=None, use_cache=False, cache=None):
        residual = tgt
        if self.normalize_before:
            tgt = self.norm1(tgt)

        if self.use_new_recompute and self.recompute_granularity == "full_attn":
            self_attn = auto.recompute(self.self_attn)
        else:
            self_attn = self.self_attn

        if use_cache is False:
            tgt = self_attn(tgt, tgt, tgt, tgt_mask, use_cache, cache)
        else:
            tgt, incremental_cache = self_attn(
                tgt, tgt, tgt, tgt_mask, use_cache, cache
            )

        tgt = residual + self.dropout1(tgt)
        if not self.normalize_before:
            tgt = self.norm1(tgt)
        residual = tgt
        if self.normalize_before:
            tgt = self.norm2(tgt)
        if _global_parallel_strategy == "mp":
            auto.shard_tensor(
                self.linear1.weight, _global_process_mesh, [None, "x"]
            )
        elif _global_parallel_strategy == "dp_mp":
            auto.shard_tensor(
                self.linear1.weight, _global_process_mesh, [None, "y"]
            )
        elif _global_parallel_strategy == "mp_pp":
            auto.shard_tensor(
                self.linear1.weight, MPPP_MESH_LIST[self.mesh_idx], [None, "x"]
            )
        if _global_parallel_strategy == "dp_mp_pp":
            auto.shard_tensor(
                self.linear1.weight,
                DPMPPP_MESH_LIST[self.mesh_idx],
                [None, "y"],
            )

        if _global_parallel_strategy == "mp":
            auto.shard_tensor(
                self.linear2.weight, _global_process_mesh, ["x", None]
            )
        elif _global_parallel_strategy == "dp_mp":
            auto.shard_tensor(
                self.linear2.weight, _global_process_mesh, ["y", None]
            )
        elif _global_parallel_strategy == "mp_pp":
            auto.shard_tensor(
                self.linear2.weight, MPPP_MESH_LIST[self.mesh_idx], ["x", None]
            )
        elif _global_parallel_strategy == "dp_mp_pp":
            auto.shard_tensor(
                self.linear2.weight,
                DPMPPP_MESH_LIST[self.mesh_idx],
                ["y", None],
            )
        tgt = self.dropout2(
            self.linear2(F.gelu(self.linear1(tgt), approximate=True))
        )
        tgt = residual + tgt
        if not self.normalize_before:
            tgt = self.norm2(tgt)
        return tgt if use_cache is False else (tgt, incremental_cache)

    def gen_cache(self, memory):
        incremental_cache = self.self_attn.gen_cache(
            memory, type=self.self_attn.Cache
        )
        return incremental_cache


class GPTEmbeddings(nn.Layer):
    """
    Include embeddings from word, position and token_type embeddings
    """

    def __init__(
        self,
        vocab_size,
        hidden_size=768,
        hidden_dropout_prob=0.1,
        max_position_embeddings=512,
        type_vocab_size=16,
        initializer_range=0.02,
    ):
        super().__init__()
        self.word_embeddings = nn.Embedding(
            vocab_size,
            hidden_size,
            weight_attr=paddle.ParamAttr(
                name="word_embeddings",
                initializer=nn.initializer.Normal(
                    mean=0.0, std=initializer_range
                ),
            ),
        )
        self.position_embeddings = nn.Embedding(
            max_position_embeddings,
            hidden_size,
            weight_attr=paddle.ParamAttr(
                name="pos_embeddings",
                initializer=nn.initializer.Normal(
                    mean=0.0, std=initializer_range
                ),
            ),
        )
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def forward(self, input_ids, position_ids=None):
        if position_ids is None:
            ones = paddle.ones_like(input_ids, dtype="int64")
            seq_length = paddle.cumsum(ones, axis=-1)
            position_ids = seq_length - ones
        input_embedings = self.word_embeddings(input_ids)
        if _global_parallel_strategy == "mp":
            auto.shard_tensor(
                self.word_embeddings.weight, _global_process_mesh, ["x", None]
            )
        elif _global_parallel_strategy == "dp_mp":
            auto.shard_tensor(
                self.word_embeddings.weight, _global_process_mesh, ["y", None]
            )
        elif _global_parallel_strategy == "mp_pp":
            auto.shard_tensor(
                self.word_embeddings.weight, MPPP_MESH_LIST[0], ["x", None]
            )
        elif _global_parallel_strategy == "dp_mp_pp":
            auto.shard_tensor(
                self.word_embeddings.weight, DPMPPP_MESH_LIST[0], ["y", None]
            )

        position_embeddings = self.position_embeddings(position_ids)
        embeddings = input_embedings + position_embeddings
        embeddings = self.dropout(embeddings)
        return embeddings


class GPTModel(nn.Layer):
    """
    The base model of gpt.
    """

    def __init__(
        self,
        vocab_size=50304,
        hidden_size=1024,
        num_hidden_layers=24,
        num_attention_heads=16,
        intermediate_size=4096,
        hidden_act="gelu",
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
        max_position_embeddings=512,
        type_vocab_size=16,
        initializer_range=0.02,
        pad_token_id=0,
        eos_token_id=7,
        bos_token_id=0,
        eol_token_id=3,
        pp_degree=None,
        use_new_recompute=False,
        recompute_granularity="full",
    ):
        super().__init__()
        self.pad_token_id = pad_token_id
        self.initializer_range = initializer_range
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.use_new_recompute = use_new_recompute
        self.recompute_granularity = recompute_granularity

        self.layer_per_stage = None
        self.pipline_mode = pp_degree is not None and pp_degree > 1
        if self.pipline_mode:
            self.layer_per_stage = num_hidden_layers // pp_degree
        self.embeddings = GPTEmbeddings(
            vocab_size,
            hidden_size,
            hidden_dropout_prob,
            max_position_embeddings,
            type_vocab_size,
            self.initializer_range,
        )

        decoder_layers = nn.LayerList()
        for i in range(num_hidden_layers):
            mesh_index = None
            DecoderLayer = TransformerDecoderLayer
            if self.layer_per_stage is not None:
                mesh_index = i // self.layer_per_stage
            decoder_layers.append(
                DecoderLayer(
                    d_model=hidden_size,
                    nhead=num_attention_heads,
                    dim_feedforward=intermediate_size,
                    dropout=hidden_dropout_prob,
                    activation=hidden_act,
                    attn_dropout=attention_probs_dropout_prob,
                    act_dropout=hidden_dropout_prob,
                    weight_attr=paddle.ParamAttr(
                        initializer=nn.initializer.Normal(
                            mean=0.0, std=self.initializer_range
                        )
                    ),
                    bias_attr=None,
                    mesh_idx=mesh_index,
                    use_new_recompute=self.use_new_recompute,
                    recompute_granularity=self.recompute_granularity,
                )
            )

        Decoder = TransformerDecoder
        self.decoder = Decoder(
            decoder_layers,
            num_hidden_layers,
            norm="LayerNorm",
            hidden_size=hidden_size,
            use_new_recompute=self.use_new_recompute,
            recompute_granularity=self.recompute_granularity,
        )
        self.checkpoints = []

    def forward(
        self,
        input_ids,
        position_ids=None,
        attention_mask=None,
        use_cache=False,
        cache=None,
    ):
        self.checkpoints = []
        if position_ids is None:
            past_length = 0
            if cache is not None:
                past_length = paddle.shape(cache[0].k)[-2]
            position_ids = paddle.arange(
                past_length,
                paddle.shape(input_ids)[-1] + past_length,
                dtype='int64',
            )
            position_ids = position_ids.unsqueeze(0)
            position_ids = paddle.expand_as(position_ids, input_ids)
        embedding_output = self.embeddings(
            input_ids=input_ids, position_ids=position_ids
        )
        if _global_parallel_strategy == "pp":
            auto.shard_tensor(
                input_ids,
                PP_MESH_LIST[0],
                [None for i in range(len(input_ids.shape))],
            )
        if _global_parallel_strategy == "dp_pp":
            auto.shard_tensor(
                input_ids,
                DPPP_MESH_LIST[0],
                ["x"] + [None for i in range(len(input_ids.shape) - 1)],
            )
        if _global_parallel_strategy == "dp_mp_pp":
            auto.shard_tensor(
                input_ids,
                DPMPPP_MESH_LIST[0],
                ["x"] + [None for i in range(len(input_ids.shape) - 1)],
            )
        encoder_outputs = self.decoder(
            embedding_output,
            memory=None,
            tgt_mask=attention_mask,
            use_cache=use_cache,
            cache=cache,
        )
        if not self.use_new_recompute:
            self.checkpoints.extend(self.decoder.checkpoints)
        return encoder_outputs


class GPTForPretraining(nn.Layer):
    """
    The pretraining model of GPT.
    It returns some logits and cached_kvs.
    """

    def __init__(
        self,
        gpt,
        vocab_size=50304,
        hidden_size=768,
        initializer_range=0.02,
    ):
        super().__init__()
        self.gpt = gpt

    def forward(
        self,
        input_ids,
        position_ids=None,
        attention_mask=None,
        masked_positions=None,
        use_cache=False,
        cache=None,
    ):
        input_ids.stop_gradient = True
        position_ids.stop_gradient = True
        attention_mask.stop_gradient = True

        outputs = self.gpt(
            input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            use_cache=use_cache,
            cache=cache,
        )
        if use_cache:
            encoder_outputs, cached_kvs = outputs[:2]
        else:
            encoder_outputs = outputs

        x = encoder_outputs
        w = self.gpt.embeddings.word_embeddings.weight

        mesh = None
        if _global_parallel_strategy == "pp":
            mesh = PP_MESH_LIST[-1]
            x_dims_mapping = [None for i in range(len(x.shape))]
            w_dims_mapping = [None for i in range(len(w.shape))]
        elif _global_parallel_strategy == "dp":
            mesh = _global_process_mesh
            x_dims_mapping = ["x"] + [None for i in range(len(x.shape) - 1)]
            w_dims_mapping = [None for i in range(len(w.shape))]
        elif _global_parallel_strategy == "mp":
            mesh = _global_process_mesh
            x_dims_mapping = [None for i in range(len(x.shape))]
            w_dims_mapping = ["x"] + [None for i in range(len(w.shape) - 1)]
        elif _global_parallel_strategy == "dp_mp":
            mesh = _global_process_mesh
            x_dims_mapping = ["x"] + [None for i in range(len(x.shape) - 1)]
            w_dims_mapping = ["y"] + [None for i in range(len(w.shape) - 1)]
        elif _global_parallel_strategy == "dp_pp":
            mesh = DPPP_MESH_LIST[-1]
            x_dims_mapping = ["x"] + [None for i in range(len(x.shape) - 1)]
            w_dims_mapping = [None for i in range(len(w.shape))]
        elif _global_parallel_strategy == "mp_pp":
            mesh = MPPP_MESH_LIST[-1]
            x_dims_mapping = [None for i in range(len(x.shape))]
            w_dims_mapping = ["x"] + [-1 for i in range(len(w.shape) - 1)]
        elif _global_parallel_strategy == "dp_mp_pp":
            mesh = DPMPPP_MESH_LIST[-1]
            x_dims_mapping = ["x"] + [None for i in range(len(x.shape) - 1)]
            w_dims_mapping = ["y"] + [None for i in range(len(w.shape) - 1)]

        if mesh:
            matmul = auto.shard_op(
                paddle.matmul, mesh, [x_dims_mapping, w_dims_mapping, None]
            )
            logits = matmul(x, w, transpose_y=True)
        else:
            logits = paddle.matmul(x, w, transpose_y=True)

        if use_cache:
            return logits, cached_kvs
        else:
            return logits


class GPTPretrainingCriterion(nn.Layer):
    """
    Criterion for GPT.
    It calculates the final loss.
    """

    def __init__(self):
        super().__init__()
        self.loss_func = paddle.nn.CrossEntropyLoss(reduction="none")

    def forward(self, prediction_scores, masked_lm_labels, loss_mask):
        masked_lm_labels.stop_gradient = True
        loss_mask.stop_gradient = True

        mesh = None
        if _global_parallel_strategy == "dp":
            mesh = _global_process_mesh
            dims_mapping = ["x"] + [
                None for i in range(len(loss_mask.shape) - 1)
            ]
        elif _global_parallel_strategy == "dp_mp":
            mesh = _global_process_mesh
            dims_mapping = ["x"] + [
                None for i in range(len(loss_mask.shape) - 1)
            ]
        elif _global_parallel_strategy == "dp_pp":
            mesh = DPPP_MESH_LIST[-1]
            dims_mapping = ["x"] + [
                None for i in range(len(loss_mask.shape) - 1)
            ]
        elif _global_parallel_strategy == "dp_mp_pp":
            mesh = DPMPPP_MESH_LIST[-1]
            dims_mapping = ["x"] + [
                None for i in range(len(loss_mask.shape) - 1)
            ]

        if mesh:
            auto.shard_tensor(loss_mask, mesh, dims_mapping)

        masked_lm_loss = self.loss_func(
            prediction_scores, masked_lm_labels.unsqueeze(2)
        )
        loss_mask = loss_mask.reshape([-1])
        masked_lm_loss = paddle.sum(masked_lm_loss.reshape([-1]) * loss_mask)
        total_loss = masked_lm_loss / loss_mask.sum()
        return total_loss
