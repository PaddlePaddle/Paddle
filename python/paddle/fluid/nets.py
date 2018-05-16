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
import layers

__all__ = [
    "simple_img_conv_pool",
    "sequence_conv_pool",
    "glu",
    "scaled_dot_product_attention",
]


def simple_img_conv_pool(input,
                         num_filters,
                         filter_size,
                         pool_size,
                         pool_stride,
                         act,
                         param_attr=None,
                         pool_type='max',
                         use_cudnn=True,
                         use_mkldnn=False):
    conv_out = layers.conv2d(
        input=input,
        num_filters=num_filters,
        filter_size=filter_size,
        param_attr=param_attr,
        act=act,
        use_cudnn=use_cudnn,
        use_mkldnn=use_mkldnn)

    pool_out = layers.pool2d(
        input=conv_out,
        pool_size=pool_size,
        pool_type=pool_type,
        pool_stride=pool_stride,
        use_cudnn=use_cudnn,
        use_mkldnn=use_mkldnn)
    return pool_out


def img_conv_group(input,
                   conv_num_filter,
                   pool_size,
                   conv_padding=1,
                   conv_filter_size=3,
                   conv_act=None,
                   param_attr=None,
                   conv_with_batchnorm=False,
                   conv_batchnorm_drop_rate=0.0,
                   pool_stride=1,
                   pool_type=None,
                   use_cudnn=True,
                   use_mkldnn=False):
    """
    Image Convolution Group, Used for vgg net.
    """
    tmp = input
    assert isinstance(conv_num_filter, list) or \
        isinstance(conv_num_filter, tuple)

    def __extend_list__(obj):
        if not hasattr(obj, '__len__'):
            return [obj] * len(conv_num_filter)
        else:
            return obj

    conv_padding = __extend_list__(conv_padding)
    conv_filter_size = __extend_list__(conv_filter_size)
    param_attr = __extend_list__(param_attr)
    conv_with_batchnorm = __extend_list__(conv_with_batchnorm)
    conv_batchnorm_drop_rate = __extend_list__(conv_batchnorm_drop_rate)

    for i in xrange(len(conv_num_filter)):
        local_conv_act = conv_act
        if conv_with_batchnorm[i]:
            local_conv_act = None

        tmp = layers.conv2d(
            input=tmp,
            num_filters=conv_num_filter[i],
            filter_size=conv_filter_size[i],
            padding=conv_padding[i],
            param_attr=param_attr[i],
            act=local_conv_act,
            use_cudnn=use_cudnn,
            use_mkldnn=use_mkldnn)

        if conv_with_batchnorm[i]:
            tmp = layers.batch_norm(input=tmp, act=conv_act, in_place=True)
            drop_rate = conv_batchnorm_drop_rate[i]
            if abs(drop_rate) > 1e-5:
                tmp = layers.dropout(x=tmp, dropout_prob=drop_rate)

    pool_out = layers.pool2d(
        input=tmp,
        pool_size=pool_size,
        pool_type=pool_type,
        pool_stride=pool_stride,
        use_cudnn=use_cudnn,
        use_mkldnn=use_mkldnn)
    return pool_out


def sequence_conv_pool(input,
                       num_filters,
                       filter_size,
                       param_attr=None,
                       act="sigmoid",
                       pool_type="max"):
    conv_out = layers.sequence_conv(
        input=input,
        num_filters=num_filters,
        filter_size=filter_size,
        param_attr=param_attr,
        act=act)

    pool_out = layers.sequence_pool(input=conv_out, pool_type=pool_type)
    return pool_out


def glu(input, dim=-1):
    """
    The gated linear unit composed by split, sigmoid activation and elementwise
    multiplication. Specifically, Split the input into two equal sized parts
    :math:`a` and :math:`b` along the given dimension and then compute as
    following:

        .. math::

            {GLU}(a, b)= a \otimes \sigma(b)

    Refer to `Language Modeling with Gated Convolutional Networks
    <https://arxiv.org/pdf/1612.08083.pdf>`_.

    Args:
        input (Variable): The input variable which is a Tensor or LoDTensor.
        dim (int): The dimension along which to split. If :math:`dim < 0`, the
            dimension to split along is :math:`rank(input) + dim`.

    Returns:
        Variable: The Tensor variable with half the size of input.

    Examples:
        .. code-block:: python

            # x is a Tensor variable with shape [3, 6, 9]
            fluid.nets.glu(input=x, dim=1)  # shape of output: [3, 3, 9]
    """

    a, b = layers.split(input, num_or_sections=2, dim=dim)
    act_b = layers.sigmoid(x=b)
    out = layers.elementwise_mul(x=a, y=act_b)
    return out


def scaled_dot_product_attention(queries,
                                 keys,
                                 values,
                                 num_heads=1,
                                 dropout_rate=0.):
    """
    The dot-product attention.

    Attention mechanism can be seen as mapping a query and a set of key-value
    pairs to an output. The output is computed as a weighted sum of the values,
    where the weight assigned to each value is computed by a compatibility
    function (dot-product here) of the query with the corresponding key.

    The dot-product attention can be implemented through (batch) matrix
    multipication as follows:

        .. math::

            Attention(Q, K, V)= softmax(QK^\mathrm{T})V

    Refer to `Attention Is All You Need
    <https://arxiv.org/pdf/1706.03762.pdf>`_.

    Args:

        queries (Variable): The input variable which should be a 3-D Tensor.
        keys (Variable): The input variable which should be a 3-D Tensor.
        values (Variable): The input variable which should be a 3-D Tensor.
        num_heads (int): Head number to compute the scaled dot product
                         attention. Default value is 1.
        dropout_rate (float): The dropout rate to drop the attention weight.
                              Default value is 0.

    Returns:

        Variable: A 3-D Tensor computed by multi-head scaled dot product \
                  attention.

    Raises:

        ValueError: If input queries, keys, values are not 3-D Tensors.

    NOTE:
        1. When num_heads > 1, three linear projections are learned respectively
        to map input queries, keys and values into queries', keys' and values'.
        queries', keys' and values' have the same shapes with queries, keys
        and values.

        1. When num_heads == 1, scaled_dot_product_attention has no learnable
        parameters.

    Examples:
        .. code-block:: python

            # Suppose q, k, v are Tensors with the following shape:
            # q: [3, 5, 9], k: [3, 6, 9], v: [3, 6, 10]

            contexts = fluid.nets.scaled_dot_product_attention(q, k, v)
            contexts.shape  # [3, 5, 10]
    """
    if not (len(queries.shape) == len(keys.shape) == len(values.shape) == 3):
        raise ValueError(
            "Inputs quries, keys and values should all be 3-D tensors.")

    if queries.shape[-1] != keys.shape[-1]:
        raise ValueError(
            "The hidden size of queries and keys should be the same.")
    if keys.shape[-2] != values.shape[-2]:
        raise ValueError(
            "The max sequence length in query batch and in key batch "
            "should be the same.")
    if keys.shape[-1] % num_heads != 0:
        raise ValueError("The hidden size of keys (%d) must be divisible "
                         "by the number of attention heads (%d)." %
                         (keys.shape[-1], num_heads))
    if values.shape[-1] % num_heads != 0:
        raise ValueError("The hidden size of values (%d) must be divisible "
                         "by the number of attention heads (%d)." %
                         (values.shape[-1], num_heads))

    def __compute_qkv(queries, keys, values, num_heads):
        """
        Add linear projection to queries, keys, and values.

        Args:
            queries(Tensor): a 3-D input Tensor.
            keys(Tensor): a 3-D input Tensor.
            values(Tensor): a 3-D input Tensor.
            num_heads(int): The number of heads. Linearly project the inputs
                            ONLY when num_heads > 1.

        Returns:
            Tensor: linearly projected output Tensors: queries', keys' and
                    values'. They have the same shapes with queries, keys and
                    values.
        """

        if num_heads == 1:
            return queries, keys, values

        q = layers.fc(input=queries, size=queries.shape[-1], num_flatten_dims=2)
        k = layers.fc(input=keys, size=keys.shape[-1], num_flatten_dims=2)
        v = layers.fc(input=values, size=values.shape[-1], num_flatten_dims=2)
        return q, k, v

    def __split_heads(x, num_heads):
        """
        Reshape the last dimension of inpunt tensor x so that it becomes two
        dimensions.

        Args:
            x(Tensor): a 3-D input Tensor.
            num_heads(int): The number of heads.

        Returns:
            Tensor: a Tensor with shape [..., n, m/num_heads], where m is size
                    of the last dimension of x.
        """
        if num_heads == 1:
            return x

        hidden_size = x.shape[-1]
        # reshape the 3-D input: [batch_size, max_sequence_length, hidden_dim]
        # into a 4-D output:
        # [batch_size, max_sequence_length, num_heads, hidden_size_per_head].
        reshaped = layers.reshape(
            x=x,
            shape=list(x.shape[:-1]) + [num_heads, hidden_size // num_heads])

        # permuate the dimensions into:
        # [batch_size, num_heads, max_sequence_len, hidden_size_per_head]
        return layers.transpose(x=reshaped, perm=[0, 2, 1, 3])

    def __combine_heads(x):
        """
        Reshape the last two dimensions of inpunt tensor x so that it becomes
        one dimension.

        Args:
            x(Tensor): a 4-D input Tensor with shape
                       [bs, num_heads, max_sequence_length, hidden_dim].

        Returns:
            Tensor: a Tensor with shape
                    [bs, max_sequence_length, num_heads * hidden_dim].
        """

        if len(x.shape) == 3: return x
        if len(x.shape) != 4:
            raise ValueError("Input(x) should be a 4-D Tensor.")

        trans_x = layers.transpose(x, perm=[0, 2, 1, 3])
        return layers.reshape(
            x=trans_x,
            shape=map(int, [
                trans_x.shape[0], trans_x.shape[1],
                trans_x.shape[2] * trans_x.shape[3]
            ]))

    q, k, v = __compute_qkv(queries, keys, values, num_heads)

    q = __split_heads(q, num_heads)
    k = __split_heads(k, num_heads)
    v = __split_heads(v, num_heads)

    key_dim_per_head = keys.shape[-1] // num_heads
    scaled_q = layers.scale(x=q, scale=key_dim_per_head**-0.5)
    product = layers.matmul(x=k, y=scaled_q, transpose_y=True)

    weights = layers.reshape(
        x=layers.reshape(
            x=product, shape=[-1, product.shape[-1]], act="softmax"),
        shape=product.shape)
    if dropout_rate:
        weights = layers.dropout(
            weights, dropout_prob=dropout_rate, is_test=False)
    ctx_multiheads = layers.matmul(weights, v)
    return __combine_heads(ctx_multiheads)
