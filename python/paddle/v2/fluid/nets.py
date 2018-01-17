#  Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserve.
#
#Licensed under the Apache License, Version 2.0 (the "License");
#you may not use this file except in compliance with the License.
#You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
#Unless required by applicable law or agreed to in writing, software
#distributed under the License is distributed on an "AS IS" BASIS,
#WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#See the License for the specific language governing permissions and
#limitations under the License.
import layers

__all__ = [
    "simple_img_conv_pool",
    "sequence_conv_pool",
    "glu",
    "",
]


def simple_img_conv_pool(input,
                         num_filters,
                         filter_size,
                         pool_size,
                         pool_stride,
                         act,
                         param_attr=None,
                         pool_type='max'):
    conv_out = layers.conv2d(
        input=input,
        num_filters=num_filters,
        filter_size=filter_size,
        param_attr=param_attr,
        act=act)

    pool_out = layers.pool2d(
        input=conv_out,
        pool_size=pool_size,
        pool_type=pool_type,
        pool_stride=pool_stride)
    return pool_out


def img_conv_group(input,
                   conv_num_filter,
                   pool_size,
                   conv_padding=1,
                   conv_filter_size=3,
                   conv_act=None,
                   param_attr=None,
                   conv_with_batchnorm=False,
                   conv_batchnorm_drop_rate=None,
                   pool_stride=1,
                   pool_type=None):
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
            act=local_conv_act)

        if conv_with_batchnorm[i]:
            tmp = layers.batch_norm(input=tmp, act=conv_act)
            drop_rate = conv_batchnorm_drop_rate[i]
            if abs(drop_rate) > 1e-5:
                tmp = layers.dropout(x=tmp, dropout_prob=drop_rate)

    pool_out = layers.pool2d(
        input=tmp,
        pool_size=pool_size,
        pool_type=pool_type,
        pool_stride=pool_stride)
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


def dot_product_attention(querys, keys, values):
    """
    The dot-product attention.

    Attention mechanism can be seen as mapping a query and a set of key-value 
    pairs to an output. The output is computed as a weighted sum of the values, 
    where the weight assigned to each value is computed by a compatibility 
    function (dot-product here) of the query with the corresponding key.
    
    The dot-product attention can be implemented through (batch) matrix 
    multipication as follows:

        .. math::

            Attention(Q, K, V)= softmax(QK^\mathrm{T})V

    Refer to `Attention Is All You Need 
    <https://arxiv.org/pdf/1706.03762.pdf>`_.

    Note that batch data containing sequences with different lengths is not 
    supported by this because of the (batch) matrix multipication.
    
    Args:
        query (Variable): The input variable which is a Tensor or LoDTensor.
        key (Variable): The input variable which is a Tensor or LoDTensor.
        value (Variable): The input variable which is a Tensor or LoDTensor.

    Returns:
        tuple: The Tensor variables representing the output and attention scores.

    Examples:
        .. code-block:: python

            # Suppose q, k, v are tensor variables with the following shape:
            # q: [3, 5, 9], k: [3, 6, 9], v: [3, 6, 10]
            out, attn_scores = fluid.nets.dot_product_attention(q, k, v)
            out.shape  # [3, 5, 10]
            attn_scores.shape  # [3, 5, 6]
    """
    assert keys.shape[-2] == values.shape[
        -2], 'The shapes of keys and values mismatch.'
    assert querys.shape[-1] == keys.shape[
        -1], 'The shapes of querys and keys mismatch.'
    product = layers.matmul(x=querys, y=keys, transpose_y=True)
    attn_scores = layers.reshape(
        x=layers.reshape(
            x=product, shape=[-1, product.shape[-1]], act='softmax'),
        shape=product.shape)
    out = layers.matmul(attn_scores, values)
    return out, attn_scores
