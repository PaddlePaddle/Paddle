#   Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

__all__ = []

from paddle import _C_ops, _legacy_C_ops
from paddle.fluid.framework import dygraph_only


@dygraph_only
def attention(query,
              key,
              value,
              sparse_mask,
              key_padding_mask=None,
              attn_mask=None,
              name=None):
    """
    Note:
        This API is only used from ``CUDA 11.7`` .

    SparseCsrTensor is used to store the intermediate result of Attention matrix
    in Transformer module, which can reduce memory usage and improve performance.
    ``sparse_mask`` express the sparse layout in CSR format.
    The calculation equation is:

    .. math::

        result = softmax(\frac{ Q * K^T }{\sqrt{d}}) * V

    where : ``Q``, ``K``, and ``V`` represent the three input parameters of the attention module.
    The shape of the three parameters are: `[batch_size, num_heads, seq_len, head_dim]`, and
    ``d`` represents ``head_dim`` .

    Args:
        query(DenseTensor): `query` in the Attention module. 4D Tensor with float32 or float64.
        key(DenseTensor): `key` in the Attention module. 4D Tensor with float32 or float64.
        value(DenseTensor): `value` in the Attention module. 4D Tensor with float32 or float64.
        sparse_mask(SparseCsrTensor): The sparse layout in the Attention module. Its dense shape
            is `[batch_size*num_heads, seq_len, seq_len]` .  `nnz` of each batch must be the same.
            dtype of `crows` and `cols` must be int64, dtype of `values` can be float32 or float64.
        key_padding_mask(DenseTensor, optional): The key padding mask tensor in the Attention module.
            2D tensor with shape: [batch_size, seq_len]. dtype can be float32 or float64. Default: None.
        attn_mask(DenseTensor, optional): The attention mask tensor in the Attention module.
            2D tensor with shape: [seq_len, seq_len]. dtype can be float32 or float64. Default: None.
        name(str, optional): The default value is None. Normally there is no need for user
                        to set this property. For more information, please refer to
                        :ref:`api_guide_Name`.

    Returns:
        4D tensor with shape: [batch_size, num_heads, seq_len, head_dim]. dtype is same with input.

    Examples:
        .. code-block:: python

            import paddle

            batch_size = 16
            num_heads = 16
            seq_len = 512
            head_dim = 32

            query = paddle.rand([batch_size, num_heads, seq_len, head_dim])
            key = paddle.rand([batch_size, num_heads, seq_len, head_dim])
            value = paddle.rand([batch_size, num_heads, seq_len, head_dim])

            query.stop_gradient = False
            key.stop_gradient = False
            value.stop_gradient = False

            mask = paddle.nn.functional.dropout(paddle.ones([seq_len, seq_len])).expand([batch_size, num_heads, seq_len, seq_len])
            sp_mask = mask.reshape([-1, seq_len, seq_len]).to_sparse_csr()

            kp_mask = paddle.randint(0, 2, [batch_size, seq_len])
            attn_mask = paddle.randint(0, 2, [seq_len, seq_len])

            output = paddle.incubate.sparse.nn.functional.attention(query, key, value, sp_mask, kp_mask, attn_mask)
            output.backward()
    """
    return _C_ops.sparse_fused_attention(query, key, value, sparse_mask,
                                         key_padding_mask, attn_mask)
