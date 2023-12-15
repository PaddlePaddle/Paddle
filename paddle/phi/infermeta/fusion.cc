/* Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/phi/infermeta/fusion.h"
#include <unordered_set>
#include <vector>
#include "paddle/common/layout.h"
#include "paddle/phi/common/scalar.h"
#include "paddle/phi/core/infermeta_utils.h"
#include "paddle/phi/core/meta_tensor.h"
#include "paddle/phi/kernels/cpu/conv_util.h"
#include "paddle/phi/kernels/funcs/blas/blas.h"
#include "paddle/phi/kernels/funcs/common_shape.h"
#include "paddle/phi/kernels/funcs/concat_funcs.h"
#include "paddle/phi/kernels/funcs/strided_slice.h"

namespace phi {

static phi::DDim BroadCastInferShape(const DDim x_dims,
                                     const DDim y_dims,
                                     int axis) {
  std::vector<int> out_dims_array(x_dims.size(), -1);
  if (x_dims != y_dims) {
    int max_dim = std::max(x_dims.size(), y_dims.size());
    if (x_dims.size() == y_dims.size()) {
      PADDLE_ENFORCE_EQ((axis == -1) || (axis == 0),
                        true,
                        phi::errors::InvalidArgument(
                            "axis should be -1 or 0 while the dimension of "
                            "tensor X (%s) is equal to the dimension of "
                            "tensor Y (%s), but received axis: %s",
                            x_dims.size(),
                            y_dims.size(),
                            axis));
    }
    PADDLE_ENFORCE_EQ((axis >= (-1 * max_dim)) && (axis < max_dim),
                      true,
                      phi::errors::InvalidArgument(
                          "The axis range must be [%s, %s), but axis is %s. "
                          "Please set the axis again.",
                          -1 * max_dim,
                          max_dim,
                          axis));
    axis = (axis < 0 ? (std::abs(x_dims.size() - y_dims.size()) + axis + 1)
                     : axis);
    std::vector<int> x_dims_array(max_dim);
    std::vector<int> y_dims_array(max_dim);
    out_dims_array.resize(max_dim);
    funcs::GetBroadcastDimsArrays(x_dims,
                                  y_dims,
                                  x_dims_array.data(),
                                  y_dims_array.data(),
                                  out_dims_array.data(),
                                  max_dim,
                                  axis);

    return common::make_ddim(out_dims_array);
  }
  return x_dims;
}

void AddActXPUInferMeta(const MetaTensor& x,
                        const MetaTensor& x_max,
                        const MetaTensor& y,
                        const MetaTensor& y_max,
                        int act_type,
                        MetaTensor* out,
                        MetaTensor* out_max) {
  int axis = -1;
  auto x_dims = x.dims();
  auto y_dims = y.dims();
  if (x_dims != y_dims) {
    auto out_dims = BroadCastInferShape(x_dims, y_dims, axis);
    out->set_dims(out_dims);
  } else {
    out->set_dims(x_dims);
  }
  out->set_dtype(x.dtype());
  out->set_layout(x.layout());
  out->share_lod(x);
  out_max->set_dims(common::make_ddim({6}));
  out_max->set_dtype(x.dtype());
  out_max->set_layout(x.layout());
}

void AddLayernormXPUInferMeta(const MetaTensor& x,
                              const MetaTensor& y,
                              const MetaTensor& scale,
                              const MetaTensor& bias,
                              int begin_norm_axis,
                              float epsilon,
                              MetaTensor* out) {
  int axis = -1;
  auto x_dims = x.dims();
  auto y_dims = y.dims();
  auto out_dims = x_dims;
  if (x_dims != y_dims) {
    out_dims = BroadCastInferShape(x_dims, y_dims, axis);
    out->set_dims(out_dims);
  } else {
    out->set_dims(out_dims);
  }
  out->set_dtype(x.dtype());
  out->set_layout(x.layout());
  out->share_lod(x);
}

void BlockMultiheadAttentionInferMeta(const MetaTensor& qkv,
                                      const MetaTensor& key_cache,
                                      const MetaTensor& value_cache,
                                      const MetaTensor& seq_lens_encoder,
                                      const MetaTensor& seq_lens_decoder,
                                      const MetaTensor& seq_lens_this_time,
                                      const MetaTensor& padding_offsets,
                                      const MetaTensor& cum_offsets,
                                      const MetaTensor& cu_seqlens_q,
                                      const MetaTensor& cu_seqlens_k,
                                      const MetaTensor& block_tables,
                                      const MetaTensor& pre_key_cache,
                                      const MetaTensor& pre_value_cache,
                                      const MetaTensor& rope_emb,
                                      const MetaTensor& mask,
                                      const MetaTensor& tgt_mask,
                                      int max_seq_len,
                                      int block_size,
                                      bool use_neox_style,
                                      MetaTensor* fmha_out,
                                      MetaTensor* qkv_out,
                                      MetaTensor* key_cache_out,
                                      MetaTensor* value_cache_out) {
  auto input_dims = qkv.dims();
  auto key_cache_dims = key_cache.dims();
  const int num_head = key_cache_dims[1];
  const int dim_head = key_cache_dims[3];

  PADDLE_ENFORCE_EQ(
      input_dims.size(),
      2UL,
      errors::InvalidArgument("The input(qkv) must be a 2D Tensor."));
  PADDLE_ENFORCE_EQ(
      key_cache_dims.size(),
      4UL,
      errors::InvalidArgument("The input(key_cache) must be a 4D Tensor."));
  PADDLE_ENFORCE_EQ(
      3 * num_head * dim_head,
      input_dims[1],
      errors::InvalidArgument(
          "The input_dims[1] must be equal to 3 * num_head * dim_head"));

  fmha_out->set_dims({input_dims[0], num_head * dim_head});
  fmha_out->set_dtype(qkv.dtype());
  qkv_out->set_dims(qkv.dims());
  qkv_out->set_dtype(qkv.dtype());
  key_cache_out->set_dims(key_cache_dims);
  key_cache_out->set_dtype(key_cache.dtype());
  value_cache_out->set_dims(key_cache_dims);
  value_cache_out->set_dtype(value_cache.dtype());
}

void Conv1dXPUInferMeta(const MetaTensor& x,
                        const MetaTensor& x_max,
                        const MetaTensor& filter,
                        const MetaTensor& filter_max,
                        const MetaTensor& bias,
                        const MetaTensor& branch,
                        const MetaTensor& branch_max,
                        const std::vector<int>& paddings,
                        const std::string& padding_algorithm,
                        int dilations,
                        int strides,
                        int groups,
                        int act_type,
                        float act_param,
                        MetaTensor* out,
                        MetaTensor* out_max) {
  auto in_dims = x.dims();
  auto filter_dims = filter.dims();
  // do some checks
  PADDLE_ENFORCE_EQ(
      in_dims.size(),
      3,
      phi::errors::InvalidArgument(
          "The input of Op(Conv_xpu) should be a 3-D Tensor. But "
          "received: input's dimension is %u, input's shape is [%s].",
          in_dims.size(),
          in_dims));

  PADDLE_ENFORCE_EQ(
      in_dims.size(),
      filter_dims.size(),
      phi::errors::InvalidArgument(
          "The input's dimension and filter's dimension of "
          "Op(Conv_xpu) should be equal. But received: the input's shape is "
          "[%s], "
          "the input's dimension is %d; the filter's shape is [%s],  "
          "the filter's dimension is %d.",
          in_dims,
          in_dims.size(),
          filter_dims,
          filter_dims.size()));

  const auto input_channels = in_dims[1];

  PADDLE_ENFORCE_GT(
      dilations,
      0,
      phi::errors::InvalidArgument(
          "The dilation of Op(Conv) should be larget than 0, but received "
          "dilation is %d.",
          dilations));

  PADDLE_ENFORCE_EQ(
      input_channels,
      filter_dims[1] * groups,
      phi::errors::InvalidArgument(
          "The number of input's channels should be equal to filter's channels "
          "* groups for Op(Conv_xpu). But received: the input's channels is "
          "%d, "
          "the input's shape is [%s]; the filter's channels is %d, the "
          "filter's shape is [%s]; the groups is %d. ",
          input_channels,
          in_dims,
          filter_dims[1],
          filter_dims,
          groups));

  PADDLE_ENFORCE_EQ(
      filter_dims[0] % groups,
      0,
      phi::errors::InvalidArgument(
          "The number of output's channels (filter's first dimension) of "
          "Op(Conv) should be divided by groups. But received: "
          "the output channels is %d, the filter's shape is [%s], "
          "the groups is %d.",
          filter_dims[0],
          filter_dims,
          groups));

  std::vector<int64_t> out_shape({in_dims[0], filter_dims[0]});
  out_shape.push_back(ConvOutSize(static_cast<int>(in_dims[2]),
                                  static_cast<int>(filter_dims[2]),
                                  dilations,
                                  paddings[0],
                                  paddings[1],
                                  strides));
  // set output and output max dims
  out->set_dims(DDim(out_shape.data(), static_cast<int>(out_shape.size())));
  out->set_dtype(x.dtype());
  out->set_layout(x.layout());
  out_max->set_dims(common::make_ddim({6}));
}

void Conv2dXPUInferMeta(const MetaTensor& x,
                        const MetaTensor& x_max,
                        const MetaTensor& filter,
                        const MetaTensor& filter_max,
                        const MetaTensor& bias,
                        const MetaTensor& branch,
                        const MetaTensor& branch_max,
                        const MetaTensor& scale_max,
                        const MetaTensor& out_max_in,
                        const std::vector<int>& paddings,
                        const std::vector<int>& dilations,
                        const std::vector<int>& strides,
                        const std::string& padding_algorithm,
                        int groups,
                        int act_type,
                        float act_param,
                        DataType out_dtype,
                        MetaTensor* out,
                        MetaTensor* out_max) {
  auto in_dims = x.dims();
  auto filter_dims = filter.dims();
  // do some checks
  PADDLE_ENFORCE_EQ(
      in_dims.size(),
      4,
      phi::errors::InvalidArgument(
          "The input of Op(Conv_xpu) should be a 4-D Tensor. But "
          "received: input's dimension is %u, input's shape is [%s].",
          in_dims.size(),
          in_dims));

  PADDLE_ENFORCE_EQ(
      in_dims.size(),
      filter_dims.size(),
      phi::errors::InvalidArgument(
          "The input's dimension and filter's dimension of "
          "Op(Conv_xpu) should be equal. But received: the input's shape is "
          "[%s], "
          "the input's dimension is %d; the filter's shape is [%s],  "
          "the filter's dimension is %d.",
          in_dims,
          in_dims.size(),
          filter_dims,
          filter_dims.size()));

  const auto input_channels = in_dims[1];
  int stride_size = static_cast<int>(strides.size());
  int in_sub_stride_size = in_dims.size() - stride_size;
  int dilation_size = static_cast<int>(dilations.size());
  PADDLE_ENFORCE_EQ(
      in_dims.size(),
      strides.size() + 2U,
      phi::errors::InvalidArgument(
          "The difference of input's dimension and Attr(strides)'s "
          "length must be euqal to 2 for Op(Conv_xpu). "
          "But received: input's dimension is %d, input's shape is [%s]; "
          "Attr(stride)'s length is %d, Attr(stride) is [%s]; "
          "difference of input's dimention and Attr(strides)'s length = %u.",
          in_dims.size(),
          in_dims,
          strides.size(),
          common::make_ddim(strides),
          in_sub_stride_size));

  for (int i = 0; i < dilation_size; ++i) {
    PADDLE_ENFORCE_GT(
        dilations[i],
        0,
        phi::errors::InvalidArgument(
            "The dilation of Op(Conv) should be larget than 0, but received "
            "dilation is %d.",
            dilations[i]));
  }

  PADDLE_ENFORCE_EQ(
      input_channels,
      filter_dims[1] * groups,
      phi::errors::InvalidArgument(
          "The number of input's channels should be equal to filter's channels "
          "* groups for Op(Conv_xpu). But received: the input's channels is "
          "%d, "
          "the input's shape is [%s]; the filter's channels is %d, the "
          "filter's shape is [%s]; the groups is %d. ",
          input_channels,
          in_dims,
          filter_dims[1],
          filter_dims,
          groups));

  PADDLE_ENFORCE_EQ(
      filter_dims[0] % groups,
      0,
      phi::errors::InvalidArgument(
          "The number of output's channels (filter's first dimension) of "
          "Op(Conv) should be divided by groups. But received: "
          "the output channels is %d, the filter's shape is [%s], "
          "the groups is %d.",
          filter_dims[0],
          filter_dims,
          groups));

  // update paddings and dilations accoring to padding_algorithm
  std::vector<int> paddings_vec = paddings;
  std::vector<int> dilations_vec = dilations;
  DDim in_data_dims = common::slice_ddim(in_dims, 2, in_dims.size());
  DDim filter_data_dims =
      common::slice_ddim(filter_dims, 2, filter_dims.size());
  std::vector<int> ksize = common::vectorize<int>(filter_data_dims);
  phi::UpdatePaddingAndDilation(&paddings_vec,
                                &dilations_vec,
                                padding_algorithm,
                                in_data_dims,
                                strides,
                                ksize);

  std::vector<int64_t> out_shape({in_dims[0], filter_dims[0]});
  for (int i = 0; i < static_cast<int>(strides.size()); ++i) {
    out_shape.push_back(ConvOutSize(static_cast<int>(in_dims[i + 2]),
                                    static_cast<int>(filter_dims[i + 2]),
                                    dilations[i],
                                    paddings_vec[i * 2],
                                    paddings_vec[i * 2 + 1],
                                    strides[i]));
  }
  // set output and output max dims
  out->set_dims(DDim(out_shape.data(), static_cast<int>(out_shape.size())));
  out_max->set_dims(common::make_ddim({6}));
  out->set_dtype(out_dtype);
}

void EmbeddingWithEltwiseAddXPUInferMeta(
    const std::vector<const MetaTensor*>& ids,
    const std::vector<const MetaTensor*>& tables,
    const MetaTensor& mask,
    MetaTensor* out,
    MetaTensor* seq_lod,
    MetaTensor* max_seq_len) {
  PADDLE_ENFORCE_GT(ids.size(),
                    0UL,
                    phi::errors::InvalidArgument(
                        "The input ids in EmbeddingWithEltwiseAddXPUInferMeta "
                        "can't be empty."));
  PADDLE_ENFORCE_GT(tables.size(),
                    0UL,
                    phi::errors::InvalidArgument(
                        "The input tables in "
                        "EmbeddingWithEltwiseAddXPUInferMeta can't be empty."));

  auto id_dims = ids[0]->dims();
  auto table_dims = tables[0]->dims();
  out->set_dims(common::make_ddim({id_dims[0], id_dims[1], table_dims[1]}));
  out->set_dtype(tables[0]->dtype());
  out->set_layout(ids[0]->layout());
}

void FcXPUInferMeta(const MetaTensor& x,
                    const MetaTensor& x_max,
                    const MetaTensor& w,
                    const MetaTensor& w_max,
                    const MetaTensor& bias,
                    const MetaTensor& scale_max,
                    const MetaTensor& out_max_in,
                    int in_num_col_dims,
                    bool transpose_x,
                    float alpha,
                    float beta,
                    int act_type,
                    float act_alpha,
                    DataType out_dtype,
                    MetaTensor* out,
                    MetaTensor* out_max) {
  std::vector<int> out_shape(in_num_col_dims + 1);
  for (int i = 0; i < in_num_col_dims; i++) {
    out_shape[i] = static_cast<int>(x.dims()[i]);
  }
  out_shape[in_num_col_dims] = static_cast<int>(w.dims()[0]);
  out->set_dims(DDim(out_shape.data(), static_cast<int>(out_shape.size())));
  out->set_dtype(out_dtype);
  out->set_layout(x.layout());
  out_max->set_dims(common::make_ddim({6}));
  out_max->set_dtype(x.dtype());
  out_max->set_layout(x.layout());
}

void FusedAttentionInferMeta(const MetaTensor& x,
                             const MetaTensor& ln_scale,
                             const MetaTensor& ln_bias,
                             const MetaTensor& qkv_weight,
                             const MetaTensor& qkv_bias,
                             const MetaTensor& cache_kv,
                             const MetaTensor& src_mask,
                             const MetaTensor& out_linear_weight,
                             const MetaTensor& out_linear_bias,
                             const MetaTensor& ln_scale_2,
                             const MetaTensor& ln_bias_2,
                             int num_heads,
                             bool transpose_qkv_wb,
                             bool pre_layer_norm,
                             float epsilon,
                             float attn_dropout_rate,
                             bool is_test,
                             bool attn_dropout_fix_seed,
                             int attn_dropout_seed,
                             const std::string& attn_dropout_implementation,
                             float dropout_rate,
                             bool dropout_fix_seed,
                             int dropout_seed,
                             const std::string& dropout_implementation,
                             float ln_epsilon,
                             bool add_residual,
                             int ring_id,
                             MetaTensor* ln_mean,
                             MetaTensor* ln_var,
                             MetaTensor* ln_out,
                             MetaTensor* qkv_out,
                             MetaTensor* qkv_bias_out,
                             MetaTensor* transpose_out_2,
                             MetaTensor* qk_out,
                             MetaTensor* qktv_out,
                             MetaTensor* softmax_out,
                             MetaTensor* attn_dropout_mask_out,
                             MetaTensor* attn_dropout_out,
                             MetaTensor* src_mask_out,
                             MetaTensor* fmha_out,
                             MetaTensor* out_linear_out,
                             MetaTensor* dropout_mask_out,
                             MetaTensor* ln_mean_2,
                             MetaTensor* ln_var_2,
                             MetaTensor* bias_dropout_residual_out,
                             MetaTensor* cache_kv_out,
                             MetaTensor* out,
                             MetaConfig config) {
  auto x_dim = x.dims();
  auto y_dim = qkv_weight.dims();

  int dim_head = 0;
  int hidden_size = 0;
  int nranks = 1;
  if (transpose_qkv_wb) {
    PADDLE_ENFORCE_EQ(y_dim.size(),
                      2,
                      phi::errors::InvalidArgument(
                          "The dimensions of qkv_weight must be 2 if enable"
                          "transpose_qkv_wb: (dim_embed, 3 * dim_embed),"
                          "but received dimensions of"
                          "Input is [%d]",
                          y_dim.size()));
    PADDLE_ENFORCE_GT(num_heads,
                      0,
                      phi::errors::InvalidArgument(
                          "The num_heads must be provided and greater than 0 "
                          "if enable transpose_qkv_wb, but we got %d.",
                          num_heads));
    PADDLE_ENFORCE_EQ(y_dim[0] % num_heads,
                      0,
                      phi::errors::InvalidArgument(
                          "First dim of qkv_w must be divisible by num heads "
                          "if enable transpose_qkv_wb, but receive first "
                          "dim of qkv_w is %d and num_heads is %d.",
                          y_dim[0],
                          num_heads));
    if (ring_id == -1) {
      PADDLE_ENFORCE_EQ(
          y_dim[0] * 3,
          y_dim[1],
          phi::errors::InvalidArgument("The dimensions of qkv_weight must be 2"
                                       "(dim_embed, 3 * dim_embed)."));
    } else {
      // compute the mp nranks
      nranks = static_cast<int>((y_dim[0] * 3) / y_dim[1]);
    }
    dim_head = static_cast<int>(y_dim[0] / (num_heads * nranks));
    hidden_size = static_cast<int>(y_dim[0]);
  } else {
    PADDLE_ENFORCE_EQ(y_dim.size(),
                      4,
                      phi::errors::InvalidArgument(
                          "The dimensions of qkv_weight must be 4 if not"
                          "enable transpose_qkv_wb: (3, num_head, dim_head, "
                          "dim_embed), but received [%d]",
                          y_dim.size()));
    PADDLE_ENFORCE_EQ(
        y_dim[0],
        3,
        phi::errors::InvalidArgument("First dim of qkv_w must be 3 if disable "
                                     "transpose_qkv_wb, but we got %d.",
                                     y_dim[0]));
    if (ring_id == -1) {
      PADDLE_ENFORCE_EQ(
          y_dim[1] * y_dim[2],
          y_dim[3],
          phi::errors::InvalidArgument("The dimensions of qkv_weight must be 4"
                                       "(3, num_head, dim_head, dim_embed),"
                                       "and must satisfy the limitations: "
                                       "(num_head * dim_head == dim_embed)"));
    }
    num_heads = static_cast<int>(y_dim[1]);
    dim_head = static_cast<int>(y_dim[2]);
    hidden_size = static_cast<int>(y_dim[3]);
  }

  PADDLE_ENFORCE_EQ(
      x_dim.size(),
      3,
      phi::errors::InvalidArgument("The dimensions of x must be 3"
                                   "(batch_size, seq_len, dim_embed),"
                                   "but received dimensions of"
                                   "Input is [%d]",
                                   x_dim.size()));

  PADDLE_ENFORCE_EQ(x_dim[2],
                    hidden_size,
                    phi::errors::InvalidArgument(
                        "ShapeError: the dimension of x_dim[2] and y_dim[3] "
                        "(y_dim[1] if enable transpose_qkv_w) "
                        "must be equal. But received: the shape "
                        "of input x = [%s], and the shape of "
                        "input qkv_weight = [%s]",
                        x_dim,
                        y_dim));

  if (pre_layer_norm) {
    ln_mean->set_dims({x_dim[0] * x_dim[1]});
    ln_var->set_dims({x_dim[0] * x_dim[1]});
    ln_out->set_dims(x.dims());
  } else {
    ln_mean_2->set_dims({x_dim[0] * x_dim[1]});
    ln_var_2->set_dims({x_dim[0] * x_dim[1]});
    bias_dropout_residual_out->set_dims(x.dims());
  }

  if (transpose_qkv_wb) {
    // [batch_size, seq_len, 3 * num_heads * dim_head]
    qkv_out->set_dims({x_dim[0], x_dim[1], 3 * num_heads * dim_head});

    if (qkv_bias) {
      qkv_bias_out->set_dims({x_dim[0], x_dim[1], 3 * num_heads * dim_head});
    }
  } else {
    // [batch_size, seq_len, 3, num_head, head_size]
    qkv_out->set_dims({x_dim[0], x_dim[1], 3, num_heads, dim_head});

    if (qkv_bias) {
      qkv_bias_out->set_dims({x_dim[0], x_dim[1], 3, num_heads, dim_head});
    }
  }

  // [3, batch_size, num_head, seq_len, head_size]
  transpose_out_2->set_dims({3, x_dim[0], num_heads, x_dim[1], dim_head});

  // cache_seq_len + seq_len if cache else seq_len
  auto out_seq_len = x_dim[1];
  if (cache_kv) {
    // [2, batch_size, num_head, cache_seq_len, head_size]
    auto c_dim = cache_kv.dims();

    PADDLE_ENFORCE_EQ(
        c_dim.size(),
        5,
        phi::errors::InvalidArgument("The CacheKV must be 5 dims, but got %d",
                                     c_dim.size()));
    PADDLE_ENFORCE_EQ(c_dim[0],
                      2,
                      phi::errors::InvalidArgument(
                          "The first dim of CacheKV must be 2, but got %d",
                          c_dim[0]));  // 2
    PADDLE_ENFORCE_EQ(c_dim[1],
                      x_dim[0],
                      phi::errors::InvalidArgument(
                          "The second dim of CacheKV must be equal with "
                          "batch size %d, but got %d",
                          x_dim[0],
                          c_dim[1]));  // batch_size
    PADDLE_ENFORCE_EQ(c_dim[2],
                      num_heads,
                      phi::errors::InvalidArgument(
                          "The third dim of CacheKV must be equal with num "
                          "head %d, but got %d",
                          num_heads,
                          c_dim[2]));  // num_head
    // In compile stage, input seq_len can be -1, in that case
    // c_dim[3] may < 0 in while
    if (config.is_runtime) {
      PADDLE_ENFORCE_GE(
          c_dim[3],
          0,
          phi::errors::InvalidArgument(
              "The forth dim of CacheKV must be greater than 0, but got %d",
              c_dim[3]));  // cache_seq_len
    }

    PADDLE_ENFORCE_EQ(c_dim[4],
                      dim_head,
                      phi::errors::InvalidArgument(
                          "The fifth dim of CacheKV must be equal with head "
                          "size %d, but got %d",
                          dim_head,
                          c_dim[4]));  // head_size

    out_seq_len += c_dim[3];
    // [3, batch_size, num_head, cache_seq_len + seq_len, head_size]
    cache_kv_out->set_dims(
        {c_dim[0], c_dim[1], c_dim[2], out_seq_len, c_dim[4]});
  }
  // [batch, num_head, seq_len, out_seq_len]
  qk_out->set_dims({x_dim[0], num_heads, x_dim[1], out_seq_len});

  if (src_mask) {
    src_mask_out->set_dims({x_dim[0], num_heads, x_dim[1], out_seq_len});
  }
  // the same as QKOut's shape.
  attn_dropout_out->set_dims({x_dim[0], num_heads, x_dim[1], out_seq_len});
  if (!is_test) {
    attn_dropout_mask_out->set_dims(
        {x_dim[0], num_heads, x_dim[1], out_seq_len});
  }
  softmax_out->set_dims({x_dim[0], num_heads, x_dim[1], out_seq_len});
  // [batch_size, num_heads, seq_len, head_dim]
  qktv_out->set_dims({x_dim[0], num_heads, x_dim[1], dim_head});
  // [batch_size, seq_len, number of heads*head size]
  fmha_out->set_dims({x_dim[0], x_dim[1], num_heads, dim_head});

  out_linear_out->set_dims(x.dims());

  if (is_test == false) {
    dropout_mask_out->set_dims(x.dims());
  }

  out->set_dims(x.dims());
}

void FusedAttentionGradInferMeta(const MetaTensor& out_grad,
                                 const MetaTensor& x,
                                 const MetaTensor& qkv_weight,
                                 const MetaTensor& qkv_bias,
                                 const MetaTensor& qkv_bias_out,
                                 const MetaTensor& src_mask,
                                 const MetaTensor& src_mask_out,
                                 const MetaTensor& out_linear_weight,
                                 const MetaTensor& out_linear_bias,
                                 const MetaTensor& ln_scale,
                                 const MetaTensor& ln_bias,
                                 const MetaTensor& ln_scale_2,
                                 const MetaTensor& ln_bias_2,
                                 const MetaTensor& ln_out,
                                 const MetaTensor& ln_mean,
                                 const MetaTensor& ln_var,
                                 const MetaTensor& ln_mean_2,
                                 const MetaTensor& ln_var_2,
                                 const MetaTensor& bias_dropout_residual_out,
                                 const MetaTensor& qkv_out,
                                 const MetaTensor& transpose_out_2,
                                 const MetaTensor& qk_out,
                                 const MetaTensor& qktv_out,
                                 const MetaTensor& softmax_out,
                                 const MetaTensor& attn_dropout_mask_out,
                                 const MetaTensor& attn_dropout_out,
                                 const MetaTensor& fmha_out,
                                 const MetaTensor& out_linear_out,
                                 const MetaTensor& dropout_mask_out,
                                 int num_heads,
                                 bool transpose_qkv_wb,
                                 bool pre_layer_norm,
                                 float epsilon,
                                 float attn_dropout_rate,
                                 bool is_test,
                                 bool attn_dropout_fix_seed,
                                 int attn_dropout_seed,
                                 const std::string& attn_dropout_implementation,
                                 float dropout_rate,
                                 bool dropout_fix_seed,
                                 int dropout_seed,
                                 const std::string& dropout_implementation,
                                 float ln_epsilon,
                                 bool add_residual,
                                 int ring_id,
                                 MetaTensor* qkv_bias_grad,
                                 MetaTensor* qkv_bias_out_grad,
                                 MetaTensor* src_mask_out_grad,
                                 MetaTensor* out_linear_bias_grad,
                                 MetaTensor* ln_scale_grad,
                                 MetaTensor* ln_bias_grad,
                                 MetaTensor* ln_scale_2_grad,
                                 MetaTensor* ln_bias_2_grad,
                                 MetaTensor* x_grad,
                                 MetaTensor* qkv_weight_grad,
                                 MetaTensor* out_linear_weight_grad,
                                 MetaTensor* ln_out_grad,
                                 MetaTensor* bias_dropout_residual_out_grad,
                                 MetaTensor* qkv_out_grad,
                                 MetaTensor* qktv_out_grad,
                                 MetaTensor* transpose_out_2_grad,
                                 MetaTensor* qk_out_grad,
                                 MetaTensor* softmax_out_grad,
                                 MetaTensor* attn_dropout_out_grad,
                                 MetaTensor* fmha_out_grad,
                                 MetaTensor* out_linear_out_grad) {
  PADDLE_ENFORCE_EQ(is_test,
                    false,
                    phi::errors::InvalidArgument(
                        "GradOp is only callable when is_test is false"));

  if (!pre_layer_norm) {
    if (ln_scale_2_grad) {
      ln_scale_2_grad->set_dims(ln_scale_2.dims());
    }
    if (ln_bias_2_grad) {
      ln_bias_2_grad->set_dims(ln_bias_2.dims());
    }
  }

  if (pre_layer_norm) {
    if (ln_scale_grad) {
      ln_scale_grad->set_dims(ln_scale.dims());
    }
    if (ln_bias_grad) {
      ln_bias_grad->set_dims(ln_bias.dims());
    }
  }

  if (x_grad) {
    x_grad->set_dims(x.dims());
  }

  if (out_linear_bias_grad) {
    out_linear_bias_grad->set_dims(out_linear_bias.dims());
  }

  if (out_linear_weight_grad) {
    out_linear_weight_grad->set_dims(out_linear_weight.dims());
  }

  if (qkv_weight_grad) {
    qkv_weight_grad->set_dims(qkv_weight.dims());
  }

  if (qkv_bias_grad) {
    qkv_bias_grad->set_dims(qkv_bias.dims());
  }

  if (pre_layer_norm) {
    if (ln_out_grad) {
      ln_out_grad->set_dims(ln_out.dims());
    }
  } else {
    if (bias_dropout_residual_out_grad) {
      bias_dropout_residual_out_grad->set_dims(
          bias_dropout_residual_out.dims());
    }
  }

  if (fmha_out_grad) {
    fmha_out_grad->set_dims(fmha_out.dims());
  }

  if (qktv_out_grad) {
    qktv_out_grad->set_dims(qktv_out.dims());
  }

  if (transpose_out_2_grad) {
    transpose_out_2_grad->set_dims(transpose_out_2.dims());
  }

  if (qk_out_grad) {
    qk_out_grad->set_dims(qk_out.dims());
  }

  if (softmax_out_grad) {
    softmax_out_grad->set_dims(softmax_out.dims());
  }

  if (attn_dropout_out_grad) {
    attn_dropout_out_grad->set_dims(attn_dropout_out.dims());
  }
  if (src_mask_out_grad) {
    src_mask_out_grad->set_dims(src_mask_out.dims());
  }
  if (qkv_out_grad) {
    qkv_out_grad->set_dims(qkv_out.dims());
  }

  if (qkv_bias_out_grad) {
    qkv_bias_out_grad->set_dims(qkv_bias_out.dims());
  }

  if (out_linear_out_grad) {
    out_linear_out_grad->set_dims(out_linear_out.dims());
  }
}

void FusedBiasDropoutResidualLnInferMeta(
    const MetaTensor& x,
    const MetaTensor& residual,
    const MetaTensor& bias,
    const MetaTensor& ln_scale,
    const MetaTensor& ln_bias,
    const float dropout_rate,
    const bool is_test,
    const bool dropout_fix_seed,
    const int dropout_seed,
    const std::string& dropout_implementation,
    const float ln_epsilon,
    MetaTensor* y,
    MetaTensor* bias_dropout_residual_out,
    MetaTensor* dropout_mask_out,
    MetaTensor* ln_mean,
    MetaTensor* ln_variance) {
  PADDLE_ENFORCE_EQ(dropout_rate >= 0.0f && dropout_rate <= 1.0f,
                    true,
                    phi::errors::InvalidArgument(
                        "'dropout_rate' must be between 0.0 and 1.0."));
  PADDLE_ENFORCE_EQ(
      dropout_implementation == "downgrade_in_infer" ||
          dropout_implementation == "upscale_in_train",
      true,
      phi::errors::InvalidArgument(
          "dropout_implementation can only be downgrade_in_infer or "
          "upscale_in_train"));
  PADDLE_ENFORCE_EQ(ln_epsilon >= 0.0f && ln_epsilon <= 0.001f,
                    true,
                    phi::errors::InvalidArgument(
                        "'epsilon' of the LayerNorm should be between "
                        "0.0 and 0.001, But received [%s].",
                        ln_epsilon));
  auto x_dim = x.dims();
  int left = 1;
  for (int i = 0; i < x_dim.size() - 1; i++) {
    left *= x_dim[i];
  }
  bias_dropout_residual_out->set_dims(x.dims());
  if (is_test == false) {
    dropout_mask_out->set_dims(x.dims());
  }
  ln_mean->set_dims({left});
  ln_variance->set_dims({left});
  y->set_dims(x.dims());
}

void FusedBiasDropoutResidualLnGradInferMeta(
    const MetaTensor& y_grad,
    const MetaTensor& x,
    const MetaTensor& residual,
    const MetaTensor& bias,
    const MetaTensor& ln_scale,
    const MetaTensor& ln_bias,
    const MetaTensor& ln_mean,
    const MetaTensor& ln_variance,
    const MetaTensor& bias_dropout_residual_out,
    const MetaTensor& dropout_mask_out,
    const float dropout_rate,
    const bool is_test,
    const bool dropout_fix_seed,
    const int dropout_seed,
    const std::string& dropout_implementation,
    const float ln_epsilon,
    MetaTensor* x_grad,
    MetaTensor* residual_grad,
    MetaTensor* bias_grad,
    MetaTensor* ln_scale_grad,
    MetaTensor* ln_bias_grad) {
  PADDLE_ENFORCE_EQ(is_test,
                    false,
                    phi::errors::InvalidArgument(
                        "GradOp is only callable when is_test is false"));
  if (ln_scale_grad) {
    ln_scale_grad->set_dims(ln_scale.dims());
    ln_scale_grad->set_dtype(y_grad.dtype());
  }
  if (ln_bias_grad) {
    ln_bias_grad->set_dims(ln_bias.dims());
    ln_bias_grad->set_dtype(y_grad.dtype());
  }
  if (residual_grad) {
    residual_grad->set_dims(residual.dims());
    residual_grad->set_dtype(y_grad.dtype());
  }
  if (bias_grad) {
    bias_grad->set_dims(bias.dims());
    bias_grad->set_dtype(y_grad.dtype());
  }
  if (x_grad) {
    x_grad->set_dims(x.dims());
    x_grad->set_dtype(y_grad.dtype());
  }
}

void FusedDotProductAttentionInferMeta(const MetaTensor& q,
                                       const MetaTensor& k,
                                       const MetaTensor& v,
                                       MetaTensor* out,
                                       MetaTensor* softmax_out,
                                       MetaTensor* rng_state) {
  // q input shape: [batch_size, q_seq_len, num_heads, head_size]
  // k, v input shape: [batch_size, kv_seq_len, num_heads, head_size]
  auto q_dim = q.dims();
  auto k_dim = k.dims();
  auto v_dim = v.dims();

  // check shape
  PADDLE_ENFORCE(q_dim.size() == 4 && k_dim.size() == 4 && v_dim.size() == 4,
                 phi::errors::InvalidArgument(
                     "The dimensions of q, k, v must be 4"
                     "(batch_size, seq_len, num_heads, head_size),"
                     "but received dimensions of"
                     "Input is [%d], [%d], [%d]",
                     q_dim.size(),
                     k_dim.size(),
                     v_dim.size()));

  PADDLE_ENFORCE(q_dim[0] == k_dim[0] && k_dim[0] == v_dim[0],
                 phi::errors::InvalidArgument(
                     "The first dimension of q, k, v must be equal"
                     "but received dimensions of"
                     "Input is [%d], [%d], [%d]",
                     q_dim[0],
                     k_dim[0],
                     v_dim[0]));

  // [batch_size, num_heads, q_seqlen, 1]
  std::vector<int64_t> softmax_out_shape({q_dim[0], q_dim[2], q_dim[1], 1});

  out->set_dims(q_dim);
  softmax_out->set_dims(
      DDim(softmax_out_shape.data(), softmax_out_shape.size()));

  // rng_state: {seed, offset}
  std::vector<int64_t> rng_state_shape({2});
  rng_state->set_dims(DDim(rng_state_shape.data(), rng_state_shape.size()));
}

void FusedDotProductAttentionGradInferMeta(const MetaTensor& q,
                                           const MetaTensor& k,
                                           const MetaTensor& v,
                                           MetaTensor* q_grad,
                                           MetaTensor* k_grad,
                                           MetaTensor* v_grad) {
  auto q_dim = q.dims();
  auto k_dim = k.dims();
  auto v_dim = v.dims();
  q_grad->set_dims(q_dim);
  k_grad->set_dims(k_dim);
  v_grad->set_dims(v_dim);
}

void FusedFeedForwardInferMeta(const MetaTensor& x,
                               const MetaTensor& dropout1_seed,
                               const MetaTensor& dropout2_seed,
                               const MetaTensor& linear1_weight,
                               const MetaTensor& linear1_bias,
                               const MetaTensor& linear2_weight,
                               const MetaTensor& linear2_bias,
                               const MetaTensor& ln1_scale,
                               const MetaTensor& ln1_bias,
                               const MetaTensor& ln2_scale,
                               const MetaTensor& ln2_bias,
                               bool pre_layer_norm,
                               float ln1_epsilon,
                               float ln2_epsilon,
                               const std::string& act_method,
                               float dropout1_prob,
                               float dropout2_prob,
                               const std::string& dropout1_implementation,
                               const std::string& dropout2_implementation,
                               bool is_test,
                               bool dropout1_fix_seed,
                               bool dropout2_fix_seed,
                               int dropout1_seed_val,
                               int dropout2_seed_val,
                               bool add_residual,
                               int ring_id,
                               MetaTensor* out,
                               MetaTensor* dropout1_mask,
                               MetaTensor* dropout2_mask,
                               MetaTensor* ln1_mean,
                               MetaTensor* ln1_variance,
                               MetaTensor* ln2_mean,
                               MetaTensor* ln2_variance,
                               MetaTensor* linear1_out,
                               MetaTensor* ln1_out,
                               MetaTensor* dropout1_out,
                               MetaTensor* dropout2_out) {
  auto dim_x = x.dims();

  auto RowMatrixFromVector = [](const DDim& x_dim) -> DDim {
    if (x_dim.size() > 1) {
      return x_dim;
    }
    return common::make_ddim({1, x_dim[0]});
  };

  auto mat_dim_x =
      funcs::CreateMatrixDescriptor(RowMatrixFromVector(dim_x), 0, false);
  // verify for the pre layer_norm, the feature size must be larger than 1
  PADDLE_ENFORCE_GT(
      mat_dim_x.width_,
      static_cast<size_t>(1),
      phi::errors::InvalidArgument("Product from the X shape[1] to "
                                   "shape[n-1] must be larger than 1!"));
  auto dim_Linear1Weight = linear1_weight.dims();
  auto tmp_dim_x = dim_x;
  tmp_dim_x[dim_x.size() - 1] = dim_Linear1Weight[dim_Linear1Weight.size() - 1];
  out->set_dims(dim_x);

  if (!is_test) {
    dropout1_mask->set_dims(tmp_dim_x);
  }
  dropout1_out->set_dims(tmp_dim_x);
  linear1_out->set_dims(tmp_dim_x);
  dropout2_out->set_dims(dim_x);

  if (!is_test) {
    dropout2_mask->set_dims(dim_x);
  }

  auto mean_dim =
      common::make_ddim({mat_dim_x.batch_size_ * mat_dim_x.height_});
  if (pre_layer_norm) {
    ln1_out->set_dims(dim_x);
    ln1_mean->set_dims(mean_dim);
    ln1_variance->set_dims(mean_dim);
  } else {
    ln2_mean->set_dims(mean_dim);
    ln2_variance->set_dims(mean_dim);
  }
  out->share_lod(x);
}

static bool IsUnaryCompound(const std::vector<std::string>& functor_list) {
  PADDLE_ENFORCE_EQ(
      functor_list.size(),
      2,
      phi::errors::InvalidArgument(
          "Invalid functor list size %d, which should be equal to %d.",
          functor_list.size(),
          2));
  static std::unordered_set<std::string> binary_fun = {"elementwise_add",
                                                       "elementwise_mul",
                                                       "elementwise_add_grad",
                                                       "elementwise_mul_grad"};
  return binary_fun.count(functor_list[1]) != 0;
}

static bool InputXCanBeAbsent(const std::vector<std::string>& functor_list) {
  PADDLE_ENFORCE_EQ(
      functor_list.size(),
      2,
      phi::errors::InvalidArgument(
          "Invalid functor list size %d, which should be equal to %d.",
          functor_list.size(),
          2));
  static std::unordered_set<std::string> binary_fun = {"elementwise_add_grad"};
  return binary_fun.count(functor_list[0]) != 0 ||
         binary_fun.count(functor_list[1]) != 0;
}

static bool IsBcastY(const phi::DDim& x_dim, const phi::DDim& y_dim) {
  bool bcast_y = x_dim.size() >= y_dim.size();
  if (x_dim.size() == y_dim.size()) {
    for (int i = 0; i < x_dim.size(); ++i) {
      if (x_dim[i] < y_dim[i]) {
        bcast_y = false;
        break;
      }
    }
  }
  return bcast_y;
}

void FusedElemwiseAddActivationInferMeta(
    const MetaTensor& x,
    const MetaTensor& y,
    const std::vector<std::string>& functor_list,
    float scale,
    int axis,
    bool save_intermediate_out,
    MetaTensor* out,
    MetaTensor* intermediate_out) {
  PADDLE_ENFORCE_NOT_NULL(
      x,
      errors::NotFound(
          "Input(X) of FusedElemwiseAddActivationOp op should not be null."));
  PADDLE_ENFORCE_NOT_NULL(
      y,
      errors::NotFound(
          "Input(Y) of FusedElemwiseAddActivationOp op should not be null."));
  PADDLE_ENFORCE_NOT_NULL(out,
                          phi::errors::InvalidArgument(
                              "Output(Out) of FusedElemwiseAddActivationOp op "
                              "should not be null."));

  auto x_dim = x.dims();
  auto y_dim = y.dims();

  // Whether the shape of Y is a continuous subsequence of X,
  // For more information please refer to the op's introduction.
  bool bcast_y = IsBcastY(x_dim, y_dim);

  auto out_dim = bcast_y ? x_dim : y_dim;
  auto out_lod = bcast_y ? x : y;
  auto out_dtype = bcast_y ? x.dtype() : y.dtype();

  PADDLE_ENFORCE_NOT_NULL(
      intermediate_out,
      errors::NotFound(
          "Output(IntermediateOut) of FusedElemwiseAddActivationOp "
          "should not be null."));

  if (IsUnaryCompound(functor_list)) {
    // for Unary(Binary(X, Y)), the shape and lod of out and
    // intermediate_out are the same.
    intermediate_out->set_dims(out_dim);
    // set the lod of intermediate_out
    intermediate_out->share_lod(out_lod);
  } else {
    // for Binary(X, Unary(Y)), the shape and lod of Y and
    // intermediate_out are the same.
    intermediate_out->set_dims(y_dim);
    // set the lod of intermediate_out
    intermediate_out->share_lod(y);
  }
  out->set_dims(out_dim);
  out->share_lod(out_lod);
  out->set_dtype(out_dtype);

  bool elemntwise_add_detected = false;
  for (auto names : functor_list) {
    if (names == "elementwise_add") {
      elemntwise_add_detected = true;
      break;
    }
  }
  PADDLE_ENFORCE_EQ(
      elemntwise_add_detected,
      true,
      phi::errors::InvalidArgument(
          "When the FusedElemwiseAddActivationOp Is used in fused pass, the "
          "elementwise_add Op must be"
          "detected and used, Please check the fuse pass pattern"));
}

void FusedElemwiseAddActivationGradInferMeta(
    const MetaTensor& x,
    const MetaTensor& y,
    const MetaTensor& out,
    const MetaTensor& intermediate_out,
    const MetaTensor& out_grad,
    const std::vector<std::string>& functor_list,
    float scale,
    int axis,
    bool save_intermediate_out,
    MetaTensor* x_grad,
    MetaTensor* y_grad) {
  PADDLE_ENFORCE_NOT_NULL(
      out_grad,
      phi::errors::InvalidArgument("Input(Out@Grad) should not be null."));

  if (save_intermediate_out) {
    PADDLE_ENFORCE_NOT_NULL(intermediate_out,
                            phi::errors::InvalidArgument(
                                "Input(IntermediateOut) should not be null."));
  } else {
    if (!InputXCanBeAbsent(functor_list)) {
      PADDLE_ENFORCE_NOT_NULL(
          x, phi::errors::InvalidArgument("Input(X) should not be null."));
    }
  }

  if (x_grad) {
    if (x) {
      x_grad->set_dtype(x.dtype());
      x_grad->set_dims(x.dims());
      x_grad->share_lod(x);
    } else {
      // Currently, only when Binary is elementwise_add or elementwise_sub,
      // the "X" could be absent.
      PADDLE_ENFORCE_EQ(
          InputXCanBeAbsent(functor_list),
          true,
          phi::errors::InvalidArgument(
              "Only when BinaryFunctor is elementwise_add, the 'X' "
              "could be absent."));

      // Node: If "X" is absence, the shape of Y should be a continuous
      // subsequence of X, otherwise, we could not infer the shape of dx.
      x_grad->set_dtype(out.dtype());
      x_grad->set_dims(out_grad.dims());
      x_grad->share_lod(out_grad);
    }
  }

  if (y_grad) {
    PADDLE_ENFORCE_NOT_NULL(
        y, phi::errors::InvalidArgument("Input(Y) should not be null."));
    y_grad->set_dims(y.dims());
    y_grad->share_lod(y);
    y_grad->set_dtype(y.dtype());
  }

  // if (intermediate_out_grad) {
  //   // For Unary(Binary(X, Y)), IntermediateOut should not be empty.
  //   if (IsUnaryCompound(functor_list)) {
  //     intermediate_out_grad->set_dims(out_grad.dims());
  //     intermediate_out_grad->share_lod(out_grad);
  //   } else {
  //     intermediate_out_grad->set_dims(y.dims());
  //     intermediate_out_grad->share_lod(y);
  //   }
  // }
  bool elemntwise_add_grad_detected = false;
  for (auto names : functor_list) {
    if (names == "elementwise_add_grad") {
      elemntwise_add_grad_detected = true;
      break;
    }
  }
  PADDLE_ENFORCE_EQ(
      elemntwise_add_grad_detected,
      true,
      phi::errors::InvalidArgument(
          "When the FusedElemwiseAddActivationOpGrad Is used in fused pass, "
          "the elementwise_add_grad Op must be"
          "detected and used, Please check the fuse pass pattern"));
}

void FusedFeedForwardGradInferMeta(const MetaTensor& out_grad,
                                   const MetaTensor& x,
                                   const MetaTensor& linear1_weight,
                                   const MetaTensor& linear1_bias,
                                   const MetaTensor& linear2_weight,
                                   const MetaTensor& dropout1_mask,
                                   const MetaTensor& dropout2_mask,
                                   const MetaTensor& linear1_out,
                                   const MetaTensor& dropout1_out,
                                   const MetaTensor& dropout2_out,
                                   const MetaTensor& ln1_scale,
                                   const MetaTensor& ln1_bias,
                                   const MetaTensor& ln1_out,
                                   const MetaTensor& ln1_mean,
                                   const MetaTensor& ln1_variance,
                                   const MetaTensor& ln2_scale,
                                   const MetaTensor& ln2_bias,
                                   const MetaTensor& ln2_mean,
                                   const MetaTensor& ln2_variance,
                                   const MetaTensor& linear2_bias,
                                   bool pre_layer_norm,
                                   float ln1_epsilon,
                                   float ln2_epsilon,
                                   const std::string& act_method,
                                   float dropout1_prob,
                                   float dropout2_prob,
                                   const std::string& dropout1_implementation,
                                   const std::string& dropout2_implementation,
                                   bool is_test,
                                   bool dropout1_fix_seed,
                                   bool dropout2_fix_seed,
                                   int dropout1_seed_val,
                                   int dropout2_seed_val,
                                   bool add_residual,
                                   int ring_id,
                                   MetaTensor* x_grad,
                                   MetaTensor* ln1_scale_grad,
                                   MetaTensor* ln1_bias_grad,
                                   MetaTensor* ln2_scale_grad,
                                   MetaTensor* ln2_bias_grad,
                                   MetaTensor* linear1_weight_grad,
                                   MetaTensor* linear1_bias_grad,
                                   MetaTensor* linear2_weight_grad,
                                   MetaTensor* linear2_bias_grad) {
  auto d_out_dim = out_grad.dims();
  x_grad->set_dims(d_out_dim);
  if (ln1_scale_grad) {
    ln1_scale_grad->set_dims(ln1_scale.dims());
  }
  if (ln1_bias_grad) {
    ln1_bias_grad->set_dims(ln1_bias.dims());
  }
  if (ln2_scale_grad) {
    ln2_scale_grad->set_dims(ln2_scale.dims());
  }
  if (ln2_bias_grad) {
    ln2_bias_grad->set_dims(ln2_bias.dims());
  }

  linear1_weight_grad->set_dims(linear1_weight.dims());
  if (linear1_bias_grad) {
    linear1_bias_grad->set_dims(linear1_bias.dims());
  }

  linear2_weight_grad->set_dims(linear2_weight.dims());
  if (linear2_bias_grad) {
    linear2_bias_grad->set_dims(linear2_bias.dims());
  }
}

void GenerateSequenceXPUInferMeta(const MetaTensor& x,
                                  DataType dtype,
                                  MetaTensor* out) {
  out->set_dims(x.dims());
  out->set_dtype(dtype);
  out->set_layout(x.layout());
}

void MultiEncoderXPUInferMeta(
    const MetaTensor& x,
    const std::vector<const MetaTensor*>& fc_weight,
    const std::vector<const MetaTensor*>& fc_weight_max,
    const std::vector<const MetaTensor*>& fc_bias,
    const std::vector<const MetaTensor*>& ln_scale,
    const std::vector<const MetaTensor*>& ln_bias,
    const MetaTensor& mask,
    const MetaTensor& seq_lod,
    const MetaTensor& max_seq_len,
    int layer_num,
    bool norm_before,
    int hidden_dim,
    int head_num,
    int size_per_head,
    int ffn_hidden_dim_scale,
    int act_type,
    int relative_type,
    int slice_idx,
    MetaTensor* out,
    MetaTensor* x_fp16,
    MetaTensor* out_fp16) {
  auto x_dims = x.dims();
  x_fp16->set_dims(x_dims);
  x_fp16->set_dtype(DataType::FLOAT16);
  x_fp16->set_layout(x.layout());
  out->set_dtype(x.dtype());
  out->set_layout(x.layout());
  out_fp16->set_dtype(DataType::FLOAT16);
  out_fp16->set_layout(x.layout());
  if (slice_idx == -1) {
    out->set_dims(x_dims);
    out_fp16->set_dims(x_dims);
  } else {
    out->set_dims({x_dims[0], x_dims[2]});
    out_fp16->set_dims({x_dims[0], x_dims[2]});
  }
}

void FusedGemmEpilogueInferMeta(const MetaTensor& x,
                                const MetaTensor& y,
                                const MetaTensor& bias,
                                bool trans_x,
                                bool trans_y,
                                const std::string& activation,
                                MetaTensor* out,
                                MetaTensor* reserve_space) {
  const auto& x_dims = x.dims();
  const auto& y_dims = y.dims();
  const auto& bias_dims = bias.dims();

  PADDLE_ENFORCE_EQ(y_dims.size(),
                    2,
                    phi::errors::InvalidArgument(
                        "The Input tensor Y's dimension of FusedGemmEpilogueOp "
                        " should be 2, but got %d.",
                        y_dims.size()));

  PADDLE_ENFORCE_GE(x_dims.size(),
                    2,
                    phi::errors::InvalidArgument(
                        "The Input tensor X's dimension of FusedGemmEpilogueOp "
                        " should be >= 2, but got %d.",
                        x_dims.size()));

  PADDLE_ENFORCE_EQ(
      bias_dims.size(),
      1,
      phi::errors::InvalidArgument(
          "The Input tensor bias's dimension of FusedGemmEpilogueOp "
          " should be == 1, but got %d.",
          bias_dims.size()));

  PADDLE_ENFORCE_EQ(bias_dims[0],
                    trans_y ? y_dims[0] : y_dims[1],
                    phi::errors::InvalidArgument(
                        "The Input tensor bias's dimension 0"
                        " should be == Y[-1], but got bias's shape = [%s] "
                        "and Y's shape = [%s]",
                        bias_dims,
                        y_dims));

  auto x_mat_dims =
      common::flatten_to_2d(x_dims, trans_x ? 1 : x_dims.size() - 1);

  int K_from_x = static_cast<int>(trans_x ? x_mat_dims[0] : x_mat_dims[1]);
  int K_from_y = static_cast<int>(trans_y ? y_dims[1] : y_dims[0]);

  PADDLE_ENFORCE_EQ(
      K_from_x,
      K_from_y,
      phi::errors::InvalidArgument(
          "The last dimension of X should be equal with Y's first dimension."
          "But received X[-1] = [%d], Y[0] = [%d].",
          K_from_x,
          K_from_y));

  std::vector<int64_t> out_dims;
  out_dims.reserve(static_cast<size_t>(x_dims.size()));
  if (trans_x) {
    for (int i = 1; i < x_dims.size(); ++i) out_dims.push_back(x_dims[i]);
  } else {
    for (int i = 0; i < x_dims.size() - 1; ++i) out_dims.push_back(x_dims[i]);
  }

  if (trans_y) {
    out_dims.push_back(y_dims[0]);
  } else {
    out_dims.push_back(y_dims[1]);
  }
  out->set_dims(common::make_ddim(out_dims));
  out->set_dtype(x.dtype());

  if (reserve_space) {
    reserve_space->set_dims(common::make_ddim(out_dims));
    reserve_space->set_dtype(x.dtype());
    if (activation == "none") {
      PADDLE_THROW(phi::errors::InvalidArgument(
          "The ReserveSpace would not be used when activation = \"none\""));
    } else {
      int min_size_of_n = activation == "relu" ? 128 : 8;
      int N_size = static_cast<int>(trans_y ? y_dims[0] : y_dims[1]);
      PADDLE_ENFORCE_EQ(N_size % min_size_of_n,
                        0,
                        phi::errors::InvalidArgument(
                            "The output dimension N (X(MxK) * Y(KxN) = C(MxN)) "
                            "should be multiple of %d when auxiliary_key given "
                            "and activation=%s, but got N = %d.",
                            min_size_of_n,
                            activation,
                            N_size));
    }
  }
}

void FusedGemmEpilogueGradInferMeta(const MetaTensor& x,
                                    const MetaTensor& y,
                                    const MetaTensor& reserve_space,
                                    const MetaTensor& out_grad,
                                    bool trans_x,
                                    bool trans_y,
                                    const std::string& activation_grad,
                                    MetaTensor* x_grad,
                                    MetaTensor* y_grad,
                                    MetaTensor* bias_grad) {
  auto x_dims = x.dims();
  auto y_dims = y.dims();
  auto dout_dims = out_grad.dims();

  PADDLE_ENFORCE_GE(
      dout_dims.size(),
      2,
      phi::errors::InvalidArgument(
          "The Input tensor DOut's dimension of FusedGemmEpilogueGradOp "
          " should be >= 2, but got %d.",
          dout_dims.size()));

  PADDLE_ENFORCE_EQ(
      y_dims.size(),
      2,
      phi::errors::InvalidArgument(
          "The Input tensor Y's dimension of FusedGemmEpilogueGradOp "
          " should be 2, but got %d.",
          y_dims.size()));

  PADDLE_ENFORCE_GE(
      x_dims.size(),
      2,
      phi::errors::InvalidArgument(
          "The Input tensor X's dimension of FusedGemmEpilogueGradOp "
          " should be >= 2, but got %d.",
          x_dims.size()));

  PADDLE_ENFORCE_EQ(
      dout_dims.size(),
      x_dims.size(),
      phi::errors::InvalidArgument(
          "The Input tensor DOut's and X's dimension of "
          "FusedGemmEpilogueGradOp "
          " should be the same, but got DOut's dim = %d and X's = %d.",
          dout_dims.size(),
          x_dims.size()));

  auto dout_mat_dims = common::flatten_to_2d(dout_dims, dout_dims.size() - 1);
  auto x_mat_dims = common::flatten_to_2d(x_dims, x_dims.size() - 1);

  PADDLE_ENFORCE_EQ(
      dout_mat_dims[1],
      trans_y ? y_dims[0] : y_dims[1],
      phi::errors::InvalidArgument(
          "The last dimension of DOut should be equal with Y's last"
          "dimension. But received DOut[-1] = [%d], Y[1] = [%d].",
          dout_mat_dims[1],
          y_dims[1]));

  PADDLE_ENFORCE_EQ(
      dout_mat_dims[0],
      trans_x ? x_mat_dims[1] : x_mat_dims[0],
      phi::errors::InvalidArgument(
          "The first dimension of DOut should be equal with X's first"
          "dimension. But received DOut[0] = [%d], Y[0] = [%d].",
          dout_mat_dims[0],
          x_mat_dims[0]));

  if (activation_grad != "none" && !reserve_space) {
    PADDLE_THROW(phi::errors::InvalidArgument(
        "The ReserveSpace should not be empty. "
        "when activation == {relu_grad, gelu_grad}."));
  }

  if (x_grad) {
    x_grad->set_dims(x_dims);
    x_grad->set_dtype(x.dtype());
  }
  y_grad->set_dims(y_dims);
  y_grad->set_dtype(y.dtype());

  if (bias_grad) {
    int64_t dbias_dim = trans_y ? y_dims[0] : y_dims[1];
    bias_grad->set_dims(common::make_ddim({dbias_dim}));
    bias_grad->set_dtype(y.dtype());
  }
}

void FusedMultiTransformerXpuInferMeta(
    const MetaTensor& x,
    const std::vector<const MetaTensor*>& ln_scale,
    const std::vector<const MetaTensor*>& ln_bias,
    const std::vector<const MetaTensor*>& qkvw,
    const std::vector<const MetaTensor*>& qkvw_max,
    const std::vector<const MetaTensor*>& qkv_bias,
    const std::vector<const MetaTensor*>& out_linear_w,
    const std::vector<const MetaTensor*>& out_linear_wmax,
    const std::vector<const MetaTensor*>& out_linear_bias,
    const std::vector<const MetaTensor*>& ffn_ln_scale,
    const std::vector<const MetaTensor*>& ffn_ln_bias,
    const std::vector<const MetaTensor*>& ffn1_weight,
    const std::vector<const MetaTensor*>& ffn1_weight_max,
    const std::vector<const MetaTensor*>& ffn1_bias,
    const std::vector<const MetaTensor*>& ffn2_weight,
    const std::vector<const MetaTensor*>& ffn2_weight_max,
    const std::vector<const MetaTensor*>& ffn2_bias,
    const std::vector<const MetaTensor*>& cache_kv,
    const std::vector<const MetaTensor*>& pre_caches,
    const MetaTensor& rotary_pos_emb,
    const MetaTensor& time_step,
    const MetaTensor& seq_lengths,
    const MetaTensor& src_mask,
    const MetaTensor& gather_index,
    const MetaTensor& max_buffer,
    bool pre_layer_norm,
    int rotary_emb_dims,
    float epsilon,
    float dropout_rate,
    bool is_test,
    const std::string& dropout_implementation,
    const std::string& act_method,
    bool trans_qkvw,
    int ring_id,
    int gather_axis,
    MetaTensor* out,
    std::vector<MetaTensor*> cache_kv_out) {
  auto x_dim = x.dims();
  auto y_dim = qkvw[0]->dims();
  PADDLE_ENFORCE_EQ(x_dim.size(),
                    3,
                    phi::errors::InvalidArgument(
                        "The dimensions of x must be 3(batch_size, seq_len, "
                        "dim_embed), but received dimensions of Input is [%d]",
                        x_dim.size()));
  PADDLE_ENFORCE_EQ(
      y_dim.size(),
      4,
      phi::errors::InvalidArgument(
          "The dimensions of qkv_weight must be 4(3, num_head, dim_head, "
          "dim_embed), but received dimensions of qkv_weight is [%d]",
          y_dim.size()));
  PADDLE_ENFORCE_EQ(
      x_dim[2],
      trans_qkvw ? y_dim[3] : y_dim[0],
      phi::errors::InvalidArgument(
          "The dimension of x_dim[2] and y_dim[3](trans_qkvw is  true) or "
          "y_dim[0](trans_qkvw is false) must be equal, but received: the "
          "shape of input x = [%s], and the shape of input qkv_weight = [%s]",
          x_dim,
          y_dim));
  if (!cache_kv.empty()) {
    const auto& c_dim = cache_kv[0]->dims();
    PADDLE_ENFORCE_EQ(
        c_dim.size(),
        5,
        phi::errors::InvalidArgument("The CacheKV must be 5 dims, but got %d",
                                     c_dim.size()));
    PADDLE_ENFORCE_EQ(c_dim[0],
                      2,
                      phi::errors::InvalidArgument(
                          "The first dim of CacheKV must be 2, but got %d",
                          c_dim[0]));  // 2
    PADDLE_ENFORCE_EQ(
        c_dim[3],
        trans_qkvw ? y_dim[1] : y_dim[2],
        phi::errors::InvalidArgument("The fourth dim of CacheKV must be equal "
                                     "with num head %d, but got %d",
                                     trans_qkvw ? y_dim[1] : y_dim[2],
                                     c_dim[3]));  // num_head
    PADDLE_ENFORCE_EQ(
        c_dim[4],
        trans_qkvw ? y_dim[2] : y_dim[3],
        phi::errors::InvalidArgument("The fifth dim of CacheKV must be equal "
                                     "with head size %d, but got %d",
                                     trans_qkvw ? y_dim[2] : y_dim[3],
                                     c_dim[4]));  // head_size
  }

  out->set_dims(x_dim);
  out->set_dtype(x.dtype());
  out->set_layout(x.layout());
}

void FusedMultiTransformerInt8XpuInferMeta(
    const MetaTensor& x,
    const std::vector<const MetaTensor*>& ln_scale,
    const std::vector<const MetaTensor*>& ln_bias,
    const std::vector<const MetaTensor*>& qkv_in_max,
    const std::vector<const MetaTensor*>& qkvw,
    const std::vector<const MetaTensor*>& qkv_bias,
    const std::vector<const MetaTensor*>& qkv_scales,
    const std::vector<const MetaTensor*>& out_linear_in_max,
    const std::vector<const MetaTensor*>& out_linear_w,
    const std::vector<const MetaTensor*>& out_linear_bias,
    const std::vector<const MetaTensor*>& out_linear_scales,
    const std::vector<const MetaTensor*>& ffn_ln_scale,
    const std::vector<const MetaTensor*>& ffn_ln_bias,
    const std::vector<const MetaTensor*>& ffn1_in_max,
    const std::vector<const MetaTensor*>& ffn1_weight,
    const std::vector<const MetaTensor*>& ffn1_bias,
    const std::vector<const MetaTensor*>& ffn1_scales,
    const std::vector<const MetaTensor*>& ffn2_in_max,
    const std::vector<const MetaTensor*>& ffn2_weight,
    const std::vector<const MetaTensor*>& ffn2_bias,
    const std::vector<const MetaTensor*>& ffn2_scales,
    const std::vector<const MetaTensor*>& cache_kv,
    const std::vector<const MetaTensor*>& pre_caches,
    const MetaTensor& rotary_pos_emb,
    const MetaTensor& time_step,
    const MetaTensor& seq_lengths,
    const MetaTensor& src_mask,
    const MetaTensor& gather_index,
    const MetaTensor& max_buffer,
    bool pre_layer_norm,
    int rotary_emb_dims,
    float epsilon,
    float dropout_rate,
    bool is_test,
    const std::string& dropout_implementation,
    const std::string& act_method,
    bool trans_qkvw,
    int ring_id,
    int gather_axis,
    MetaTensor* out,
    std::vector<MetaTensor*> cache_kv_out) {
  auto x_dim = x.dims();
  auto y_dim = qkvw[0]->dims();
  PADDLE_ENFORCE_EQ(x_dim.size(),
                    3,
                    phi::errors::InvalidArgument(
                        "The dimensions of x must be 3(batch_size, seq_len, "
                        "dim_embed), but received dimensions of Input is [%d]",
                        x_dim.size()));
  PADDLE_ENFORCE_EQ(
      y_dim.size(),
      4,
      phi::errors::InvalidArgument(
          "The dimensions of qkv_weight must be 4(3, num_head, dim_head, "
          "dim_embed), but received dimensions of qkv_weight is [%d]",
          y_dim.size()));
  PADDLE_ENFORCE_EQ(
      x_dim[2],
      trans_qkvw ? y_dim[3] : y_dim[0],
      phi::errors::InvalidArgument(
          "The dimension of x_dim[2] and y_dim[3](trans_qkvw is  true) or "
          "y_dim[0](trans_qkvw is false) must be equal, but received: the "
          "shape of input x = [%s], and the shape of input qkv_weight = [%s]",
          x_dim,
          y_dim));
  if (!cache_kv.empty()) {
    const auto& c_dim = cache_kv[0]->dims();
    PADDLE_ENFORCE_EQ(
        c_dim.size(),
        5,
        phi::errors::InvalidArgument("The CacheKV must be 5 dims, but got %d",
                                     c_dim.size()));
    PADDLE_ENFORCE_EQ(c_dim[0],
                      2,
                      phi::errors::InvalidArgument(
                          "The first dim of CacheKV must be 2, but got %d",
                          c_dim[0]));  // 2
    PADDLE_ENFORCE_EQ(
        c_dim[3],
        trans_qkvw ? y_dim[1] : y_dim[2],
        phi::errors::InvalidArgument("The fourth dim of CacheKV must be equal "
                                     "with num head %d, but got %d",
                                     trans_qkvw ? y_dim[1] : y_dim[2],
                                     c_dim[3]));  // num_head
    PADDLE_ENFORCE_EQ(
        c_dim[4],
        trans_qkvw ? y_dim[2] : y_dim[3],
        phi::errors::InvalidArgument("The fifth dim of CacheKV must be equal "
                                     "with head size %d, but got %d",
                                     trans_qkvw ? y_dim[2] : y_dim[3],
                                     c_dim[4]));  // head_size
  }

  out->set_dims(x_dim);
  out->set_dtype(x.dtype());
  out->set_layout(x.layout());
}

void YoloBoxXPUInferMeta(const MetaTensor& x,
                         const MetaTensor& x_max,
                         const MetaTensor& grid,
                         const MetaTensor& stride,
                         const MetaTensor& anchor_grid,
                         float offset,
                         MetaTensor* out,
                         MetaTensor* out_max) {
  auto x_dims = x.dims();
  auto x_dims_size = x_dims.size();
  PADDLE_ENFORCE_GT(
      x_dims[x_dims_size - 1],
      4,
      phi::errors::InvalidArgument(
          "The last dim of x should be larget than 4, but received "
          " is %d.",
          x_dims[x_dims_size - 1]));
  // compute left out_dims
  // y[..., 0:2] = (x[..., 0:2] * 2 + self.grid[i]) * self.stride[i]  # xy
  std::vector<int> axes_ = {x_dims_size - 1};
  std::vector<int> infer_flags_ = {1};
  std::vector<int> decrease_axis_ = {-1};
  std::vector<int64_t> strides_ = {1};
  std::vector<int64_t> starts_l = {0};
  std::vector<int64_t> ends_l = {2};
  std::vector<int64_t> left_slice_out_dims_vector(x_dims_size, -1);
  phi::funcs::StridedSliceOutDims(starts_l,
                                  ends_l,
                                  strides_,
                                  axes_,
                                  infer_flags_,
                                  x_dims,
                                  decrease_axis_,
                                  left_slice_out_dims_vector.data(),
                                  1,
                                  true);
  auto left_slice_out_dims = common::make_ddim(left_slice_out_dims_vector);
  auto grid_dims = grid.dims();
  auto left_add_out_dims =
      BroadCastInferShape(left_slice_out_dims, grid_dims, -1);
  auto stride_dims = stride.dims();
  auto left_mul_out_dims =
      BroadCastInferShape(left_add_out_dims, stride_dims, -1);
  // compute mid out_dims
  // wh = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]             # wh
  std::vector<int64_t> starts_m = {2};
  std::vector<int64_t> ends_m = {4};
  std::vector<int64_t> mid_slice_out_dims_vector(x_dims_size, -1);
  phi::funcs::StridedSliceOutDims(starts_m,
                                  ends_m,
                                  strides_,
                                  axes_,
                                  infer_flags_,
                                  x_dims,
                                  decrease_axis_,
                                  mid_slice_out_dims_vector.data(),
                                  1,
                                  true);
  auto mid_slice_out_dims = common::make_ddim(mid_slice_out_dims_vector);
  auto anchor_grid_dims = anchor_grid.dims();
  auto mid_mul_out_dims =
      BroadCastInferShape(mid_slice_out_dims, anchor_grid_dims, -1);
  // compute right out_dims
  std::vector<int64_t> starts_r = {4};
  std::vector<int64_t> ends_r = {2147483647};
  std::vector<int64_t> right_slice_out_dims_vector(x_dims_size, -1);
  phi::funcs::StridedSliceOutDims(starts_r,
                                  ends_r,
                                  strides_,
                                  axes_,
                                  infer_flags_,
                                  x_dims,
                                  decrease_axis_,
                                  right_slice_out_dims_vector.data(),
                                  1,
                                  true);
  auto right_slice_out_dims = common::make_ddim(right_slice_out_dims_vector);
  // compute concat out_dims
  std::vector<phi::DDim> in_dims;
  in_dims.reserve(3);
  in_dims.emplace_back(left_mul_out_dims);
  in_dims.emplace_back(mid_mul_out_dims);
  in_dims.emplace_back(right_slice_out_dims);
  phi::DDim out_dim =
      phi::funcs::ComputeAndCheckShape(false, in_dims, x_dims_size - 1);

  out->set_dims(out_dim);
  out->set_dtype(x.dtype());
  out->set_layout(x.layout());
  out_max->set_dims(common::make_ddim({6}));
  out_max->set_dtype(x.dtype());
  out_max->set_layout(x.layout());
}

void ConvTransposeXPUInferMeta(const MetaTensor& x,
                               const MetaTensor& filter,
                               const std::vector<int>& strides,
                               const std::vector<int>& paddings,
                               const std::vector<int>& output_padding,
                               const std::vector<int>& output_size,
                               const std::string& padding_algorithm,
                               int groups,
                               const std::vector<int>& dilations,
                               const std::string& data_format,
                               MetaTensor* out,
                               MetaTensor* out_max) {
  auto x_dims = x.dims();
  auto filter_dims = filter.dims();
  std::vector<int> paddings_ = paddings;
  std::vector<int> dilations_ = dilations;
  PADDLE_ENFORCE_EQ(
      x_dims.size() == 4,
      true,
      errors::InvalidArgument("Input of Op(conv_transpose) should be 4-D "
                              "Tensor. But received: %u-D Tensor, "
                              "the shape of input is [%s]",
                              x_dims.size(),
                              x_dims));
  PADDLE_ENFORCE_EQ(
      x_dims.size(),
      filter_dims.size(),
      errors::InvalidArgument(
          "The input's dimension size and filter's dimension size of "
          "Op (conv_transpose) should be equal. But received: the shape of "
          "input is [%s], the dimension size of input is [%d], the shape "
          "of filter is [%s],  the dimension size of filter is [%d]. ",
          x_dims,
          x_dims.size(),
          filter_dims,
          filter_dims.size()));

  int stride_size = static_cast<int>(strides.size());
  for (int i = 0; i < stride_size; ++i) {
    PADDLE_ENFORCE_GT(
        strides[i],
        0,
        errors::InvalidArgument(
            "The stride of Op(Conv) should be larget than 0, but received "
            "stride is %d.",
            strides[i]));
  }

  int in_sub_stride_size = x_dims.size() - stride_size;

  PADDLE_ENFORCE_EQ(
      x_dims.size() - strides.size(),
      2U,
      errors::InvalidArgument(
          "The input's dimension size minus Attr(stride)'s size must "
          "be euqal to 2 for Op(conv_transpose). But received: [%d], the "
          "input's dimension size is [%d], the shape of input "
          "is [%s], the Attr(stride)'s size is [%d].",
          in_sub_stride_size,
          x_dims.size(),
          x_dims,
          strides.size()));
  if (!output_size.empty())
    PADDLE_ENFORCE_EQ(
        output_size.size(),
        strides.size(),
        errors::InvalidArgument(
            "The Attr(output_size) and Attr(stride) of Op(conv_transpose) "
            "should be the same."));
  if (!output_padding.empty())
    PADDLE_ENFORCE_EQ(
        output_padding.size(),
        strides.size(),
        errors::InvalidArgument(
            "The Attr(output_padding) and Attr(stride) of Op(conv_transpose) "
            "should be the same."));

  const int64_t C =
      (data_format != "NHWC" ? x_dims[1] : x_dims[x_dims.size() - 1]);
  PADDLE_ENFORCE_EQ(
      C,
      filter_dims[0],
      errors::InvalidArgument(
          "The number of input channels should be equal to filter channels "
          "for Op(conv_transpose). But received: the input's channels is "
          "[%d], the shape of input is [%s], the filter's channels is [%d], "
          "the shape of filter is [%s]. The data_format is %s."
          "The error may come from wrong data_format setting.",
          C,
          x_dims,
          filter_dims[0],
          filter_dims,
          data_format));

  DDim x_data_dims;
  if (data_format != "NHWC") {
    x_data_dims = slice_ddim(x_dims, 2, x_dims.size());
  } else {
    x_data_dims = slice_ddim(x_dims, 1, x_dims.size() - 1);
  }
  DDim filter_data_dims = slice_ddim(filter_dims, 2, filter_dims.size());
  std::vector<int> ksize = common::vectorize<int>(filter_data_dims);
  UpdatePaddingAndDilation(
      &paddings_, &dilations_, padding_algorithm, x_data_dims, strides, ksize);

  std::vector<int64_t> output_shape({x_dims[0]});
  if (data_format != "NHWC") {
    output_shape.push_back(filter_dims[1] * groups);
  }
  const int offset = (data_format != "NHWC" ? 2 : 1);
  for (int i = 0; i < static_cast<int>(strides.size()); ++i) {
    auto filter_extent = dilations_[i] * (filter_dims[i + 2] - 1) + 1;
    auto infer_shape = (x_dims[i + offset] > 0)
                           ? (x_dims[i + offset] - 1) * strides[i] -
                                 paddings_[2 * i] - paddings_[2 * i + 1] +
                                 filter_extent
                           : -1;
    if (!output_size.empty()) {
      output_shape.push_back(output_size[i]);
    } else if (!output_padding.empty()) {
      output_shape.push_back((infer_shape + output_padding[i]));
    } else {
      output_shape.push_back(infer_shape);
    }
  }
  if (data_format == "NHWC") {
    output_shape.push_back(filter_dims[1] * groups);
  }

  out->set_dims(common::make_ddim(output_shape));
  out->set_dtype(x.dtype());
  out_max->set_dims(common::make_ddim({6}));
}

void Conv2dTransposeXPUInferMeta(const MetaTensor& x,
                                 const MetaTensor& x_max,
                                 const MetaTensor& filter,
                                 const MetaTensor& filter_max,
                                 const MetaTensor& bias,
                                 const std::vector<int>& strides,
                                 const std::vector<int>& paddings,
                                 const std::vector<int>& output_padding,
                                 const IntArray& output_size,
                                 const std::string& padding_algorithm,
                                 int groups,
                                 const std::vector<int>& dilations,
                                 const std::string& data_format,
                                 bool has_bias,
                                 bool with_act,
                                 const std::string& act_type,
                                 MetaTensor* out,
                                 MetaTensor* out_max) {
  std::vector<int32_t> vec_output_size(output_size.GetData().begin(),
                                       output_size.GetData().end());
  ConvTransposeXPUInferMeta(x,
                            filter,
                            strides,
                            paddings,
                            output_padding,
                            vec_output_size,
                            padding_algorithm,
                            groups,
                            dilations,
                            data_format,
                            out,
                            out_max);
}

void FastWhereXPUInferMeta(const MetaTensor& condition,
                           const MetaTensor& x,
                           const MetaTensor& y,
                           MetaTensor* out) {
  out->set_dims(x.dims());
  out->set_dtype(x.dtype());
}

void FastLayernormXPUInferMeta(const MetaTensor& x,
                               const MetaTensor& scale,
                               const MetaTensor& bias,
                               int begin_norm_axis,
                               float epsilon,
                               MetaTensor* out) {
  out->set_dims(x.dims());
  out->set_dtype(x.dtype());
  out->set_layout(x.layout());
}

void BNActXPUInferMeta(const MetaTensor& x,
                       const MetaTensor& mean,
                       const MetaTensor& variance,
                       const MetaTensor& scale,
                       const MetaTensor& bias,
                       float momentum,
                       float epsilon,
                       const std::string& data_layout,
                       int act_type,
                       MetaTensor* y,
                       MetaConfig config) {
  const auto x_dims = x.dims();
  for (int i = 0; i < x_dims.size(); i++) {
    PADDLE_ENFORCE_EQ(
        (x_dims[i] == -1) || (x_dims[i] > 0),
        true,
        phi::errors::InvalidArgument(
            "Each dimension of input tensor is expected to be -1 or a "
            "positive number, but received %d. Input's shape is [%s].",
            x_dims[i],
            x_dims));
  }

  const DataLayout data_layout_str = common::StringToDataLayout(data_layout);

  PADDLE_ENFORCE_GE(
      x_dims.size(),
      2,
      phi::errors::InvalidArgument(
          "ShapeError: the dimension of input "
          "X must greater than or equal to 2. But received: the shape of input "
          "X = [%s], the dimension of input X =[%d]",
          x_dims,
          x_dims.size()));
  PADDLE_ENFORCE_LE(
      x_dims.size(),
      5,
      phi::errors::InvalidArgument(
          "ShapeError: the dimension of input X "
          "must smaller than or equal to 5. But received: the shape of input X "
          "= [%s], the dimension of input X = [%d]",
          x_dims,
          x_dims.size()));

  const int64_t C = ((config.is_run_mkldnn_kernel == true) ||
                             (data_layout_str == DataLayout::kNCHW)
                         ? x_dims[1]
                         : x_dims[x_dims.size() - 1]);
  auto scale_dim = scale.dims();
  auto bias_dim = bias.dims();

  PADDLE_ENFORCE_EQ(
      scale_dim.size(),
      1UL,
      phi::errors::InvalidArgument(
          "ShapeError: the dimension of scale must equal to 1."
          "But received: the shape of scale is [%s], the dimension "
          "of scale is [%d]",
          scale_dim,
          scale_dim.size()));
  PADDLE_ENFORCE_EQ(bias_dim.size(),
                    1UL,
                    phi::errors::InvalidArgument(
                        "ShapeError: the dimension of bias must equal to 1."
                        "But received: the shape of bias is [%s],the dimension "
                        "of bias is [%d]",
                        bias_dim,
                        bias_dim.size()));

  bool check = true;
  if ((!config.is_runtime) &&
      (common::product(scale_dim) <= 0 || common::product(bias_dim) <= 0)) {
    check = false;
  }

  if (check) {
    PADDLE_ENFORCE_EQ(scale_dim[0],
                      C,
                      phi::errors::InvalidArgument(
                          "ShapeError: the shape of scale must equal to [%d]"
                          "But received: the shape of scale is [%d]",
                          C,
                          scale_dim[0]));
    PADDLE_ENFORCE_EQ(bias_dim[0],
                      C,
                      phi::errors::InvalidArgument(
                          "ShapeError: the shape of bias must equal to [%d]"
                          "But received: the shape of bias is [%d]",
                          C,
                          bias_dim[0]));
  }
  y->set_dims(x_dims);
  y->share_lod(x);
  y->set_dtype(x.dtype());
}
void AddCMulXPUInferMeta(const MetaTensor& x,
                         const MetaTensor& y,
                         const MetaTensor& w,
                         MetaTensor* out) {
  out->set_dims(x.dims());
  out->set_dtype(x.dtype());
  out->set_layout(x.layout());
}

void LayerNormActXPUInferMeta(const MetaTensor& x,
                              const MetaTensor& scale,
                              const MetaTensor& bias,
                              int begin_norm_axis,
                              float epsilon,
                              int act_type,
                              float act_param,
                              MetaTensor* y) {
  y->set_dims(x.dims());
  //   y->share_lod(x);
  y->set_dtype(x.dtype());
  y->set_layout(x.layout());
}

void FusedScaleBiasReluConvBnInferMeta(const MetaTensor& x,
                                       const MetaTensor& w,
                                       const MetaTensor& scale,
                                       const MetaTensor& bias,
                                       const MetaTensor& bn_scale,
                                       const MetaTensor& bn_bias,
                                       const MetaTensor& input_running_mean,
                                       const MetaTensor& input_running_var,
                                       const std::vector<int>& paddings,
                                       const std::vector<int>& dilations,
                                       const std::vector<int>& strides,
                                       const std::string& padding_algorithm,
                                       int groups,
                                       const std::string& data_format,
                                       float momentum,
                                       float epsilon,
                                       bool fuse_prologue,
                                       bool exhaustive_search,
                                       int64_t accumulation_count,
                                       MetaTensor* out,
                                       MetaTensor* out_running_mean,
                                       MetaTensor* out_running_var,
                                       MetaTensor* saved_mean,
                                       MetaTensor* saved_var,
                                       MetaTensor* eq_scale,
                                       MetaTensor* eq_bias) {
  auto in_dims = x.dims();
  auto filter_dims = w.dims();
  // do some checks
  PADDLE_ENFORCE_EQ(
      in_dims.size(),
      4,
      phi::errors::InvalidArgument(
          "The input of Op(FusedScaleBiasReluConvBn) should be a 4-D "
          "Tensor. But "
          "received: input's dimension is %u, input's shape is [%s].",
          in_dims.size(),
          in_dims));

  PADDLE_ENFORCE_EQ(
      in_dims.size(),
      filter_dims.size(),
      phi::errors::InvalidArgument(
          "The input's dimension and filter's dimension of "
          "Op(FusedScaleBiasReluConvBn) should be equal. But received: "
          "the input's"
          " shape is [%s], "
          "the input's dimension is %d; the filter's shape is [%s],  "
          "the filter's dimension is %d.",
          in_dims,
          in_dims.size(),
          filter_dims,
          filter_dims.size()));

  // Check if data format is NHWC
  PADDLE_ENFORCE_EQ(
      data_format,
      "NHWC",
      phi::errors::InvalidArgument(
          "Operator(FusedScaleBiasReluConvBn) only supports data format "
          "of "
          "channel last (NHWC) now. But recieved: data_format = '%s'.",
          data_format));

  PADDLE_ENFORCE_EQ(
      groups,
      1,
      phi::errors::InvalidArgument("Expect group to be 1, got %d.", groups));

  const auto input_channels = in_dims[in_dims.size() - 1];
  int dilation_size = static_cast<int>(dilations.size());
  for (int i = 0; i < dilation_size; ++i) {
    PADDLE_ENFORCE_GT(
        dilations[i],
        0,
        phi::errors::InvalidArgument(
            "The dilation of Op(Conv) should be larget than 0, but received "
            "dilation is %d.",
            dilations[i]));
  }

  PADDLE_ENFORCE_EQ(
      input_channels,
      filter_dims[1] * groups,
      phi::errors::InvalidArgument(
          "The number of input's channels should be equal to filter's channels "
          "* groups for Op(FusedScaleBiasReluConvBn). But received: the "
          "input's"
          " channels is %d, "
          "the input's shape is [%s]; the filter's channels is %d, the "
          "filter's shape is [%s]; the groups is %d. ",
          input_channels,
          in_dims,
          filter_dims[1],
          filter_dims,
          groups));

  // update paddings and dilations accoring to padding_algorithm
  std::vector<int> paddings_vec = paddings;
  std::vector<int> dilations_vec = dilations;
  // get "HW" from "NHWC"
  DDim in_data_dims = common::slice_ddim(in_dims, 1, in_dims.size() - 1);
  DDim filter_data_dims =
      common::slice_ddim(filter_dims, 2, filter_dims.size());
  std::vector<int> ksize = common::vectorize<int>(filter_data_dims);
  phi::UpdatePaddingAndDilation(&paddings_vec,
                                &dilations_vec,
                                padding_algorithm,
                                in_data_dims,
                                strides,
                                ksize);

  std::vector<int64_t> out_shape({in_dims[0]});
  for (int i = 0; i < static_cast<int>(strides.size()); ++i) {
    out_shape.push_back(ConvOutSize(static_cast<int>(in_dims[i + 1]),
                                    static_cast<int>(filter_dims[i + 2]),
                                    dilations[i],
                                    paddings_vec[i * 2],
                                    paddings_vec[i * 2 + 1],
                                    strides[i]));
  }
  out_shape.push_back(filter_dims[0]);
  // make shape for other outputs
  auto c_dims = common::make_ddim({filter_dims[0]});
  // set output and output max dims
  out->set_dims(DDim(out_shape.data(), static_cast<int>(out_shape.size())));
  out_running_mean->set_dims(c_dims);
  out_running_var->set_dims(c_dims);
  saved_mean->set_dims(c_dims);
  saved_var->set_dims(c_dims);
  eq_scale->set_dims(c_dims);
  eq_bias->set_dims(c_dims);
}

void FusedScaleBiasAddReluInferMeta(const MetaTensor& x1,
                                    const MetaTensor& scale1,
                                    const MetaTensor& bias1,
                                    const MetaTensor& x2,
                                    const MetaTensor& scale2,
                                    const MetaTensor& bias2,
                                    bool fuse_dual,
                                    bool exhaustive_search,
                                    MetaTensor* out) {
  // check optional inputs
  if (fuse_dual) {
    bool has_scale2 = !!scale2;
    bool has_bias2 = !!bias2;
    PADDLE_ENFORCE(has_scale2 && has_bias2,
                   phi::errors::InvalidArgument(
                       "Argument scale2 and bias2 should be provided when "
                       "fuse_dual is set, but got has_scale2=%d, has_bias2=%d, "
                       "fuse_dual=%d.",
                       has_scale2,
                       has_bias2,
                       fuse_dual));
  }
  // set output dims
  out->set_dims(x1.dims());
  out->set_dtype(x1.dtype());
  out->set_layout(x1.layout());
}

void FusedDconvDreluDbnInferMeta(const MetaTensor& grad_output,
                                 const MetaTensor& weight,
                                 const MetaTensor& grad_output_add,
                                 const MetaTensor& residual_input,
                                 const MetaTensor& bn1_eqscale,
                                 const MetaTensor& bn1_eqbias,
                                 const MetaTensor& conv_input,
                                 const MetaTensor& bn1_mean,
                                 const MetaTensor& bn1_inv_std,
                                 const MetaTensor& bn1_gamma,
                                 const MetaTensor& bn1_beta,
                                 const MetaTensor& bn1_input,
                                 const MetaTensor& bn2_mean,
                                 const MetaTensor& bn2_inv_std,
                                 const MetaTensor& bn2_gamma,
                                 const MetaTensor& bn2_beta,
                                 const MetaTensor& bn2_input,
                                 const std::vector<int>& paddings,
                                 const std::vector<int>& dilations,
                                 const std::vector<int>& strides,
                                 const std::string& padding_algorithm,
                                 int groups,
                                 const std::string& data_format,
                                 bool fuse_shortcut,
                                 bool fuse_dual,
                                 bool fuse_add,
                                 bool exhaustive_search,
                                 MetaTensor* grad_weight,
                                 MetaTensor* grad_bn1_input,
                                 MetaTensor* grad_bn1_gamma,
                                 MetaTensor* grad_bn1_beta,
                                 MetaTensor* grad_bn2_input,
                                 MetaTensor* grad_bn2_gamma,
                                 MetaTensor* grad_bn2_beta) {
  // Check if data format is NHWC
  PADDLE_ENFORCE_EQ(
      data_format,
      "NHWC",
      phi::errors::InvalidArgument(
          "Operator(FusedScaleBiasReluConvBnstats) only supports data format "
          "of "
          "channel last (NHWC) now. But recieved: data_format = '%s'.",
          data_format));

  PADDLE_ENFORCE_EQ(
      groups,
      1,
      phi::errors::InvalidArgument("Expect group to be 1, got %d.", groups));

  PADDLE_ENFORCE_EQ(
      fuse_shortcut && fuse_dual,
      0,
      phi::errors::InvalidArgument(
          "fuse_shortcut and fuse_dual should not be set at the same time."
          "Got fuse_shortcut=%d, fuse_dual=%d.",
          fuse_shortcut,
          fuse_dual));

  if (fuse_add) {
    PADDLE_ENFORCE_EQ(
        !!grad_output_add,
        true,
        phi::errors::InvalidArgument(
            "grad_output_add must be provided when fuse_add = true."
            "Got fuse_add=%d, grad_output_add=%d.",
            fuse_add,
            !!grad_output_add));
  }
  if (fuse_shortcut) {
    PADDLE_ENFORCE_EQ(
        !!residual_input,
        true,
        phi::errors::InvalidArgument(
            "residual_input must be provided when fuse_shortcut = true."
            "Got fuse_shortcut =%d, residual_input=%d.",
            fuse_shortcut,
            !!residual_input));
  }
  if (fuse_shortcut || fuse_dual) {
    PADDLE_ENFORCE_EQ(
        !!conv_input,
        true,
        phi::errors::InvalidArgument(
            "conv_input must be provided when either fuse_shortcut "
            "or fuse_dual is set to true. Got conv_input=%d, fuse_shortcut=%d, "
            "fuse_dual=%d.",
            !!conv_input,
            fuse_shortcut,
            fuse_dual));
  } else {
    PADDLE_ENFORCE_EQ(
        bn1_eqscale && bn1_eqbias,
        true,
        phi::errors::InvalidArgument(
            "bn1_eqscale and bn1_eqbias must be provided when neither "
            "fuse_shortcut "
            "or fuse_dual is set. Got bn1_eqscale=%d, bn1_eqbias=%d.",
            !!bn1_eqscale,
            !!bn1_eqbias));
  }
  if (fuse_dual) {
    PADDLE_ENFORCE_EQ(
        bn2_mean && bn2_inv_std && bn2_gamma && bn2_beta && bn2_input,
        true,
        phi::errors::InvalidArgument("bn2_mean, bn2_inv_std, bn2_gamma, "
                                     "bn2_beta, bn2_input must be provided "
                                     "when fuse_dual is set. Got bn2_mean=%d, "
                                     "bn2_inv_std=%d, bn2_gamma=%d, "
                                     "bn2_beta=%d, bn2_input=%d.",
                                     !!bn2_mean,
                                     !!bn2_inv_std,
                                     !!bn2_gamma,
                                     !!bn2_beta,
                                     !!bn2_input));
  }

  auto set_unchanged_meta = [](MetaTensor* out, const MetaTensor& input) {
    out->set_dims(input.dims());
    out->set_dtype(input.dtype());
    out->set_layout(input.layout());
  };

  set_unchanged_meta(grad_weight, weight);
  set_unchanged_meta(grad_bn1_input, bn1_input);
  set_unchanged_meta(grad_bn1_gamma, bn1_gamma);
  set_unchanged_meta(grad_bn1_beta, bn1_beta);
  if (grad_bn2_input) {
    set_unchanged_meta(grad_bn2_input, bn1_input);
  }
  if (grad_bn2_gamma) {
    set_unchanged_meta(grad_bn2_gamma, bn1_gamma);
  }
  if (grad_bn2_beta) {
    set_unchanged_meta(grad_bn2_beta, bn1_beta);
  }
}

void SqueezeExcitationInferMeta(const MetaTensor& x,
                                const MetaTensor& filter,
                                const MetaTensor& filter_max,
                                const MetaTensor& bias,
                                const MetaTensor& branch,
                                const std::vector<int>& act_type,
                                const std::vector<float>& act_param,
                                const std::vector<int>& filter_dims,
                                MetaTensor* out) {
  auto in_dims = x.dims();
  // do some checks
  PADDLE_ENFORCE_EQ(
      in_dims.size(),
      4,
      phi::errors::InvalidArgument(
          "The input should be a 4-D Tensor. But "
          "received: input's dimension is %u, input's shape is [%s].",
          in_dims.size(),
          in_dims));
  std::vector<int64_t> out_shape(
      {in_dims[0], filter_dims[1], in_dims[2], in_dims[3]});
  // set output dims
  out->set_dims(DDim(out_shape.data(), static_cast<int>(out_shape.size())));
}

void FusedEmbeddingEltWiseLayerNormInferMeta(
    const std::vector<const MetaTensor*>& ids,
    const std::vector<const MetaTensor*>& embs,
    const MetaTensor& bias,
    const MetaTensor& scale,
    const float epsilon,
    MetaTensor* out) {
  PADDLE_ENFORCE_EQ(
      ids.size(),
      embs.size(),
      phi::errors::InvalidArgument(
          "Two inputs of EmbeddingEltWiseLayerNormOp shoube be "
          "the same size, but received the size of input Ids = %d,"
          " the size of input Embs = %d",
          ids.size(),
          embs.size()));
  PADDLE_ENFORCE_GE(embs.size(),
                    2UL,
                    phi::errors::InvalidArgument(
                        "Input Embs of EmbeddingEltWiseLayerNormOp should "
                        "have at least 2 tensors"));
  PADDLE_ENFORCE_GE(ids.size(),
                    2UL,
                    phi::errors::InvalidArgument(
                        "Input Ids of EmbeddingEltWiseLayerNormOp should "
                        "have at least 2 tensors"));

  // batch * seq_len * 1
  std::vector<DDim> ids_dims, embs_dims;
  ids_dims.reserve(ids.size());
  std::transform(ids.begin(),
                 ids.end(),
                 std::back_inserter(ids_dims),
                 [](const MetaTensor* var) { return var->dims(); });
  // word_num * hidden
  embs_dims.reserve(embs.size());
  std::transform(embs.begin(),
                 embs.end(),
                 std::back_inserter(embs_dims),
                 [](const MetaTensor* var) { return var->dims(); });
  // hidden
  DDim dims_bias = bias.dims();

  int batch = ids_dims[0][0];
  int seq_len = ids_dims[0][1];
  int hidden = embs_dims[0][1];
  for (auto& embs_dim : embs_dims) {
    PADDLE_ENFORCE_EQ(
        embs_dim.size(),
        2,
        phi::errors::InvalidArgument(
            "The Emb dim's size shoule be 2, but found %d.", embs_dim.size()));
    PADDLE_ENFORCE_EQ(
        embs_dim[1],
        dims_bias[0],
        phi::errors::InvalidArgument(
            "The second dims (%d) of the Embedding should be equal "
            "to the Bias's size(%d).",
            embs_dim[1],
            dims_bias[0]));
    PADDLE_ENFORCE_EQ(
        embs_dim[1],
        hidden,
        phi::errors::InvalidArgument(
            "The second dimension size(%d) of the Embedding should be "
            "equal to the hidden's size(%d)",
            embs_dim[1],
            hidden));
  }

  auto dim_output = common::make_ddim({batch, seq_len, hidden});
  out->set_dims(dim_output);
  out->share_lod(*ids[0]);
  out->set_dtype((*embs[0]).dtype());
}

void FusionTransposeFlattenConcatInferMeta(
    const std::vector<const MetaTensor*>& x,
    const std::vector<int>& trans_axis,
    const int flatten_axis,
    const int concat_axis,
    MetaTensor* out) {
  PADDLE_ENFORCE_GE(
      x.size(),
      1UL,
      phi::errors::InvalidArgument(
          "Inputs(X) of TransposeFlattenConcat op should not be empty."));

  std::vector<DDim> ins;
  ins.reserve(x.size());
  std::transform(
      x.begin(), x.end(), std::back_inserter(ins), [](const MetaTensor* var) {
        return var->dims();
      });
  const size_t n = ins.size();
  PADDLE_ENFORCE_GT(n,
                    0,
                    phi::errors::InvalidArgument(
                        "The size of Inputs(X)'s dimension should be greater "
                        " than 0, but received %d.",
                        n));

  size_t x_rank = ins[0].size();
  size_t trans_axis_size = trans_axis.size();
  PADDLE_ENFORCE_EQ(x_rank,
                    trans_axis_size,
                    phi::errors::InvalidArgument(
                        "The input tensor's rank(%d) "
                        "should be equal to the permutation axis's size(%d)",
                        x_rank,
                        trans_axis_size));

  auto dims0 = phi::funcs::GetFlattenShape(
      flatten_axis, phi::funcs::GetPermuteShape(trans_axis, ins[0]));
  std::vector<int> out_dims(dims0);
  for (size_t i = 1; i < n; i++) {
    auto dimsi = phi::funcs::GetFlattenShape(
        flatten_axis, phi::funcs::GetPermuteShape(trans_axis, ins[i]));
    for (int j = 0; j < static_cast<int>(dims0.size()); j++) {
      if (j == concat_axis) {
        out_dims[concat_axis] += dimsi[j];
      } else {
        PADDLE_ENFORCE_EQ(out_dims[j],
                          dimsi[j],
                          phi::errors::InvalidArgument(
                              "After flatting, the %d-th dim should be save "
                              "except the specify axis.",
                              j));
      }
    }
  }
  if (out_dims[concat_axis] < 0) {
    out_dims[concat_axis] = -1;
  }
  out->set_dims(common::make_ddim(out_dims));
  out->set_dtype((*x[0]).dtype());
}

void FusedFCElementwiseLayerNormInferMeta(const MetaTensor& x,
                                          const MetaTensor& w,
                                          const MetaTensor& y,
                                          const MetaTensor& bias0,
                                          const MetaTensor& scale,
                                          const MetaTensor& bias1,
                                          const int x_num_col_dims,
                                          const std::string& activation_type,
                                          const float epsilon,
                                          const int begin_norm_axis,
                                          MetaTensor* out,
                                          MetaTensor* mean,
                                          MetaTensor* variance,
                                          MetaConfig config) {
  DDim w_dims = w.dims();
  PADDLE_ENFORCE_EQ(
      w_dims.size(),
      2,
      phi::errors::InvalidArgument(
          "The input Weight of fc is expected to be a 2-D tensor. "
          "But received the number of Weight's dimensions is %d, ",
          "Weight's shape is %s.",
          w_dims.size(),
          w_dims));

  if (bias0) {
    DDim bias0_dims = bias0.dims();

    PADDLE_ENFORCE_LE(bias0_dims.size(),
                      2,
                      phi::errors::InvalidArgument(
                          "The input Bias of fc is expected to be an 1-D or "
                          "2-D tensor. But received the number of Bias's "
                          "dimensions is %d, Bias's shape is %s.",
                          bias0_dims.size(),
                          bias0_dims));

    PADDLE_ENFORCE_EQ(
        bias0_dims[bias0_dims.size() - 1],
        w_dims[1],
        phi::errors::InvalidArgument(
            "The last dimension of input Bias is expected be equal "
            "to the actual width of input Weight. But received the last "
            "dimension of Bias is %d, Bias's shape is %s; "
            "the actual width of Weight is %d, Weight's shape is %s.",
            bias0_dims[bias0_dims.size() - 1],
            bias0_dims,
            w_dims[1],
            w_dims));

    if (bias0_dims.size() == 2) {
      PADDLE_ENFORCE_EQ(
          bias0_dims[0],
          1,
          phi::errors::InvalidArgument(
              "The first dimension of input Bias is expected to be 1, "
              "but received %d, Bias's shape is %s.",
              bias0_dims[0],
              bias0_dims));
    }
  }

  DDim x_dims = x.dims();
  PADDLE_ENFORCE_LT(
      x_num_col_dims,
      x_dims.size(),
      phi::errors::InvalidArgument(
          "The attribute x_num_col_dims used to flatten input X to "
          "a 2-D tensor, is expected to be less than the number of "
          "input X's dimensions. But received x_num_col_dims is %d, "
          "the number of input X's dimensions is %d, input X's shape is %s.",
          x_num_col_dims,
          x_dims.size(),
          x_dims));

  auto x_mat_dims = common::flatten_to_2d(x_dims, x_num_col_dims);
  PADDLE_ENFORCE_EQ(
      x_mat_dims[1],
      w_dims[0],
      phi::errors::InvalidArgument(
          "The input's second dimension and weight's first dimension is "
          "expected to be the same. But received input's second dimension is "
          "%d, input's shape is %s; weight's first dimension is %d, weight's "
          "shape is %s.",
          x_mat_dims[1],
          x_mat_dims,
          w_dims[0],
          w_dims));

  std::vector<int64_t> fc_out_dims;
  for (int i = 0; i < x_num_col_dims; ++i) {
    fc_out_dims.push_back(x_dims[i]);
  }
  fc_out_dims.push_back(w_dims[1]);

  DDim y_dims = y.dims();
  PADDLE_ENFORCE_EQ(common::make_ddim(fc_out_dims),
                    y_dims,
                    phi::errors::InvalidArgument(
                        "The output's shape of fc is expected to be equal to "
                        "that of input Y. But received output's shape of fc "
                        "is %s, input Y's shape is %s.",
                        common::make_ddim(fc_out_dims),
                        y_dims));

  PADDLE_ENFORCE_LT(
      begin_norm_axis,
      y_dims.size(),
      phi::errors::InvalidArgument(
          "The attribute begin_norm_axis used to flatten input Y to a 2-D "
          "tensor, is expected to be less than the number of input Y's "
          "dimensions. But received begin_norm_axis is %d, the number of "
          "input Y's dimensions is %d, input Y's shape is %s.",
          begin_norm_axis,
          y_dims.size(),
          y_dims));

  auto y_mat_dim = common::flatten_to_2d(y_dims, begin_norm_axis);
  int64_t dim_0 = y_mat_dim[0];
  int64_t dim_1 = y_mat_dim[1];
  if (scale) {
    DDim scale_dims = scale.dims();
    PADDLE_ENFORCE_EQ(scale_dims.size(),
                      1,
                      phi::errors::InvalidArgument(
                          "The input Scale is expected to be an 1-D tensor. "
                          "But received the number of input Scale's "
                          "dimensions is %d, input Scale's shape is %s.",
                          scale_dims.size(),
                          scale_dims));

    if (config.is_runtime) {
      PADDLE_ENFORCE_EQ(
          scale_dims[0],
          dim_1,
          phi::errors::InvalidArgument(
              "The first dimension of input Scale is expected to be equal to "
              "the second dimension of input Y after flattened. "
              "But received the first dimension of input Scale is %d, input "
              "Scale's shape is %s; the second dimension of flattened input "
              "Y is %d, input Y's shape is %s, flattened axis is %d.",
              scale_dims[0],
              scale_dims,
              dim_1,
              y_dims,
              begin_norm_axis));
    }
  }
  if (bias1) {
    DDim bias1_dims = bias1.dims();
    PADDLE_ENFORCE_EQ(
        bias1_dims.size(),
        1,
        phi::errors::InvalidArgument(
            "The input Bias1 is expected to be an 1-D tensor. "
            "But received the number of input Bias1's dimension is %d, "
            "input Bias1's shape is %s.",
            bias1_dims.size(),
            bias1_dims));

    if (config.is_runtime) {
      PADDLE_ENFORCE_EQ(
          bias1_dims[0],
          dim_1,
          phi::errors::InvalidArgument(
              "The first dimension of input Bias1 is expected to be equal to "
              "the second dimension of input Y after flattened. "
              "But received the first dimension of input Bias1 is %d, input "
              "Bias1's shape is %s; the second dimension of flatten input "
              "Y is %d, input Y's shape is %s, flattened axis is %d.",
              bias1_dims[0],
              bias1_dims,
              dim_1,
              y_dims,
              begin_norm_axis));
    }
  }

  out->set_dims(y_dims);
  out->set_dtype(x.dtype());
  if (mean) {
    mean->set_dtype(x.dtype());
    mean->set_dims({dim_0});
  }
  if (variance) {
    variance->set_dims({dim_0});
    variance->set_dtype(x.dtype());
  }
  out->share_lod(x);
}

void FusedConv2dAddActInferMeta(const MetaTensor& input,
                                const MetaTensor& filter,
                                const MetaTensor& bias,
                                const MetaTensor& residual_data,
                                const std::vector<int>& strides,
                                const std::vector<int>& paddings,
                                const std::string& padding_algorithm,
                                const std::vector<int>& dilations,
                                int groups,
                                const std::string& data_format,
                                const std::string& activation,
                                const std::vector<int>& split_channels,
                                MetaTensor* output,
                                std::vector<MetaTensor*> outputs,
                                MetaConfig config) {
  // TODO(liuyuanle): mkldnn seems only support nchw.
  const bool channel_last = (data_format == "NHWC" || data_format == "NDHWC");
  std::vector<int64_t> out_shape = ComputeOutputShape(input,
                                                      filter,
                                                      bias,
                                                      strides,
                                                      paddings,
                                                      padding_algorithm,
                                                      dilations,
                                                      groups,
                                                      data_format,
                                                      channel_last,
                                                      config);
  output->set_dims(common::make_ddim(out_shape));
  output->set_dtype(input.dtype());
  if (data_format == "NHWC") {
    output->set_layout(phi::DataLayout::NHWC);
  } else if (data_format == "NDHWC") {
    output->set_layout(phi::DataLayout::NDHWC);
  }

  output->share_lod(input);

  if (split_channels.size()) {
    PADDLE_ENFORCE_EQ(
        outputs.size(),
        split_channels.size(),
        phi::errors::InvalidArgument(
            "The number of Output(Outputs) of operator 'fused_conv2d_add_act' "
            "is "
            "expected to be equal to the length of Attr(split_channels). But "
            "reiceved: the number of Output(Outputs) = %u; the length of "
            "Attr(split_channels) = %u, the content = [%s].",
            outputs.size(),
            split_channels.size(),
            common::make_ddim(split_channels)));

    int split_channels_sum = 0;
    std::vector<phi::DDim> output_shapes(split_channels.size());
    for (size_t i = 0; i < split_channels.size(); ++i) {
      split_channels_sum += split_channels[i];
      if (channel_last) {
        output_shapes[i] = common::make_ddim(
            {out_shape[0], out_shape[1], out_shape[2], split_channels[i]});
      } else {
        output_shapes[i] = common::make_ddim(
            {out_shape[0], split_channels[i], out_shape[2], out_shape[3]});
      }
    }
    int output_channels = out_shape[1];
    // for NHWC
    if (channel_last) output_channels = out_shape[3];
    PADDLE_ENFORCE_EQ(
        split_channels_sum,
        output_channels,
        phi::errors::InvalidArgument(
            "The sum of Attr(split_channels) is expected to be equal to "
            "the "
            "total output split_channels. But received: the sum of "
            "Attr(split_channels) = %d, the total output split_channels = %d.",
            split_channels_sum,
            output_channels));
    for (size_t i = 0; i < outputs.size(); ++i) {
      if (outputs[i]) {
        outputs[i]->set_dims(output_shapes[i]);
        outputs[i]->set_dtype(input.dtype());
        if (data_format == "NHWC") {
          outputs[i]->set_layout(phi::DataLayout::NHWC);
        } else if (data_format == "NDHWC") {
          outputs[i]->set_layout(phi::DataLayout::NDHWC);
        }

        outputs[i]->share_lod(input);
      }
    }
  }
}

void FusionRepeatedFCReluInferMeta(const MetaTensor& x,
                                   const std::vector<const MetaTensor*>& w,
                                   const std::vector<const MetaTensor*>& bias,
                                   std::vector<MetaTensor*> relu_out,
                                   MetaTensor* out) {
  auto sz = w.size();
  PADDLE_ENFORCE_GT(sz,
                    1UL,
                    phi::errors::InvalidArgument(
                        "Inputs(W) of FusionRepeatedFCReluOp should "
                        "be greater than 1, but received value is %d.",
                        sz));
  PADDLE_ENFORCE_EQ(
      bias.size(),
      sz,
      phi::errors::InvalidArgument(
          "Size of inputs(Bias) of FusionRepeatedFCReluOp should be "
          "equal to inputs size %d, but received value is %d.",
          sz,
          bias.size()));
  PADDLE_ENFORCE_EQ(
      relu_out.size(),
      sz - 1,
      phi::errors::InvalidArgument(
          "Size of output(ReluOut) of FusionRepeatedFCReluOp should "
          "be equal to inputs size minus one %d, but received value is %d",
          sz - 1,
          relu_out.size()));

  auto i_dims = x.dims();
  PADDLE_ENFORCE_EQ(
      i_dims.size(),
      2,
      phi::errors::InvalidArgument(
          "Input shape size should be 2, but received value is %d.",
          i_dims.size()));

  std::vector<DDim> w_dims, b_dims;
  w_dims.reserve(w.size());
  std::transform(w.begin(),
                 w.end(),
                 std::back_inserter(w_dims),
                 [](const MetaTensor* var) { return var->dims(); });

  b_dims.reserve(bias.size());
  std::transform(bias.begin(),
                 bias.end(),
                 std::back_inserter(b_dims),
                 [](const MetaTensor* var) { return var->dims(); });

  PADDLE_ENFORCE_EQ(w_dims.size(),
                    b_dims.size(),
                    phi::errors::InvalidArgument(
                        "Shape size of weight and bias should be equal, but "
                        "weight size is %d, bias size is %d.",
                        w_dims.size(),
                        b_dims.size()));
  PADDLE_ENFORCE_EQ(i_dims[1],
                    w_dims[0][0],
                    phi::errors::InvalidArgument(
                        "input width should be equal to weight height, but "
                        "input width is %d, weight height is %d.",
                        i_dims[1],
                        w_dims[0][0]));

  for (size_t i = 1; i < sz; ++i) {
    PADDLE_ENFORCE_EQ(w_dims[i].size(),
                      2,
                      phi::errors::InvalidArgument(
                          "Every weight shape size should be 2, but received "
                          "w_dims[%d].size() = %d.",
                          i,
                          w_dims[i].size()));
    PADDLE_ENFORCE_EQ(
        common::product(b_dims[i]),
        w_dims[i][1],
        phi::errors::InvalidArgument(
            "The length of Bias must be equal with w_dims[1], but received "
            "product(b_dims[%d]) = %d, w_dims[%d][1] = %d.",
            i,
            common::product(b_dims[i]),
            i,
            w_dims[i][1]));
  }
  out->set_dims({i_dims[0], w_dims[sz - 1][1]});
  out->share_lod(x);
  out->set_dtype(x.dtype());
}

void FusionSquaredMatSubInferMeta(const MetaTensor& x,
                                  const MetaTensor& y,
                                  const float scalar,
                                  MetaTensor* squared_x,
                                  MetaTensor* squared_y,
                                  MetaTensor* squared_xy,
                                  MetaTensor* out) {
  auto x_dims = x.dims();
  auto y_dims = y.dims();
  PADDLE_ENFORCE_EQ(
      x_dims.size(),
      y_dims.size(),
      phi::errors::InvalidArgument("The input tensor X's dims size should "
                                   "be equal to Y's. But received X's "
                                   "dims size = %d, Y's dims size = %d.",
                                   x_dims.size(),
                                   y_dims.size()));
  PADDLE_ENFORCE_EQ(x_dims.size(),
                    2,
                    phi::errors::InvalidArgument(
                        "The input tensor X's dims size should be 2. But "
                        "received X's dims size = %d.",
                        x_dims.size()));
  PADDLE_ENFORCE_EQ(
      x_dims[1],
      y_dims[0],
      phi::errors::InvalidArgument("The input tensor X's dims[1] should "
                                   "be equal to Y's dims[0]. But received "
                                   "X's dims[1] = %d, Y's dims[0] = %d.",
                                   x_dims[1],
                                   y_dims[0]));
  squared_x->set_dims(x_dims);
  squared_x->set_dtype(x.dtype());
  squared_y->set_dims(y_dims);
  squared_y->set_dtype(x.dtype());
  squared_xy->set_dims({x_dims[0], y_dims[1]});
  squared_xy->set_dtype(x.dtype());
  out->set_dims({x_dims[0], y_dims[1]});
  out->set_dtype(x.dtype());
}

void FusionGRUInferMeta(const MetaTensor& x,
                        const MetaTensor& h0,
                        const MetaTensor& weight_x,
                        const MetaTensor& weight_h,
                        const MetaTensor& bias,
                        const std::string& activation,
                        const std::string& gate_activation,
                        const bool is_reverse,
                        const bool use_seq,
                        const bool origin_mode,
                        const bool use_mkldnn,
                        const std::string& mkldnn_data_type,
                        const float scale_data,
                        const float shift_data,
                        const std::vector<float>& scale_weights,
                        const bool force_fp32_output,
                        MetaTensor* reordered_h0,
                        MetaTensor* xx,
                        MetaTensor* batched_input,
                        MetaTensor* batched_out,
                        MetaTensor* hidden) {
  std::string mkldnn_data_type_list[] = {"float32", "int8", "bfloat16"};
  PADDLE_ENFORCE_EQ(
      std::find(std::begin(mkldnn_data_type_list),
                std::end(mkldnn_data_type_list),
                mkldnn_data_type) != std::end(mkldnn_data_type_list),
      true,
      phi::errors::InvalidArgument("The mkldnn_data_type shoule be [float32, "
                                   "int8, bfloat16], but found %s.",
                                   mkldnn_data_type.c_str()));

  DDim x_dims = x.dims();
  auto x_mat_dims = (x_dims.size() == 3 && x_dims[1] == 1)
                        ? common::flatten_to_2d(x_dims, 1)
                        : x_dims;
  PADDLE_ENFORCE_EQ(
      x_mat_dims.size(),
      2,
      phi::errors::InvalidArgument("The size of input X dims should be 2, "
                                   "or 3 with second dimension equal to "
                                   "1, but now Input X dim is:[%s] ",
                                   x_dims));

  auto wx_dims = weight_x.dims();
  PADDLE_ENFORCE_EQ(wx_dims.size(),
                    2,
                    phi::errors::InvalidArgument(
                        "The rank of Input(WeightX) should be 2, but received "
                        "WeightX dim size is:%d, WeightX dim is:[%s] ",
                        wx_dims.size(),
                        wx_dims));
  PADDLE_ENFORCE_EQ(
      wx_dims[0],
      x_mat_dims[1],
      phi::errors::InvalidArgument(
          "The first dimension of flattened WeightX"
          "should equal to last dimension of flattened input X, but "
          "received fattened WeightX dimension is:%d, flattened X dimension "
          "is:%d",
          wx_dims[0],
          x_mat_dims[1]));

  int frame_size = static_cast<int>(wx_dims[1] / 3);
  auto wh_dims = weight_h.dims();

  PADDLE_ENFORCE_EQ(wh_dims.size(),
                    2,
                    phi::errors::InvalidArgument(
                        "The rank of Input(WeightH) should be 2, but received "
                        "WeightH dim size is:%d, WeightH dim is:[%s]",
                        wh_dims.size(),
                        wh_dims));
  PADDLE_ENFORCE_EQ(wh_dims[0],
                    frame_size,
                    phi::errors::InvalidArgument(
                        "The first dimension of WeightH "
                        "should equal to frame_size, but received WeightH's "
                        "first dimension is: "
                        "%d, frame size is:%d",
                        wh_dims[0],
                        frame_size));
  PADDLE_ENFORCE_EQ(wh_dims[1],
                    3 * frame_size,
                    phi::errors::InvalidArgument(
                        "The second dimension of Input(WeightH) "
                        "should equal to 3 * frame_size, but received WeightH "
                        "is:%d, frame size is:%d",
                        wh_dims[1],
                        frame_size));

  if (h0) {
    auto h0_dims = h0.dims();
    PADDLE_ENFORCE_EQ(h0_dims[1],
                      frame_size,
                      phi::errors::InvalidArgument(
                          "The width of H0 must be equal to frame_size, but "
                          "receiced the width of H0 is:%d, frame size is:%d",
                          h0_dims[1],
                          frame_size));
    reordered_h0->set_dtype(x.dtype());
  }
  if (bias) {
    auto b_dims = bias.dims();
    PADDLE_ENFORCE_EQ(b_dims.size(),
                      2,
                      phi::errors::InvalidArgument(
                          "The rank of Input(Bias) should be 2, but received "
                          "Bias rank is:%d, Bias dim is:[%s]",
                          b_dims.size(),
                          b_dims));
    PADDLE_ENFORCE_EQ(b_dims[0],
                      1,
                      phi::errors::InvalidArgument(
                          "The first dimension of Input(Bias) should be 1, but "
                          "received Bias first dim is:%d, Bias dim is:[%s]",
                          b_dims[0],
                          b_dims));
    PADDLE_ENFORCE_EQ(b_dims[1],
                      frame_size * 3,
                      phi::errors::InvalidArgument(
                          "The shape of Bias must be [1, frame_size * 3], but "
                          "received bias dim is:[%s], frame size is:%d",
                          b_dims,
                          frame_size));
  }
  DDim out_dims({x_mat_dims[0], frame_size});
  hidden->set_dims(out_dims);
  hidden->share_lod(x);
  hidden->set_dtype(x.dtype());
  int xx_width = 0;
  if (use_seq) {
    xx_width = static_cast<int>(wx_dims[1]);
  } else {
    xx_width = static_cast<int>(x_mat_dims[1] > wx_dims[1] ? wx_dims[1]
                                                           : x_mat_dims[1]);
    batched_input->set_dims({x_mat_dims[0], wx_dims[1]});
    batched_input->set_dtype(x.dtype());
    batched_out->set_dims(out_dims);
    batched_out->set_dtype(x.dtype());
  }
  xx->set_dims({x_mat_dims[0], xx_width});
  xx->set_dtype(x.dtype());
  xx->share_lod(x);
}

void FusionSeqConvEltAddReluInferMeta(const MetaTensor& x,
                                      const MetaTensor& filter,
                                      const MetaTensor& bias,
                                      const int context_length,
                                      const int context_start,
                                      const int context_stride,
                                      MetaTensor* out,
                                      MetaTensor* col_mat) {
  auto x_dims = x.dims();
  auto w_dims = filter.dims();
  PADDLE_ENFORCE_GT(
      context_length,
      0,
      phi::errors::InvalidArgument("context_length should be greater than 0, "
                                   "but received context_length is: %d",
                                   context_length));
  PADDLE_ENFORCE_EQ(context_stride,
                    1,
                    phi::errors::InvalidArgument(
                        "Currently, FusionSeqConvEltAddReluOp only supports "
                        "contextStride=1, but received value is: %d.",
                        context_stride));

  PADDLE_ENFORCE_EQ(
      x_dims.size(),
      2,
      phi::errors::InvalidArgument(
          "Input(X) should be 2-D tensor, but reveiced value is: %d.",
          x_dims.size()));

  PADDLE_ENFORCE_EQ(
      w_dims.size(),
      2,
      phi::errors::InvalidArgument(
          "Filter should be 2-D tensor, but reveiced value is: %d.",
          w_dims.size()));

  PADDLE_ENFORCE_EQ(w_dims[0],
                    context_length * x_dims[1],
                    phi::errors::InvalidArgument(
                        "Filter's height should be equal to context_length * "
                        "input_hidden_size, but received Filter height is: %d,"
                        "context_length is: %d, input_hidden_size is: %d.",
                        w_dims[0],
                        context_length,
                        x_dims[1]));

  PADDLE_ENFORCE_GT(
      context_length + context_start,
      0,
      phi::errors::InvalidArgument(
          "contextStart size should be smaller than contextLength, "
          "but received context_length is: %d, contextStart is: "
          "%d.",
          context_length,
          context_start));
  out->set_dims({x_dims[0], w_dims[1]});
  col_mat->set_dims({x_dims[0], w_dims[0]});
  out->share_lod(x);
  col_mat->set_dtype(x.dtype());
  out->set_dtype(x.dtype());
}

void FusionSeqExpandConcatFCInferMeta(const std::vector<const MetaTensor*>& x,
                                      const MetaTensor& fc_weight,
                                      const MetaTensor& fc_bias,
                                      const std::string& fc_activation,
                                      MetaTensor* out,
                                      MetaTensor* fc_out) {
  PADDLE_ENFORCE_GT(x.size(),
                    1UL,
                    phi::errors::InvalidArgument(
                        "Inputs(X) of FusionSeqExpandConcatFCOp should larger "
                        "than 1, but received value is: %d.",
                        x.size()));

  std::vector<DDim> ins_dims;
  ins_dims.reserve(x.size());
  std::transform(x.begin(),
                 x.end(),
                 std::back_inserter(ins_dims),
                 [](const MetaTensor* var) { return var->dims(); });

  auto w_dims = fc_weight.dims();  // (M0+M1+M2+..) x D
  PADDLE_ENFORCE_EQ(
      w_dims.size(),
      2,
      phi::errors::InvalidArgument(
          "Input(FCWeight)'s rank must be 2, but received value is: %d.",
          w_dims.size()));
  const int D = static_cast<int>(w_dims[1]);
  int sum = static_cast<int>(ins_dims[0][1]);
  for (size_t i = 1; i < ins_dims.size(); ++i) {
    sum += static_cast<int>(ins_dims[i][1]);
  }
  PADDLE_ENFORCE_EQ(
      sum,
      w_dims[0],
      phi::errors::InvalidArgument("FC height should be sum of all inputs "
                                   "width, but received FC height is: %d, "
                                   "sum of all inputs width is: %d.",
                                   w_dims[0],
                                   sum));
  if (fc_bias) {
    auto b_dims = fc_bias.dims();
    PADDLE_ENFORCE_EQ(
        b_dims.size() == 1 || b_dims.size() == 2,
        true,
        phi::errors::InvalidArgument(
            "FCBias dim should be 1 or 2, but received value is: %d.",
            b_dims.size()));
    if (b_dims.size() == 1) {
      PADDLE_ENFORCE_EQ(b_dims[0],
                        D,
                        phi::errors::InvalidArgument(
                            "FCBias shapes must be %d when FCBias dim = 1, but "
                            "received value is: %d.",
                            D,
                            b_dims[0]));
    } else {
      PADDLE_ENFORCE_EQ(b_dims[0],
                        1,
                        phi::errors::InvalidArgument(
                            "FCBias shapes must be 1x%d, when FCBias dim = 2, "
                            "but received dim[0] is: %d.",
                            D,
                            b_dims[0]));
      PADDLE_ENFORCE_EQ(b_dims[1],
                        D,
                        phi::errors::InvalidArgument(
                            "FCBias shapes must be 1x%d, when FCBias dim = 2, "
                            "but received dim[1] is: %d.",
                            D,
                            b_dims[1]));
    }
  }
  fc_out->set_dtype((*x[0]).dtype());
  out->set_dims({ins_dims[0][0], D});
  out->set_dtype((*x[0]).dtype());
  // fcout should be reshape when run since can not get lod in infershape
  // explicit share the ref lod
  out->share_lod(*x[0]);
}

void FCInferMeta(const MetaTensor& input,
                 const MetaTensor& w,
                 const MetaTensor& bias,
                 const int in_num_col_dims,
                 const std::string& activation_type,
                 const bool use_mkldnn,
                 const bool padding_weights,
                 const bool use_quantizer,
                 const std::string& mkldnn_data_type,
                 const float scale_in,
                 const std::vector<float>& sclae_weights,
                 const float scale_out,
                 const bool force_fp32_output,
                 MetaTensor* out) {
  PADDLE_ENFORCE_GE(
      in_num_col_dims,
      1,
      phi::errors::InvalidArgument(
          "The in_num_col_dims is expected to equal or greater than 1. "
          "But received the in_num_col_dims is %d. ",
          in_num_col_dims));
  std::string mkldnn_data_type_list[] = {"float32", "int8", "bfloat16"};
  PADDLE_ENFORCE_EQ(
      std::find(std::begin(mkldnn_data_type_list),
                std::end(mkldnn_data_type_list),
                mkldnn_data_type) != std::end(mkldnn_data_type_list),
      true,
      phi::errors::InvalidArgument("The mkldnn_data_type shoule be [float32, "
                                   "int8, bfloat16], but found %s.",
                                   mkldnn_data_type.c_str()));
  auto w_dims = w.dims();
  PADDLE_ENFORCE_EQ(
      w_dims.size(),
      2,
      phi::errors::InvalidArgument(
          "The input Weight of fc is expected to be a 2-D tensor. "
          "But received the number of Weight's dimensions is %d, "
          "Weight's shape is %s.",
          w_dims.size(),
          w_dims));

  if (bias) {
    auto bias_dims = bias.dims();
    auto w_dims1 = padding_weights ? w_dims[1] - 4 : w_dims[1];

    PADDLE_ENFORCE_LE(
        bias_dims.size(),
        2,
        phi::errors::InvalidArgument(
            "The input Bias of fc is expected to be a 1-D or 2-D tensor. But "
            "received the number of Bias's dimensions is %d, "
            "Bias's shape is %s.",
            bias_dims.size(),
            bias_dims));

    PADDLE_ENFORCE_EQ(
        bias_dims[bias_dims.size() - 1],
        w_dims1,
        phi::errors::InvalidArgument(
            "The last dimension of input Bias is expected be equal "
            "to the actual width of input Weight. But received the last "
            "dimension of Bias is %d, Bias's shape is %s; "
            "the actual width of Weight is %d, Weight's shape is %s.",
            bias_dims[bias_dims.size() - 1],
            bias_dims,
            w_dims1,
            w_dims));

    if (bias_dims.size() == 2) {
      PADDLE_ENFORCE_EQ(
          bias_dims[0],
          1,
          phi::errors::InvalidArgument(
              "The first dimension of input Bias is expected to be 1, "
              "but received %d, Bias's shape is %s.",
              bias_dims[0],
              bias_dims));
    }
  }

  auto in_dims = input.dims();
  PADDLE_ENFORCE_LT(
      in_num_col_dims,
      in_dims.size(),
      phi::errors::InvalidArgument(
          "The attribute in_num_col_dims used to flatten Input to "
          "a 2-D tensor, is expected to be less than the number of "
          "Input's dimensions. But received in_num_col_dims is %d, "
          "the number of Input's dimensions is %d, Input's shape is %s.",
          in_num_col_dims,
          in_dims.size(),
          in_dims));

  if (!activation_type.empty()) {
    PADDLE_ENFORCE_EQ(activation_type,
                      "relu",
                      phi::errors::InvalidArgument(
                          "The attribute activation_type of fc is expected "
                          "to be \"relu\", but received %s.",
                          activation_type.c_str()));
  }

  if (use_mkldnn) {
    PADDLE_ENFORCE_EQ(
        in_dims.size() >= 2 && in_dims.size() <= 4,
        true,
        phi::errors::Unimplemented(
            "The Input of fc is expected to be a 2-D, 3-D or 4-D tensor when "
            "use_mkldnn is set. But received the number of Input's "
            "dimensions is %d, Input's shape is %s.",
            in_dims.size(),
            in_dims));
  }

  std::vector<int64_t> output_dims;
  phi::funcs::FCOutputSize(
      in_dims, w_dims, output_dims, in_num_col_dims, padding_weights);

  out->set_dims(common::make_ddim(output_dims));
  out->share_lod(input);
  out->set_dtype(input.dtype());
}

void SelfDPAttenInferMeta(const MetaTensor& x,
                          const float alpha,
                          const int head_number,
                          MetaTensor* out) {
  auto dim_input = x.dims();
  PADDLE_ENFORCE_EQ(
      dim_input.size(),
      5,
      phi::errors::InvalidArgument("The size of input X dims should be 5, "
                                   "[batchsize, tokensize, 3, nhead, headsize] "
                                   ", but now Input X dim is:[%s] ",
                                   dim_input));
  DDim out_dims({dim_input[0], dim_input[1], dim_input[3], dim_input[4]});
  out->set_dims(out_dims);
  out->share_lod(x);
  out->set_dtype(x.dtype());
}

void SkipLayerNormInferMeta(const MetaTensor& x,
                            const MetaTensor& y,
                            const MetaTensor& scale,
                            const MetaTensor& bias,
                            const float epsilon,
                            const int begin_norm_axis,
                            MetaTensor* out) {
  auto dim_input = x.dims();
  out->set_dims(dim_input);
  out->share_lod(x);
  out->set_dtype(x.dtype());
}

void VariableLengthMemoryEfficientAttentionInferMeta(
    const MetaTensor& query,
    const MetaTensor& key,
    const MetaTensor& value,
    const MetaTensor& seq_lens,
    const MetaTensor& kv_seq_lens,
    const MetaTensor& mask,
    float scale,
    bool causal,
    int pre_cache_length,
    MetaTensor* out) {
  PADDLE_ENFORCE_EQ(
      query.dims().size(),
      4,
      phi::errors::InvalidArgument("Query should be a 4-D tensor"
                                   "But received Query dimension(%s)",
                                   query.dims().size()));
  PADDLE_ENFORCE_EQ(
      key.dims().size(),
      4,
      phi::errors::InvalidArgument("Key should be a 4-D tensor"
                                   "But received Key dimension(%s)",
                                   key.dims().size()));
  PADDLE_ENFORCE_EQ(
      value.dims().size(),
      4,
      phi::errors::InvalidArgument("Value should be a 4-D tensor"
                                   "But received Value dimension(%s)",
                                   value.dims().size()));

  const int64_t query_batch_size = query.dims()[0];
  const int64_t query_num_head = query.dims()[1];
  const int64_t query_seq_length = query.dims()[2];
  const int64_t query_head_size = query.dims()[3];

  const int64_t key_batch_size = key.dims()[0];
  const int64_t key_num_head = key.dims()[1];
  const int64_t key_seq_length = key.dims()[2];
  const int64_t key_head_size = key.dims()[3];

  const int64_t value_batch_size = value.dims()[0];
  const int64_t value_num_head = value.dims()[1];
  const int64_t value_seq_length = value.dims()[2];
  const int64_t value_head_size = value.dims()[3];

  PADDLE_ENFORCE_EQ(
      ((query_batch_size == key_batch_size) &&
       (key_batch_size == value_batch_size)),
      true,
      phi::errors::InvalidArgument(
          "The batch size of Query, Key, Value should be equal."));

  PADDLE_ENFORCE_EQ((key_num_head == value_num_head),
                    true,
                    phi::errors::InvalidArgument(
                        "The head number of Key, Value should be equal."));

  PADDLE_ENFORCE_EQ(
      query_num_head % key_num_head,
      0,
      errors::InvalidArgument(
          "The num_head of query must be divisible by the num_head of key, but "
          "recived num_head of query is %d, and the num_head of key is %d",
          query_num_head,
          key_num_head));

  PADDLE_ENFORCE_EQ(query_head_size == key_head_size,
                    true,
                    phi::errors::InvalidArgument(
                        "The head size of Query, Key should be equal."));

  PADDLE_ENFORCE_EQ(key_seq_length == value_seq_length,
                    true,
                    phi::errors::InvalidArgument(
                        "The seq length of Key, Value should be equal."));

  std::vector<int64_t> out_dims(
      {query_batch_size, query_num_head, query_seq_length, value_head_size});

  out->set_dims(phi::make_ddim(out_dims));
  out->set_dtype(query.dtype());
  out->set_layout(query.layout());
}

}  // namespace phi
