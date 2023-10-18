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
#include <vector>
#include "paddle/phi/common/layout.h"
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

    return phi::make_ddim(out_dims_array);
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
  out_max->set_dims(phi::make_ddim({6}));
  out_max->set_dtype(x.dtype());
  out_max->set_layout(x.layout());
}

void AddLayernormXPUInferMeta(const MetaTensor& x,
                              const MetaTensor& y,
                              const MetaTensor& scale,
                              const MetaTensor& bias,
                              int begin_norm_axis,
                              float epsilon,
                              int act_type,
                              float act_param,
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

inline int ConvOutSize(int input_size,
                       int filter_size,
                       int dilation,
                       int pad_left,
                       int pad_right,
                       int stride) {
  const int dkernel = dilation * (filter_size - 1) + 1;
  int output_size =
      (input_size + (pad_left + pad_right) - dkernel) / stride + 1;

  return output_size;
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
  out_max->set_dims(phi::make_ddim({6}));
}

void Conv2dXPUInferMeta(const MetaTensor& x,
                        const MetaTensor& x_max,
                        const MetaTensor& filter,
                        const MetaTensor& filter_max,
                        const MetaTensor& bias,
                        const MetaTensor& branch,
                        const MetaTensor& branch_max,
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
          phi::make_ddim(strides),
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
  DDim in_data_dims = phi::slice_ddim(in_dims, 2, in_dims.size());
  DDim filter_data_dims = phi::slice_ddim(filter_dims, 2, filter_dims.size());
  std::vector<int> ksize = phi::vectorize<int>(filter_data_dims);
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
  out_max->set_dims(phi::make_ddim({6}));
  out->set_dtype(out_dtype);
}

inline int PoolOutputSize(int input_size,
                          int k_size,
                          int padding_left,
                          int padding_right,
                          int stride) {
  PADDLE_ENFORCE_NE(
      stride,
      0,
      phi::errors::InvalidArgument(
          "The stride of MaxPool shall not be 0, but received %d.", stride));
  int output_size =
      (input_size - k_size + padding_left + padding_right) / stride + 1;
  return output_size;
}

void Conv2dPoolingXPUInferMeta(const MetaTensor& x,
                               const MetaTensor& x_max,
                               const MetaTensor& filter,
                               const MetaTensor& filter_max,
                               const MetaTensor& bias,
                               const std::vector<int>& paddings,
                               const std::vector<int>& dilations,
                               const std::vector<int>& strides,
                               const std::string& padding_algorithm,
                               int groups,
                               int act_type,
                               float act_param,
                               DataType out_dtype,
                               const std::vector<int>& pool2d_paddings,
                               const std::vector<int>& pool2d_strides,
                               const std::vector<int>& pool2d_ksize,
                               bool is_avg,
                               MetaTensor* out,
                               MetaTensor* out_max) {
  auto in_dims = x.dims();
  auto filter_dims = filter.dims();
  // do some checks
  PADDLE_ENFORCE_EQ(
      in_dims.size(),
      4,
      phi::errors::InvalidArgument(
          "The input of Op(conv_pooling_xpu) should be a 4-D Tensor. But "
          "received: input's dimension is %u, input's shape is [%s].",
          in_dims.size(),
          in_dims));

  PADDLE_ENFORCE_EQ(
      in_dims.size(),
      filter_dims.size(),
      phi::errors::InvalidArgument(
          "The input's dimension and filter's dimension of "
          "Op(conv_pooling_xpu) should be equal. But received: the input's "
          "shape is "
          "[%s], "
          "the input's dimension is %d; the filter's shape is [%s],  "
          "the filter's dimension is %d.",
          in_dims,
          in_dims.size(),
          filter_dims,
          filter_dims.size()));

  const auto input_channels = in_dims[1];
  int stride_size = strides.size();
  int in_sub_stride_size = in_dims.size() - stride_size;
  int dilation_size = dilations.size();
  PADDLE_ENFORCE_EQ(
      in_dims.size(),
      strides.size() + 2U,
      phi::errors::InvalidArgument(
          "The difference of input's dimension and Attr(strides)'s "
          "length must be euqal to 2 for Op(conv_pooling_xpu). "
          "But received: input's dimension is %d, input's shape is [%s]; "
          "Attr(stride)'s length is %d, Attr(stride) is [%s]; "
          "difference of input's dimention and Attr(strides)'s length = %u.",
          in_dims.size(),
          in_dims,
          strides.size(),
          phi::make_ddim(strides),
          in_sub_stride_size));

  for (int i = 0; i < dilation_size; ++i) {
    PADDLE_ENFORCE_GT(
        dilations[i],
        0,
        phi::errors::InvalidArgument("The dilation of Op(conv_pooling_xpu) "
                                     "should be larget than 0, but received "
                                     "dilation is %d.",
                                     dilations[i]));
  }

  PADDLE_ENFORCE_EQ(
      input_channels,
      filter_dims[1] * groups,
      phi::errors::InvalidArgument(
          "The number of input's channels should be equal to filter's channels "
          "* groups for Op(conv_pooling_xpu). But received: the input's "
          "channels is "
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
          "Op(conv_pooling_xpu) should be divided by groups. But received: "
          "the output channels is %d, the filter's shape is [%s], "
          "the groups is %d.",
          filter_dims[0],
          filter_dims,
          groups));

  // update paddings and dilations accoring to padding_algorithm
  std::vector<int> paddings_vec = paddings;
  std::vector<int> dilations_vec = dilations;
  DDim in_data_dims = phi::slice_ddim(in_dims, 2, in_dims.size());
  DDim filter_data_dims = phi::slice_ddim(filter_dims, 2, filter_dims.size());
  std::vector<int> ksize = phi::vectorize<int>(filter_data_dims);
  phi::UpdatePaddingAndDilation(&paddings_vec,
                                &dilations_vec,
                                padding_algorithm,
                                in_data_dims,
                                strides,
                                ksize);

  std::vector<int64_t> out_shape({in_dims[0], filter_dims[0]});
  for (size_t i = 0; i < strides.size(); ++i) {
    out_shape.push_back(ConvOutSize(in_dims[i + 2],
                                    filter_dims[i + 2],
                                    dilations[i],
                                    paddings_vec[i * 2],
                                    paddings_vec[i * 2 + 1],
                                    strides[i]));
  }
  // compute pool2d output size
  for (size_t i = 0; i < pool2d_strides.size(); ++i) {
    auto insize = out_shape[i + 2];
    out_shape[i + 2] = PoolOutputSize(insize,
                                      pool2d_ksize[i],
                                      pool2d_paddings[i * 2],
                                      pool2d_paddings[i * 2 + 1],
                                      pool2d_strides[i]);
  }
  // set output and output max dims
  out->set_dims(DDim(out_shape.data(), out_shape.size()));
  out->set_dtype(out_dtype);
  out_max->set_dims(phi::make_ddim({6}));
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
  out->set_dims(phi::make_ddim({id_dims[0], id_dims[1], table_dims[1]}));
  out->set_dtype(tables[0]->dtype());
  out->set_layout(ids[0]->layout());
}

void FcXPUInferMeta(const MetaTensor& x,
                    const MetaTensor& x_max,
                    const MetaTensor& w,
                    const MetaTensor& w_max,
                    const MetaTensor& bias,
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
  out_max->set_dims(phi::make_ddim({6}));
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
    return make_ddim({1, x_dim[0]});
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

  auto mean_dim = phi::make_ddim({mat_dim_x.batch_size_ * mat_dim_x.height_});
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

  auto x_mat_dims = phi::flatten_to_2d(x_dims, trans_x ? 1 : x_dims.size() - 1);

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
  out->set_dims(phi::make_ddim(out_dims));
  out->set_dtype(x.dtype());

  if (reserve_space) {
    reserve_space->set_dims(phi::make_ddim(out_dims));
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

  auto dout_mat_dims = phi::flatten_to_2d(dout_dims, dout_dims.size() - 1);
  auto x_mat_dims = phi::flatten_to_2d(x_dims, x_dims.size() - 1);

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
    bias_grad->set_dims(phi::make_ddim({dbias_dim}));
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
  auto left_slice_out_dims = phi::make_ddim(left_slice_out_dims_vector);
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
  auto mid_slice_out_dims = phi::make_ddim(mid_slice_out_dims_vector);
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
  auto right_slice_out_dims = phi::make_ddim(right_slice_out_dims_vector);
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
  out_max->set_dims(phi::make_ddim({6}));
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
  std::vector<int> ksize = vectorize<int>(filter_data_dims);
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

  out->set_dims(make_ddim(output_shape));
  out->set_dtype(x.dtype());
  out_max->set_dims(phi::make_ddim({6}));
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

  const DataLayout data_layout_str = phi::StringToDataLayout(data_layout);

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
      (phi::product(scale_dim) <= 0 || phi::product(bias_dim) <= 0)) {
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

void FusedScaleBiasReluConvBnstatsInferMeta(
    const MetaTensor& x,
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
          "The input of Op(FusedScaleBiasReluConvBnstats) should be a 4-D "
          "Tensor. But "
          "received: input's dimension is %u, input's shape is [%s].",
          in_dims.size(),
          in_dims));

  PADDLE_ENFORCE_EQ(
      in_dims.size(),
      filter_dims.size(),
      phi::errors::InvalidArgument(
          "The input's dimension and filter's dimension of "
          "Op(FusedScaleBiasReluConvBnstats) should be equal. But received: "
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
          "Operator(FusedScaleBiasReluConvBnstats) only supports data format "
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
          "* groups for Op(FusedScaleBiasReluConvBnstats). But received: the "
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
  DDim in_data_dims = phi::slice_ddim(in_dims, 1, in_dims.size() - 1);
  DDim filter_data_dims = phi::slice_ddim(filter_dims, 2, filter_dims.size());
  std::vector<int> ksize = phi::vectorize<int>(filter_data_dims);
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
  auto c_dims = phi::make_ddim({filter_dims[0]});
  // set output and output max dims
  out->set_dims(DDim(out_shape.data(), static_cast<int>(out_shape.size())));
  out_running_mean->set_dims(c_dims);
  out_running_var->set_dims(c_dims);
  saved_mean->set_dims(c_dims);
  saved_var->set_dims(c_dims);
  eq_scale->set_dims(c_dims);
  eq_bias->set_dims(c_dims);
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

  auto dim_output = phi::make_ddim({batch, seq_len, hidden});
  out->set_dims(dim_output);
  // out->share_lod(ids);
  // context->ShareLoD("Ids", /*->*/ "Out");
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
  out->set_dims(phi::make_ddim(out_dims));
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

  auto x_mat_dims = phi::flatten_to_2d(x_dims, x_num_col_dims);
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
  PADDLE_ENFORCE_EQ(phi::make_ddim(fc_out_dims),
                    y_dims,
                    phi::errors::InvalidArgument(
                        "The output's shape of fc is expected to be equal to "
                        "that of input Y. But received output's shape of fc "
                        "is %s, input Y's shape is %s.",
                        phi::make_ddim(fc_out_dims),
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

  auto y_mat_dim = phi::flatten_to_2d(y_dims, begin_norm_axis);
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
  if (mean) {
    mean->set_dims({dim_0});
  }
  if (variance) {
    variance->set_dims({dim_0});
  }
  out->share_lod(x);
}

}  // namespace phi
