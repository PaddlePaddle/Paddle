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

void AddCMulXPUInferMeta(const MetaTensor& x,
                         const MetaTensor& y,
                         const MetaTensor& w,
                         MetaTensor* out) {
  out->set_dims(x.dims());
  out->set_dtype(x.dtype());
  out->set_layout(x.layout());
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
  out->set_dims(DDim(out_shape.data(), out_shape.size()));
}

}  // namespace phi
