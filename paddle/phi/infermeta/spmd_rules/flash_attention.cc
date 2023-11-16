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

#include "paddle/phi/infermeta/spmd_rules/flash_attention.h"

#include "glog/logging.h"

#include "paddle/phi/core/enforce.h"
#include "paddle/phi/infermeta/spmd_rules/utils.h"

namespace phi {
namespace distributed {
using phi::distributed::auto_parallel::str_join;

SpmdInfo FlashAttInferSpmd(const DistMetaTensor& q,
                           const DistMetaTensor& k,
                           const DistMetaTensor& v,
                           const DistMetaTensor& fixed_seed_offset,
                           const DistMetaTensor& attn_mask,
                           float dropout,
                           bool causal,
                           bool return_softmax,
                           bool is_test,
                           const std::string& rng_name) {
  // q
  // [batch_size, seq_len_q, num_heads, head_dim]
  auto q_shape = phi::vectorize(q.dims());
  int q_ndim = q_shape.size();
  auto q_dist_attr = q.dist_attr();
  int q_dims_mapping_size = q_dist_attr.dims_mapping().size();

  PADDLE_ENFORCE_EQ(
      q_ndim,
      4,
      phi::errors::InvalidArgument("The Tensor q's shape must be [batch_size, "
                                   "seq_len_q, num_heads, head_dim]"));

  auto batch_size = q_shape[0];
  auto seq_len_q = q_shape[1];
  auto num_heads = q_shape[2];
  auto head_dim = q_shape[3];

  PADDLE_ENFORCE_EQ(
      q_ndim,
      q_dims_mapping_size,
      phi::errors::InvalidArgument("The Tensor q's rank [%d] and Its "
                                   "dims_mapping size [%d] are not matched.",
                                   q_ndim,
                                   q_dims_mapping_size));

  // k
  // [batch_size, seq_len_kv, num_heads, head_dim]
  auto k_shape = phi::vectorize(k.dims());
  int k_ndim = k_shape.size();
  auto k_dist_attr = k.dist_attr();
  int k_dims_mapping_size = k_dist_attr.dims_mapping().size();
  PADDLE_ENFORCE_EQ(
      k_ndim,
      4,
      phi::errors::InvalidArgument("The Tensor k's shape must be [batch_size, "
                                   "seq_len_kv, num_heads, head_dim]"));

  auto k_batch_size = q_shape[0];
  auto k_seq_len = q_shape[1];
  auto k_num_heads = q_shape[2];
  auto k_head_dim = q_shape[3];

  PADDLE_ENFORCE_EQ(
      batch_size,
      k_batch_size,
      phi::errors::InvalidArgument(
          "The Tensor q and k's batch size [%d]  vs [%d] are not matched.",
          batch_size,
          k_batch_size));

  PADDLE_ENFORCE_EQ(
      num_heads,
      k_num_heads,
      phi::errors::InvalidArgument(
          "The Tensor q and k's k_num_heads [%d] vs [%d] are not matched.",
          num_heads,
          k_num_heads));

  PADDLE_ENFORCE_EQ(
      head_dim,
      k_head_dim,
      phi::errors::InvalidArgument(
          "The Tensor q and k's head_dim [%d] vs [%d] are not matched.",
          head_dim,
          k_head_dim));

  PADDLE_ENFORCE_EQ(
      k_ndim,
      k_dims_mapping_size,
      phi::errors::InvalidArgument("The Tensor q's rank [%d] and Its "
                                   "dims_mapping size [%d] are not matched.",
                                   k_ndim,
                                   k_dims_mapping_size));

  // v
  // [batch_size, seq_len_kv, num_heads, head_dim]
  auto v_shape = phi::vectorize(v.dims());
  int v_ndim = v_shape.size();
  auto v_dist_attr = v.dist_attr();
  int v_dims_mapping_size = v_dist_attr.dims_mapping().size();
  PADDLE_ENFORCE_EQ(
      v_ndim,
      4,
      phi::errors::InvalidArgument("The Tensor v's shape must be [batch_size, "
                                   "seq_len_kv, num_heads, head_dim_v]"));

  auto v_batch_size = v_shape[0];
  auto v_seq_len = v_shape[1];
  auto v_num_heads = v_shape[2];
  auto v_head_dim = v_shape[3];

  PADDLE_ENFORCE_EQ(
      batch_size,
      v_batch_size,
      phi::errors::InvalidArgument(
          "The Tensor q and v's batch size [%d] vs [%d] are not matched.",
          batch_size,
          v_batch_size));

  PADDLE_ENFORCE_EQ(
      num_heads,
      v_num_heads,
      phi::errors::InvalidArgument(
          "The Tensor q and v's k_num_heads [%d] vs [%d] are not matched.",
          num_heads,
          v_num_heads));

  PADDLE_ENFORCE_EQ(
      k_seq_len,
      v_seq_len,
      phi::errors::InvalidArgument(
          "The Tensor k and v's seq_len [%d] vs [%d] are not matched.",
          k_seq_len,
          v_seq_len));

  PADDLE_ENFORCE_EQ(
      v_ndim,
      v_dims_mapping_size,
      phi::errors::InvalidArgument("The Tensor q's rank [%d] and Its "
                                   "dims_mapping size [%d] are not matched.",
                                   v_ndim,
                                   v_dims_mapping_size));

  // fixed_seed_offset

  // attn_mask
  auto mask_shape = phi::vectorize(attn_mask.dims());
  int mask_ndim = mask_shape.size();
  auto mask_dist_attr = attn_mask.dist_attr();
  int mask_dims_mapping_size = mask_dist_attr.dims_mapping().size();
  PADDLE_ENFORCE_EQ(
      mask_ndim,
      mask_dims_mapping_size,
      phi::errors::InvalidArgument("The Tensor q's rank [%d] and Its "
                                   "dims_mapping size [%d] are not matched.",
                                   mask_ndim,
                                   mask_dims_mapping_size));

  std::string alphabet = "abcdefghijklmnopqrstuvwxyz";
  int used_axes_index = 0;
  char batch_axis = alphabet[used_axes_index++];
  char seq_len_q_axis = alphabet[used_axes_index++];
  char num_heads_axis = alphabet[used_axes_index++];
  char head_dim_axis = alphabet[used_axes_index++];
  char seq_len_kv_axis = alphabet[used_axes_index++];
  char head_dim_v_axis = alphabet[used_axes_index++];

  // [batch_size, seq_len_q, num_heads, head_dim]
  std::string q_axes = {
      batch_axis, seq_len_q_axis, num_heads_axis, head_dim_axis};
  // [batch_size, seq_len_kv, num_heads, head_dim]
  std::string k_axes = {
      batch_axis, seq_len_kv_axis, num_heads_axis, head_dim_axis};
  // [batch_size, seq_len_kv, num_heads, head_dim_v]
  std::string v_axes = {
      batch_axis, seq_len_kv_axis, num_heads_axis, head_dim_v_axis};
  // [batch_size, seq_len_q, num_heads, head_dim_v]
  std::string out_axes = {
      batch_axis, seq_len_q_axis, num_heads_axis, head_dim_v_axis};
  // [batch_size,  num_heads, seq_len_q, seq_len_kv]
  std::string softmax_axes = {
      batch_axis, num_heads_axis, seq_len_q_axis, seq_len_kv_axis};
  // [batch_size,  num_heads, seq_len_q, seq_len_kv]
  std::string softmax_lse_axes = {
      batch_axis, num_heads_axis, seq_len_q_axis, seq_len_kv_axis};

  std::string q_axes_align = q_axes;
  q_axes_align[1] = alphabet[used_axes_index++];
  q_axes_align[3] = alphabet[used_axes_index++];

  std::string k_axes_align = k_axes;
  k_axes_align[1] = alphabet[used_axes_index++];
  k_axes_align[3] = alphabet[used_axes_index++];

  std::string v_axes_align = v_axes;
  v_axes_align[1] = alphabet[used_axes_index++];
  v_axes_align[3] = alphabet[used_axes_index++];

  std::vector<std::pair<std::string, std::vector<int64_t>>> axes_sharding_info;

  axes_sharding_info.emplace_back(q_axes_align, q_dist_attr.dims_mapping());
  axes_sharding_info.emplace_back(k_axes_align, k_dist_attr.dims_mapping());
  axes_sharding_info.emplace_back(v_axes_align, k_dist_attr.dims_mapping());

  auto axis_to_dim_map = ShardingMergeForTensors(axes_sharding_info);

  auto q_dist_attr_dst = CopyTensorDistAttrForOutput(q_dist_attr);
  auto q_dims_mapping = GetDimsMappingForAxes(q_axes, axis_to_dim_map, true);
  q_dist_attr_dst.set_dims_mapping(q_dims_mapping);
  auto k_dist_attr_dst = CopyTensorDistAttrForOutput(k_dist_attr);
  auto k_dims_mapping = GetDimsMappingForAxes(k_axes, axis_to_dim_map, true);
  k_dist_attr_dst.set_dims_mapping(k_dims_mapping);
  auto v_dist_attr_dst = CopyTensorDistAttrForOutput(v_dist_attr);
  auto v_dims_mapping = GetDimsMappingForAxes(v_axes, axis_to_dim_map, true);
  v_dist_attr_dst.set_dims_mapping(v_dims_mapping);

  // TODO(liuzhenhai): process fixed_seed_offset and attn_mask
  auto fixed_seed_offset_dist_attr = fixed_seed_offset.dist_attr();
  auto attn_mask_dist_attr = attn_mask.dist_attr();

  TensorDistAttr out;
  auto out_dims_mapping =
      GetDimsMappingForAxes(out_axes, axis_to_dim_map, true);
  out.set_dims_mapping(out_dims_mapping);
  TensorDistAttr softmax;
  softmax.set_dims_mapping(
      GetDimsMappingForAxes(softmax_axes, axis_to_dim_map, true));
  TensorDistAttr softmax_lse;
  softmax_lse.set_dims_mapping(
      GetDimsMappingForAxes(softmax_lse_axes, axis_to_dim_map, true));
  TensorDistAttr seed_offset =
      CopyTensorDistAttrForOutput(fixed_seed_offset_dist_attr);
  // same as input
  seed_offset.set_dims_mapping(fixed_seed_offset_dist_attr.dims_mapping());

  VLOG(4) << "FlashAttInferSpmd:";
  VLOG(4) << "Einsum Notation: " << q_axes << "," << k_axes << "," << v_axes
          << "-->" << out_axes << "," << softmax_axes << ","
          << softmax_lse_axes;

  VLOG(4) << "q";
  VLOG(4) << "Input shape: [" << str_join(q_shape) << "] "
          << "src_dims_mapping: [" << str_join(q_dist_attr.dims_mapping())
          << "] "
          << "dst_dims_mapping: [" << str_join(q_dims_mapping) << "]";

  VLOG(4) << "k";
  VLOG(4) << "Input shape: [" << str_join(k_shape) << "] "
          << "src_dims_mapping: [" << str_join(k_dist_attr.dims_mapping())
          << "] "
          << "dst_dims_mapping: [" << str_join(v_dims_mapping) << "]";

  VLOG(4) << "v";
  VLOG(4) << "Input shape: [" << str_join(v_shape) << "] "
          << "src_dims_mapping: [" << str_join(v_dist_attr.dims_mapping())
          << "] "
          << "dst_dims_mapping: [" << str_join(v_dims_mapping) << "]";

  VLOG(4) << "Output"
          << " dims_mapping: [" << str_join(out_dims_mapping) << "]";
  VLOG(4) << std::endl;

  return {{q_dist_attr_dst,
           k_dist_attr_dst,
           v_dist_attr_dst,
           fixed_seed_offset_dist_attr,
           attn_mask_dist_attr},
          {out, softmax, softmax_lse, seed_offset}};
}

SpmdInfo FlashAttGradInferSpmd(const DistMetaTensor& q,
                               const DistMetaTensor& k,
                               const DistMetaTensor& v,
                               const DistMetaTensor& out,
                               const DistMetaTensor& softmax_lse,
                               const DistMetaTensor& seed_offset,
                               const DistMetaTensor& attn_mask,
                               const DistMetaTensor& out_grad,
                               float dropout,
                               bool causal) {
  // q
  // [batch_size, seq_len_q, num_heads, head_dim]
  auto q_shape = phi::vectorize(q.dims());
  int q_ndim = q_shape.size();
  auto q_dist_attr = q.dist_attr();
  int q_dims_mapping_size = q_dist_attr.dims_mapping().size();

  PADDLE_ENFORCE_EQ(
      q_ndim,
      4,
      phi::errors::InvalidArgument("The Tensor q's shape must be [batch_size, "
                                   "seq_len_q, num_heads, head_dim]"));

  auto batch_size = q_shape[0];
  auto seq_len_q = q_shape[1];
  auto num_heads = q_shape[2];
  auto head_dim = q_shape[3];

  PADDLE_ENFORCE_EQ(
      q_ndim,
      q_dims_mapping_size,
      phi::errors::InvalidArgument("The Tensor q's rank [%d] and Its "
                                   "dims_mapping size [%d] are not matched.",
                                   q_ndim,
                                   q_dims_mapping_size));

  // k
  // [batch_size, seq_len_kv, num_heads, head_dim]
  auto k_shape = phi::vectorize(k.dims());
  int k_ndim = k_shape.size();
  auto k_dist_attr = k.dist_attr();
  int k_dims_mapping_size = k_dist_attr.dims_mapping().size();
  PADDLE_ENFORCE_EQ(
      k_ndim,
      4,
      phi::errors::InvalidArgument("The Tensor k's shape must be [batch_size, "
                                   "seq_len_kv, num_heads, head_dim]"));

  auto k_batch_size = q_shape[0];
  auto k_seq_len = q_shape[1];
  auto k_num_heads = q_shape[2];
  auto k_head_dim = q_shape[3];

  PADDLE_ENFORCE_EQ(
      batch_size,
      k_batch_size,
      phi::errors::InvalidArgument(
          "The Tensor q and k's batch size [%d]  vs [%d] are not matched.",
          batch_size,
          k_batch_size));

  PADDLE_ENFORCE_EQ(
      num_heads,
      k_num_heads,
      phi::errors::InvalidArgument(
          "The Tensor q and k's k_num_heads [%d] vs [%d] are not matched.",
          num_heads,
          k_num_heads));

  PADDLE_ENFORCE_EQ(
      head_dim,
      k_head_dim,
      phi::errors::InvalidArgument(
          "The Tensor q and k's head_dim [%d] vs [%d] are not matched.",
          head_dim,
          k_head_dim));

  PADDLE_ENFORCE_EQ(
      k_ndim,
      k_dims_mapping_size,
      phi::errors::InvalidArgument("The Tensor q's rank [%d] and Its "
                                   "dims_mapping size [%d] are not matched.",
                                   k_ndim,
                                   k_dims_mapping_size));

  // v
  // [batch_size, seq_len_kv, num_heads, head_dim]
  auto v_shape = phi::vectorize(v.dims());
  int v_ndim = v_shape.size();
  auto v_dist_attr = v.dist_attr();
  int v_dims_mapping_size = v_dist_attr.dims_mapping().size();
  PADDLE_ENFORCE_EQ(
      v_ndim,
      4,
      phi::errors::InvalidArgument("The Tensor v's shape must be [batch_size, "
                                   "seq_len_kv, num_heads, head_dim_v]"));

  auto v_batch_size = v_shape[0];
  auto v_seq_len = v_shape[1];
  auto v_num_heads = v_shape[2];
  auto v_head_dim = v_shape[3];

  PADDLE_ENFORCE_EQ(
      batch_size,
      v_batch_size,
      phi::errors::InvalidArgument(
          "The Tensor q and v's batch size [%d] vs [%d] are not matched.",
          batch_size,
          v_batch_size));

  PADDLE_ENFORCE_EQ(
      num_heads,
      v_num_heads,
      phi::errors::InvalidArgument(
          "The Tensor q and v's k_num_heads [%d] vs [%d] are not matched.",
          num_heads,
          v_num_heads));

  PADDLE_ENFORCE_EQ(
      k_seq_len,
      v_seq_len,
      phi::errors::InvalidArgument(
          "The Tensor k and v's seq_len [%d] vs [%d] are not matched.",
          k_seq_len,
          v_seq_len));

  PADDLE_ENFORCE_EQ(
      v_ndim,
      v_dims_mapping_size,
      phi::errors::InvalidArgument("The Tensor q's rank [%d] and Its "
                                   "dims_mapping size [%d] are not matched.",
                                   v_ndim,
                                   v_dims_mapping_size));

  // fixed_seed_offset

  // attn_mask
  auto mask_shape = phi::vectorize(attn_mask.dims());
  int mask_ndim = mask_shape.size();
  auto mask_dist_attr = attn_mask.dist_attr();
  int mask_dims_mapping_size = mask_dist_attr.dims_mapping().size();
  PADDLE_ENFORCE_EQ(
      mask_ndim,
      mask_dims_mapping_size,
      phi::errors::InvalidArgument("The Tensor q's rank [%d] and Its "
                                   "dims_mapping size [%d] are not matched.",
                                   mask_ndim,
                                   mask_dims_mapping_size));

  auto out_shape = phi::vectorize(out.dims());
  auto out_dist_attr = attn_mask.dist_attr();

  auto out_grad_shape = phi::vectorize(out_grad.dims());
  auto out_grad_dist_attr = out_grad.dist_attr();

  std::string alphabet = "abcdefghijklmnopqrstuvwxyz";
  int used_axes_index = 0;
  char batch_axis = alphabet[used_axes_index++];
  char seq_len_q_axis = alphabet[used_axes_index++];
  char num_heads_axis = alphabet[used_axes_index++];
  char head_dim_axis = alphabet[used_axes_index++];
  char seq_len_kv_axis = alphabet[used_axes_index++];
  char head_dim_v_axis = alphabet[used_axes_index++];

  // [batch_size, seq_len_q, num_heads, head_dim]
  std::string q_axes = {
      batch_axis, seq_len_q_axis, num_heads_axis, head_dim_axis};
  // [batch_size, seq_len_kv, num_heads, head_dim]
  std::string k_axes = {
      batch_axis, seq_len_kv_axis, num_heads_axis, head_dim_axis};
  // [batch_size, seq_len_kv, num_heads, head_dim_v]
  std::string v_axes = {
      batch_axis, seq_len_kv_axis, num_heads_axis, head_dim_v_axis};
  // [batch_size, seq_len_q, num_heads, head_dim_v]
  std::string out_axes = {
      batch_axis, seq_len_q_axis, num_heads_axis, head_dim_v_axis};
  // [batch_size,  num_heads, seq_len_q, seq_len_kv]
  std::string softmax_axes = {
      batch_axis, num_heads_axis, seq_len_q_axis, seq_len_kv_axis};
  // [batch_size,  num_heads, seq_len_q, seq_len_kv]
  std::string softmax_lse_axes = {
      batch_axis, num_heads_axis, seq_len_q_axis, seq_len_kv_axis};

  std::string q_axes_align = q_axes;
  q_axes_align[1] = alphabet[used_axes_index++];
  q_axes_align[3] = alphabet[used_axes_index++];

  std::string k_axes_align = k_axes;
  k_axes_align[1] = alphabet[used_axes_index++];
  k_axes_align[3] = alphabet[used_axes_index++];

  std::string v_axes_align = v_axes;
  v_axes_align[1] = alphabet[used_axes_index++];
  v_axes_align[3] = alphabet[used_axes_index++];

  std::string out_axes_align = out_axes;
  out_axes_align[1] = alphabet[used_axes_index++];
  out_axes_align[3] = alphabet[used_axes_index++];

  std::string out_grad_axes_align = out_axes;
  out_grad_axes_align[1] = alphabet[used_axes_index++];
  out_grad_axes_align[3] = alphabet[used_axes_index++];

  std::vector<std::pair<std::string, std::vector<int64_t>>> axes_sharding_info;

  axes_sharding_info.emplace_back(q_axes_align, q_dist_attr.dims_mapping());
  axes_sharding_info.emplace_back(k_axes_align, k_dist_attr.dims_mapping());
  axes_sharding_info.emplace_back(v_axes_align, k_dist_attr.dims_mapping());
  axes_sharding_info.emplace_back(out_axes_align, out_dist_attr.dims_mapping());
  axes_sharding_info.emplace_back(out_grad_axes_align,
                                  out_grad_dist_attr.dims_mapping());

  auto axis_to_dim_map = ShardingMergeForTensors(axes_sharding_info);

  auto q_dist_attr_dst = CopyTensorDistAttrForOutput(q_dist_attr);
  auto q_dims_mapping = GetDimsMappingForAxes(q_axes, axis_to_dim_map, true);
  q_dist_attr_dst.set_dims_mapping(q_dims_mapping);
  auto k_dist_attr_dst = CopyTensorDistAttrForOutput(k_dist_attr);
  auto k_dims_mapping = GetDimsMappingForAxes(k_axes, axis_to_dim_map, true);
  k_dist_attr_dst.set_dims_mapping(k_dims_mapping);
  auto v_dist_attr_dst = CopyTensorDistAttrForOutput(v_dist_attr);
  auto v_dims_mapping = GetDimsMappingForAxes(v_axes, axis_to_dim_map, true);
  v_dist_attr_dst.set_dims_mapping(v_dims_mapping);
  auto out_dist_attr_dst = CopyTensorDistAttrForOutput(out_dist_attr);
  auto out_dims_mapping =
      GetDimsMappingForAxes(out_axes, axis_to_dim_map, true);
  out_dist_attr_dst.set_dims_mapping(out_dims_mapping);

  // TODO(liuzhenhai): process fixed_seed_offset and attn_mask
  auto fixed_seed_offset_dist_attr = seed_offset.dist_attr();
  auto attn_mask_dist_attr = attn_mask.dist_attr();

  auto out_grad_dist_attr_dst = CopyTensorDistAttrForOutput(out_grad_dist_attr);
  auto out_grad_dims_mapping =
      GetDimsMappingForAxes(out_axes, axis_to_dim_map, true);
  v_dist_attr_dst.set_dims_mapping(out_grad_dims_mapping);

  TensorDistAttr q_grad;
  auto q_grad_dims_mapping =
      GetDimsMappingForAxes(q_axes, axis_to_dim_map, true);
  q_grad.set_dims_mapping(q_grad_dims_mapping);

  TensorDistAttr k_grad;
  auto k_grad_dims_mapping =
      GetDimsMappingForAxes(k_axes, axis_to_dim_map, true);
  k_grad.set_dims_mapping(k_grad_dims_mapping);

  TensorDistAttr v_grad;
  auto v_grad_dims_mapping =
      GetDimsMappingForAxes(v_axes, axis_to_dim_map, true);
  v_grad.set_dims_mapping(v_grad_dims_mapping);

  VLOG(4) << "FlashAttInferSpmd:";
  VLOG(4) << "Einsum Notation: " << q_axes << "," << k_axes << "," << v_axes
          << "-->" << out_axes << "," << softmax_axes << ","
          << softmax_lse_axes;

  VLOG(4) << "q";
  VLOG(4) << "Input shape: [" << str_join(q_shape) << "] "
          << "src_dims_mapping: [" << str_join(q_dist_attr.dims_mapping())
          << "] "
          << "dst_dims_mapping: [" << str_join(q_dims_mapping) << "]";

  VLOG(4) << "k";
  VLOG(4) << "Input shape: [" << str_join(k_shape) << "] "
          << "src_dims_mapping: [" << str_join(k_dist_attr.dims_mapping())
          << "] "
          << "dst_dims_mapping: [" << str_join(v_dims_mapping) << "]";

  VLOG(4) << "v";
  VLOG(4) << "Input shape: [" << str_join(v_shape) << "] "
          << "src_dims_mapping: [" << str_join(v_dist_attr.dims_mapping())
          << "] "
          << "dst_dims_mapping: [" << str_join(v_dims_mapping) << "]";

  VLOG(4) << "out";
  VLOG(4) << "Input shape: [" << str_join(out_shape) << "] "
          << "src_dims_mapping: [" << str_join(out_dist_attr.dims_mapping())
          << "] "
          << "dst_dims_mapping: [" << str_join(out_dims_mapping) << "]";

  VLOG(4) << "out_grad";
  VLOG(4) << "Input shape: [" << str_join(out_grad_shape) << "] "
          << "src_dims_mapping: ["
          << str_join(out_grad_dist_attr.dims_mapping()) << "] "
          << "dst_dims_mapping: [" << str_join(out_grad_dims_mapping) << "]";

  VLOG(4) << "q_grad"
          << " dims_mapping: [" << str_join(q_grad_dims_mapping) << "]";
  VLOG(4) << "k_grad"
          << " dims_mapping: [" << str_join(k_grad_dims_mapping) << "]";
  VLOG(4) << "v_grad"
          << " dims_mapping: [" << str_join(v_grad_dims_mapping) << "]";

  VLOG(4) << std::endl;

  return {{q_dist_attr_dst,
           k_dist_attr_dst,
           v_dist_attr_dst,
           out_dist_attr_dst,
           fixed_seed_offset_dist_attr,
           attn_mask_dist_attr,
           out_grad_dist_attr_dst},
          {q_grad, k_grad, v_grad}};
}

}  // namespace distributed
}  // namespace phi
