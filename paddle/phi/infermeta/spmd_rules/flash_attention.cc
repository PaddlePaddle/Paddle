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

namespace phi::distributed {
const int kNumHeadsDimIndex = 2;

#define LOG_SPMD_INPUT(name)                                                  \
  do {                                                                        \
    VLOG(4) << #name;                                                         \
    VLOG(4) << "shape: [" << str_join(name##_shape) << "] "                   \
            << "src_dist_attr: [" << name##_dist_attr.to_string() << "] "     \
            << "src_dist_attr: [" << name##_dist_attr_dst.to_string() << "]"; \
  } while (0)

#define LOG_SPMD_OUTPUT(name)                                 \
  do {                                                        \
    VLOG(4) << #name;                                         \
    VLOG(4) << "src_dist_attr: [" << name.to_string() << "]"; \
  } while (0)

using phi::distributed::auto_parallel::str_join;

TensorDistAttr MapDims(
    const TensorDistAttr& src,
    const std::unordered_map<std::string, int64_t>& axes_mapping,
    const std::string& axes) {
  auto dst = CopyTensorDistAttrForOutput(src);
  auto dims_mapping = GetDimsMappingForAxes(axes, axes_mapping, true);
  dst.set_dims_mapping(dims_mapping);
  return dst;
}

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
  auto q_shape = common::vectorize(q.dims());
  int q_ndim = q_shape.size();
  auto q_dist_attr = q.dist_attr();
  int q_dims_mapping_size = q_dist_attr.dims_mapping().size();

  PADDLE_ENFORCE_EQ(q_ndim,
                    4,
                    common::errors::InvalidArgument(
                        "The Tensor q's shape must be [batch_size, "
                        "seq_len_q, num_heads, head_dim]"));

  auto batch_size = q_shape[0];
  auto num_heads = q_shape[2];
  auto head_dim = q_shape[3];

  PADDLE_ENFORCE_EQ(
      q_ndim,
      q_dims_mapping_size,
      common::errors::InvalidArgument("The Tensor q's rank [%d] and Its "
                                      "dims_mapping size [%d] are not matched.",
                                      q_ndim,
                                      q_dims_mapping_size));

  // k
  // [batch_size, seq_len_kv, num_heads, head_dim]
  auto k_shape = common::vectorize(k.dims());
  int k_ndim = k_shape.size();
  auto k_dist_attr = k.dist_attr();
  int k_dims_mapping_size = k_dist_attr.dims_mapping().size();
  PADDLE_ENFORCE_EQ(k_ndim,
                    4,
                    common::errors::InvalidArgument(
                        "The Tensor k's shape must be [batch_size, "
                        "seq_len_kv, num_heads, head_dim]"));

  auto k_batch_size = k_shape[0];
  auto k_seq_len = k_shape[1];
  auto k_num_heads = k_shape[2];
  auto k_head_dim = k_shape[3];

  PADDLE_ENFORCE_EQ(
      batch_size,
      k_batch_size,
      common::errors::InvalidArgument(
          "The Tensor q and k's batch size [%d]  vs [%d] are not matched.",
          batch_size,
          k_batch_size));

  PADDLE_ENFORCE_EQ(
      num_heads % k_num_heads == 0,
      true,
      common::errors::InvalidArgument(
          "The num_heads of q must be divisible by k's, but [%d] vs [%d].",
          num_heads,
          k_num_heads));

  PADDLE_ENFORCE_EQ(
      head_dim,
      k_head_dim,
      common::errors::InvalidArgument(
          "The Tensor q and k's head_dim [%d] vs [%d] are not matched.",
          head_dim,
          k_head_dim));

  PADDLE_ENFORCE_EQ(
      k_ndim,
      k_dims_mapping_size,
      common::errors::InvalidArgument("The Tensor q's rank [%d] and Its "
                                      "dims_mapping size [%d] are not matched.",
                                      k_ndim,
                                      k_dims_mapping_size));

  bool is_divisible = true;
  int64_t num_head_mesh_dim = k_dist_attr.dims_mapping()[kNumHeadsDimIndex];
  if (num_head_mesh_dim != -1) {
    int64_t num_head_split_size =
        k_dist_attr.process_mesh().dim_size(num_head_mesh_dim);
    is_divisible = k_num_heads % num_head_split_size == 0;
  }

  // v
  // [batch_size, seq_len_kv, num_heads, head_dim]
  auto v_shape = common::vectorize(v.dims());
  int v_ndim = v_shape.size();
  auto v_dist_attr = v.dist_attr();
  int v_dims_mapping_size = v_dist_attr.dims_mapping().size();
  PADDLE_ENFORCE_EQ(v_ndim,
                    4,
                    common::errors::InvalidArgument(
                        "The Tensor v's shape must be [batch_size, "
                        "seq_len_kv, num_heads, head_dim_v]"));

  auto v_batch_size = v_shape[0];
  auto v_seq_len = v_shape[1];
  auto v_num_heads = v_shape[2];

  PADDLE_ENFORCE_EQ(
      batch_size,
      v_batch_size,
      common::errors::InvalidArgument(
          "The Tensor q and v's batch size [%d] vs [%d] are not matched.",
          batch_size,
          v_batch_size));

  PADDLE_ENFORCE_EQ(
      num_heads % v_num_heads == 0,
      true,
      common::errors::InvalidArgument(
          "The num_heads of q must be divisible by v's, but [%d] vs [%d].",
          num_heads,
          v_num_heads));

  bool is_same_num_heads = num_heads == v_num_heads;

  PADDLE_ENFORCE_EQ(
      k_seq_len,
      v_seq_len,
      common::errors::InvalidArgument(
          "The Tensor k and v's seq_len [%d] vs [%d] are not matched.",
          k_seq_len,
          v_seq_len));

  PADDLE_ENFORCE_EQ(
      v_ndim,
      v_dims_mapping_size,
      common::errors::InvalidArgument("The Tensor v's rank [%d] and Its "
                                      "dims_mapping size [%d] are not matched.",
                                      v_ndim,
                                      v_dims_mapping_size));

  // fixed_seed_offset
  // TODO(liuzhenhai): process fixed_seed_offset and attn_mask
  auto fixed_seed_offset_dist_attr = fixed_seed_offset.dist_attr();
  auto fixed_seed_offset_shape = common::vectorize(fixed_seed_offset.dims());
  // attn_mask
  auto attn_mask_shape = common::vectorize(attn_mask.dims());
  int mask_ndim = attn_mask_shape.size();
  auto attn_mask_dist_attr = attn_mask.dist_attr();
  int mask_dims_mapping_size = attn_mask_dist_attr.dims_mapping().size();
  if (!IsEmpty(attn_mask_shape)) {
    PADDLE_ENFORCE_EQ(mask_ndim,
                      mask_dims_mapping_size,
                      common::errors::InvalidArgument(
                          "The Tensor mask's rank [%d] and Its "
                          "dims_mapping size [%d] are not matched.",
                          mask_ndim,
                          mask_dims_mapping_size));
  }

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
  // [batch_size, num_heads, seq_len_q, seq_len_kv]
  std::string attn_mask_axes = {
      batch_axis, num_heads_axis, seq_len_q_axis, seq_len_kv_axis};
  // [batch_size, seq_len_q, num_heads, head_dim_v]
  std::string out_axes = {
      batch_axis, seq_len_q_axis, num_heads_axis, head_dim_v_axis};
  // [batch_size,  num_heads, seq_len_q, seq_len_kv]
  std::string softmax_axes = {
      batch_axis, num_heads_axis, seq_len_q_axis, seq_len_kv_axis};
  // [batch_size,  num_heads, seq_len_q]
  std::string softmax_lse_axes = {batch_axis, num_heads_axis, seq_len_q_axis};

  auto q_dist_attr_dst = UnShardTensorDims(q_dist_attr, {1, 3});
  auto k_dist_attr_dst = UnShardTensorDims(k_dist_attr, {1, 3});
  auto v_dist_attr_dst = UnShardTensorDims(k_dist_attr, {1, 3});
  auto attn_mask_dist_attr_dst = attn_mask_dist_attr;
  if (!IsEmpty(attn_mask_shape)) {
    attn_mask_dist_attr_dst = UnShardTensorDims(attn_mask_dist_attr, {2, 3});
  }

  if (!is_same_num_heads && !is_divisible) {
    q_dist_attr_dst = UnShardTensorDims(q_dist_attr_dst, {2});
    k_dist_attr_dst = UnShardTensorDims(k_dist_attr_dst, {2});
    v_dist_attr_dst = UnShardTensorDims(v_dist_attr_dst, {2});
    if (!IsEmpty(attn_mask_shape)) {
      attn_mask_dist_attr_dst = UnShardTensorDims(attn_mask_dist_attr_dst, {1});
    }
  }

  std::vector<std::pair<std::string, std::vector<int64_t>>> axes_sharding_info;

  axes_sharding_info.emplace_back(q_axes, q_dist_attr_dst.dims_mapping());
  axes_sharding_info.emplace_back(k_axes, k_dist_attr_dst.dims_mapping());
  axes_sharding_info.emplace_back(v_axes, v_dist_attr_dst.dims_mapping());
  if (!IsEmpty(attn_mask_shape)) {
    axes_sharding_info.emplace_back(attn_mask_axes,
                                    attn_mask_dist_attr_dst.dims_mapping());
  }

  auto axis_to_dim_map = ShardingMergeForTensors(axes_sharding_info);

  q_dist_attr_dst = MapDims(q_dist_attr, axis_to_dim_map, q_axes);
  k_dist_attr_dst = MapDims(k_dist_attr, axis_to_dim_map, k_axes);
  v_dist_attr_dst = MapDims(v_dist_attr, axis_to_dim_map, v_axes);
  if (!IsEmpty(attn_mask_shape)) {
    attn_mask_dist_attr_dst =
        MapDims(attn_mask_dist_attr, axis_to_dim_map, attn_mask_axes);
    if (attn_mask_shape[1] == 1) {
      attn_mask_dist_attr_dst = UnShardTensorDims(attn_mask_dist_attr_dst, {1});
    }
  }

  // TODO(liuzhenhai): process fixed_seed
  auto fixed_seed_offset_dist_attr_dst = fixed_seed_offset_dist_attr;

  auto out = MapDims(q_dist_attr, axis_to_dim_map, out_axes);
  auto softmax = MapDims(q_dist_attr, axis_to_dim_map, softmax_axes);
  auto softmax_lse = MapDims(q_dist_attr, axis_to_dim_map, softmax_lse_axes);

  TensorDistAttr seed_offset = fixed_seed_offset_dist_attr;
  seed_offset.set_dims_mapping({-1});
  seed_offset.set_process_mesh(out.process_mesh());

  VLOG(4) << "FlashAttInferSpmd:";
  VLOG(4) << "Einsum Notation: " << q_axes << "," << k_axes << "," << v_axes
          << "-->" << out_axes << "," << softmax_axes << ","
          << softmax_lse_axes;

  LOG_SPMD_INPUT(q);
  LOG_SPMD_INPUT(k);
  LOG_SPMD_INPUT(v);
  LOG_SPMD_INPUT(fixed_seed_offset);
  LOG_SPMD_INPUT(attn_mask);
  VLOG(4) << "Outputs:";
  LOG_SPMD_OUTPUT(out);
  LOG_SPMD_OUTPUT(softmax);
  LOG_SPMD_OUTPUT(softmax_lse);
  LOG_SPMD_OUTPUT(seed_offset);
  VLOG(4) << std::endl;

  return {{q_dist_attr_dst,
           k_dist_attr_dst,
           v_dist_attr_dst,
           fixed_seed_offset_dist_attr_dst,
           attn_mask_dist_attr_dst},
          {out, softmax, softmax_lse, seed_offset}};
}

SpmdInfo FlashAttInferSpmdStatic(const DistMetaTensor& q,
                                 const DistMetaTensor& k,
                                 const DistMetaTensor& v,
                                 const DistMetaTensor& fixed_seed_offset,
                                 const DistMetaTensor& attn_mask,
                                 float dropout,
                                 bool causal,
                                 bool return_softmax,
                                 bool is_test) {
  return FlashAttInferSpmd(q,
                           k,
                           v,
                           fixed_seed_offset,
                           attn_mask,
                           dropout,
                           causal,
                           return_softmax,
                           is_test);
}

SpmdInfo FlashAttInferSpmdReverse(const DistMetaTensor& q,
                                  const DistMetaTensor& k,
                                  const DistMetaTensor& v,
                                  const DistMetaTensor& fixed_seed_offset,
                                  const DistMetaTensor& attn_mask,
                                  const DistMetaTensor& out,
                                  const DistMetaTensor& softmax,
                                  const DistMetaTensor& softmax_lse,
                                  const DistMetaTensor& seed_offset,
                                  float dropout,
                                  bool causal,
                                  bool return_softmax,
                                  bool is_test) {
  // q
  // [batch_size, seq_len_q, num_heads, head_dim]
  auto q_shape = common::vectorize(q.dims());
  auto q_dist_attr = q.dist_attr();

  // k
  // [batch_size, seq_len_kv, num_heads, head_dim]
  auto k_shape = common::vectorize(k.dims());
  auto k_dist_attr = k.dist_attr();

  // v
  // [batch_size, seq_len_kv, num_heads, head_dim]
  auto v_shape = common::vectorize(v.dims());
  auto v_dist_attr = v.dist_attr();

  // fixed_seed_offset
  // TODO(liuzhenhai): process fixed_seed_offset、seed_offset、 and attn_mask
  auto fixed_seed_offset_dist_attr = fixed_seed_offset.dist_attr();
  auto fixed_seed_offset_shape = common::vectorize(fixed_seed_offset.dims());
  // attn_mask
  auto attn_mask_shape = common::vectorize(attn_mask.dims());
  int mask_ndim = attn_mask_shape.size();
  auto attn_mask_dist_attr = attn_mask.dist_attr();
  int mask_dims_mapping_size = attn_mask_dist_attr.dims_mapping().size();
  if (!IsEmpty(attn_mask_shape)) {
    PADDLE_ENFORCE_EQ(mask_ndim,
                      mask_dims_mapping_size,
                      common::errors::InvalidArgument(
                          "The Tensor mask's rank [%d] and Its "
                          "dims_mapping size [%d] are not matched.",
                          mask_ndim,
                          mask_dims_mapping_size));
  }

  // out
  // [batch_size, seq_len_q, num_heads, head_dim_v]
  auto out_shape = common::vectorize(out.dims());
  int out_ndim = out_shape.size();
  auto out_dist_attr = v.dist_attr();
  int out_dims_mapping_size = out_dist_attr.dims_mapping().size();
  PADDLE_ENFORCE_EQ(out_ndim,
                    4,
                    common::errors::InvalidArgument(
                        "The Tensor out's shape must be [batch_size, "
                        "seq_len_q, num_heads, head_dim_v]"));

  auto batch_size = out_shape[0];
  auto seq_len_q = out_shape[1];
  auto num_heads = out_shape[2];

  PADDLE_ENFORCE_EQ(
      out_ndim,
      out_dims_mapping_size,
      common::errors::InvalidArgument("The Tensor out's rank [%d] and Its "
                                      "dims_mapping size [%d] are not matched.",
                                      out_ndim,
                                      out_dims_mapping_size));

  // softmax_lse
  // [batch_size,  num_heads, seq_len_q, seq_len_kv]
  auto softmax_lse_shape = common::vectorize(softmax_lse.dims());
  int softmax_lse_ndim = softmax_lse_shape.size();
  auto softmax_lse_dist_attr = softmax_lse.dist_attr();
  int softmax_lse_dims_mapping_size =
      softmax_lse_dist_attr.dims_mapping().size();
  PADDLE_ENFORCE_EQ(out_ndim,
                    4,
                    common::errors::InvalidArgument(
                        "The Tensor softmax_lse's shape must be [batch_size, "
                        "num_heads, seq_len_q, seq_len_kv]"));

  PADDLE_ENFORCE_EQ(softmax_lse_ndim,
                    softmax_lse_dims_mapping_size,
                    common::errors::InvalidArgument(
                        "The Tensor softmax_lse's rank [%d] and Its "
                        "dims_mapping size [%d] are not matched.",
                        softmax_lse_ndim,
                        softmax_lse_dims_mapping_size));

  auto batch_size_2 = softmax_lse_shape[0];
  auto num_heads_2 = softmax_lse_shape[1];
  auto seq_len_q_2 = softmax_lse_shape[2];

  PADDLE_ENFORCE_EQ(
      batch_size,
      batch_size_2,
      common::errors::InvalidArgument(
          "batch size of Tensor out and softmax_lse is not matched: [] vs []",
          batch_size,
          batch_size_2));

  PADDLE_ENFORCE_EQ(
      num_heads,
      num_heads_2,
      common::errors::InvalidArgument(
          "num heads of Tensor out and softmax_lse is not matched: [] vs []",
          num_heads,
          num_heads_2));

  PADDLE_ENFORCE_EQ(
      seq_len_q,
      seq_len_q_2,
      common::errors::InvalidArgument(
          "seq_len_q of Tensor out and softmax_lse is not matched: [] vs []",
          seq_len_q,
          seq_len_q_2));

  TensorDistAttr seed_offset_dist_attr = fixed_seed_offset.dist_attr();
  auto seed_offset_shape = common::vectorize(seed_offset.dims());

  TensorDistAttr softmax_dist_attr = softmax.dist_attr();
  auto softmax_shape = common::vectorize(softmax.dims());

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

  // [batch_size, num_heads, seq_len_q, seq_len_kv]
  std::string attn_mask_axes = {
      batch_axis, num_heads_axis, seq_len_q_axis, seq_len_kv_axis};

  // [batch_size, seq_len_q, num_heads, head_dim_v]
  std::string out_axes = {
      batch_axis, seq_len_q_axis, num_heads_axis, head_dim_v_axis};
  // [batch_size,  num_heads, seq_len_q, seq_len_kv]
  std::string softmax_axes = {
      batch_axis, num_heads_axis, seq_len_q_axis, seq_len_kv_axis};
  // [batch_size,  num_heads, seq_len_q]
  std::string softmax_lse_axes = {batch_axis, num_heads_axis, seq_len_q_axis};

  auto out_dist_attr_dst = UnShardTensorDims(out_dist_attr, {1, 3});
  auto softmax_lse_dist_attr_dst =
      UnShardTensorDims(softmax_lse_dist_attr, {2});

  bool is_same_num_heads = q_shape[2] == k_shape[2];
  bool is_divisible = true;
  int64_t num_head_mesh_dim = k_dist_attr.dims_mapping()[kNumHeadsDimIndex];
  if (num_head_mesh_dim != -1) {
    int64_t num_head_split_size =
        k_dist_attr.process_mesh().dim_size(num_head_mesh_dim);
    is_divisible = k_shape[2] % num_head_split_size == 0;
  }

  if (!is_same_num_heads && !is_divisible) {
    out_dist_attr_dst = UnShardTensorDims(out_dist_attr_dst, {2});
    softmax_lse_dist_attr_dst =
        UnShardTensorDims(softmax_lse_dist_attr_dst, {1});
  }

  std::vector<std::pair<std::string, std::vector<int64_t>>> axes_sharding_info;

  axes_sharding_info.emplace_back(out_axes, out_dist_attr_dst.dims_mapping());
  axes_sharding_info.emplace_back(softmax_lse_axes,
                                  softmax_lse_dist_attr_dst.dims_mapping());

  auto axis_to_dim_map = ShardingMergeForTensors(axes_sharding_info);

  auto q_dist_attr_dst = MapDims(q_dist_attr, axis_to_dim_map, q_axes);
  auto k_dist_attr_dst = MapDims(k_dist_attr, axis_to_dim_map, k_axes);
  auto v_dist_attr_dst = MapDims(v_dist_attr, axis_to_dim_map, v_axes);
  out_dist_attr_dst = MapDims(out_dist_attr_dst, axis_to_dim_map, out_axes);
  softmax_lse_dist_attr_dst =
      MapDims(softmax_lse_dist_attr_dst, axis_to_dim_map, softmax_lse_axes);
  auto attn_mask_dist_attr_dst = attn_mask_dist_attr;
  if (!IsEmpty(attn_mask_shape)) {
    attn_mask_dist_attr_dst =
        MapDims(attn_mask_dist_attr, axis_to_dim_map, attn_mask_axes);
    if (attn_mask_shape[1] == 1) {
      attn_mask_dist_attr_dst = UnShardTensorDims(attn_mask_dist_attr_dst, {1});
    }
  }

  // TODO(liuzhenhai): process fixed_seed
  auto fixed_seed_offset_dist_attr_dst = fixed_seed_offset_dist_attr;
  auto softmax_dist_attr_dst = softmax_dist_attr;
  auto seed_offset_dist_attr_dst = seed_offset_dist_attr;

  VLOG(4) << "FlashAttInferSpmd:";
  VLOG(4) << "Einsum Notation: " << q_axes << "," << k_axes << "," << v_axes
          << "-->" << out_axes << "," << softmax_axes << ","
          << softmax_lse_axes;

  LOG_SPMD_INPUT(q);
  LOG_SPMD_INPUT(k);
  LOG_SPMD_INPUT(v);
  LOG_SPMD_INPUT(fixed_seed_offset);
  LOG_SPMD_INPUT(attn_mask);

  VLOG(4) << "Outputs:";
  LOG_SPMD_INPUT(out);
  LOG_SPMD_INPUT(softmax);
  LOG_SPMD_INPUT(softmax_lse);
  LOG_SPMD_INPUT(seed_offset);
  VLOG(4) << std::endl;

  return {{q_dist_attr_dst,
           k_dist_attr_dst,
           v_dist_attr_dst,
           fixed_seed_offset_dist_attr_dst,
           attn_mask_dist_attr_dst},
          {out_dist_attr_dst,
           softmax_dist_attr_dst,
           softmax_lse_dist_attr_dst,
           seed_offset_dist_attr_dst}};
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
  auto q_shape = common::vectorize(q.dims());
  int q_ndim = q_shape.size();
  auto q_dist_attr = q.dist_attr();
  int q_dims_mapping_size = q_dist_attr.dims_mapping().size();

  PADDLE_ENFORCE_EQ(q_ndim,
                    4,
                    common::errors::InvalidArgument(
                        "The Tensor q's shape must be [batch_size, "
                        "seq_len_q, num_heads, head_dim]"));

  auto batch_size = q_shape[0];
  auto num_heads = q_shape[2];
  auto head_dim = q_shape[3];

  PADDLE_ENFORCE_EQ(
      q_ndim,
      q_dims_mapping_size,
      common::errors::InvalidArgument("The Tensor q's rank [%d] and Its "
                                      "dims_mapping size [%d] are not matched.",
                                      q_ndim,
                                      q_dims_mapping_size));

  // k
  // [batch_size, seq_len_kv, num_heads, head_dim]
  auto k_shape = common::vectorize(k.dims());
  int k_ndim = k_shape.size();
  auto k_dist_attr = k.dist_attr();
  int k_dims_mapping_size = k_dist_attr.dims_mapping().size();
  PADDLE_ENFORCE_EQ(k_ndim,
                    4,
                    common::errors::InvalidArgument(
                        "The Tensor k's shape must be [batch_size, "
                        "seq_len_kv, num_heads, head_dim]"));

  auto k_batch_size = k_shape[0];
  auto k_seq_len = k_shape[1];
  auto k_num_heads = k_shape[2];
  auto k_head_dim = k_shape[3];

  PADDLE_ENFORCE_EQ(
      batch_size,
      k_batch_size,
      common::errors::InvalidArgument(
          "The Tensor q and k's batch size [%d]  vs [%d] are not matched.",
          batch_size,
          k_batch_size));

  PADDLE_ENFORCE_EQ(
      num_heads % k_num_heads == 0,
      true,
      common::errors::InvalidArgument(
          "The num_heads of q must be divisible by k's, but [%d] vs [%d].",
          num_heads,
          k_num_heads));

  PADDLE_ENFORCE_EQ(
      head_dim,
      k_head_dim,
      common::errors::InvalidArgument(
          "The Tensor q and k's head_dim [%d] vs [%d] are not matched.",
          head_dim,
          k_head_dim));

  PADDLE_ENFORCE_EQ(
      k_ndim,
      k_dims_mapping_size,
      common::errors::InvalidArgument("The Tensor k's rank [%d] and Its "
                                      "dims_mapping size [%d] are not matched.",
                                      k_ndim,
                                      k_dims_mapping_size));

  // v
  // [batch_size, seq_len_kv, num_heads, head_dim]
  auto v_shape = common::vectorize(v.dims());
  int v_ndim = v_shape.size();
  auto v_dist_attr = v.dist_attr();
  int v_dims_mapping_size = v_dist_attr.dims_mapping().size();
  PADDLE_ENFORCE_EQ(v_ndim,
                    4,
                    common::errors::InvalidArgument(
                        "The Tensor v's shape must be [batch_size, "
                        "seq_len_kv, num_heads, head_dim_v]"));

  auto v_batch_size = v_shape[0];
  auto v_seq_len = v_shape[1];
  auto v_num_heads = v_shape[2];

  PADDLE_ENFORCE_EQ(
      batch_size,
      v_batch_size,
      common::errors::InvalidArgument(
          "The Tensor q and v's batch size [%d] vs [%d] are not matched.",
          batch_size,
          v_batch_size));

  PADDLE_ENFORCE_EQ(
      num_heads % v_num_heads == 0,
      true,
      common::errors::InvalidArgument(
          "The num_head of q must be divisible by v's, but [%d] vs [%d].",
          num_heads,
          v_num_heads));

  PADDLE_ENFORCE_EQ(
      k_seq_len,
      v_seq_len,
      common::errors::InvalidArgument(
          "The Tensor k and v's seq_len [%d] vs [%d] are not matched.",
          k_seq_len,
          v_seq_len));

  PADDLE_ENFORCE_EQ(
      v_ndim,
      v_dims_mapping_size,
      common::errors::InvalidArgument("The Tensor v's rank [%d] and Its "
                                      "dims_mapping size [%d] are not matched.",
                                      v_ndim,
                                      v_dims_mapping_size));

  // fixed_seed_offset
  auto seed_offset_dist_attr = seed_offset.dist_attr();
  auto seed_offset_shape = common::vectorize(seed_offset.dims());

  // attn_mask
  auto attn_mask_shape = common::vectorize(attn_mask.dims());
  int mask_ndim = attn_mask_shape.size();
  auto attn_mask_dist_attr = attn_mask.dist_attr();
  int mask_dims_mapping_size = attn_mask_dist_attr.dims_mapping().size();
  if (!IsEmpty(attn_mask_shape)) {
    PADDLE_ENFORCE_EQ(mask_ndim,
                      mask_dims_mapping_size,
                      common::errors::InvalidArgument(
                          "The Tensor mask's rank [%d] and Its "
                          "dims_mapping size [%d] are not matched.",
                          mask_ndim,
                          mask_dims_mapping_size));
  }

  auto out_shape = common::vectorize(out.dims());
  auto out_dist_attr = out.dist_attr();

  auto softmax_lse_shape = common::vectorize(softmax_lse.dims());
  auto softmax_lse_dist_attr = softmax_lse.dist_attr();

  auto out_grad_shape = common::vectorize(out_grad.dims());
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
  // [batch_size, num_heads, seq_len_q, seq_len_kv]
  std::string attn_mask_axes = {
      batch_axis, num_heads_axis, seq_len_q_axis, seq_len_kv_axis};
  // [batch_size, seq_len_q, num_heads, head_dim_v]
  std::string out_axes = {
      batch_axis, seq_len_q_axis, num_heads_axis, head_dim_v_axis};
  // [batch_size,  num_heads, seq_len_q, seq_len_kv]
  std::string softmax_axes = {
      batch_axis, num_heads_axis, seq_len_q_axis, seq_len_kv_axis};
  // [batch_size,  num_heads, seq_len_q]
  std::string softmax_lse_axes = {batch_axis, num_heads_axis, seq_len_q_axis};

  auto q_dist_attr_dst = UnShardTensorDims(q_dist_attr, {1, 3});
  auto k_dist_attr_dst = UnShardTensorDims(k_dist_attr, {1, 3});
  auto v_dist_attr_dst = UnShardTensorDims(k_dist_attr, {1, 3});
  auto out_dist_attr_dst = UnShardTensorDims(out_dist_attr, {1, 3});
  auto out_grad_dist_attr_dst = UnShardTensorDims(out_grad_dist_attr, {1, 3});
  auto softmax_lse_dist_attr_dst =
      UnShardTensorDims(softmax_lse_dist_attr, {2});

  bool is_same_num_heads = num_heads == v_num_heads;
  bool is_divisible = true;
  int64_t num_head_mesh_dim = k_dist_attr.dims_mapping()[kNumHeadsDimIndex];
  if (num_head_mesh_dim != -1) {
    int64_t num_head_split_size =
        k_dist_attr.process_mesh().dim_size(num_head_mesh_dim);
    is_divisible = k_shape[2] % num_head_split_size == 0;
  }
  if (!is_same_num_heads && !is_divisible) {
    q_dist_attr_dst = UnShardTensorDims(q_dist_attr_dst, {2});
    k_dist_attr_dst = UnShardTensorDims(k_dist_attr_dst, {2});
    v_dist_attr_dst = UnShardTensorDims(v_dist_attr_dst, {2});
    out_dist_attr_dst = UnShardTensorDims(out_dist_attr_dst, {2});
    out_grad_dist_attr_dst = UnShardTensorDims(out_grad_dist_attr_dst, {2});
    softmax_lse_dist_attr_dst =
        UnShardTensorDims(softmax_lse_dist_attr_dst, {1});
  }

  std::vector<std::pair<std::string, std::vector<int64_t>>> axes_sharding_info;
  axes_sharding_info.emplace_back(q_axes, q_dist_attr_dst.dims_mapping());
  axes_sharding_info.emplace_back(k_axes, k_dist_attr_dst.dims_mapping());
  axes_sharding_info.emplace_back(v_axes, v_dist_attr_dst.dims_mapping());
  axes_sharding_info.emplace_back(out_axes, out_dist_attr_dst.dims_mapping());
  axes_sharding_info.emplace_back(out_axes,
                                  out_grad_dist_attr_dst.dims_mapping());
  axes_sharding_info.emplace_back(softmax_lse_axes,
                                  softmax_lse_dist_attr_dst.dims_mapping());
  auto axis_to_dim_map = ShardingMergeForTensors(axes_sharding_info);

  q_dist_attr_dst = MapDims(q_dist_attr, axis_to_dim_map, q_axes);
  k_dist_attr_dst = MapDims(k_dist_attr, axis_to_dim_map, k_axes);
  v_dist_attr_dst = MapDims(v_dist_attr, axis_to_dim_map, v_axes);
  out_dist_attr_dst = MapDims(out_dist_attr, axis_to_dim_map, out_axes);
  softmax_lse_dist_attr_dst =
      MapDims(softmax_lse_dist_attr, axis_to_dim_map, softmax_lse_axes);
  auto attn_mask_dist_attr_dst = attn_mask_dist_attr;
  if (!IsEmpty(attn_mask_shape)) {
    attn_mask_dist_attr_dst =
        MapDims(attn_mask_dist_attr, axis_to_dim_map, attn_mask_axes);
    if (attn_mask_shape[1] == 1) {
      attn_mask_dist_attr_dst = UnShardTensorDims(attn_mask_dist_attr_dst, {1});
    }
  }

  // TODO(liuzhenhai): process seed and  attn_mask
  auto& seed_offset_dist_attr_dst = seed_offset_dist_attr;
  out_grad_dist_attr_dst = MapDims(out_dist_attr, axis_to_dim_map, out_axes);

  auto q_grad = MapDims(q_dist_attr, axis_to_dim_map, q_axes);
  auto k_grad = MapDims(k_dist_attr, axis_to_dim_map, k_axes);
  auto v_grad = MapDims(v_dist_attr, axis_to_dim_map, v_axes);

  VLOG(4) << "FlashAttInferSpmd:";
  VLOG(4) << "Einsum Notation: " << q_axes << "," << k_axes << "," << v_axes
          << "-->" << out_axes << "," << softmax_axes << "," << softmax_lse_axes
          << std::endl;
  VLOG(4) << "Inputs:" << std::endl;
  LOG_SPMD_INPUT(q);
  LOG_SPMD_INPUT(k);
  LOG_SPMD_INPUT(v);
  LOG_SPMD_INPUT(out);
  LOG_SPMD_INPUT(softmax_lse);
  LOG_SPMD_INPUT(seed_offset);
  LOG_SPMD_INPUT(attn_mask);
  LOG_SPMD_INPUT(out_grad);
  VLOG(4) << "Outputs:" << std::endl;
  LOG_SPMD_OUTPUT(q_grad);
  LOG_SPMD_OUTPUT(k_grad);
  LOG_SPMD_OUTPUT(v_grad);

  return {{q_dist_attr_dst,
           k_dist_attr_dst,
           v_dist_attr_dst,
           out_dist_attr_dst,
           softmax_lse_dist_attr_dst,
           seed_offset_dist_attr_dst,
           attn_mask_dist_attr_dst,
           out_grad_dist_attr_dst},
          {q_grad, k_grad, v_grad}};
}

}  // namespace phi::distributed
