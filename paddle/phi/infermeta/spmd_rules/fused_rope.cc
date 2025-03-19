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

#include "paddle/phi/infermeta/spmd_rules/fused_rope.h"

#include "glog/logging.h"

#include "paddle/phi/core/distributed/auto_parallel/dist_attr.h"
#include "paddle/phi/core/distributed/auto_parallel/inferspmd_utils.h"
#include "paddle/phi/core/distributed/auto_parallel/utils.h"
#include "paddle/phi/infermeta/spmd_rules/utils.h"

namespace phi::distributed {

using auto_parallel::str_join;
const int kNumHeadsDimIndex = 2;
const int kHeadDimIndex = 3;

void check_q(const DistMetaTensor& q) {
  std::vector<int64_t> q_shape = common::vectorize(q.dims());
  int q_ndim = q_shape.size();
  const TensorDistAttr& q_dist_attr_src = q.dist_attr();
  int q_dims_mapping_size = q_dist_attr_src.dims_mapping().size();

  PADDLE_ENFORCE_EQ(q_ndim,
                    4,
                    common::errors::InvalidArgument(
                        "The Tensor q's ndim must be 4 with shape [batch_size, "
                        "seq_len_q, num_heads, head_dim]"));
  PADDLE_ENFORCE_EQ(
      q_ndim,
      q_dims_mapping_size,
      common::errors::InvalidArgument("The Tensor q's rank [%d] and Its "
                                      "dims_mapping size [%d] are not matched.",
                                      q_ndim,
                                      q_dims_mapping_size));
}

// check k/v's shape equal to q's shape
void check_k_or_v(const DistMetaTensor& k_or_v,
                  const std::vector<int64_t>& q_shape) {
  std::vector<int64_t> shape = common::vectorize(k_or_v.dims());
  int ndim = shape.size();
  int dims_mapping_size = k_or_v.dist_attr().dims_mapping().size();
  PADDLE_ENFORCE_EQ(ndim,
                    4,
                    common::errors::InvalidArgument(
                        "The Tensor k/v's shape must be [batch_size, "
                        "seq_len_kv, num_heads, head_dim]"));

  PADDLE_ENFORCE_EQ(
      ndim,
      dims_mapping_size,
      common::errors::InvalidArgument("The Tensor k/v's rank [%d] and Its "
                                      "dims_mapping size [%d] are not matched.",
                                      ndim,
                                      dims_mapping_size));

  int64_t k_num_head = shape[kNumHeadsDimIndex];
  int64_t q_num_head = q_shape[kNumHeadsDimIndex];
  PADDLE_ENFORCE_EQ(q_num_head % k_num_head == 0,
                    true,
                    common::errors::InvalidArgument(
                        "The num_head of q must be divisible by k "
                        "and v, but got [%d] vs [%d]",
                        q_num_head,
                        k_num_head));

  for (size_t i = 0; i <= kHeadDimIndex; ++i) {
    if (i == kNumHeadsDimIndex) {
      PADDLE_ENFORCE_EQ(q_shape[i] % shape[i] == 0,
                        true,
                        common::errors::InvalidArgument(
                            "The num_head of q must be divisible by "
                            "k and v, but got [%d] vs [%d]",
                            q_shape[i],
                            shape[i]));
    } else {
      PADDLE_ENFORCE_EQ(q_shape[i],
                        shape[i],
                        common::errors::InvalidArgument(
                            "The shape except for num_head of q "
                            "must be same as k and v, but got [%d] vs [%d]",
                            str_join(q_shape),
                            str_join(shape)));
    }
  }
}

void check_sin_cos(const DistMetaTensor& sin,
                   const DistMetaTensor& cos,
                   const DistMetaTensor& position_ids,
                   const int64_t batch_size,
                   const int64_t seq_len,
                   const int64_t head_dim) {
  PADDLE_ENFORCE_EQ(sin.dims(),
                    cos.dims(),
                    common::errors::InvalidArgument(
                        "The dims of sin and cos must be the same. But "
                        "received sin's dims is {%s}, cos's dims is {%s}.",
                        sin.dims(),
                        sin.dims()));

  std::vector<int64_t> shape = common::vectorize(sin.dims());
  int ndim = shape.size();
  PADDLE_ENFORCE_EQ(
      (ndim == 2 || ndim == 4),
      true,
      common::errors::InvalidArgument(
          "The Tensor sin/cos's ndim must be 2 or 4. but given [%d]", ndim));

  int seq_len_dim_index = ndim == 2 ? 0 : 1;
  int head_dim_index = ndim == 2 ? 1 : 3;
  if (ndim == 4) {
    PADDLE_ENFORCE_EQ((shape[0] == 1 && shape[kNumHeadsDimIndex] == 1),
                      true,
                      common::errors::InvalidArgument(
                          "The batch_size and num_heads of sin/cos "
                          "must be 1, but given [%d], [%d]",
                          shape[0],
                          shape[kNumHeadsDimIndex]));
  }

  const std::vector<int64_t> position_ids_shape =
      common::vectorize(position_ids.dims());
  if (!IsEmpty(position_ids_shape)) {
    PADDLE_ENFORCE_EQ(
        (shape[seq_len_dim_index] >= seq_len &&
         shape[head_dim_index] == head_dim),
        true,
        common::errors::InvalidArgument(
            "The seq_len of sin/cos must be greater or equal to q's seq_len, "
            "and head_dim must be equal to q's head_dim, but given [%d], [%d]",
            shape[seq_len_dim_index],
            shape[head_dim_index]));

    PADDLE_ENFORCE_EQ(position_ids_shape.size(),
                      2,
                      common::errors::InvalidArgument(
                          "The position_ids's ndim must be 2, but given [%d]",
                          position_ids_shape.size()));

    PADDLE_ENFORCE_EQ(
        (position_ids_shape[0] == batch_size &&
         position_ids_shape[1] == seq_len),
        true,
        common::errors::InvalidArgument(
            "The batch_size and seq_len of position_ids must be the same as "
            "those of q. But received position_ids's "
            "shape is {%s}, q's batch_size is {%d}, q's seq_len is {%d}.",
            str_join(position_ids_shape),
            batch_size,
            seq_len));
  } else {
    PADDLE_ENFORCE_EQ(
        (shape[seq_len_dim_index] == seq_len &&
         shape[head_dim_index] == head_dim),
        true,
        common::errors::InvalidArgument(
            "The seq_len and head_dim of sin/cos must be equal to q's shape"
            ", but given [%d], [%d]",
            shape[seq_len_dim_index],
            shape[head_dim_index]));
  }
}

void infer_sin_cos(const DistMetaTensor& sin,
                   const DistMetaTensor& cos,
                   const DistMetaTensor& position_ids,
                   const TensorDistAttr& q_dist_attr_dst,
                   const std::vector<int64_t>& q_shape,
                   bool time_major,
                   bool enable_sequence_parallel,
                   TensorDistAttr* sin_dist_attr_dst,
                   TensorDistAttr* cos_dist_attr_dst) {
  const TensorDistAttr& sin_dist_attr_src = sin.dist_attr();
  const TensorDistAttr& cos_dist_attr_src = cos.dist_attr();

  *sin_dist_attr_dst = CopyTensorDistAttrForOutput(sin_dist_attr_src);
  *cos_dist_attr_dst = CopyTensorDistAttrForOutput(cos_dist_attr_src);

  // check sin, cos and position_ids shape
  const std::vector<int64_t> sin_shape = common::vectorize(sin.dims());
  const std::vector<int64_t> cos_shape = common::vectorize(cos.dims());
  // if one of sin cos is empty, they are all useless in kernel
  if (!IsEmpty(sin_shape) && !IsEmpty(cos_shape)) {
    // check sin, cos, position_ids's shape
    const int kBatchDimIndex = time_major ? 1 : 0;
    const int kSeqlenDimIndex = time_major ? 0 : 1;
    int batch_size = q_shape[kBatchDimIndex];
    int seq_len = q_shape[kSeqlenDimIndex];
    int head_dim = q_shape[kHeadDimIndex];

    int seq_len_dim_index = sin_shape.size() == 4 ? 1 : 0;
    int head_dim_index = sin_shape.size() == 4 ? 3 : 1;

    check_sin_cos(sin, cos, position_ids, batch_size, seq_len, head_dim);

    *sin_dist_attr_dst =
        enable_sequence_parallel
            ? UnShardTensorDims(sin_dist_attr_src, {head_dim_index})
            : UnShardTensorDims(sin_dist_attr_src,
                                {seq_len_dim_index, head_dim_index});
    *cos_dist_attr_dst =
        enable_sequence_parallel
            ? UnShardTensorDims(sin_dist_attr_src, {head_dim_index})
            : UnShardTensorDims(cos_dist_attr_src,
                                {seq_len_dim_index, head_dim_index});

    if (enable_sequence_parallel) {
      // shard on seq_len dimension
      std::vector<int64_t> sin_dims_mapping = sin_dist_attr_dst->dims_mapping();
      sin_dims_mapping[seq_len_dim_index] =
          q_dist_attr_dst.dims_mapping()[kSeqlenDimIndex];
      sin_dist_attr_dst->set_dims_mapping(sin_dims_mapping);

      std::vector<int64_t> cos_dims_mapping = cos_dist_attr_dst->dims_mapping();
      cos_dims_mapping[seq_len_dim_index] =
          q_dist_attr_dst.dims_mapping()[kSeqlenDimIndex];
      cos_dist_attr_dst->set_dims_mapping(cos_dims_mapping);
    }
  }
}

SpmdInfo FusedRopeInferSpmd(const DistMetaTensor& q,
                            const DistMetaTensor& k,
                            const DistMetaTensor& v,
                            const DistMetaTensor& sin,
                            const DistMetaTensor& cos,
                            const DistMetaTensor& position_ids,
                            bool use_neox_rotary_style,
                            bool time_major,
                            float rotary_emb_base) {
  check_q(q);

  std::vector<std::pair<std::string, std::vector<int64_t>>>
      inputs_sharding_info;
  std::string qkv_axes = "abcd";
  const TensorDistAttr& q_dist_attr_src = q.dist_attr();
  inputs_sharding_info.emplace_back(qkv_axes, q_dist_attr_src.dims_mapping());

  const TensorDistAttr& k_dist_attr_src = k.dist_attr();
  // q_shape equals [bs, seq_len, num_heads, head_dim] if time_major is False,
  // otherwise [seq_len, bs, num_heads, head_dim]
  std::vector<int64_t> q_shape = common::vectorize(q.dims());
  std::vector<int64_t> k_shape = common::vectorize(k.dims());
  std::vector<int64_t> v_shape = common::vectorize(v.dims());
  bool is_k_none = IsEmpty(common::vectorize(k.dims()));
  // except for q, all other inputs are optional.
  bool is_same_num_heads = true;
  bool is_divisible = true;
  if (!is_k_none) {
    check_k_or_v(k, q_shape);
    inputs_sharding_info.emplace_back(qkv_axes, k_dist_attr_src.dims_mapping());
    is_same_num_heads =
        q_shape[kNumHeadsDimIndex] == k_shape[kNumHeadsDimIndex];
    int64_t num_head_shape = k_shape[kNumHeadsDimIndex];
    int64_t num_head_mesh_dim =
        k_dist_attr_src.dims_mapping()[kNumHeadsDimIndex];
    if (num_head_mesh_dim != -1) {
      int64_t num_head_split_size =
          k_dist_attr_src.process_mesh().dim_size(num_head_mesh_dim);
      is_divisible = num_head_shape % num_head_split_size == 0;
    }
  }

  const TensorDistAttr& v_dist_attr_src = v.dist_attr();
  bool is_v_none = IsEmpty(common::vectorize(v.dims()));
  if (!is_v_none) {
    check_k_or_v(v, q_shape);
    inputs_sharding_info.emplace_back(qkv_axes, v_dist_attr_src.dims_mapping());
    is_same_num_heads =
        q_shape[kNumHeadsDimIndex] == v_shape[kNumHeadsDimIndex];
    int64_t num_head_shape = v_shape[kNumHeadsDimIndex];
    int64_t num_head_mesh_dim =
        v_dist_attr_src.dims_mapping()[kNumHeadsDimIndex];
    if (num_head_mesh_dim != -1) {
      int64_t num_head_split_size =
          v_dist_attr_src.process_mesh().dim_size(num_head_mesh_dim);
      is_divisible = num_head_shape % num_head_split_size == 0;
    }
  }

  if (!is_k_none && !is_v_none) {
    PADDLE_ENFORCE_EQ(
        k_shape,
        v_shape,
        common::errors::InvalidArgument("The shape of k and v must be same, "
                                        "but [%d]  vs [%d]",
                                        str_join(k_shape),
                                        str_join(v_shape)));
  }

  const TensorDistAttr& position_ids_dist_attr_src = position_ids.dist_attr();
  std::string position_ids_axes = time_major ? "ba" : "ab";
  bool is_ids_none = IsEmpty(common::vectorize(position_ids.dims()));
  if (!is_ids_none) {
    inputs_sharding_info.emplace_back(
        position_ids_axes, position_ids_dist_attr_src.dims_mapping());
  }
  std::unordered_map<std::string, int64_t> axis_to_dim_map =
      ShardingMergeForTensors({inputs_sharding_info});

  std::vector<int64_t> out_dims_mapping =
      GetDimsMappingForAxes(qkv_axes, axis_to_dim_map);
  TensorDistAttr q_dist_attr_dst = CopyTensorDistAttrForOutput(q_dist_attr_src);
  q_dist_attr_dst.set_dims_mapping(out_dims_mapping);

  const int kSeqlenDimIndex = time_major ? 0 : 1;
  // if one of sin cos is empty, they are all useless in kernel
  bool is_sin_cos_none = IsEmpty(common::vectorize(sin.dims())) ||
                         IsEmpty(common::vectorize(cos.dims()));

  // Enable sharding on seq_len dimension only if sin/cos is not None and
  // position_ids is None
  bool enable_sequence_parallel =
      !is_sin_cos_none && is_ids_none &&
      IsDimSharded(q_dist_attr_dst, kSeqlenDimIndex);
  if (enable_sequence_parallel) {
    // Sharded along seq_len dimension
    q_dist_attr_dst = UnShardTensorDims(q_dist_attr_dst, {kHeadDimIndex});
  } else {
    q_dist_attr_dst =
        UnShardTensorDims(q_dist_attr_dst, {kSeqlenDimIndex, kHeadDimIndex});
  }

  if (!is_same_num_heads && !is_divisible) {
    q_dist_attr_dst = UnShardTensorDims(q_dist_attr_dst, {kNumHeadsDimIndex});
  }

  TensorDistAttr k_dist_attr_dst = CopyTensorDistAttrForOutput(k_dist_attr_src);
  k_dist_attr_dst.set_process_mesh(q_dist_attr_dst.process_mesh());
  if (!is_k_none) {
    k_dist_attr_dst = q_dist_attr_dst;
  }

  TensorDistAttr v_dist_attr_dst = CopyTensorDistAttrForOutput(v_dist_attr_src);
  v_dist_attr_dst.set_process_mesh(q_dist_attr_dst.process_mesh());
  if (!is_v_none) {
    v_dist_attr_dst = q_dist_attr_dst;
  }

  TensorDistAttr sin_dist_attr_dst;
  TensorDistAttr cos_dist_attr_dst;
  infer_sin_cos(sin,
                cos,
                position_ids,
                q_dist_attr_dst,
                q_shape,
                time_major,
                enable_sequence_parallel,
                &sin_dist_attr_dst,
                &cos_dist_attr_dst);

  std::vector<int64_t> position_ids_dims_mapping =
      GetDimsMappingForAxes(position_ids_axes, axis_to_dim_map);
  TensorDistAttr position_ids_dist_attr_dst =
      CopyTensorDistAttrForOutput(position_ids_dist_attr_src);
  if (!is_ids_none) {
    position_ids_dist_attr_dst.set_dims_mapping(position_ids_dims_mapping);
    position_ids_dist_attr_dst =
        UnShardTensorDims(position_ids_dist_attr_dst, {1});
  }

  return {{q_dist_attr_dst,
           k_dist_attr_dst,
           v_dist_attr_dst,
           sin_dist_attr_dst,
           cos_dist_attr_dst,
           position_ids_dist_attr_dst},
          {q_dist_attr_dst, k_dist_attr_dst, v_dist_attr_dst}};
}

SpmdInfo FusedRopeInferSpmdReverse(const DistMetaTensor& q,
                                   const DistMetaTensor& k,
                                   const DistMetaTensor& v,
                                   const DistMetaTensor& sin,
                                   const DistMetaTensor& cos,
                                   const DistMetaTensor& position_ids,
                                   const DistMetaTensor& out_q,
                                   const DistMetaTensor& out_k,
                                   const DistMetaTensor& out_v,
                                   bool use_neox_rotary_style,
                                   bool time_major,
                                   float rotary_emb_base) {
  check_q(out_q);
  std::vector<std::pair<std::string, std::vector<int64_t>>>
      outputs_sharding_info;
  std::string qkv_axes = "abcd";
  const TensorDistAttr& out_q_dist_attr_src = out_q.dist_attr();
  outputs_sharding_info.emplace_back(qkv_axes,
                                     out_q_dist_attr_src.dims_mapping());

  const TensorDistAttr& out_k_dist_attr_src = out_k.dist_attr();
  // out_q shape = [bs, seq_len, num_heads, head_dim]
  std::vector<int64_t> out_q_shape = common::vectorize(out_q.dims());
  std::vector<int64_t> out_k_shape = common::vectorize(out_k.dims());
  std::vector<int64_t> out_v_shape = common::vectorize(out_v.dims());
  bool is_k_none = IsEmpty(common::vectorize(out_k.dims()));
  // except for q, all other inputs are optional.
  bool is_same_num_heads = true;
  bool is_divisible = true;

  if (!is_k_none) {
    check_k_or_v(out_k, out_q_shape);
    outputs_sharding_info.emplace_back(qkv_axes,
                                       out_k_dist_attr_src.dims_mapping());
    is_same_num_heads =
        out_q_shape[kHeadDimIndex] == out_k_shape[kHeadDimIndex];

    int64_t num_head_shape = out_k_shape[kNumHeadsDimIndex];
    int64_t num_head_mesh_dim =
        out_k_dist_attr_src.dims_mapping()[kNumHeadsDimIndex];
    if (num_head_mesh_dim != -1) {
      int64_t num_head_split_size =
          out_k_dist_attr_src.process_mesh().dim_size(num_head_mesh_dim);
      is_divisible = num_head_shape % num_head_split_size == 0;
    }
  }

  const TensorDistAttr& out_v_dist_attr_src = out_v.dist_attr();
  bool is_v_none = IsEmpty(common::vectorize(v.dims()));
  if (!is_v_none) {
    check_k_or_v(out_v, out_q_shape);
    outputs_sharding_info.emplace_back(qkv_axes,
                                       out_v_dist_attr_src.dims_mapping());
    is_same_num_heads =
        out_q_shape[kHeadDimIndex] == out_v_shape[kHeadDimIndex];

    int64_t num_head_shape = out_v_shape[kNumHeadsDimIndex];
    int64_t num_head_mesh_dim =
        out_v_dist_attr_src.dims_mapping()[kNumHeadsDimIndex];
    if (num_head_mesh_dim != -1) {
      int64_t num_head_split_size =
          out_v_dist_attr_src.process_mesh().dim_size(num_head_mesh_dim);
      is_divisible = num_head_shape % num_head_split_size == 0;
    }
  }

  if (!is_k_none && !is_v_none) {
    PADDLE_ENFORCE_EQ(
        out_k_shape,
        out_v_shape,
        common::errors::InvalidArgument("The shape of k and v must be same, "
                                        "but [%d]  vs [%d]",
                                        str_join(out_k_shape),
                                        str_join(out_v_shape)));
  }

  std::unordered_map<std::string, int64_t> axis_to_dim_map =
      ShardingMergeForTensors({outputs_sharding_info});

  std::vector<int64_t> dims_mapping =
      GetDimsMappingForAxes(qkv_axes, axis_to_dim_map);

  TensorDistAttr q_dist_attr_dst =
      CopyTensorDistAttrForOutput(out_q_dist_attr_src);
  q_dist_attr_dst.set_dims_mapping(dims_mapping);

  const int kSeqlenDimIndex = time_major ? 0 : 1;
  // if one of sin cos is empty, they are all useless in kernel
  bool is_sin_cos_none = IsEmpty(common::vectorize(sin.dims())) ||
                         IsEmpty(common::vectorize(cos.dims()));
  bool is_ids_none = IsEmpty(common::vectorize(position_ids.dims()));

  // Enable sharding on seq_len dimension only if sin/cos is not None and
  // position_ids is None
  bool enable_sequence_parallel =
      !is_sin_cos_none && is_ids_none &&
      IsDimSharded(q_dist_attr_dst, kSeqlenDimIndex);
  if (enable_sequence_parallel) {
    // Sharded along seq_len dimension
    q_dist_attr_dst = UnShardTensorDims(q_dist_attr_dst, {kHeadDimIndex});
  } else {
    q_dist_attr_dst =
        UnShardTensorDims(q_dist_attr_dst, {kSeqlenDimIndex, kHeadDimIndex});
  }

  if (!is_same_num_heads && !is_divisible) {
    q_dist_attr_dst = UnShardTensorDims(q_dist_attr_dst, {kNumHeadsDimIndex});
  }

  TensorDistAttr out_q_dist_attr_dst = q_dist_attr_dst;

  TensorDistAttr k_dist_attr_dst = CopyTensorDistAttrForOutput(k.dist_attr());
  k_dist_attr_dst.set_process_mesh(q_dist_attr_dst.process_mesh());
  TensorDistAttr out_k_dist_attr_dst = k_dist_attr_dst;
  if (!is_k_none) {
    k_dist_attr_dst = q_dist_attr_dst;
    out_k_dist_attr_dst = q_dist_attr_dst;
  }

  TensorDistAttr v_dist_attr_dst = CopyTensorDistAttrForOutput(v.dist_attr());
  v_dist_attr_dst.set_process_mesh(q_dist_attr_dst.process_mesh());
  TensorDistAttr out_v_dist_attr_dst = v_dist_attr_dst;
  if (!is_v_none) {
    v_dist_attr_dst = q_dist_attr_dst;
    out_v_dist_attr_dst = q_dist_attr_dst;
  }

  TensorDistAttr sin_dist_attr_dst;
  TensorDistAttr cos_dist_attr_dst;
  infer_sin_cos(sin,
                cos,
                position_ids,
                out_q_dist_attr_dst,
                out_q_shape,
                time_major,
                enable_sequence_parallel,
                &sin_dist_attr_dst,
                &cos_dist_attr_dst);

  std::string position_ids_axes = time_major ? "ba" : "ab";
  std::vector<int64_t> position_ids_dims_mapping =
      GetDimsMappingForAxes(position_ids_axes, axis_to_dim_map);
  TensorDistAttr position_ids_dist_attr_dst =
      CopyTensorDistAttrForOutput(position_ids.dist_attr());

  if (!is_ids_none) {
    position_ids_dist_attr_dst.set_dims_mapping(position_ids_dims_mapping);
    position_ids_dist_attr_dst =
        UnShardTensorDims(position_ids_dist_attr_dst, {1});
  }

  return {{q_dist_attr_dst,
           k_dist_attr_dst,
           v_dist_attr_dst,
           sin_dist_attr_dst,
           cos_dist_attr_dst,
           position_ids_dist_attr_dst},
          {out_q_dist_attr_dst, out_k_dist_attr_dst, out_v_dist_attr_dst}};
}

SpmdInfo FusedRopeGradInferSpmd(const DistMetaTensor& sin,
                                const DistMetaTensor& cos,
                                const DistMetaTensor& position_ids,
                                const DistMetaTensor& out_q_grad,
                                const DistMetaTensor& out_k_grad,
                                const DistMetaTensor& out_v_grad,
                                bool use_neox_rotary_style,
                                bool time_major,
                                float rotary_emb_base) {
  // NOTE(zhonghui): The forward and backward kernels of fuse rope are same, so
  // the spmd rules can be shared.
  SpmdInfo spmd_info = FusedRopeInferSpmd(out_q_grad,
                                          out_k_grad,
                                          out_v_grad,
                                          sin,
                                          cos,
                                          position_ids,
                                          use_neox_rotary_style,
                                          time_major);
  std::vector<ArgDistAttr> dist_attrs;
  std::vector<int> order = {3, 4, 5, 0, 1, 2};
  dist_attrs.reserve(order.size());
  for (int ind : order) {
    dist_attrs.emplace_back(spmd_info.first[ind]);
  }
  return {dist_attrs, spmd_info.second};
}

}  // namespace phi::distributed
