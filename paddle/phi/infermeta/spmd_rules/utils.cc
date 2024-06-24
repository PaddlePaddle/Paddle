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

#include "paddle/phi/infermeta/spmd_rules/utils.h"

#include <queue>

#include "glog/logging.h"

#include "paddle/phi/core/distributed/auto_parallel/dist_attr.h"
#include "paddle/phi/core/distributed/auto_parallel/utils.h"
#include "paddle/phi/core/enforce.h"

namespace phi::distributed {

using phi::distributed::auto_parallel::str_join;

std::string GetBroadcastAxes(const int64_t& tensor_ndim,
                             const int64_t& broadcast_ndim,
                             const std::string& alphabet) {
  PADDLE_ENFORCE_GE(
      alphabet.size(),
      broadcast_ndim,
      phi::errors::InvalidArgument(
          "The size of alphabet [%d] is less than broadcast ndim [%d]",
          alphabet.size(),
          broadcast_ndim));
  PADDLE_ENFORCE_GE(broadcast_ndim,
                    tensor_ndim,
                    phi::errors::InvalidArgument(
                        "The broadcast ndim [%d] is less than tensor ndim [%d]",
                        broadcast_ndim,
                        tensor_ndim));
  if (tensor_ndim <= 0) {
    return std::string();
  }
  return alphabet.substr(broadcast_ndim - tensor_ndim, tensor_ndim);
}

// Rule1: A repicated dimension could be merged by any sharded dimension.
// Rule2: A tensor axis could at most be sharded by one mesh dimension.
// (TODO trigger heuristics cost model and reshard to handle axis sharded by
// multiple dimension case.)
int64_t ShardingMergeForAxis(const std::string& axis,
                             const int64_t& mesh_dim1,
                             const int64_t& mesh_dim2) {
  if (mesh_dim1 != mesh_dim2) {
    if (mesh_dim1 == -1) {
      return mesh_dim2;
    } else if (mesh_dim2 == -1) {
      return mesh_dim1;
    } else {
      // (TODO) local cost model here.
      PADDLE_THROW(
          phi::errors::Unimplemented("Tensor Axis[%s] is Sharded by two "
                                     "different mesh dimension [%d] and [%d].",
                                     axis,
                                     mesh_dim1,
                                     mesh_dim2));
    }

  } else {
    return mesh_dim1;
  }
}

std::unordered_map<std::string, int64_t> ShardingMergeForTensors(
    const std::vector<std::pair<std::string, std::vector<int64_t>>>&
        tensor_axes_to_dim_pairs,
    const bool merge_conflicts) {
  std::unordered_map<std::string, int64_t> axis_to_dim_map;
  std::unordered_map<int64_t, std::string> dim_to_axis_map;
  int64_t merge_dim;

  for (auto& pair : tensor_axes_to_dim_pairs) {
    for (size_t i = 0; i < pair.second.size(); ++i) {
      auto tensor_axis = pair.first.substr(i, 1);
      auto mesh_dim = pair.second[i];

      if (axis_to_dim_map.count(tensor_axis) == 0) {
        merge_dim = mesh_dim;
      } else {
        merge_dim = ShardingMergeForAxis(
            tensor_axis, mesh_dim, axis_to_dim_map[tensor_axis]);
      }
      axis_to_dim_map[tensor_axis] = merge_dim;
      if (merge_dim != -1) {
        if (dim_to_axis_map.count(merge_dim) == 0) {
          dim_to_axis_map.insert({merge_dim, tensor_axis});
        } else if (dim_to_axis_map[merge_dim].find(tensor_axis) ==
                   std::string::npos) {
          dim_to_axis_map[merge_dim] += tensor_axis;
        }
      }
    }
  }

  // Resolute "mesh_dim shard by more than one axis" confict.
  // Now we just naive pick the first axis naively.
  // (TODO) use local cost model to pick the axis with lowest cost(in concern of
  // memory or communication or computation).
  for (auto& it : dim_to_axis_map) {
    if (it.second.size() > 1) {
      if (merge_conflicts) {
        VLOG(4) << "Sharding Conflict: Mesh_Dim [" << it.first
                << "] are Sharding Multiple Tensor Axis: [" << it.second
                << "]. The Axis: [" << it.second[0] << "] is Picked.";
        for (size_t i = 1; i < it.second.size(); ++i) {
          axis_to_dim_map[it.second.substr(i, 1)] = -1;
        }
      } else {
        PADDLE_THROW(phi::errors::PreconditionNotMet(
            "Multiple Tensor Axes [%s] is sharded by same mesh dimension [%d].",
            str_join(it.second),
            it.first));
      }
    }
  }

  return axis_to_dim_map;
}

TensorDistAttr CopyTensorDistAttrForOutput(
    const TensorDistAttr& src_dist_attr) {
  TensorDistAttr new_dist_attr = TensorDistAttr();
  new_dist_attr.set_process_mesh(src_dist_attr.process_mesh());
  new_dist_attr.set_batch_dim(src_dist_attr.batch_dim());
  // new_dist_attr.set_dynamic_dims(src_dist_attr.dynamic_dims());
  // new_dist_attr.set_annotated(false); TODO unset field is false by default.
  new_dist_attr.clean_partial_status();  // in partial-stage I, partial is
                                         // not allowed to propagate

  return new_dist_attr;
}

std::vector<int64_t> ResoluteOutputPartialDimension(
    const std::unordered_map<std::string, int64_t>& axis_to_dim_map,
    const std::string& tensor_axes) {
  std::vector<int64_t> partial_on_dims;

  for (auto& it : axis_to_dim_map) {
    if (tensor_axes.find(it.first) == std::string::npos) {
      if (it.second > -1) {
        partial_on_dims.push_back(it.second);
      }
    }
  }
  return partial_on_dims;
}

TensorDistAttr GetReplicatedDistAttr(const TensorDistAttr& dist_attr) {
  TensorDistAttr dst_dist_attr = CopyTensorDistAttrForOutput(dist_attr);
  std::vector<int64_t> dims_mapping(dist_attr.dims_mapping().size(), -1);
  dst_dist_attr.set_dims_mapping(dims_mapping);
  return dst_dist_attr;
}

TensorDistAttr ReplicateTensorDim(const TensorDistAttr& dist_attr, int dim) {
  TensorDistAttr dst_dist_attr = CopyTensorDistAttrForOutput(dist_attr);
  std::vector<int64_t> dims_mapping = dist_attr.dims_mapping();
  int64_t n_dim = dims_mapping.size();
  dim = dim < 0 ? n_dim + dim : dim;
  dims_mapping[dim] = kReplicateDim;
  dst_dist_attr.set_dims_mapping(dims_mapping);
  return dst_dist_attr;
}

TensorDistAttr UnShardTensorDim(const TensorDistAttr& dist_attr, int dim) {
  TensorDistAttr dst_dist_attr = CopyTensorDistAttrForOutput(dist_attr);
  std::vector<int64_t> dims_mapping = dist_attr.dims_mapping();
  int64_t n_dim = dims_mapping.size();
  dim = dim < 0 ? n_dim + dim : dim;
  dims_mapping[dim] = kReplicateDim;
  dst_dist_attr.set_dims_mapping(dims_mapping);
  return dst_dist_attr;
}

bool IsDimSharded(const TensorDistAttr& dist_attr, int dim) {
  return dist_attr.is_shard(-1, dim);
}

bool PlacementEqual(const std::shared_ptr<PlacementStatus>& a,
                    const std::shared_ptr<PlacementStatus>& b) {
  if (a->is_partial()) {
    if (!b->is_partial()) {
      return false;
    }
    auto a_partial = std::dynamic_pointer_cast<PartialStatus>(a);
    auto b_partial = std::dynamic_pointer_cast<PartialStatus>(b);
    return a_partial->get_reduce_type() == b_partial->get_reduce_type();
  }
  if (a->is_replicated()) {
    if (b->is_replicated()) {
      return true;
    }
    return false;
  }
  if (!b->is_shard()) {
    return false;
  }

  auto a_shard = std::dynamic_pointer_cast<ShardStatus>(a);
  auto b_shard = std::dynamic_pointer_cast<ShardStatus>(b);
  return a_shard->get_axis() == b_shard->get_axis();
}

void AlignDimsSharding(
    std::vector<TensorDistAttr>* input_attrs_ptr,
    const std::vector<std::vector<int64_t>>& tensor_shapes,  // NOLINT
    const std::vector<std::string>& axis_names,
    const std::set<int64_t>& skip_mesh_dims,  // NOLINT
    const std::string& align_axis,            // NOLINT
    bool allow_partial) {                     // NOLINT
  std::vector<std::pair<std::string, std::vector<int64_t>>> axes_sharding_info;
  int num_inputs = input_attrs_ptr->size();
  for (int i = 0; i < num_inputs; ++i) {
    auto& input_attr = (*input_attrs_ptr)[i];
    auto& out_axes = axis_names[i];
    auto& dims_mapping = input_attr.dims_mapping();
    axes_sharding_info.emplace_back(std::make_pair(out_axes, out_dims_mapping));
  }
  std::unordered_map<std::string, int64_t> axis_to_dim_map =
      ShardingMergeForTensors(axes_sharding_info);

  int num_inputs = input_attrs_ptr->size();
  for (int i = 0; i < num_inputs; ++i) {
    auto& input_attr = (*input_attrs_ptr)[i];
    input_attr.clean_partial_status();
    auto& out_axes = axis_names[i];
    auto dims_mapping = GetDimsMappingForAxes(out_axes, axis_to_dim_map);
    input_attr[i].set_dims_mapping(dims_mapping);
  }
}

TensorDistAttr FromPlacements(
    const TensorDistAttr& dist_attr,
    const std::vector<std::shared_ptr<PlacementStatus>>& placements) {
  TensorDistAttr dst_dist_attr = CopyTensorDistAttrForOutput(dist_attr);
  std::vector<int64_t> dims_mapping(dist_attr.dims_mapping().size(), -1);
  paddle::flat_hash_map<int64_t, ReduceType> partial_status;

  for (size_t mesh_dim = 0; mesh_dim < placements.size(); mesh_dim++) {
    auto& placement = placements[mesh_dim];
    if (placement->is_shard()) {
      auto shard_placement = std::dynamic_pointer_cast<ShardStatus>(placement);
      dims_mapping[shard_placement->get_axis()] =
          static_cast<int64_t>(mesh_dim);
    }
    if (placement->is_partial()) {
      auto partial_placement =
          std::dynamic_pointer_cast<PartialStatus>(placement);
      auto reduce_type = partial_placement->get_reduce_type();
      partial_status[mesh_dim] = reduce_type;  // NOLINT
    }
  }
  dst_dist_attr.set_dims_mapping(dims_mapping);
  dst_dist_attr.set_partial_status(partial_status);
  return dst_dist_attr;
}

TensorDistAttr UnShardTensorDims(const TensorDistAttr& dist_attr,
                                 std::vector<int64_t> dims) {
  TensorDistAttr dst_dist_attr = CopyTensorDistAttrForOutput(dist_attr);
  std::vector<int64_t> dims_mapping = dist_attr.dims_mapping();
  int64_t n_dim = dims_mapping.size();
  for (auto dim : dims) {
    dim = dim < 0 ? n_dim + dim : dim;
    dims_mapping[dim] = kReplicateDim;
  }
  dst_dist_attr.set_dims_mapping(dims_mapping);
  return dst_dist_attr;
}

std::vector<ArgDistAttr> ToArgDistAttr(
    const std::vector<TensorDistAttr>& dist_attrs) {
  std::vector<ArgDistAttr> items_dist_attrs;
  std::transform(
      dist_attrs.begin(),
      dist_attrs.end(),
      std::back_inserter(items_dist_attrs),
      [](const TensorDistAttr& attr) -> ArgDistAttr { return {attr}; });
  return items_dist_attrs;
}

std::vector<int64_t> GetLocalShape(
    const std::vector<int64_t> shape,
    const ProcessMesh& mesh,
    const std::vector<std::shared_ptr<PlacementStatus>>& placements) {
  auto local_shape = shape;
  auto n_placement = placements.size();
  for (size_t i = 0; i < n_placement; i++) {
    auto& placement = placements.at(i);
    if (placement->is_shard()) {
      auto mesh_dim_size = mesh.dim_size(i);  // NOLINT
      auto shard_dim =
          std::dynamic_pointer_cast<ShardStatus>(placement)->get_axis();
      auto split_size =
          (shape.at(shard_dim) + mesh_dim_size - 1) / mesh_dim_size;
      local_shape[shard_dim] = split_size;
    }
  }
  return local_shape;
}

std::vector<int64_t> GetDimsMappingForAxes(
    const std::string& axes,
    const std::unordered_map<std::string, int64_t>& axis_to_dim_map,
    const bool unsharded_miss_axis) {
  std::vector<int64_t> dims_mapping;
  for (int64_t i = 0, n = static_cast<int64_t>(axes.size()); i < n; i++) {
    std::string axis = axes.substr(i, 1);
    if (axis == "1") {
      dims_mapping.emplace_back(-1);
    } else {
      auto iter = axis_to_dim_map.find(axis);
      if (iter == axis_to_dim_map.end()) {
        if (unsharded_miss_axis) {
          dims_mapping.emplace_back(-1);
        } else {
          phi::errors::InvalidArgument(
              "Tensor axis [%s] of not in axis_to_dim_map.", axis);
        }
      } else {
        dims_mapping.emplace_back(iter->second);
      }
    }
  }
  return dims_mapping;
}

void DebugInfoForInferSpmd(const std::string& rule_name,
                           const SpmdInfo& infer_result) {
  VLOG(4) << "The infer spmd result of " << rule_name << " is as below:";
  auto dist_attr_for_inputs = infer_result.first;
  VLOG(4) << "======= The dist attr of inputs after inferspmd =======";
  for (size_t i = 0; i < dist_attr_for_inputs.size(); ++i) {
    if (paddle::holds_alternative<TensorDistAttr>(dist_attr_for_inputs[i])) {
      VLOG(4) << "The dist attr of the " << i << "th input need to be "
              << PADDLE_GET(TensorDistAttr, dist_attr_for_inputs[i]);
    } else if (paddle::holds_alternative<std::vector<TensorDistAttr>>(
                   dist_attr_for_inputs[i])) {
      auto& dist_attr_vec =
          PADDLE_GET(std::vector<TensorDistAttr>, dist_attr_for_inputs[i]);
      for (size_t j = 0; j < dist_attr_vec.size(); j++) {
        VLOG(4) << "The dist attr of the " << i << "th input[" << j
                << "] need to be " << dist_attr_vec[j];
      }
    } else {
      PADDLE_THROW(phi::errors::InvalidArgument(
          "The dist attr of the %d th input should be TensorDistAttr "
          "or std::vector<TensorDistAttr>.",
          i));
    }
  }
  VLOG(4) << "======= The dist attr of outputs after inferspmd =======";
  auto dist_attr_for_outputs = infer_result.second;
  for (size_t i = 0; i < dist_attr_for_outputs.size(); ++i) {
    if (paddle::holds_alternative<TensorDistAttr>(dist_attr_for_outputs[i])) {
      VLOG(4) << "The dist attr of the " << i << "th output need to be "
              << PADDLE_GET(TensorDistAttr, dist_attr_for_outputs[i]);
    } else if (paddle::holds_alternative<std::vector<TensorDistAttr>>(
                   dist_attr_for_outputs[i])) {
      auto& dist_attr_vec =
          PADDLE_GET(std::vector<TensorDistAttr>, dist_attr_for_outputs[i]);
      for (size_t j = 0; j < dist_attr_vec.size(); j++) {
        VLOG(4) << "The dist attr of the " << i << "th output[" << j
                << "] need to be " << dist_attr_vec[j];
      }
    } else {
      PADDLE_THROW(phi::errors::InvalidArgument(
          "The dist attr of the %d th output should be TensorDistAttr "
          "or std::vector<TensorDistAttr>.",
          i));
    }
  }
}

TensorDistAttr ReduceGradBroadCastDims(const TensorDistAttr& input,
                                       const ArgDistAttr& grad) {
  const auto& grad_in = PADDLE_GET_CONST(TensorDistAttr, grad);
  return ReduceGradBroadCastDims(input, grad_in);
}

TensorDistAttr ReduceGradBroadCastDims(int64_t input_dims,
                                       const TensorDistAttr& grad) {
  TensorDistAttr input;
  std::vector<int64_t> dim_mapping(input_dims, -1);
  input.set_dims_mapping(dim_mapping);
  return ReduceGradBroadCastDims(input, grad);
}

TensorDistAttr ReduceGradBroadCastDims(const TensorDistAttr& input,
                                       const TensorDistAttr& grad) {
  auto grad_dim = grad.dims_mapping().size();
  auto input_dim = input.dims_mapping().size();
  PADDLE_ENFORCE_GE(
      grad_dim,
      input_dim,
      phi::errors::InvalidArgument("grad dim must ge than input dim, but we "
                                   "got grad_dim [%d], input_dim[%d]",
                                   grad_dim,
                                   input_dim));
  if (grad_dim == input_dim) {
    return grad;
  }
  size_t broadcast_dim = grad_dim - input_dim;
  // gather partial status
  auto partial_dims = grad.partial_dims();
  auto& grad_dims_mapping = grad.dims_mapping();
  auto dims_mapping = input.dims_mapping();
  for (size_t i = 0; i < grad_dim; ++i) {
    auto mapping = grad_dims_mapping[i];
    if (i < broadcast_dim) {
      if (mapping >= 0) {
        partial_dims.insert(mapping);
      }
    } else {
      dims_mapping[i - broadcast_dim] = mapping;
    }
  }
  auto grad_out = CopyTensorDistAttrForOutput(input);
  grad_out.set_dims_mapping(dims_mapping);
  std::vector<int64_t> partial_status(partial_dims.begin(), partial_dims.end());
  grad_out.set_partial_status(partial_status);
  return grad_out;
}

}  // namespace phi::distributed
