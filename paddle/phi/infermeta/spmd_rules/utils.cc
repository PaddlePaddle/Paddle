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
      common::errors::InvalidArgument(
          "The size of alphabet [%d] is less than broadcast ndim [%d]",
          alphabet.size(),
          broadcast_ndim));
  PADDLE_ENFORCE_GE(broadcast_ndim,
                    tensor_ndim,
                    common::errors::InvalidArgument(
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
      PADDLE_THROW(common::errors::Unimplemented(
          "Tensor Axis[%s] is Sharded by two "
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

  // Resolute "mesh_dim shard by more than one axis" conflict.
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
        PADDLE_THROW(common::errors::PreconditionNotMet(
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

void AlignDimsSharding(std::vector<TensorDistAttr>* input_attrs_ptr,
                       const std::vector<std::vector<int64_t>>& tensor_shapes,
                       const std::vector<std::string>& axis_names,
                       const std::set<int64_t>& skip_mesh_dims,
                       const std::string& align_axis,
                       bool allow_partial) {
  auto& input_attrs = *input_attrs_ptr;
  size_t n_inputs = input_attrs.size();
  PADDLE_ENFORCE_EQ(n_inputs,
                    tensor_shapes.size(),
                    common::errors::InvalidArgument(
                        "n_inputs[%d] and tensor_shapes size [%d] not equal",
                        n_inputs,
                        tensor_shapes.size()));
  PADDLE_ENFORCE_EQ(n_inputs,
                    axis_names.size(),
                    common::errors::InvalidArgument(
                        "n_inputs[%d] and axis_names size [%d] not equal",
                        n_inputs,
                        axis_names.size()));

  PADDLE_ENFORCE_EQ(align_axis.empty(),
                    false,
                    common::errors::InvalidArgument("align_axis is empty"));

  std::map<std::pair<int64_t, char>, int64_t> axis_name_to_dim;

  for (size_t i = 0; i < n_inputs; i++) {
    // 1、check all inputs have the align_axis
    for (char axi : align_axis) {
      if (axis_names[i].find(axi) == std::string::npos) {
        PADDLE_THROW(common::errors::PreconditionNotMet(
            "[%s] some axis not in  input [%d],[%s]",
            align_axis,
            i,
            axis_names[i]));
      }
    }
    // 2、build axis map
    for (size_t j = 0; j < axis_names[i].size(); j++) {
      auto axi = axis_names[i][j];
      axis_name_to_dim[{i, axi}] = j;
    }
  }
  // 3、check all inputs have the same align_axis
  auto non_empty_iter =
      std::find_if(tensor_shapes.begin(), tensor_shapes.end(), [](auto& shape) {
        return !IsEmpty(shape);
      });
  auto non_empty_index = non_empty_iter - tensor_shapes.begin();

  // 3、align non-concat dimensions according to cost
  std::vector<std::vector<std::shared_ptr<PlacementStatus>>> inputs_placements;
  std::transform(
      input_attrs.begin(),
      input_attrs.end(),
      std::back_inserter(inputs_placements),
      [](const TensorDistAttr& attr) { return attr.to_placement(); });

  const auto& process_mess = input_attrs[non_empty_index].process_mesh();
  auto has_mismatch = [&](int32_t mesh_dim) {
    for (size_t i = 0; i < n_inputs; i++) {
      if (IsEmpty(tensor_shapes[i])) {
        continue;
      }
      auto& p_a = inputs_placements[non_empty_index][mesh_dim];
      auto& p_b = inputs_placements[i][mesh_dim];
      if (p_a->is_shard() && p_b->is_shard()) {
        auto a_shard = std::dynamic_pointer_cast<ShardStatus>(p_a);
        auto b_shard = std::dynamic_pointer_cast<ShardStatus>(p_b);
        auto a_axis = axis_names[non_empty_index][a_shard->get_axis()];
        auto b_axis = axis_names[i][b_shard->get_axis()];
        if (a_axis != b_axis) {
          return true;
        }
      }

      if (!PlacementEqual(p_a, p_b)) {
        return true;
      }
    }
    return false;
  };

  // a dim can not be sharded twice along different mesh_dim
  std::set<char> sharded_axis;
  std::map<int32_t, ReduceType> partial_dim_to_type;
  std::map<int32_t, char> mesh_dim_to_axis;

  // 4、find already shard axis
  for (int32_t mesh_dim = 0; mesh_dim < process_mess.ndim(); ++mesh_dim) {
    if (!has_mismatch(mesh_dim)) {
      auto& old = inputs_placements[non_empty_index][mesh_dim];
      if (old->is_shard()) {
        auto shard_placement = std::dynamic_pointer_cast<ShardStatus>(old);
        auto axis_name =
            axis_names[non_empty_index][shard_placement->get_axis()];
        if (align_axis.find(axis_name) == std::string::npos) {
          continue;
        }
        sharded_axis.insert(axis_name);
        mesh_dim_to_axis[mesh_dim] = axis_name;
      } else if (old->is_partial()) {
        auto partial_placement = std::dynamic_pointer_cast<PartialStatus>(old);
        auto reduce_type = partial_placement->get_reduce_type();
        if (allow_partial && (reduce_type == ReduceType::kRedSum ||
                              reduce_type == ReduceType::kRedAvg)) {
          partial_dim_to_type[mesh_dim] = reduce_type;
        }
      }
    }
  }
  // 4、align axis
  for (int32_t mesh_dim = 0; mesh_dim < process_mess.ndim(); ++mesh_dim) {
    if (!has_mismatch(mesh_dim)) {
      continue;
    }
    if (skip_mesh_dims.count(mesh_dim)) {
      continue;
    }
    if (partial_dim_to_type.count(mesh_dim)) {
      continue;
    }
    std::priority_queue<std::pair<double, char>,
                        std::vector<std::pair<double, char>>,
                        std::greater<>>
        cost_queue;

    for (auto axis_name : align_axis) {
      double cost = std::numeric_limits<double>::infinity();
      if (!sharded_axis.count(axis_name)) {
        cost = 0.0;
        for (size_t i = 0; i < n_inputs; i++) {
          auto& tensor_shape = tensor_shapes[i];
          auto& tensor_dist_attr = input_attrs[i];
          if (IsEmpty(tensor_shape)) {
            continue;
          }
          auto shard_dim = axis_name_to_dim[{i, axis_name}];
          if (tensor_shape[shard_dim] < process_mess.dim_size(mesh_dim)) {
            // should not be selected
            cost += std::numeric_limits<double>::infinity();
            continue;
          }
          if (IsDimSharded(tensor_dist_attr, shard_dim)) {
            continue;
          }
          int64_t num = std::accumulate(
              tensor_shape.begin(), tensor_shape.end(), 1, std::multiplies<>());
          if (num == static_cast<int64_t>(0)) {
            continue;
          }
          std::vector<int64_t> local_shape =
              GetLocalShape(tensor_shape, process_mess, inputs_placements[i]);
          cost += std::accumulate(local_shape.begin(),
                                  local_shape.end(),
                                  1,
                                  std::multiplies<>()) *
                  process_mess.dim_size(mesh_dim);
        }
      }
      cost_queue.push(std::make_pair(cost, axis_name));
    }
    while (!cost_queue.empty()) {
      auto cost_axis = cost_queue.top();
      cost_queue.pop();
      if (sharded_axis.count(cost_axis.second)) {
        continue;
      }
      if (cost_axis.first == std::numeric_limits<double>::infinity()) {
        continue;
      }
      sharded_axis.insert(cost_axis.second);
      mesh_dim_to_axis[mesh_dim] = cost_axis.second;
      break;
    }
  }
  std::vector<TensorDistAttr> new_input_attrs;
  for (size_t i = 0; i < n_inputs; i++) {
    auto& e = input_attrs[i];
    std::vector<std::shared_ptr<PlacementStatus>> placements(
        process_mess.ndim(), std::make_shared<ReplicatedStatus>());
    for (auto pair : mesh_dim_to_axis) {
      auto shard_dim = axis_name_to_dim[{i, pair.second}];
      placements[pair.first] = std::make_shared<ShardStatus>(shard_dim);
    }
    for (auto pair : partial_dim_to_type) {
      placements[pair.first] = std::make_shared<PartialStatus>(pair.second);
    }
    new_input_attrs.emplace_back(FromPlacements(e, placements));  // NOLINT
  }
  std::swap(input_attrs, new_input_attrs);
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
          common::errors::InvalidArgument(
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
      PADDLE_THROW(common::errors::InvalidArgument(
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
      PADDLE_THROW(common::errors::InvalidArgument(
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
      common::errors::InvalidArgument("grad dim must ge than input dim, but we "
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
