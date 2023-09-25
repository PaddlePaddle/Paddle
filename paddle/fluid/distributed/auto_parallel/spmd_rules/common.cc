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

#include "paddle/fluid/distributed/auto_parallel/spmd_rules/common.h"

#include <glog/logging.h>

#include "paddle/fluid/distributed/auto_parallel/spmd_rules/rules.h"
#include "paddle/phi/core/distributed/auto_parallel/utils.h"

namespace paddle {
namespace distributed {
namespace auto_parallel {

using phi::distributed::auto_parallel::str_join;

std::pair<std::vector<TensorDistAttr>, std::vector<TensorDistAttr>>
SPMDRuleBase::InferForward(const std::vector<DistTensorSpec>& input_specs,
                           const paddle::framework::AttributeMap& attrs) {
  PADDLE_THROW(
      phi::errors::Unimplemented("InferForward should be called from a "
                                 "derived class of SPMDRuleBase !"));
}

std::pair<std::vector<TensorDistAttr>, std::vector<TensorDistAttr>>
SPMDRuleBase::InferBackward(const std::vector<DistTensorSpec>& input_specs,
                            const std::vector<DistTensorSpec>& output_specs,
                            const paddle::framework::AttributeMap& attrs) {
  PADDLE_THROW(
      phi::errors::Unimplemented("InferBackward should be called from a "
                                 "derived class of SPMDRuleBase !"));
}

// deprecated
std::pair<std::vector<TensorDistAttr>, std::vector<TensorDistAttr>>
SPMDRuleBase::InferBackward(const std::vector<DistTensorSpec>& output_specs,
                            const paddle::framework::AttributeMap& attrs) {
  PADDLE_THROW(
      phi::errors::Unimplemented("InferBackward should be called from a "
                                 "derived class of SPMDRuleBase !"));
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

TensorDistAttr CopyTensorDistAttrForOutput(
    const TensorDistAttr& src_dist_attr) {
  TensorDistAttr new_dist_attr = TensorDistAttr();
  new_dist_attr.set_process_mesh(src_dist_attr.process_mesh());
  new_dist_attr.set_batch_dim(src_dist_attr.batch_dim());
  new_dist_attr.set_dynamic_dims(src_dist_attr.dynamic_dims());
  // new_dist_attr.set_annotated(false); TODO unset field is false by default.
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

std::string GetBroadcastAxes(const int64_t& tenosr_ndim,
                             const int64_t& broadcast_ndim,
                             const std::string& alphabet) {
  PADDLE_ENFORCE_GE(
      alphabet.size(),
      broadcast_ndim,
      phi::errors::InvalidArgument(
          "size of alphabet [%d] is less than broadcast ndim [%d]",
          alphabet.size(),
          broadcast_ndim));
  PADDLE_ENFORCE_GE(broadcast_ndim,
                    tenosr_ndim,
                    phi::errors::InvalidArgument(
                        "broadcast ndim [%d] is less than tenosr ndim [%d]",
                        broadcast_ndim,
                        tenosr_ndim));
  if (tenosr_ndim <= 0) {
    return std::string();
  }
  return alphabet.substr(broadcast_ndim - tenosr_ndim, tenosr_ndim);
}

TensorDistAttr ReplicatedOnMesh(const TensorDistAttr& src_dist_attr) {
  TensorDistAttr replicated_dist_attr = src_dist_attr;
  replicated_dist_attr.clear_annotated();
  size_t tensor_ndim = replicated_dist_attr.dims_mapping().size();
  replicated_dist_attr.set_dims_mapping(std::vector<int64_t>(tensor_ndim, -1));
  return replicated_dist_attr;
}

void VerifySpecs(const std::vector<DistTensorSpec>& specs,
                 const std::string& op_name) {
  for (size_t i = 0, n = specs.size(); i < n; ++i) {
    const std::vector<int64_t>& shape = specs[i].shape();
    const std::vector<int64_t>& dims_mapping = specs[i].dims_mapping();
    PADDLE_ENFORCE_EQ(shape.size(),
                      dims_mapping.size(),
                      phi::errors::InvalidArgument(
                          "Mismatch in %s, spec[%d]'s tensor size: [%d] and "
                          "spec[%d]'s dims_mapping size [%d].",
                          op_name,
                          i,
                          shape.size(),
                          i,
                          dims_mapping.size()));
  }
}

std::vector<std::pair<std::string, std::vector<int64_t>>>
GetAxesDimsMappingPair(const std::vector<std::string>& tensor_axes,
                       const std::vector<DistTensorSpec>& specs) {
  std::vector<std::pair<std::string, std::vector<int64_t>>> res;
  size_t ntensor = specs.size();
  for (size_t i = 0; i < ntensor; ++i) {
    res.emplace_back(tensor_axes[i], specs[i].dims_mapping());
  }
  return res;
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

// SPMDRuleMap
SPMDRuleMap& SPMDRuleMap::Instance() {
  static SPMDRuleMap g_spmd_rule_map;
  return g_spmd_rule_map;
}

// To enable default replicated spmd rule for op that are NOT registered
// which all tensors of inputs and outputs will be replicated in all ranks of
// the mesh.
SPMDRuleBase* SPMDRuleMap::Get(const std::string& op_type) const {
  auto rule_ptr = GetNullable(op_type);
  if (rule_ptr == nullptr) {
    std::string str;
    for (const auto& item : map_) {
      str += item.first + ", ";
    }
    VLOG(4) << "Size of current map [" << map_.size() << "]";
    VLOG(4) << "Keys are [" << str << "]";
  }
  PADDLE_ENFORCE_NOT_NULL(
      rule_ptr,
      platform::errors::NotFound(
          "NO SPMD Rule has been registered for Operator [%s].", op_type));
  return rule_ptr;
}

SPMDRuleBase* SPMDRuleMap::GetNullable(const std::string& op_type) const {
  auto it = map_.find(op_type);
  if (it == map_.end()) {
    return nullptr;
  } else {
    return it->second.get();
  }
}

int SPMDRuleMap::Insert(const std::string& op_type,
                        std::unique_ptr<SPMDRuleBase> rule) {
  VLOG(4) << "Call SPMDRuleMap::Insert!";
  PADDLE_ENFORCE_NE(
      Has(op_type),
      true,
      platform::errors::AlreadyExists(
          "SPMD Rule for Operator [%s] has been registered.", op_type));
  map_.insert({op_type, std::move(rule)});

  return 1;
}

}  // namespace auto_parallel
}  // namespace distributed
}  // namespace paddle
