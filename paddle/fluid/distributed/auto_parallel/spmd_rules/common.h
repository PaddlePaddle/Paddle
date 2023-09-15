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

#pragma once

#include <iterator>
#include <map>
#include <string>
#include <utility>
#include <vector>

#include "paddle/fluid/distributed/auto_parallel/spmd_rules/dist_tensor_spec.h"
#include "paddle/fluid/framework/attribute.h"
#include "paddle/fluid/framework/type_defs.h"
#include "paddle/fluid/platform/enforce.h"
#include "paddle/phi/core/distributed/auto_parallel/dist_attr.h"
#include "paddle/utils/flat_hash_map.h"

namespace paddle {
namespace distributed {
namespace auto_parallel {

using paddle::framework::Attribute;

class SPMDRuleBase {
 public:
  virtual ~SPMDRuleBase() {}

  // Based on the information of Input Tensors and Op Attribute:
  // 1. Merge the Sharding (dims_mapping) among Input Tensors.
  // 2. Infer the Sharding (dims_mapping) for Output Tensors.
  // The Info of input tensors (Shape and DistAttr) are wrapped as
  // DistTensorSpec, and  op attribtue should be given as AttributeMap. The
  // Output is a pair consist of two vectors:
  // 1. The first vector: the merged DistAttr of input tensors.
  // 2. The infered DistAttr of output tensors.
  // The Merged DistAttr might be different from the original Intput DistAttrs,
  // which means that the corressponding input tensor need to be reshard.
  virtual std::pair<std::vector<TensorDistAttr>, std::vector<TensorDistAttr>>
  InferForward(const std::vector<DistTensorSpec>& input_specs,
               const paddle::framework::AttributeMap& attrs);

  // Based on the information of Input & Output Tensors and Op Attribute:
  // 1. Merge the Sharding (dims_mapping) among Output Tensors.
  // 2. Infer the Sharding (dims_mapping) for Input Tensors.
  // The Info of output tensors (Shape and DistAttr) are wrapped as
  // DistTensorSpec, and  op attribtue should be given as AttributeMap. The
  // Output is a pair consist of two vectors:
  // 1. The first vector: the merged DistAttr of output tensors.
  // 2. The infered DistAttr of Input tensors.
  virtual std::pair<std::vector<TensorDistAttr>, std::vector<TensorDistAttr>>
  InferBackward(const std::vector<DistTensorSpec>& input_specs,
                const std::vector<DistTensorSpec>& output_specs,
                const paddle::framework::AttributeMap& attrs);

  // deprecated, to be remove in future
  virtual std::pair<std::vector<TensorDistAttr>, std::vector<TensorDistAttr>>
  InferBackward(const std::vector<DistTensorSpec>& output_specs,
                const paddle::framework::AttributeMap& attrs);

  template <typename T>
  inline const T ExtractAttr(
      const std::string& name,
      const paddle::framework::AttributeMap& attrs) const {
    auto attr = GetAttr(name, attrs);
    return *paddle::framework::ExtractAttribute<T>(name)(attr);
  }

  Attribute GetAttr(const std::string& name,
                    const paddle::framework::AttributeMap& attrs) const {
    auto iter = attrs.find(name);
    PADDLE_ENFORCE_NE(iter,
                      attrs.end(),
                      paddle::platform::errors::NotFound(
                          "(%s) is not found in AttributeMap.", name));
    return iter->second;
  }
};

// Merge sharding specification (dims mapping) of given tensors.
// The same axes of different tensors will be merged.
std::unordered_map<std::string, int64_t> ShardingMergeForTensors(
    const std::vector<std::pair<std::string, std::vector<int64_t>>>&
        tensor_axes_to_dim_pairs,
    const bool merge_conflicts = true);

// Merge the sharding specification (dims mapping) for one tensor Axis.
// Rule1: A repicated dimension could be merged by any sharded dimension.
// Rule2: A tensor axis could at most be sharded by one mesh dimension.
// (TODO trigger heuristics cost model and reshard to handle axis sharded by
// multiple dimension case.)
int64_t ShardingMergeForAxis(const std::string& axis,
                             const int64_t& mesh_dim1,
                             const int64_t& mesh_dim2);

// Intend to use for generating the TensorDistAttr of output based on the input
// activation TensorDistAttr. The process_mesh, batch_dim, dynamic_dim are
// copied with annotated is forced to False, and dims_mapping is leave to be
// null.
TensorDistAttr CopyTensorDistAttrForOutput(const TensorDistAttr& src_dist_attr);

// Resolute the partial mesh dimension of a output tensor, giving the
// merged sharding specifcation of input tensors and the axis names of output
// tensor. Input are
std::vector<int64_t> ResoluteOutputPartialDimension(
    const std::unordered_map<std::string, int64_t>& axis_to_dim_map,
    const std::string& tensor_axes);

// Generate the axis notation of tensor for the einsum notation of a broadcast
// operation(alignment star from the rightmost axis). tenosr_ndim: the size of
// the tensor. broadcast_ndim: the maxium size of tensors in this broadcast
// operation. alphabet: the characters used to represent the axes of tensor.
// length of alphabet should >= broadcast_ndim.
std::string GetBroadcastAxes(const int64_t& tenosr_ndim,
                             const int64_t& broadcast_ndim,
                             const std::string& alphabet);

// Return a NEW TensorDistAttr whose dims mapping is consist of "-1"
// (unsharded).
TensorDistAttr ReplicatedOnMesh(const TensorDistAttr& src_dist_attr);

// Check whether the given DistTensorSpec objects are valid. For each
// DistTensorSpec, the rank of its dims mapping must be equal to the rank of its
// corresponding tensor shape. the parameter op_name is used for logging error
// message.
void VerifySpecs(const std::vector<DistTensorSpec>& specs,
                 const std::string& op_name);

// Get dims mapping for the given tensors. Return the pair of each
// tensor's einsum notation and the corresponding dims mapping.
std::vector<std::pair<std::string, std::vector<int64_t>>>
GetAxesDimsMappingPair(const std::vector<std::string>& tensor_axes,
                       const std::vector<DistTensorSpec>& specs);

// Get dims mapping for the given axes according to sharding information of
// the annotated axes after inferring forward or backward. The parameter axis
// stores the axes of the tensor. "1" is a special axis, for the axis "1", set
// its dims mapping to -1.
// if unsharded_miss_axis, "-1" is assigend to axes that has no key in
// axis_to_dim_map.
std::vector<int64_t> GetDimsMappingForAxes(
    const std::string& axes,
    const std::unordered_map<std::string, int64_t>& axis_to_dim_map,
    const bool unsharded_miss_axis = false);

// The static map that stores and initializes all the registered SPMD rules.
class SPMDRuleMap {
 public:
  ~SPMDRuleMap() = default;

  // A singleton
  static SPMDRuleMap& Instance();

  // Returns the spmd rule for the given op_type
  SPMDRuleBase* Get(const std::string& op_type) const;

  // Returns the spmd by name or nullptr if not registered
  SPMDRuleBase* GetNullable(const std::string& op_type) const;

  // Register a spmd for an op_type.
  int Insert(const std::string& op_type, std::unique_ptr<SPMDRuleBase> rule);

  bool Has(const std::string& op_type) const {
    return map_.find(op_type) != map_.end();
  }

 private:
  SPMDRuleMap() = default;
  paddle::flat_hash_map<std::string, std::unique_ptr<SPMDRuleBase>> map_;
  DISABLE_COPY_AND_ASSIGN(SPMDRuleMap);
};

#define REGISTER_SPMD_RULE(op_type, rule_class, ...)                        \
  UNUSED static int __spmd_rule_holder_##op_type =                          \
      ::paddle::distributed::auto_parallel::SPMDRuleMap::Instance().Insert( \
          #op_type, std::make_unique<rule_class>(__VA_ARGS__))

}  // namespace auto_parallel
}  // namespace distributed
}  // namespace paddle
