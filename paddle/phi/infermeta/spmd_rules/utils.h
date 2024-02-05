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

#include <set>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "paddle/phi/core/attribute.h"
#include "paddle/phi/core/distributed/auto_parallel/dist_meta_tensor.h"
#include "paddle/phi/core/distributed/type_defs.h"

namespace phi {
namespace distributed {
class TensorDistAttr;

inline bool IsEmpty(const std::vector<int64_t>& shape) {
  return shape.empty() || shape.at(0) == 0;
}

// Generate the axis notation of tensor for the einsum notation of a broadcast
// operation(alignment star from the rightmost axis). tensor_ndim: the size of
// the tensor. broadcast_ndim: the maximum size of tensors in this broadcast
// operation. alphabet: the characters used to represent the axes of tensor.
// length of alphabet should >= broadcast_ndim.
std::string GetBroadcastAxes(const int64_t& tensor_ndim,
                             const int64_t& broadcast_ndim,
                             const std::string& alphabet);

// Merge the sharding specification (dims mapping) for one tensor Axis.
// Rule1: A replicated dimension could be merged by any sharded dimension.
// Rule2: A tensor axis could at most be sharded by one mesh dimension.
// (TODO trigger heuristics cost model and reshard to handle axis sharded by
// multiple dimension case.)
int64_t ShardingMergeForAxis(const std::string& axis,
                             const int64_t& mesh_dim1,
                             const int64_t& mesh_dim2);

// Merge sharding specification (dims mapping) of given tensors.
// The same axes of different tensors will be merged.
std::unordered_map<std::string, int64_t> ShardingMergeForTensors(
    const std::vector<std::pair<std::string, std::vector<int64_t>>>&
        tensor_axes_to_dim_pairs,
    const bool merge_conflicts = true);

// Intend to use for generating the TensorDistAttr of output based on the input
// activation TensorDistAttr. The process_mesh, batch_dim, dynamic_dim are
// copied with annotated is forced to False, and dims_mapping is leave to be
// null.
TensorDistAttr CopyTensorDistAttrForOutput(const TensorDistAttr& src_dist_attr);

TensorDistAttr UnShardTensorDims(const TensorDistAttr& dist_attr,
                                 std::vector<int64_t> dims);

// Resolute the partial mesh dimension of a output tensor, giving the
// merged sharding specification of input tensors and the axis names of output
// tensor. Input are
std::vector<int64_t> ResoluteOutputPartialDimension(
    const std::unordered_map<std::string, int64_t>& axis_to_dim_map,
    const std::string& tensor_axes);

// Construct a DistAttr from the incoming DistAttr corresponding to the
// Replicated state
TensorDistAttr GetReplicatedDistAttr(const TensorDistAttr& dist_attr);

bool IsDimSharded(const TensorDistAttr& dist_attr, int dim);

std::vector<int64_t> GetLocalShape(
    const std::vector<int64_t> shape,
    const ProcessMesh& mesh,
    const std::vector<std::shared_ptr<PlacementStatus>>& placements);

TensorDistAttr FromPlacements(
    const TensorDistAttr& dist_attr,
    const std::vector<std::shared_ptr<PlacementStatus>>& placements);

std::vector<ArgDistAttr> ToArgDistAttr(
    const std::vector<TensorDistAttr>& dist_attrs);

TensorDistAttr ReplicateTensorDim(const TensorDistAttr& dist_attr, int dim);

TensorDistAttr UnShardTensorDim(const TensorDistAttr& dist_attr, int dim);

bool PlacementEqual(const std::shared_ptr<PlacementStatus>& a,
                    const std::shared_ptr<PlacementStatus>& b);

void AlignDimsSharding(std::vector<TensorDistAttr>* input_attrs,
                       const std::vector<std::vector<int64_t>>& tensor_shapes,
                       const std::vector<std::string>& axis_names,
                       const std::set<int64_t>& skip_mesh_dims,
                       const std::string& align_axis,
                       bool allow_partial);

// Adaptor for variadic arguments
template <typename Functor>
struct ArgsIterator {
  template <typename... Args>
  inline Functor& apply() {
    return self();
  }

  template <typename T, typename... Args>
  inline Functor& apply(T&& arg, Args&&... args) {
    self()(std::forward<T>(arg));
    if (self().short_circuit()) {
      return self();
    } else {
      return apply(std::forward<Args>(args)...);
    }
  }

  constexpr bool short_circuit() const { return false; }

 private:
  inline Functor& self() { return *static_cast<Functor*>(this); }
};

using SpmdFn = SpmdInfo (*)(const std::vector<const DistMetaTensor*>& ins,
                            const std::vector<const DistMetaTensor*>& outs);

namespace detail {
template <SpmdFn Fn>
struct VariadicSpmdRuleArgumentParser
    : public ArgsIterator<VariadicSpmdRuleArgumentParser<Fn>> {
  std::vector<const DistMetaTensor*> inputs;
  std::vector<const DistMetaTensor*> outputs;

  // deal with inputs
  void operator()(const DistMetaTensor& x) { inputs.emplace_back(&x); }

  void operator()(const std::vector<const DistMetaTensor*>& x) {
    for (auto t : x) {
      inputs.emplace_back(t);
    }
  }

  void operator()(const std::vector<DistMetaTensor>& x) {
    for (auto& t : x) {
      inputs.emplace_back(&t);
    }
  }

  // deal with outputs
  void operator()(DistMetaTensor* out) { outputs.emplace_back(out); }

  void operator()(std::vector<DistMetaTensor*> out) {
    for (auto t : out) {
      outputs.emplace_back(t);
    }
  }

  SpmdInfo InferForward() { return Fn(inputs, outputs); }

  SpmdInfo InferBackward() { return Fn(inputs, outputs); }
};

using DynamicSpmdFn = SpmdInfo (*)(
    const std::vector<paddle::variant<const DistMetaTensor*,
                                      const std::vector<DistMetaTensor>*>>&);

template <DynamicSpmdFn Fn>
struct ReplicateInferSpmdDynamicHelper
    : public ArgsIterator<ReplicateInferSpmdDynamicHelper<Fn>> {
  SpmdInfo Infer() { return Fn(inputs); }

  void operator()(const DistMetaTensor& x) { inputs.emplace_back(&x); }
  void operator()(const std::vector<DistMetaTensor>& x) {
    inputs.emplace_back(&x);
  }

  void operator()(std::vector<DistMetaTensor>&& x) = delete;
  void operator()(DistMetaTensor&& x) = delete;

  std::vector<paddle::variant<const DistMetaTensor*,
                              const std::vector<DistMetaTensor>*>>
      inputs;
};
}  // namespace detail

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

void DebugInfoForInferSpmd(const std::string& rule_name,
                           const SpmdInfo& infer_result);

TensorDistAttr ReduceGradBroadCastDims(const TensorDistAttr& input,
                                       const ArgDistAttr& grad);

TensorDistAttr ReduceGradBroadCastDims(const TensorDistAttr& input,
                                       const TensorDistAttr& grad);

TensorDistAttr ReduceGradBroadCastDims(int64_t input_dims,
                                       const TensorDistAttr& grad);

}  // namespace distributed
}  // namespace phi
