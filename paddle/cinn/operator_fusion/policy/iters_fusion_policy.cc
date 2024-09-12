// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "paddle/cinn/operator_fusion/policy/iters_fusion_policy.h"
#include "paddle/cinn/operator_fusion/pattern.h"

namespace cinn::fusion {

bool ItersFusionPolicy::CanFuseSource2Target(const PatternNodePtr& source,
                                             const PatternNodePtr& target) {
  const auto iters_transforms = GetItersTransformRouteImpl(
      source->fusion_iters(), target->fusion_iters());
  if (iters_transforms != std::nullopt) {
    VLOG(4) << "Find iters transform: "
            << DebugStrItersTransformRoute(iters_transforms.value());
    routes_[source][target] = iters_transforms.value();
    return true;
  }
  return false;
}

std::optional<ItersTransformRoute> ItersFusionPolicy::GetItersTransformRoute(
    const PatternNodePtr& source, const PatternNodePtr& target) {
  if (routes_.count(source) == 0 || routes_[source].count(target) == 0) {
    return std::nullopt;
  } else {
    PADDLE_ENFORCE_GT(routes_[source][target].size(),
                      0,
                      ::common::errors::InvalidArgument(
                          "Iters transform route should not be empty."));
    return routes_[source][target];
  }
}

FusionItersSignature ItersFusionPolicy::SingleDownstreamItersFusion(
    const PatternNodePtr& upstream, const PatternNodePtr& downstream) {
  return iters_manager_->SingleDownstreamItersFusion(
      upstream->fusion_iters(), downstream->fusion_iters());
}

FusionItersSignature ItersFusionPolicy::MultiDownstreamItersFusion(
    const PatternNodePtr& upstream, const PatternNodePtr& downstream) {
  return iters_manager_->MultiDownstreamItersFusion(upstream->fusion_iters(),
                                                    downstream->fusion_iters());
}

std::optional<ItersTransformRoute>
ItersFusionPolicy::GetItersTransformRouteImpl(
    const FusionItersSignature& source, const FusionItersSignature& target) {
  const auto source_iters = source.loop_iters;
  const auto target_iters = target.loop_iters;
  VLOG(4) << "Search iters transform route from "
          << cinn::utils::Join(source_iters, ",") << " to "
          << cinn::utils::Join(target_iters, ",");
  // ItesTransform can not support decreasing iters in multi downstream fusion
  if (source_iters.size() > target_iters.size()) {
    return std::nullopt;
  }
  ItersTransformRoute iters_transforms;

  // Search ItersTransform including reduce iters
  if (source.reduce_iter_nums && target.reduce_iter_nums) {
    // Reduce -> Reduce ItersTransform
    VLOG(4) << "Unimplemented case.";
    return std::nullopt;
  } else if (!source.reduce_iter_nums && target.reduce_iter_nums) {
    // Trivial -> Reduce ItersTransform
    VLOG(4) << "Unimplemented case.";
    return std::nullopt;
  } else if (source.reduce_iter_nums && !target.reduce_iter_nums) {
    // Can not support Reduce -> Trivial ItersTransform
    return std::nullopt;
  }
  // else: Trivial -> Trivial ItersTransform

  if (source_iters == target_iters) {
    // Can apply IdentityItersTransform
    iters_transforms.push_back(IdentityItersTransform());
    return iters_transforms;
  }

  const auto source_iters_set = ToSet(source_iters);
  const auto target_iters_set = ToSet(target_iters);
  PADDLE_ENFORCE_EQ(
      source_iters_set.size(),
      source_iters.size(),
      ::common::errors::InvalidArgument(
          "The source iters should not contain duplicate elements."));
  PADDLE_ENFORCE_EQ(
      target_iters_set.size(),
      target_iters.size(),
      ::common::errors::InvalidArgument(
          "The target iters should not contain duplicate elements."));

  if (source_iters_set == target_iters_set) {
    // Source and target have the same iters but different order
    const auto perm = GetTransposePerm<int32_t>(source_iters, target_iters);
    iters_transforms.push_back(TransposeItersTransform(perm));
    return iters_transforms;
  }

  const auto shared_iters = SetIntersection(source_iters_set, target_iters_set);
  const auto source_unique_iters =
      SetDifference(source_iters_set, shared_iters);
  const auto target_unique_iters =
      SetDifference(target_iters_set, shared_iters);

  FusionIters reused_source_iters = source_iters;
  if (!source_unique_iters.empty() && !target_unique_iters.empty()) {
    // Search whether exist iters in source can reuse target iter
    std::unordered_map<std::string, std::string> reuse_target_to_source;
    for (const auto& source_iter : source_unique_iters) {
      for (const auto& target_iter : target_unique_iters) {
        if (iters_manager_->IterSymbolEqual(source_iter, target_iter) &&
            !reuse_target_to_source.count(target_iter)) {
          reuse_target_to_source[target_iter] = source_iter;
          break;
        }
      }
    }
    if (reuse_target_to_source.size() != source_unique_iters.size()) {
      // Exist iters in source can not reuse target iter
      return std::nullopt;
    }
    for (const auto& [target_iter, source_iter] : reuse_target_to_source) {
      const auto it = std::find(
          reused_source_iters.begin(), reused_source_iters.end(), source_iter);
      if (it != reused_source_iters.end()) {
        *it = target_iter;
      }
    }
    iters_transforms.push_back(ReuseItersTransform(reuse_target_to_source));
  }

  const auto reused_source_iters_set = ToSet(reused_source_iters);
  PADDLE_ENFORCE_EQ(
      SetDifference(reused_source_iters_set, target_iters_set).empty(),
      true,
      ::common::errors::PreconditionNotMet("The reused source iters should not "
                                           "contains element not in target."));
  if (reused_source_iters_set != target_iters_set) {
    // Source iters are a subset of target iters
    std::vector<int32_t> append_axis;
    std::vector<symbol::DimExpr> append_symbols;
    std::vector<std::string> decreased_target = target_iters;
    for (const auto& iter : target_unique_iters) {
      auto it =
          std::find(decreased_target.begin(), decreased_target.end(), iter);
      append_axis.insert(append_axis.begin(), it - decreased_target.begin());
      append_symbols.insert(append_symbols.begin(),
                            iters_manager_->GetIterSymbol(iter));
      decreased_target.erase(it);
    }
    if (decreased_target != source_iters) {
      iters_transforms.push_back(TransposeItersTransform(
          GetTransposePerm<int32_t>(source_iters, decreased_target)));
    }
    iters_transforms.push_back(
        AppendItersTransform(append_axis, append_symbols));
  }

  return iters_transforms;
}

std::ostream& operator<<(std::ostream& os, const ItersTransform& trans) {
  os << std::visit([](auto&& arg) { return arg.DebugStr(); }, trans);
  return os;
}

std::string DebugStrItersTransformRoute(const ItersTransformRoute& route) {
  return cinn::utils::Join(route, " -> ");
}

}  // namespace cinn::fusion
