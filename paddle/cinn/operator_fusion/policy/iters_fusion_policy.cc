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
  const auto source_iters = source->fusion_iters().loop_iters;
  const auto target_iters = target->fusion_iters().loop_iters;
  const auto iters_transforms = GetItersTransforms(source_iters, target_iters);
  if (iters_transforms != std::nullopt) {
    VLOG(4) << "Find iters transform: "
            << DebugStrItersTransformRoute(iters_transforms.value());
    routes_[source][target] = iters_transforms.value();
    return true;
  }
  return false;
}

ItersTransformRoute ItersFusionPolicy::GetItersTransformRoute(
    const PatternNodePtr& source, const PatternNodePtr& target) {
  PADDLE_ENFORCE(
      routes_.count(source) && routes_[source].count(target),
      ::common::errors::InvalidArgument("Can not find iters transform route."));
  PADDLE_ENFORCE_NE(routes_[source][target].size(),
                    0,
                    ::common::errors::InvalidArgument(
                        "Iters transform route should not be empty."));
  return routes_[source][target];
}

FusionItersSignature ItersFusionPolicy::FuseItersSignature(
    const PatternNodePtr& upstream,
    const PatternNodePtr& downstream,
    bool is_sink) {
  return iters_manager_->FuseItersSignature(
      upstream->fusion_iters(), downstream->fusion_iters(), is_sink);
}

std::optional<ItersTransformRoute> GetItersTransforms(
    const FusionIters& source, const FusionIters& target) {
  VLOG(4) << "Search iters transform route from "
          << cinn::utils::Join(source, ",") << " to "
          << cinn::utils::Join(target, ",");
  // ItesTransform can not support decreasing iters
  if (source.size() > target.size()) {
    return std::nullopt;
  }

  // Can apply IdentityItersTransform
  if (source == target) {
    ItersTransformRoute iters_transforms;
    iters_transforms.push_back(IdentityItersTransform());
    return iters_transforms;
  }

  std::set<std::string> source_iters(source.begin(), source.end());
  std::set<std::string> target_iters(target.begin(), target.end());
  PADDLE_ENFORCE_EQ(
      source.size(),
      source_iters.size(),
      ::common::errors::InvalidArgument(
          "The source iters should not contain duplicate elements."));
  PADDLE_ENFORCE_EQ(
      target.size(),
      target_iters.size(),
      ::common::errors::InvalidArgument(
          "The target iters should not contain duplicate elements."));

  // Source and target have the same iters but different order
  if (source_iters == target_iters) {
    ItersTransformRoute iters_transforms;
    auto perm = GetTransposePerm<int32_t>(source, target);
    iters_transforms.push_back(TransposeItersTransform(perm));
    return iters_transforms;
  }

  std::set<std::string> shared_iters =
      SetIntersection(source_iters, target_iters);
  std::set<std::string> source_unique_iters =
      SetDifference(source_iters, shared_iters);
  std::set<std::string> target_unique_iters =
      SetDifference(target_iters, shared_iters);

  if (source_unique_iters.empty() && !target_unique_iters.empty()) {
    // Source iters are a subset of target iters
    std::vector<int32_t> append_axis;
    std::vector<std::string> decreased_target(target.begin(), target.end());
    for (const auto& iter : target_unique_iters) {
      auto idx =
          std::find(decreased_target.begin(), decreased_target.end(), iter);
      append_axis.push_back(idx - decreased_target.begin());
      decreased_target.erase(idx);
    }

    ItersTransformRoute iters_transforms;
    if (decreased_target != source) {
      iters_transforms.push_back(TransposeItersTransform(
          GetTransposePerm<int32_t>(source, decreased_target)));
    }
    iters_transforms.push_back(AppendItersTransform(append_axis));
    return iters_transforms;
  } else if (!source_unique_iters.empty() && !target_unique_iters.empty()) {
    VLOG(4) << "Unimplement condition: !source_unique_iters.empty() && "
               "!target_unique_iters.empty()";
    return std::nullopt;
  } else {
    PADDLE_THROW(
        ::common::errors::PreconditionNotMet("Unexpected iters condition."));
  }
}

std::ostream& operator<<(std::ostream& os, const ItersTransform& trans) {
  os << std::visit([](auto&& arg) { return arg.DebugStr(); }, trans);
  return os;
}

std::string DebugStrItersTransformRoute(const ItersTransformRoute& route) {
  return cinn::utils::Join(route, " -> ");
}

}  // namespace cinn::fusion
