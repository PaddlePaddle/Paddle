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
  VLOG(4) << "Search iters transform route from "
          << source->fusion_iters().DebugStr() << " to "
          << target->fusion_iters().DebugStr();
  const auto iters_transforms =
      SearchItersTransformRoute(source->fusion_iters(), target->fusion_iters());
  if (iters_transforms != std::nullopt) {
    VLOG(4) << "Find iters transforms: "
            << DebugStrItersTransformRoute(iters_transforms.value());
    routes_[source][target] = iters_transforms.value();
    return true;
  } else {
    VLOG(4) << "Can't find iters transform route.";
    return false;
  }
}

std::optional<ItersTransformRoute> ItersFusionPolicy::GetItersTransformRoute(
    const PatternNodePtr& source, const PatternNodePtr& target) {
  if (routes_.count(source) && routes_[source].count(target)) {
    PADDLE_ENFORCE_GT(routes_[source][target].size(),
                      0,
                      ::common::errors::InvalidArgument(
                          "Iters transform route should not be empty."));
    return routes_[source][target];
  } else {
    return std::nullopt;
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

std::optional<ItersTransform> ItersFusionPolicy::GetReuseItersTransform(
    FusionIters* source_iters, const FusionIters& target_iters) {
  const auto [shared_iters, source_unique_iters] =
      SplitFirstWhetherInSecond(*source_iters, target_iters);
  const auto target_unique_iters =
      GatherFirstNotInSecond(target_iters, shared_iters);

  if (!source_unique_iters.empty() && !target_unique_iters.empty()) {
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
    // Replace source iters with reused target iters
    for (const auto& [target_iter, source_iter] : reuse_target_to_source) {
      const auto it =
          std::find(source_iters->begin(), source_iters->end(), source_iter);
      if (it != source_iters->end()) {
        *it = target_iter;
      }
    }
    return ReuseItersTransform(reuse_target_to_source);
  } else {
    // No need to reuse iters
    return IdentityItersTransform();
  }
}

std::optional<ItersTransformRoute>
ItersFusionPolicy::SearchTransformRouteFromReduce2Reduce(
    const FusionItersSignature& source, const FusionItersSignature& target) {
  VLOG(4) << "Start search transform Route from reduce to reduce.";
  if (source.loop_iters.size() == target.loop_iters.size() &&
      source.reduce_iter_nums == target.reduce_iter_nums) {
    // Currenly only support fusion with same iter_nums and same reduce axis
    // TODO(huangjiyi): Analysis fusion with different non reduce axis
    auto [source_flatten_iters, source_reduce_iters] = SplitReduceIters(source);
    auto [target_flatten_iters, target_reduce_iters] = SplitReduceIters(target);

    ItersTransformRoute route;
    // 1. Apply ReuseItersTransform
    const auto flatten_reuse_iters_transform =
        GetReuseItersTransform(&source_flatten_iters, target_flatten_iters);
    const auto reduce_reuse_iters_transform =
        GetReuseItersTransform(&source_reduce_iters, target_reduce_iters);
    if (flatten_reuse_iters_transform == std::nullopt ||
        reduce_reuse_iters_transform == std::nullopt) {
      return std::nullopt;
    }
    route.push_back(flatten_reuse_iters_transform.value());
    route.push_back(reduce_reuse_iters_transform.value());

    // 2. Apply TransposeItersTransform
    if (source_flatten_iters == target_flatten_iters &&
        source_reduce_iters == target_reduce_iters) {
      return route;
    } else if (source_flatten_iters != target_flatten_iters &&
               source_reduce_iters == target_reduce_iters) {
      const auto flatten_perm =
          GetTransposePerm<int32_t>(source_flatten_iters, target_flatten_iters);
      const auto perm = ConcatVector(
          flatten_perm,
          ArangeVector<int32_t>(flatten_perm.size(), target.loop_iters.size()));
      route.push_back(TransposeItersTransform(perm));
      return route;
    } else {
      // TODO(huangjiyi): Support tranpose reduce axis
      return std::nullopt;
    }
  } else {
    return std::nullopt;
  }
}

std::optional<ItersTransformRoute> ItersFusionPolicy::SearchItersTransformRoute(
    const FusionItersSignature& source, const FusionItersSignature& target) {
  auto source_iters = source.loop_iters;
  const auto target_iters = target.loop_iters;
  PADDLE_ENFORCE_EQ(
      ToSet(source_iters).size(),
      source_iters.size(),
      ::common::errors::InvalidArgument(
          "The source iters should not contain duplicate elements."));
  PADDLE_ENFORCE_EQ(
      ToSet(target_iters).size(),
      target_iters.size(),
      ::common::errors::InvalidArgument(
          "The target iters should not contain duplicate elements."));

  // ItesTransform can not support decreasing iters in multi downstream fusion
  if (source_iters.size() > target_iters.size()) {
    return std::nullopt;
  }

  // Search ItersTransform including reduce iters
  if (source.reduce_iter_nums && target.reduce_iter_nums) {
    // Reduce -> Reduce ItersTransform
    return SearchTransformRouteFromReduce2Reduce(source, target);
  } else if (source.reduce_iter_nums && !target.reduce_iter_nums) {
    // Can not transform iters from Reduce to Trivial
    return std::nullopt;
  }
  // else: Search Trivial -> Reduce ItersTransform

  ItersTransformRoute iters_transforms;
  // STEP1: Remove Ones from source
  auto source_ones = MapVectorIfTrue<std::pair<std::string, int>, int>(
      Enumerate(source_iters),
      [this](std::pair<std::string, int> p) { return p.second; },
      [this](std::pair<std::string, int> p) {
        return this->iters_manager_->IterSymbolEqualOne(p.first);
      });
  iters_transforms.emplace_back(RemoveOnesTransform(source_ones));
  source_iters = GatherVectorExcept(source_iters, source_ones);

  // STEP2: Do transpose and axes reuse analysis.
  // 1. Apply IdentityItersTransform if source iters are equal to target
  if (source_iters == target_iters) {
    iters_transforms.push_back(IdentityItersTransform());
    return iters_transforms;
  }

  // 2. Apply ReuseItersTransform
  // if all source unique iters can reuse target iters
  FusionIters reused_source_iters = source_iters;
  const auto reuse_iters_transform =
      GetReuseItersTransform(&reused_source_iters, target_iters);
  if (reuse_iters_transform == std::nullopt) {
    return std::nullopt;
  } else {
    iters_transforms.push_back(reuse_iters_transform.value());
  }
  VLOG(4) << "source iters after reuse: "
          << PrintFusionIters(reused_source_iters);

  PADDLE_ENFORCE_EQ(
      AllFirstInSecond(reused_source_iters, target_iters),
      true,
      ::common::errors::PreconditionNotMet("The reused source iters should not "
                                           "contains element not in target."));
  const auto reused_target_unique_iters =
      GatherFirstNotInSecond(target_iters, reused_source_iters);

  // 3. Apply AppendItersTransform
  // if exist iters in target can not find in source
  FusionIters appended_source_iters = reused_source_iters;
  if (!reused_target_unique_iters.empty()) {
    std::vector<int32_t> append_axis;
    std::vector<symbol::DimExpr> append_symbols;
    for (const auto& iter : reused_target_unique_iters) {
      const size_t pos =
          std::find(target_iters.begin(), target_iters.end(), iter) -
          target_iters.begin();
      append_axis.push_back(pos);
      append_symbols.push_back(iters_manager_->GetIterSymbol(iter));
      appended_source_iters.insert(appended_source_iters.begin() + pos, iter);
    }
    iters_transforms.push_back(
        AppendItersTransform(append_axis, append_symbols));
    if (appended_source_iters == target_iters) {
      return iters_transforms;
    }
  }
  VLOG(4) << "source iters after reuse and append: "
          << PrintFusionIters(appended_source_iters);

  PADDLE_ENFORCE_EQ(ToSet(appended_source_iters),
                    ToSet(target_iters),
                    ::common::errors::PreconditionNotMet(
                        "The source iters after reuse and append "
                        "should have same elements as target."));

  // 4. Apply TransposeItersTransform
  // if source iters after reuse and append are not equal to target
  if (appended_source_iters != target_iters) {
    const auto perm =
        GetTransposePerm<int32_t>(appended_source_iters, target_iters);
    iters_transforms.push_back(TransposeItersTransform(perm));
    return iters_transforms;
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
