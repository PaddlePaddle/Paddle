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

#include "paddle/cinn/operator_fusion/policy/anchor_search_policy.h"
#include "paddle/cinn/operator_fusion/pattern.h"

namespace cinn::fusion {

bool IsLegalRoute(const AnchorTransformRoute& route) {
  // TODO(@wuzhanfei) we need to judge if this tranform route will reduce
  // performance
  return true;
}

std::optional<AnchorTransformRoute> SearchAnchorTransformRecursively(
    const pir::Value& begin,
    const pir::Value& end,
    AnchorTransformRoute* cur_route,
    std::unordered_set<pir::Value>* visited,
    const std::unordered_set<pir::Operation*>& ops) {
  VLOG(4) << "[SearchAnchorTransformRecursively] Curent Route:\n"
          << DebugStrOfAnchorTransformRoute(*cur_route);
  auto transforms = PossibleTransform(begin, ops);
  for (auto anchor_transform : transforms) {
    auto info = GetTransformInfo(anchor_transform);
    auto dst_value = info.DstValue();

    if (std::holds_alternative<UnsupportTransformPtr>(anchor_transform) ||
        ops.find(info.op) == ops.end() ||
        visited->find(dst_value) != visited->end() || !IsLegalRoute(*cur_route))
      continue;

    visited->emplace(dst_value);
    if (dst_value == end) {
      return *cur_route;
    }

    cur_route->emplace_back(anchor_transform);
    auto recursive_result = SearchAnchorTransformRecursively(
        dst_value, end, cur_route, visited, ops);
    cur_route->pop_back();

    if (recursive_result != std::nullopt) {
      return recursive_result;
    }
  }

  return std::nullopt;
}

std::optional<AnchorTransformRoute> FindAnchorTransformRoute(
    pir::Value begin, pir::Value end, std::unordered_set<pir::Operation*> ops) {
  AnchorTransformRoute cur_route;
  std::unordered_set<pir::Value> visited;
  visited.emplace(begin);

  auto result =
      SearchAnchorTransformRecursively(begin, end, &cur_route, &visited, ops);
  if (VLOG_IS_ON(4)) {
    if (result == std::nullopt) {
      VLOG(4) << "FindAnchorTransformRoute: Not Found";
    } else {
      VLOG(4) << "FindAnchorTransformRoute: ";
      for (const auto& trans : result.value()) {
        VLOG(4) << DebugStrOfAnchorTransform(trans);
      }
    }
  }
  return result;
}

std::optional<AnchorTransformRoute>
AnchorSearchPolicy::FindUpstreamAnchorTransformRoute(
    const PatternNodePtr& upstream, const PatternNodePtr& downstream) {
  const auto& upstream_anchor_pattern =
      std::get<AnchorPattern>(upstream->stmt_pattern());
  const auto& downstream_anchor_pattern =
      std::get<AnchorPattern>(downstream->stmt_pattern());

  return FindAnchorTransformRoute(
      upstream_anchor_pattern.anchor(),
      downstream_anchor_pattern.anchor(),
      ToUnorderedSet(ConcatVector(upstream_anchor_pattern.ops(),
                                  downstream_anchor_pattern.ops())));
}

std::optional<AnchorTransformRoute>
AnchorSearchPolicy::FindDownstreamAnchorTransformRoute(
    const PatternNodePtr& upstream, const PatternNodePtr& downstream) {
  const auto& upstream_anchor_pattern =
      std::get<AnchorPattern>(upstream->stmt_pattern());
  const auto& downstream_anchor_pattern =
      std::get<AnchorPattern>(downstream->stmt_pattern());

  return FindAnchorTransformRoute(
      downstream_anchor_pattern.anchor(),
      upstream_anchor_pattern.anchor(),
      ToUnorderedSet(ConcatVector(upstream_anchor_pattern.ops(),
                                  downstream_anchor_pattern.ops())));
}

bool AnchorSearchPolicy::HasUpstreamAnchor(const PatternNodePtr& upstream,
                                           const PatternNodePtr& downstream) {
  auto result =
      FindUpstreamAnchorTransformRoute(upstream, downstream) != std::nullopt;
  VLOG(4) << "[AnchorSearchPolicy] HasUpstreamAnchor between " << upstream
          << ", " << downstream << " : " << result;
  return result;
}

bool AnchorSearchPolicy::HasDownstreamAnchor(const PatternNodePtr& upstream,
                                             const PatternNodePtr& downstream) {
  auto result =
      FindDownstreamAnchorTransformRoute(upstream, downstream) != std::nullopt;
  VLOG(4) << "[AnchorSearchPolicy] HasDownstreamAnchor between " << upstream
          << ", " << downstream << " : " << result;
  return result;
}

}  // namespace cinn::fusion
