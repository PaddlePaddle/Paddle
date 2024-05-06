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
#include "paddle/cinn/operator_fusion/backend/pattern.h"
#include "paddle/cinn/operator_fusion/frontend/pattern.h"

namespace cinn::fusion {

bool IsLegalRoute(const AnchorTransformRoute& route) {
  // TODO(@wuzhanfei) we need to judge if this tranform route will reduce
  // performance
  return True;
}

std::optional<AnchorTransformRoute> SearchAnchorTransformRecursively(
    const pir::Value& begin,
    const pir::Value& end,
    AnchorTransformRoute* cur_route,
    std::unordered_set<pir::Value>* visited,
    const std::unordered_set<pir::Operation*>& ops) {
  auto transforms = PossibleTransform(begin);
  for (auto anchor_transform : transforms) {
    auto info = GetTransformInfo(anchor_transform);
    auto dst_value = info.DstValue();
    cur_route->emplace_back(anchor_transform);

    if (std::holds_alternative<UnsupportTransformPtr>(anchor_transform) ||
        ops.find(info.op) == ops.end() ||
        visited->find(dst_value) != visited->end() || !IsLegalRoute(*cur_route))
      continue;

    visited->emplace(dst_value);
    if (dst_value == end) {
      return *cur_route;
    }

    auto recursive_result = SearchAnchorTransformRecursively(
        dst_value, end, cur_route, visited, ops);
    if (recursive_result != std::nullopt) {
      return recursive_result;
    }

    cur_route->pop_back();
  }

  return std::nullopt;
}

std::optional<AnchorTransformRoute> FindAnchorTransformRoute(
    pir::Value begin, pir::Value end, std::unordered_set<pir::Operation*> ops) {
  AnchorTransformRoute cur_route;
  std::unordered_set<pir::Value> visited;
  visited.emplace(begin);

  return SearchAnchorTransformRecursively(
      begin, end, &cur_route, &visited, ops);
}

template <typename T>
std::optional<AnchorTransformRoute>
AnchorSearchPolicy<T>::FindUpstreamAnchorTransformRoute(
    const PatternNodePtr<T>& upstream, const PatternNodePtr<T>& downstream) {
  const auto& upstream_anchor_pattern =
      std::get<AnchorPattern<T>>(upstream->stmt_pattern());
  const auto& downstream_anchor_pattern =
      std::get<AnchorPattern<T>>(downstream->stmt_pattern());

  return FindAnchorTransformRoute(
      upstream_anchor_pattern.anchor(),
      downstream_anchor_pattern.anchor(),
      ToSet(ConcatVector(upstream_anchor_pattern.ops(),
                         downstream_anchor_pattern.ops())));
}

template <typename T>
std::optional<AnchorTransformRoute>
AnchorSearchPolicy<T>::FindDownstreamAnchorTransformRoute(
    const PatternNodePtr<T>& upstream, const PatternNodePtr<T>& downstream) {
  const auto& upstream_anchor_pattern =
      std::get<AnchorPattern<T>>(upstream->stmt_pattern());
  const auto& downstream_anchor_pattern =
      std::get<AnchorPattern<T>>(downstream->stmt_pattern());

  return FindAnchorTransformRoute(
      downstream_anchor_pattern.anchor(),
      upstream_anchor_pattern.anchor(),
      ToSet(ConcatVector(upstream_anchor_pattern.ops(),
                         downstream_anchor_pattern.ops())));
}

template <typename T>
bool AnchorSearchPolicy<T>::HasUpstreamAnchor(
    const PatternNodePtr<T>& upstream, const PatternNodePtr<T>& downstream) {
  return FindUpstreamAnchorTransformRoute(upstream, downstream).value() !=
         std::nullopt;
}

template <typename T>
bool AnchorSearchPolicy<T>::HasDownstreamAnchor(
    const PatternNodePtr<T>& upstream, const PatternNodePtr<T>& downstream) {
  return FindDownstreamAnchorTransformRoute(upstream, downstream).value() !=
         std::nullopt;
}

template <typename T>
AnchorState<T> AnchorSearchPolicy<T>::MergeAnchorState(
    const AnchorState<T>& source, const AnchorState<T>& dest) {}

template class AnchorSearchPolicy<FrontendStage>;
template class AnchorSearchPolicy<BackendStage>;
}  // namespace cinn::fusion
