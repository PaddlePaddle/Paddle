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

#pragma once
#include <functional>
#include "paddle/cinn/operator_fusion/pattern_node.h"
#include "paddle/cinn/operator_fusion/policy/policy_base.h"
#include "paddle/cinn/operator_fusion/utils.h"
#include "paddle/common/enforce.h"

namespace cinn::fusion {

struct ItersFusionPolicy final : public PolicyBase {
  ItersFusionPolicy(std::shared_ptr<FusionItersManager> iters_manager)
      : iters_manager_(iters_manager) {
    PADDLE_ENFORCE_NOT_NULL(iters_manager,
                            "iters_manager should not be nullptr.");
  }
  static constexpr PolicyKind Kind = PolicyKind::ItersFusion;
  std::string Name() { return "ItersFusionPolicy"; }
  std::shared_ptr<FusionItersManager> iters_manager() { return iters_manager_; }

  bool CanFuseSource2Target(const PatternNodePtr& source,
                            const PatternNodePtr& target);
  std::optional<ItersTransformRoute> GetItersTransformRoute(
      const PatternNodePtr& source, const PatternNodePtr& target);
  FusionItersSignature SingleDownstreamItersFusion(
      const PatternNodePtr& upstream, const PatternNodePtr& downstream);
  FusionItersSignature MultiDownstreamItersFusion(
      const PatternNodePtr& upstream,
      const PatternNodePtr& downstream,
      const FusionItersManager::FusionDirection& direction);

  std::pair<std::vector<symbol::DimExpr>, std::vector<bool>> GetLoopDims(
      const FusionItersSignature& sig);

 private:
  std::optional<ItersTransform> GetReuseItersTransform(
      FusionIters* source_iters, const FusionIters& target_iters);
  std::optional<ItersTransformRoute> SearchTransformRouteFromReduce2Reduce(
      const FusionItersSignature& source, const FusionItersSignature& target);
  std::optional<ItersTransformRoute> SearchItersTransformRoute(
      const FusionItersSignature& source, const FusionItersSignature& target);

  using NodeRouteMap = std::unordered_map<PatternNodePtr, ItersTransformRoute>;
  std::unordered_map<PatternNodePtr, NodeRouteMap> routes_;
  std::shared_ptr<FusionItersManager> iters_manager_;
};

std::string DebugStrItersTransformRoute(const ItersTransformRoute& route);

}  // namespace cinn::fusion
