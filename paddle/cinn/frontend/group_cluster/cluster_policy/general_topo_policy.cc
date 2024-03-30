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

#include "paddle/cinn/frontend/group_cluster/cluster_policy/general_topo_policy.h"

namespace cinn::frontend::group_cluster::policy {

bool IsDownstreamNode(const PatternNodePtr start, const PatternNodePtr target) {
  if (start == target) return true;
  for (const auto& down_node : start->downstream_) {
    if (IsDownstreamNode(down_node, target)) return true;
  }
  return false;
}

bool IsIndirectDownstreamNode(const PatternNodePtr start,
                              const PatternNodePtr target) {
  for (const auto& node : start->downstream_) {
    if (node == target) continue;
    if (IsDownstreamNode(node, target)) return true;
  }
  return false;
}

bool GeneralTopoPolicy::CanFuse(const PatternNodePtr& first,
                                const PatternNodePtr& second) {
  VLOG(4) << "Start GeneralTopoPolicy";
  return !(IsIndirectDownstreamNode(first, second) ||
           IsIndirectDownstreamNode(second, first));
}

}  // namespace cinn::frontend::group_cluster::policy
