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

#include "paddle/cinn/frontend/group_cluster/cluster_policy/policy_manager.h"
#include "paddle/common/enforce.h"

namespace cinn::frontend::group_cluster::policy {

bool PolicyManager::CanFuse(const PatternNodePtr& upstream,
                            const PatternNodePtr& downstream) const {
  for (const auto& policy : policies_) {
    if (!policy->CanFuse(upstream, downstream)) return false;
  }
  return true;
}

std::vector<size_t> PolicyManager::GetFakeReduceIterIdx(
    const PatternNodePtr& upstream, const PatternNodePtr& downstream) const {
  for (const auto& policy : policies_) {
    if (policy->Name() == "RelativeJudgePolicy") {
      return policy->GetFakeReduceIterIdx(upstream, downstream);
    }
  }
  return {};
}

}  // namespace cinn::frontend::group_cluster::policy
