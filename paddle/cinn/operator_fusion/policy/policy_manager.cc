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

#include "paddle/cinn/operator_fusion/policy/policy_manager.h"
#include "paddle/cinn/operator_fusion/backend/pattern.h"
#include "paddle/cinn/operator_fusion/frontend/pattern.h"
#include "paddle/common/enforce.h"

namespace cinn::fusion {

template <typename T>
bool PolicyManager<T>::CanFuse(const PatternNodePtr<T>& upstream,
                               const PatternNodePtr<T>& downstream) const {
  for (const auto& policy : policies_) {
    if (!policy->CanFuse(upstream, downstream)) return false;
  }
  return true;
}

template <typename T>
std::vector<size_t> PolicyManager<T>::GetFakeReduceIterIdx(
    const PatternNodePtr<T>& upstream,
    const PatternNodePtr<T>& downstream) const {
  for (const auto& policy : policies_) {
    if (policy->Name() == "RelativeJudgePolicy") {
      return policy->GetFakeReduceIterIdx(upstream, downstream);
    }
  }
  return {};
}

template class PolicyManager<FrontendStage>;
template class PolicyManager<BackendStage>;

}  // namespace cinn::fusion
