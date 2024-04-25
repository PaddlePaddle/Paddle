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

#include "paddle/cinn/operator_fusion/pattern_node.h"
#include "paddle/cinn/operator_fusion/policy/general_topo_policy.h"
#include "paddle/cinn/operator_fusion/policy/policy_base.h"
#include "paddle/cinn/operator_fusion/policy/relative_judge_policy.h"

namespace cinn::fusion {

template <typename T>
class PolicyManager {
 public:
  PolicyManager() {}

  template <typename U>
  PolicyKind GetKey() {
    return U::Kind;
  }

  template <template <typename> typename POLICY>
  void SetPolicy(const std::shared_ptr<POLICY<T>>& policy) {
    policies[GetKey<POLICY<T>>()] =
        std::static_pointer_cast<PolicyBase<T>>(policy);
  }

  template <template <typename> typename POLICY>
  std::shared_ptr<POLICY<T>> GetPolicy() {
    PADDLE_ENFORCE_NE(
        policies.find(GetKey<POLICY<T>>()),
        policies.end(),
        phi::errors::NotFound(
            "The upstream nodes of the merged node are not unique."));
    return std::static_pointer_cast<POLICY<T>>(policies[GetKey<POLICY<T>>()]);
  }

 private:
  PolicyMap<T> policies;
};

}  // namespace cinn::fusion
