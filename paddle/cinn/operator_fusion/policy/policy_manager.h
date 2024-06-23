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

namespace cinn::fusion {

template <typename T>
class Policy {
 public:
  virtual std::string Name() = 0;
  virtual bool CanFuse(const PatternNodePtr<T>& upstream,
                       const PatternNodePtr<T>& downstream) = 0;
  virtual std::vector<size_t> GetFakeReduceIterIdx(
      const PatternNodePtr<T>& upstream, const PatternNodePtr<T>& downstream) {
    return {};
  }
};

template <typename T>
using PolicyPtr = std::shared_ptr<Policy<T>>;

template <typename T>
class PolicyManager {
 public:
  explicit PolicyManager(const std::vector<PolicyPtr<T>>& policies)
      : policies_(policies) {}
  bool CanFuse(const PatternNodePtr<T>& upstream,
               const PatternNodePtr<T>& downstream) const;
  std::vector<size_t> GetFakeReduceIterIdx(
      const PatternNodePtr<T>& upstream,
      const PatternNodePtr<T>& downstream) const;

  PolicyPtr<T> find_policy(const std::string& name) const {
    for (auto& p : policies_) {
      VLOG(4) << "Find policy: " << p->Name();
      if (p->Name() == name) return p;
    }
    return nullptr;
  }

 private:
  std::vector<PolicyPtr<T>> policies_;
};

}  // namespace cinn::fusion
