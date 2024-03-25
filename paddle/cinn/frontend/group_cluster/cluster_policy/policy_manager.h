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

#include "paddle/cinn/frontend/group_cluster/pattern_node.h"

namespace cinn::frontend::group_cluster::policy {

class Policy {
 public:
  virtual bool CanFuse(const PatternNodePtr upstream,
                       const PatternNodePtr downstream) = 0;
};

using PolicyPtr = std::shared_ptr<Policy>;

class PolicyManager {
 public:
  explicit PolicyManager(const std::vector<PolicyPtr>& policies)
      : policies_(policies) {}
  bool CanFuse(const PatternNodePtr upstream, const PatternNodePtr downstream);

 private:
  std::vector<PolicyPtr> policies_;
};

}  // namespace cinn::frontend::group_cluster::policy
