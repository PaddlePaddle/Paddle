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
#include <memory>
#include <unordered_map>

namespace cinn::fusion {

enum PolicyKind { GeneralTopo = 1, RelativeJudge = 2, ItersFusion = 3 };

struct PolicyKindHash {
  std::size_t operator()(const PolicyKind& t) const {
    return static_cast<std::size_t>(t);
  }
};

class PolicyBase {};

using PolicyPtr = std::shared_ptr<PolicyBase>;

using PolicyMap = std::unordered_map<PolicyKind, PolicyPtr, PolicyKindHash>;

}  // namespace cinn::fusion
