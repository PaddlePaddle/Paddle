//   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

#include <type_traits>
#include <vector>

namespace paddle {
namespace framework {
namespace details {

template <typename ResultType, typename ElemType>
std::vector<ResultType*> DynamicCast(const std::vector<ElemType*>& container) {
  static_assert(std::is_base_of<ElemType, ResultType>::value,
                "ElementType must be a base class of ResultType");
  std::vector<ResultType*> res;
  for (auto* ptr : container) {
    auto* derived = dynamic_cast<ResultType*>(ptr);
    if (derived) {
      res.emplace_back(derived);
    }
  }
  return res;
}

}  // namespace details
}  // namespace framework
}  // namespace paddle
