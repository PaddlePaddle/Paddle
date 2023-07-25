// Copyright (c) 2023 CINN Authors. All Rights Reserved.
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
#include <string>
#include <typeindex>
#include <unordered_map>

#include "glog/logging.h"

namespace cinn {
namespace hlir {
namespace drr {

class TensorInterface;

class MatchContextImpl {
 public:
  virtual ~MatchContextImpl() = default;

  virtual const TensorInterface& Tensor(
      const std::string& tensor_name) const = 0;

  template <typename T>
  const T& Node4OpCall(const std::string& op_call_name) const {
    CHECK(TypeIndex4Node() == typeid(T));
    const T* ptr = reinterpret_cast<const T*>(Node4OpCall(op_call_name));
    CHECK(ptr != nullptr) << op_call_name << " should not be null";
    return *ptr;
  }

 protected:
  virtual const void* Node4OpCall(const std::string& op_call_name) const = 0;
  virtual std::type_index TypeIndex4Node() const = 0;
};

}  // namespace drr
}  // namespace hlir
}  // namespace cinn
