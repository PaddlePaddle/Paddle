// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
#include <vector>

#include "paddle/fluid/jit/ivalue.h"

namespace paddle {
namespace jit {
class ClassType;

namespace internal {

class Object {
 public:
  Object(const std::shared_ptr<ClassType>& type, size_t num_slot)
      : type_(type) {
    slots_.resize(num_slot);
  }

  static std::unique_ptr<Object> Create(std::shared_ptr<ClassType> type,
                                        size_t num_slot) {
    return std::make_unique<Object>(type, num_slot);
  }

  std::shared_ptr<ClassType> Type() const { return type_; }

  void SetSlot(size_t slot, IValue val) {
    if (slot >= slots_.size()) {
      slots_.resize(slot);
    }
    slots_[slot] = std::move(val);
  }

  const IValue& GetSlot(size_t slot) {
    // TODO(dev): Add ENFORCE_LT(slot, size());
    return slots_[slot];
  }

  IValue GetAttrtr(const std::string& name) const;

  void SetAttr(const std::string& name, IValue val);

 private:
  std::shared_ptr<ClassType> type_;
  // Store Tensors and Attributes
  std::vector<IValue> slots_;
};

}  // namespace internal
}  // namespace jit
}  // namespace paddle
