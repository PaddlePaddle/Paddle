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
#include <unordered_map>
#include "paddle/common/enforce.h"
#include "paddle/pir/core/value.h"

namespace pir {
class Block;
class Operation;

class IrMapping {
 public:
  template <typename T>
  void Add(T from, T to) {
    if (!from) return;
    MutableMap<T>()[from] = to;
  }

  template <typename T>
  T Lookup(T from) const {
    if (!from) return static_cast<T>(nullptr);
    IR_ENFORCE(Map<T>().count(from) > 0, "Not found key in IRMapping.");
    return Map<T>().at(from);
  }

  template <typename T>
  void Earse(T from) {
    MutableMap<T>().erase(from);
  }

  void Clear() {
    value_map_.clear();
    block_map_.clear();
    operation_map_.clear();
  }

  template <typename T>
  using MapType = std::unordered_map<T, T>;

  template <typename T>
  const MapType<T> &Map() const {
    if constexpr (std::is_convertible<T, Value>::value)
      return value_map_;
    else if constexpr (std::is_convertible<T, Block *>::value)
      return block_map_;
    else if constexpr (std::is_convertible<T, Operation *>::value)
      return operation_map_;
    else
      IR_THROW("Not support type in IRMapping.");
  }

  template <typename T>
  MapType<T> &MutableMap() {
    if constexpr (std::is_convertible<T, Value>::value)
      return value_map_;
    else if constexpr (std::is_convertible<T, Block *>::value)
      return block_map_;
    else if constexpr (std::is_convertible<T, Operation *>::value)
      return operation_map_;
    else
      IR_THROW("Not support type in IRMapping.");
  }

  const std::unordered_map<Value, Value> &value_map() const {
    return value_map_;
  }

 private:
  MapType<Value> value_map_;
  MapType<Block *> block_map_;
  MapType<Operation *> operation_map_;
};

}  // namespace pir
