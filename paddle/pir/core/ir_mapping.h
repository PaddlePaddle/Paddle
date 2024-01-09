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
  template <typename T, typename Enable = void>
  struct ElementTypeTraits;

  template <typename T>
  struct ElementTypeTraits<
      T,
      std::enable_if_t<std::is_convertible<T, Value>::value>> {
    using Type = Value;
  };

  template <typename T>
  struct ElementTypeTraits<
      T,
      std::enable_if_t<std::is_convertible<T, const Block *>::value>> {
    using Type = const Block *;
  };

  template <typename T>
  struct ElementTypeTraits<
      T,
      std::enable_if_t<std::is_convertible<T, const Operation *>::value>> {
    using Type = const Operation *;
  };

  template <typename T>
  using ElementType = typename ElementTypeTraits<T>::Type;

  template <typename T>
  using MapType = std::unordered_map<T, T>;

  template <typename T>
  const MapType<T> &Map() const {
    if constexpr (std::is_same<T, Value>::value)
      return value_map_;
    else if constexpr (std::is_same<T, const Block *>::value)
      return block_map_;
    else if constexpr (std::is_same<T, const Operation *>::value)
      return operation_map_;
    else
      IR_THROW("Not support type in IRMapping.");
  }

  template <typename T>
  MapType<T> &MutableMap() {
    if constexpr (std::is_same<T, Value>::value)
      return value_map_;
    else if constexpr (std::is_same<T, const Block *>::value)
      return block_map_;
    else if constexpr (std::is_same<T, const Operation *>::value)
      return operation_map_;
    else
      IR_THROW("Not support type in IRMapping.");
  }

  template <typename T1, typename T2, typename ElementT = ElementType<T1>>
  void Add(T1 from, T2 to) {
    if (!from) return;
    MutableMap<ElementT>()[static_cast<ElementT>(from)] =
        static_cast<ElementT>(to);
  }

  template <typename T, typename ElementT = ElementType<T>>
  ElementT Lookup(T from) const {
    if (!from) return static_cast<ElementT>(nullptr);
    IR_ENFORCE(Map<ElementT>().count(from) > 0, "Not found key in IRMapping.");
    return Map<ElementT>().at(static_cast<ElementT>(from));
  }

  template <typename T, typename ElementT = ElementType<T>>
  void Earse(T from) {
    MutableMap<ElementT>().erase(static_cast<ElementT>(from));
  }

  void Clear() {
    value_map_.clear();
    block_map_.clear();
    operation_map_.clear();
  }

 private:
  MapType<Value> value_map_;
  MapType<const Block *> block_map_;
  MapType<const Operation *> operation_map_;
};

}  // namespace pir
