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

#include <optional>
#include <unordered_map>
#include "paddle/ap/include/axpr/adt.h"
#include "paddle/ap/include/axpr/error.h"
#include "paddle/ap/include/axpr/type.h"

namespace ap::axpr {

template <typename ValueT>
struct AttrMapImpl {
  std::unordered_map<std::string, ValueT> storage;

  size_t size() const { return storage.size(); }

  void clear() { storage.clear(); }

  Result<ValueT> Get(const std::string& var) const {
    const auto& iter = storage.find(var);
    if (iter == storage.end()) {
      return AttributeError{"object has no attribute '" + var + "'"};
    }
    return iter->second;
  }

  bool Has(const std::string& var) const {
    return storage.find(var) != storage.end();
  }

  template <typename T>
  Result<T> Get(const std::string& var) const {
    ADT_LET_CONST_REF(val, this->Get(var));
    ADT_CHECK(val.template Has<T>());
    return val.template Get<T>();
  }

  template <typename T>
  Result<T> TryGet(const std::string& var) const {
    return this->template Get<T>(var);
  }

  template <typename T>
  Result<std::optional<T>> OptGet(const std::string& var) const {
    if (!this->Has(var)) {
      return std::nullopt;
    }
    ADT_LET_CONST_REF(val, this->Get(var));
    ADT_CHECK(val.template Has<T>());
    return val.template Get<T>();
  }

  std::optional<ValueT> OptGet(const std::string& var) const {
    const auto& iter = storage.find(var);
    if (iter == storage.end()) {
      return std::nullopt;
    }
    return iter->second;
  }

  void Set(const std::string& var, const ValueT& val) {
    this->storage[var] = val;
  }

  bool Emplace(const std::string& var, const ValueT& val) {
    return this->storage.emplace(var, val).second;
  }

  bool operator==(const AttrMapImpl& other) const { return &other == this; }
};

template <typename ValueT>
ADT_DEFINE_RC(AttrMap, AttrMapImpl<ValueT>);

template <typename ValueT>
struct TypeImpl<AttrMap<ValueT>> : public std::monostate {
  using value_type = AttrMap<ValueT>;

  const char* Name() const { return "AttrMap"; }
};

}  // namespace ap::axpr
