/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once

#include <cassert>
#include <map>
#include <mutex>
#include <string>
#include <vector>

#include "paddle/phi/core/utils/type_info.h"

namespace phi {

template <typename BaseT>
class TypeRegistry {
 public:
  TypeRegistry(const TypeRegistry&) = delete;
  TypeRegistry& operator=(const TypeRegistry&) = delete;

  static TypeRegistry& GetInstance();

  TypeInfo<BaseT> RegisterType(const std::string& type);
  const std::string& GetTypeName(TypeInfo<BaseT> info) const;

 private:
  TypeRegistry() = default;
  mutable std::mutex mutex_;
  std::vector<std::string> names_;
  std::map<std::string, int8_t> name_to_id_;
};

template <typename BaseT>
TypeRegistry<BaseT>& TypeRegistry<BaseT>::GetInstance() {
  static TypeRegistry<BaseT> registry;
  return registry;
}

template <typename BaseT>
TypeInfo<BaseT> TypeRegistry<BaseT>::RegisterType(const std::string& type) {
  std::lock_guard<std::mutex> guard(mutex_);
  assert(name_to_id_.find(type) == name_to_id_.end());
  assert(names_.size() < static_cast<decltype(names_.size())>(
                             std::numeric_limits<int8_t>::max()));
  int8_t id = static_cast<int8_t>(names_.size());
  names_.emplace_back(type);
  name_to_id_[type] = id;
  return TypeInfo<BaseT>(id);
}

template <typename BaseT>
const std::string& TypeRegistry<BaseT>::GetTypeName(
    TypeInfo<BaseT> info) const {
  std::lock_guard<std::mutex> guard(mutex_);
  int8_t id = info.id();
  assert(id >= 0);
  assert(static_cast<size_t>(id) < names_.size());
  return names_[id];
}

template <typename BaseT>
TypeInfo<BaseT> RegisterStaticType(const std::string& type) {
  return TypeRegistry<BaseT>::GetInstance().RegisterType(type);
}

template <typename BaseT>
const std::string& TypeInfo<BaseT>::name() const {
  return TypeRegistry<BaseT>::GetInstance().GetTypeName(*this);
}

template <typename BaseT>
const TypeInfo<BaseT> TypeInfo<BaseT>::kUnknownType =
    RegisterStaticType<BaseT>("Unknown");

}  // namespace phi
