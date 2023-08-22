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

#include <string>

namespace phi {

template <typename BaseT>
class TypeRegistry;

template <typename BaseT>
class TypeInfo {
 public:
  const std::string& name() const;

  int8_t id() const { return id_; }

  bool operator==(TypeInfo other) const { return id_ == other.id(); }
  bool operator!=(TypeInfo other) const { return id_ != other.id(); }

  static const TypeInfo kUnknownType;

 private:
  friend class TypeRegistry<BaseT>;
  explicit TypeInfo(int8_t id) : id_(id) {}
  int8_t id_;
};

template <typename BaseT, typename DerivedT>
class TypeInfoTraits {
 public:
  static const TypeInfo<BaseT> kType;
  TypeInfoTraits();
  static bool classof(const BaseT* obj);
};

template <typename BaseT>
TypeInfo<BaseT> RegisterStaticType(const std::string& type);

}  // namespace phi
