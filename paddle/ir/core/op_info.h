// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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
#include <functional>
#include <unordered_map>
#include "paddle/ir/core/type_id.h"

namespace ir {
class OpInfoImpl;
class IrContext;
class OpResult;
class Type;
class Attribute;

class OpInfo {
 public:
  constexpr OpInfo() = default;

  OpInfo(const OpInfo &other) = default;

  OpInfo &operator=(const OpInfo &other) = default;

  bool operator==(OpInfo other) const { return impl_ == other.impl_; }

  bool operator!=(OpInfo other) const { return impl_ != other.impl_; }

  explicit operator bool() const { return impl_; }

  bool operator!() const { return impl_ == nullptr; }

  IrContext *ir_context() const;

  const char *name() const;

  TypeId id() const;

  void Verify(const std::vector<OpResult> &inputs,
              const std::vector<Type> &outputs,
              const std::unordered_map<std::string, Attribute> &attributes);

  template <typename Trait>
  bool HasTrait() const {
    return HasTrait(TypeId::get<Trait>());
  }

  bool HasTrait(TypeId trait_id) const;

  template <typename Interface>
  bool HasInterface() const {
    return HasInterface(TypeId::get<Interface>());
  }

  bool HasInterface(TypeId interface_id) const;

  template <typename Interface>
  typename Interface::Concept *GetInterfaceImpl() const;

  void *AsOpaquePointer() const { return impl_; }
  static OpInfo RecoverFromOpaquePointer(void *impl) {
    return static_cast<OpInfoImpl *>(impl);
  }

  friend class OpInfoImpl;
  friend struct std::hash<OpInfo>;

 private:
  OpInfo(OpInfoImpl *impl) : impl_(impl) {}  // NOLINT
  void *GetInterfaceImpl(TypeId interface_id) const;

 private:
  OpInfoImpl *impl_{nullptr};  // not owned
};

template <typename Interface>
typename Interface::Concept *OpInfo::GetInterfaceImpl() const {
  void *model = GetInterfaceImpl(TypeId::get<Interface>());
  return reinterpret_cast<typename Interface::Concept *>(model);
}

}  // namespace ir

namespace std {
template <>
struct hash<ir::OpInfo> {
  std::size_t operator()(const ir::OpInfo &obj) const {
    return std::hash<const ir::OpInfoImpl *>()(obj.impl_);
  }
};
}  // namespace std
