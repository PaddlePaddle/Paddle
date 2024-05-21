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
#include <string>
#include <unordered_map>

#include "paddle/pir/include/core/type_id.h"

namespace pir {
class OpInfoImpl;
class IrContext;
class Type;
class Attribute;
class Dialect;
class Operation;
class InterfaceValue;

typedef void (*VerifyPtr)(Operation *op);

class IR_API OpInfo {
 public:
  OpInfo(std::nullptr_t ptr = nullptr){};  // NOLINT

  OpInfo(const OpInfo &other) = default;

  OpInfo &operator=(const OpInfo &other) = default;

  bool operator==(OpInfo other) const { return impl_ == other.impl_; }

  bool operator!=(OpInfo other) const { return impl_ != other.impl_; }

  explicit operator bool() const { return impl_; }

  bool operator!() const { return impl_ == nullptr; }

  IrContext *ir_context() const;
  Dialect *dialect() const;

  const char *name() const;

  TypeId id() const;

  void Verify(Operation *) const;

  void VerifySig(Operation *) const;

  void VerifyRegion(Operation *) const;

  template <typename Trait>
  bool HasTrait() const {
    return HasTrait(TypeId::get<Trait>());
  }

  bool HasTrait(TypeId trait_id) const;

  template <typename InterfaceT>
  bool HasInterface() const {
    return HasInterface(TypeId::get<InterfaceT>());
  }

  bool HasInterface(TypeId interface_id) const;

  void AttachInterface(InterfaceValue &&interface_value);

  template <typename InterfaceT>
  typename InterfaceT::Concept *GetInterfaceImpl() const;

  operator void *() const { return impl_; }
  static OpInfo RecoverFromVoidPointer(void *pointer) {
    return OpInfo(static_cast<OpInfoImpl *>(pointer));
  }

  std::vector<std::string> GetAttributesName() const;

  friend class OpInfoImpl;

 private:
  explicit OpInfo(OpInfoImpl *impl) : impl_(impl) {}
  void *GetInterfaceImpl(TypeId interface_id) const;

 private:
  /// The internal implementation of the operation name.
  /// Not owned.
  OpInfoImpl *impl_{nullptr};
};

///
/// \brief Returns an instance of the concept object for the given interface if
/// it was registered to this operation, null otherwise.
///
template <typename InterfaceT>
typename InterfaceT::Concept *OpInfo::GetInterfaceImpl() const {
  void *model = GetInterfaceImpl(TypeId::get<InterfaceT>());
  return reinterpret_cast<typename InterfaceT::Concept *>(model);
}

}  // namespace pir

namespace std {
template <>
struct hash<pir::OpInfo> {
  std::size_t operator()(const pir::OpInfo &obj) const {
    return std::hash<void *>()(obj);
  }
};
}  // namespace std
