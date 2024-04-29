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

#include "paddle/pir/include/core/type_id.h"

namespace pir {
class Dialect;
class IrContext;
///
/// \brief DialectInterface
///
template <typename ConcreteType, typename BaseT>
class DialectInterfaceBase : public BaseT {
 public:
  using Base = DialectInterfaceBase<ConcreteType, BaseT>;

  /// Get a unique id for the derived interface type.
  static TypeId id() { return TypeId::get<ConcreteType>(); }

 protected:
  explicit DialectInterfaceBase(Dialect *dialect) : BaseT(dialect, id()) {}
};

class IR_API DialectInterface {
 public:
  virtual ~DialectInterface();

  /// The base class used for all derived interface types. This class provides
  /// utilities necessary for registration.
  template <typename ConcreteType>
  using Base = DialectInterfaceBase<ConcreteType, DialectInterface>;

  /// Return the dialect that this interface represents.
  Dialect *dialect() const { return dialect_; }

  /// Return the context that holds the parent dialect of this interface.
  IrContext *ir_context() const;

  /// Return the derived interface id.
  TypeId interface_id() const { return interface_id_; }

 protected:
  DialectInterface(Dialect *dialect, TypeId id)
      : dialect_(dialect), interface_id_(id) {}

 private:
  /// The dialect that represents this interface.
  Dialect *dialect_;

  /// The unique identifier for the derived interface type.
  TypeId interface_id_;
};

}  // namespace pir
