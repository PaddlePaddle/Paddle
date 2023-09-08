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
#include <ostream>

#include "paddle/ir/core/attribute.h"
#include "paddle/ir/core/attribute_base.h"
#include "paddle/ir/core/dialect_interface.h"
#include "paddle/ir/core/enforce.h"
#include "paddle/ir/core/ir_context.h"
#include "paddle/ir/core/op_base.h"
#include "paddle/ir/core/type_base.h"

namespace ir {

class Operation;
class IrPrinter;

class DialectInterface;
///
/// \brief Dialect can basically be understood as a namespace. In Dialect, we
/// can define a series of types, attributes, operations, etc. An instance of
/// the dialect object will be loaded into the global IrContext. Specific
/// compilers only need to combine existing dialects and add their own
/// extensions or customizations.
///
class IR_API Dialect {
 public:
  Dialect(std::string name, IrContext *context, TypeId id);

  virtual ~Dialect();

  const std::string &name() const { return name_; }

  IrContext *ir_context() const { return context_; }

  TypeId id() const { return id_; }

  ///
  /// \brief Register all types contained in the template parameter Args.
  /// To register only one Type, you can use the RegisterType template function.
  ///
  template <typename... Args>
  void RegisterTypes() {
    (void)std::initializer_list<int>{0, (RegisterType<Args>(), 0)...};
  }

  template <typename T>
  void RegisterType() {
    ir_context()->RegisterAbstractType(TypeId::get<T>(),
                                       AbstractType::get<T>(*this));
    TypeManager::RegisterType<T>(ir_context());
  }

  ///
  /// \brief Register all attributes contained in the template parameter Args.
  /// To register only one Attribute, you can use the RegisterAttribute template
  /// function.
  ///
  template <typename... Args>
  void RegisterAttributes() {
    (void)std::initializer_list<int>{0, (RegisterAttribute<Args>(), 0)...};
  }

  template <typename T>
  void RegisterAttribute() {
    ir_context()->RegisterAbstractAttribute(TypeId::get<T>(),
                                            AbstractAttribute::get<T>(*this));
    AttributeManager::RegisterAttribute<T>(ir_context());
  }

  ///
  /// \brief Register Ops.
  ///
  template <typename... Args>
  void RegisterOps() {
    (void)std::initializer_list<int>{0, (RegisterOp<Args>(), 0)...};
  }

  template <typename ConcreteOp>
  void RegisterOp() {
    ir_context()->RegisterOpInfo(this,
                                 TypeId::get<ConcreteOp>(),
                                 ConcreteOp::name(),
                                 ConcreteOp::GetInterfaceMap(),
                                 ConcreteOp::GetTraitSet(),
                                 ConcreteOp::attributes_num,
                                 ConcreteOp::attributes_name,
                                 ConcreteOp::VerifyInvariants);
  }

  void RegisterOp(const std::string &name, OpInfoImpl *op_info);

  ///
  /// \brief Register interface methods.
  ///

  DialectInterface *GetRegisteredInterface(TypeId id) {
    auto it = registered_interfaces_.find(id);
    return it != registered_interfaces_.end() ? it->second.get() : nullptr;
  }

  template <typename InterfaceT>
  InterfaceT *GetRegisteredInterface() {
    return static_cast<InterfaceT *>(
        GetRegisteredInterface(TypeId::get<InterfaceT>()));
  }

  /// Register a dialect interface with this dialect instance.
  void RegisterInterface(std::unique_ptr<DialectInterface> interface);

  /// Register a set of dialect interfaces with this dialect instance.
  template <typename... Args>
  void RegisterInterfaces() {
    (void)std::initializer_list<int>{
        0, (RegisterInterface(std::make_unique<Args>(this)), 0)...};
  }

  template <typename InterfaceT, typename... Args>
  InterfaceT &RegisterInterface(Args &&...args) {
    InterfaceT *interface = new InterfaceT(this, std::forward<Args>(args)...);
    RegisterInterface(std::unique_ptr<DialectInterface>(interface));
    return *interface;
  }

  virtual void PrintType(Type type, std::ostream &os) const {
    IR_THROW("dialect has no registered type printing hook");
  }

  virtual void PrintAttribute(Attribute attr, std::ostream &os) const {
    IR_THROW("dialect has no registered attribute printing hook");
  }

  virtual void PrintOperation(Operation *op,
                              IrPrinter &printer) const;  // NOLINT

 private:
  Dialect(const Dialect &) = delete;

  Dialect &operator=(Dialect &) = delete;

  std::string name_;

  IrContext *context_;  // not owned

  TypeId id_;

  std::unordered_map<TypeId, std::unique_ptr<DialectInterface>>
      registered_interfaces_;
};
}  // namespace ir
