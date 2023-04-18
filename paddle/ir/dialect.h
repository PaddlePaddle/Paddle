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

#include "paddle/ir/attribute_base.h"
#include "paddle/ir/ir_context.h"
#include "paddle/ir/op_info_impl.h"
#include "paddle/ir/type_base.h"

namespace ir {
///
/// \brief Dialect can basically be understood as a namespace. In Dialect, we
/// can define a series of types, attributes, operations, etc. An instance of
/// the dialect object will be loaded into the global IrContext. Specific
/// compilers only need to combine existing dialects and add their own
/// extensions or customizations.
///
class Dialect {
 public:
  Dialect(std::string name, ir::IrContext *context, ir::TypeId id);

  const std::string &name() const { return name_; }

  ir::IrContext *ir_context() const { return context_; }

  ir::TypeId id() const { return id_; }

  ///
  /// \brief Register all types contained in the template parameter Args.
  /// To register only one Type, you can use the RegisterType template function.
  ///
  template <typename... Args>
  void RegisterTypes() {
    (void)std::initializer_list<int>{0, (RegisterType<Args>(), 0)...};
  }

  ///
  /// \brief Register type of class T.
  ///
  template <typename T>
  void RegisterType() {
    VLOG(4) << "Type registered into Dialect. --->";
    ir::AbstractType *abstract_type =
        new ir::AbstractType(std::move(ir::AbstractType::get<T>(*this)));
    this->ir_context()->RegisterAbstractType(ir::TypeId::get<T>(),
                                             abstract_type);
    ir::TypeManager::RegisterType<T>(this->ir_context());
    VLOG(4) << "----------------------------------";
  }

  ///
  /// \brief Register abstract_type into context.
  /// NOTE: It's not recommended to use this interface directly. This interface
  /// only registers abstract_type. To register TypeStorage into context, you
  /// need to call ir::TypeManager::RegisterType<T>() additionally,
  /// RegisterType<T>() is recommended to use.
  ///
  void RegisterType(ir::AbstractType &&abstract_type);

  ///
  /// \brief Register all attributes contained in the template parameter Args.
  /// To register only one Attribute, you can use the RegisterAttribute template
  /// function.
  ///
  template <typename... Args>
  void RegisterAttributes() {
    (void)std::initializer_list<int>{0, (RegisterAttribute<Args>(), 0)...};
  }

  ///
  /// \brief Register attribute of class T.
  ///
  template <typename T>
  void RegisterAttribute() {
    VLOG(4) << "Attribute registered into Dialect. --->";
    ir::AbstractAttribute *abstract_attribute = new ir::AbstractAttribute(
        std::move(ir::AbstractAttribute::get<T>(*this)));
    this->ir_context()->RegisterAbstractAttribute(ir::TypeId::get<T>(),
                                                  abstract_attribute);
    ir::AttributeManager::RegisterAttribute<T>(this->ir_context());
    VLOG(4) << "----------------------------------";
  }

  ///
  /// \brief Register abstract_attribute into context.
  ///
  void RegisterAttribute(ir::AbstractAttribute &&abstract_attribute);

  ///
  /// \brief Register all Operation.
  ///
  template <typename... Args>
  void RegisterOperations() {
    (void)std::initializer_list<int>{0, (RegisterOperation<Args>(), 0)...};
  }

  ///
  /// \brief Register Operation of class T.
  ///
  template <typename ConcertOp>
  void RegisterOperation() {
    VLOG(4) << "Operation registered into Dialect. --->";
    ir::OpInfoImpl *op_info = ir::OpInfoImpl::create<ConcertOp>();
    this->ir_context()->RegisterOperation(ir::TypeId::get<ConcertOp>(),
                                          op_info);
    VLOG(4) << "----------------------------------";
  }

  ///
  /// \brief Register operation into context.
  ///
  void RegisterOperation(ir::TypeId id, OpInfoImpl *op_info);

 private:
  std::string name_;

  ir::IrContext *context_;  // not owned

  ir::TypeId id_;
};
}  // namespace ir
