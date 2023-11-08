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
#include "paddle/pir/core/builder.h"
#include "paddle/pir/core/op_base.h"

namespace pir {
class IR_API YieldOp : public Op<YieldOp> {
 public:
  using Op::Op;
  static const char *name() { return "cf.yield"; }
  static constexpr uint32_t attributes_num = 0;
  static constexpr const char **attributes_name = nullptr;

  static void Build(Builder &builder,             // NOLINT
                    OperationArgument &argument,  // NOLINT
                    const std::vector<Value> &Value);
  void VerifySig() {}
};
class PushBackOp;
class PopBackOp;
class IR_API CreateStackOp : public Op<CreateStackOp> {
 public:
  using Op::Op;
  static const char *name() { return "cf.create_stack"; }
  static constexpr uint32_t attributes_num = 0;
  static constexpr const char **attributes_name = nullptr;
  static void Build(Builder &builder,              // NOLINT
                    OperationArgument &argument);  // NOLINT
  void VerifySig();

  Value stack() { return result(0); }
  Value inlet() { return result(1); }
  Value outlet() { return result(2); }
  std::tuple<Value, Value, Value> out() { return {stack(), inlet(), outlet()}; }

  size_t stack_size();
  Value inlet_element(size_t index);
  Value outlet_element(size_t index);
  PushBackOp push_op();
  PopBackOp pop_op();

  void Print(pir::IrPrinter &printer);  // NOLINT
};

class IR_API PushBackOp : public Op<PushBackOp> {
 public:
  using Op::Op;
  static const char *name() { return "cf.push_back"; }
  static constexpr uint32_t attributes_num = 0;
  static constexpr const char **attributes_name = nullptr;

  static void Build(Builder &builder,             // NOLINT
                    OperationArgument &argument,  // NOLINT
                    Value inlet,
                    const std::vector<Value> &elements);
  static void Build(Builder &builder,             // NOLINT
                    OperationArgument &argument,  // NOLINT
                    Value inlet,
                    std::initializer_list<Value> element_list);
  void VerifySig();

  Value stack() { return create_op().stack(); }
  Value inlet() { return operand_source(0); }
  Value outlet() { return create_op().outlet(); }
  size_t stack_size();
  Value inlet_element(size_t index) { return operand_source(index + 1u); }
  Value outlet_element(size_t index) {
    return create_op().outlet_element(index);
  }
  CreateStackOp create_op() { return inlet().defining_op<CreateStackOp>(); }
  PopBackOp pop_op();
};

class IR_API PopBackOp : public Op<PopBackOp> {
 public:
  using Op::Op;
  static const char *name() { return "cf.pop_back"; }
  static constexpr uint32_t attributes_num = 0;
  static constexpr const char **attributes_name = nullptr;

  static void Build(Builder &builder,             // NOLINT
                    OperationArgument &argument,  // NOLINT
                    Value outlet);
  void VerifySig();

  Value stack() { return create_op().stack(); }
  Value inlet() { return create_op().inlet(); }
  Value outlet() { return operand_source(0); }

  size_t stack_size() { return num_results(); }
  Value inlet_element(size_t index) { return push_op().inlet_element(index); }
  Value outlet_element(size_t index) { return result(index); }
  CreateStackOp create_op() { return outlet().defining_op<CreateStackOp>(); }
  PushBackOp push_op() { return create_op().push_op(); }
};

class IR_API HasElementsOp : public Op<HasElementsOp> {
 public:
  using Op::Op;
  static const char *name() { return "cf.has_elements"; }
  static constexpr uint32_t attributes_num = 0;
  static constexpr const char **attributes_name = nullptr;

  static void Build(Builder &builder,             // NOLINT
                    OperationArgument &argument,  // NOLINT
                    Value stack);
  void VerifySig();
  Value out() { return result(0); }
};

}  // namespace pir

IR_EXPORT_DECLARE_EXPLICIT_TYPE_ID(pir::YieldOp);
IR_EXPORT_DECLARE_EXPLICIT_TYPE_ID(pir::CreateStackOp);
IR_EXPORT_DECLARE_EXPLICIT_TYPE_ID(pir::PushBackOp);
IR_EXPORT_DECLARE_EXPLICIT_TYPE_ID(pir::PopBackOp);
IR_EXPORT_DECLARE_EXPLICIT_TYPE_ID(pir::HasElementsOp);
