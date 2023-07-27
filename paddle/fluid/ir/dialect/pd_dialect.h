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

#include "paddle/fluid/framework/variable.h"
#include "paddle/ir/core/dialect.h"
#include "paddle/ir/core/enforce.h"
#include "paddle/ir/core/parameter.h"
#include "paddle/ir/core/program.h"

namespace paddle {
namespace dialect {
class ParameterConvertInterface
    : public ir::DialectInterface::Base<ParameterConvertInterface> {
 public:
  explicit ParameterConvertInterface(ir::Dialect* dialect) : Base(dialect) {}

  // NOTE(zhangbo): Only support new a CPU Variable.
  std::shared_ptr<paddle::framework::Variable> ParameterToVariable(
      ir::Parameter* parameter);

  std::unique_ptr<ir::Parameter> VariableToParameter(
      paddle::framework::Variable* var);
};

class PaddleDialect : public ir::Dialect {
 public:
  explicit PaddleDialect(ir::IrContext* context);

  static const char* name() { return "pd"; }

  void PrintType(ir::Type type, std::ostream& os) const;
  void PrintAttribute(ir::Attribute type, std::ostream& os) const;

 private:
  void initialize();
};

///
/// \brief APIBuilder is used in IR API for building op
///
class APIBuilder {
 public:
  static APIBuilder& Instance() {
    static APIBuilder api_builder;
    return api_builder;
  }
  void SetProgram(ir::Program* program) {
    builder_ = std::make_shared<ir::Builder>(ctx_, program->block());
  }

  /// Set the insertion point to the specified operation, which will cause
  /// subsequent insertions to go right before it.
  void SetInsertionPoint(ir::Operation* op) {
    IR_ENFORCE(builder_ != nullptr,
               "builder doesn't hold program, please call SetProgram for "
               "initialization.");
    builder_->SetInsertionPoint(op);
  }

  void ResetInsertionPointToStart() {
    IR_ENFORCE(builder_ != nullptr,
               "builder doesn't hold program, please call SetProgram for "
               "initialization.");
    builder_->SetInsertionPointToStart(builder_->block());
  }

  void ResetInsertionPointToEnd() {
    IR_ENFORCE(builder_ != nullptr,
               "builder doesn't hold program, please call SetProgram for "
               "initialization.");
    builder_->SetInsertionPointToEnd(builder_->block());
  }

  std::shared_ptr<ir::Builder> GetBuilder() { return builder_; }

 private:
  APIBuilder() : builder_(nullptr) {
    ctx_ = ir::IrContext::Instance();
    ctx_->GetOrRegisterDialect<paddle::dialect::PaddleDialect>();
  }
  ir::IrContext* ctx_;
  std::shared_ptr<ir::Builder> builder_;
};

}  // namespace dialect
}  // namespace paddle

IR_DECLARE_EXPLICIT_TYPE_ID(paddle::dialect::PaddleDialect)
IR_DECLARE_EXPLICIT_TYPE_ID(paddle::dialect::ParameterConvertInterface)
