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

#include "paddle/fluid/pir/dialect/operator/ir/api_builder.h"
#include "paddle/pir/core/enforce.h"
#include "paddle/pir/core/ir_context.h"

namespace paddle {
namespace dialect {

APIBuilder::APIBuilder() : builder_(nullptr) {
  ctx_ = pir::IrContext::Instance();
}

void APIBuilder::SetProgram(pir::Program* program) {
  builder_ = std::make_shared<pir::Builder>(ctx_, program->block());
}

void APIBuilder::SetInsertionPoint(pir::Operation* op) {
  IR_ENFORCE(builder_ != nullptr,
             "builder doesn't hold program, please call SetProgram for "
             "initialization.");
  builder_->SetInsertionPoint(op);
}

void APIBuilder::ResetInsertionPointToStart() {
  IR_ENFORCE(builder_ != nullptr,
             "builder doesn't hold program, please call SetProgram for "
             "initialization.");
  builder_->SetInsertionPointToStart(builder_->block());
}

void APIBuilder::ResetInsertionPointToEnd() {
  IR_ENFORCE(builder_ != nullptr,
             "builder doesn't hold program, please call SetProgram for "
             "initialization.");
  builder_->SetInsertionPointToEnd(builder_->block());
}

pir::Parameter* APIBuilder::GetParameter(const std::string& name) const {
  pir::Program* program = builder_->block()->GetParentOp()->GetParentProgram();
  return program->GetParameter(name);
}

void APIBuilder::SetParameter(const std::string& name,
                              std::unique_ptr<pir::Parameter>&& parameter) {
  pir::Program* program = builder_->block()->GetParentOp()->GetParentProgram();
  program->SetParameter(name, std::move(parameter));
}

}  // namespace dialect
}  // namespace paddle
