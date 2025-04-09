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
#include "paddle/common/enforce.h"
#include "paddle/pir/include/core/ir_context.h"

namespace paddle::dialect {

ApiBuilder::ApiBuilder()
    : ctx_(pir::IrContext::Instance()),
      builder_(std::make_shared<pir::Builder>(ctx_)) {
  PADDLE_ENFORCE_NE(
      builder_,
      nullptr,
      common::errors::InvalidArgument("api builder construct error!"));
}

void ApiBuilder::SetProgram(pir::Program* program) {
  PADDLE_ENFORCE_NE(
      program,
      nullptr,
      common::errors::InvalidArgument("argument of program is nullptr"));
  builder_->SetInsertionPointToBlockEnd(program->block());
}

void ApiBuilder::ResetInsertionPointToStart() {
  builder_->SetInsertionPointToStart(builder_->block());
}

void ApiBuilder::ResetInsertionPointToEnd() {
  builder_->SetInsertionPointToBlockEnd(builder_->block());
}

pir::Parameter* ApiBuilder::GetParameter(const std::string& name) const {
  pir::Program* program = builder_->block()->GetParentOp()->GetParentProgram();
  return program->GetParameter(name);
}

void ApiBuilder::SetParameter(const std::string& name,
                              std::unique_ptr<pir::Parameter>&& parameter) {
  pir::Program* program = builder_->block()->GetParentOp()->GetParentProgram();
  program->SetParameter(name, std::move(parameter));
}

void ApiBuilder::LoadInsertionPoint() {
  PADDLE_ENFORCE_EQ(
      !insertion_point_stack_.empty(),
      true,
      common::errors::InvalidArgument("insertion_point_stack_ is empty."));
  builder_->set_insertion_point(insertion_point_stack_.top());
  insertion_point_stack_.pop();
}

}  // namespace paddle::dialect
