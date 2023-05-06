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

#include "paddle/ir/program.h"
#include "paddle/ir/ir_context.h"

namespace ir {
void Program::InsertOp(Operation* op) {
  if (op->parent_program() != this) {
    throw("op parent program is not this program");
  }
  ops_.push_back(op);
}
Parameter* Program::GetParameter(std::string name) const {
  ir::IrContext* ctx = ir::IrContext::Instance();
  ir::StrAttribute parameter_name = ir::StrAttribute::get(ctx, name);
  if (parameters_.count(parameter_name) != 0) {
    return parameters_.at(parameter_name).get();
  }
  return nullptr;
}

void Program::SetParameter(std::string name,
                           std::unique_ptr<Parameter> parameter) {
  ir::IrContext* ctx = ir::IrContext::Instance();
  ir::StrAttribute parameter_name = ir::StrAttribute::get(ctx, name);
  parameters_.emplace(parameter_name, parameter);
}

}  // namespace ir
