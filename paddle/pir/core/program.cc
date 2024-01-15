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

#include "paddle/pir/core/program.h"
#include "paddle/pir/core/ir_context.h"

namespace pir {

Program::Program(IrContext* context) {
  module_ = ModuleOp::Create(context, this);
}

Program::~Program() {
  if (module_) {
    module_.Destroy();
  }
}

std::shared_ptr<Program> Program::Clone(IrMapping& ir_mapping) const {
  pir::IrContext* ctx = pir::IrContext::Instance();
  auto new_program = std::make_shared<Program>(ctx);
  auto clone_options = CloneOptions::All();
  for (const auto& op : *block()) {
    auto* new_op = op.Clone(ir_mapping, clone_options);
    new_program->block()->push_back(new_op);
  }
  return new_program;
}

Parameter* Program::GetParameter(const std::string& name) const {
  if (parameters_.count(name) != 0) {
    return parameters_.at(name).get();
  }
  return nullptr;
}

void Program::SetParameter(const std::string& name,
                           std::unique_ptr<Parameter>&& parameter) {
  parameters_[name].reset(parameter.release());
}

}  // namespace pir
