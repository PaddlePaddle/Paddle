// Copyright (c) 2023 CINN Authors. All Rights Reserved.
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

#include "test/cpp/cinn/program_builder.h"

namespace cinn {
namespace tests {

OpBuilder::OpBuilder(const std::string& op_name)
    : ProgramBuilder(op_name), op_name_(op_name) {}

frontend::Program OpBuilder::Build(
    const std::vector<VariableInfo>& inputs_varinfo,
    const utils::AttributeMap& attrs) {
  std::vector<frontend::Variable> inputs;
  for (auto&& item : inputs_varinfo) {
    inputs.emplace_back(builder_.CreateInput(item.type, item.shape, item.id));
  }
  outputs_ = builder_.CustomInstr(op_name_, inputs, attrs);
  return builder_.Build();
}

PaddleModelBuilder::PaddleModelBuilder(const std::string& model_path,
                                       const common::Target& target)
    : ProgramBuilder("test_paddle_model"),
      model_path_(model_path),
      target_(target) {}

frontend::Program PaddleModelBuilder::Build(
    const std::vector<VariableInfo>& inputs_varinfo,
    const utils::AttributeMap& attrs) {
  // build a name to shape map of input
  CHECK(!inputs_varinfo.empty());
  auto scope = std::make_shared<hlir::framework::Scope>();
  std::unordered_map<std::string, std::vector<int>> input_name2shape;
  for (auto&& item : inputs_varinfo) {
    input_name2shape[item.id] = item.shape;
  }

  auto loadedProgram = cinn::frontend::LoadPaddleProgram(
      model_path_, scope.get(), input_name2shape, true, target_);
  auto& program = std::get<0>(loadedProgram);
  auto& varmap = std::get<1>(loadedProgram);
  VLOG(3) << "loaded program: " << *program;
  CHECK(!varmap.empty());

  // fetch input variables and set to program
  std::vector<frontend::Variable> input_vars;
  for (auto&& item : inputs_varinfo) {
    input_vars.emplace_back(varmap.at(item.id));
    input_vars.back()->shape = item.shape;
  }

  program->SetInputs(input_vars);
  program->Validate();
  return *program;
}

}  // namespace tests
}  // namespace cinn
