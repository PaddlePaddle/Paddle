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

#include "paddle/fluid/framework/new_executor/instruction/instruction_base.h"

namespace ir {
class Operation;
}  // namespace ir

namespace paddle {
namespace framework {
class Scope;
class Value;
class InterpreterBaseImpl;

class CondInstruction : public InstructionBase {
 public:
  CondInstruction(
      size_t id,
      const platform::Place& place,
      ::ir::Operation* op,
      Scope* scope,
      Scope* local_scope,
      const std::unordered_map<::ir::Value, std::string>& value_2_var_name,
      const std::map<std::string, int>& var_name_2_id,
      const std::unordered_map<const paddle::framework::Variable*, std::string>&
          variable_2_var_name);

  void Run() override;

  const std::string& Name() const override { return cond_name_; }

 private:
  std::string cond_name_{"cond_instruction"};

  Variable* cond_var;

  InterpreterBaseImpl* true_branch_inter;
  InterpreterBaseImpl* false_branch_inter;
};

}  // namespace framework
}  // namespace paddle
