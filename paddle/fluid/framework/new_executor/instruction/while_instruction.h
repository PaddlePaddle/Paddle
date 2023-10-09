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
class NewIRInterpreter;
class ValueExecutionInfo;

class WhileInstruction : public InstructionBase {
 public:
  WhileInstruction(
      size_t id,
      const platform::Place& place,
      ::pir::Operation* op,
      Scope* scope,
      Scope* local_scope,
      ValueExecutionInfo* parent_exe_info,
      const std::map<pir::Block*, paddle::framework::Scope*>& sub_blocks);

  void Run() override;

  const std::string& Name() const override { return cond_name_; }

  ::pir::Operation* Operation() const override { return op_; }

 private:
  void CopyStepOutput();

  void CopyWhileInputToBlockArgs(const NewIRInterpreter* inter,
                                 ::pir::Block* block);

  void CopyStepOutputToBlockArgs(const NewIRInterpreter* inter,
                                 ::pir::Block* block);

  std::string cond_name_{"while_instruction"};

  Variable* cond_var;

  std::vector<Variable*> while_op_inputs_;
  std::vector<Variable*> while_op_outputs_;

  NewIRInterpreter* cond_inter_;
  NewIRInterpreter* body_inter_;

  std::vector<std::string> cond_skip_gc_names_;
  std::vector<std::string> body_skip_gc_names_;

  ::pir::Block* cond_block_;
  ::pir::Block* body_block_;

  ::pir::Operation* op_;
};

}  // namespace framework
}  // namespace paddle
