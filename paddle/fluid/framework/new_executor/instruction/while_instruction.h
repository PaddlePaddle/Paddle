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
class PirInterpreter;
class ValueExecutionInfo;

/// The execute semantics of while op ['output' = while_op('cond', 'intput')]
/// is:
///   'output' = 'input';
///   while('cond') {
///      'cond', 'output' = body_block('output');
///  }
class WhileInstruction : public InstructionBase {
 public:
  WhileInstruction(size_t id,
                   const platform::Place& place,
                   ::pir::Operation* op,
                   ValueExecutionInfo* parent_exe_info,
                   const std::set<std::string>& skip_gc_vars);

  void Run() override;

  const std::string& Name() const override { return name_; }

  ::pir::Operation* Operation() const override { return op_; }

  PirInterpreter* BodyInterpreter() const { return body_inter_.get(); }

 private:
  // 'output' = 'input'
  void ShareInputsToOutputs();

  // Pass argument to body_block for execution.
  void CopyOutputsToBlockArgs();

  // Get return value from body_block after each execution.
  void ShareDatasToOutputs();

  std::string name_{"while_instruction"};

  Variable* cond_var_;

  std::vector<Variable*> inputs_;
  std::vector<Variable*> outputs_;

  std::unique_ptr<PirInterpreter> body_inter_;
  std::vector<std::string> body_outputs_;
  std::vector<std::string> body_skip_gc_names_;

  ::pir::Block* body_block_;

  ::pir::Operation* op_;
};

}  // namespace framework
}  // namespace paddle
