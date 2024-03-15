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
#include "paddle/fluid/framework/new_executor/interpreter/execution_config.h"

namespace ir {
class Operation;
}  // namespace ir

namespace paddle {
namespace framework {
class Scope;
class Value;
class PirInterpreter;
class ValueExecutionInfo;

class IfInstruction : public InstructionBase {
 public:
  IfInstruction(size_t id,
                const platform::Place& place,
                ::pir::Operation* op,
                ValueExecutionInfo* value_exe_info,
                interpreter::ExecutionConfig execution_config);

  ~IfInstruction();

  void Run() override;

  const std::string& Name() const override { return cond_name_; }

  ::pir::Operation* Operation() const override { return op_; }

  PirInterpreter* TrueBranchInterpreter() const { return true_branch_inter_; }

  PirInterpreter* FalseBranchInterpreter() const { return false_branch_inter_; }

 private:
  ::pir::Operation* op_;

  std::string cond_name_{"if_instruction"};

  Variable* cond_var_;

  std::vector<Variable*> output_vars_;

  PirInterpreter* true_branch_inter_ = nullptr;

  PirInterpreter* false_branch_inter_ = nullptr;

  std::vector<std::string> true_branch_outputs_;

  std::vector<std::string> false_branch_outputs_;

  // TODO(zhangbo): Currently, only the output of IfOp is included. In the
  // future, need to consider how to support IfGradOp using IfOp value.
  std::vector<std::string> true_skip_gc_names_;

  std::vector<std::string> false_skip_gc_names_;
};

}  // namespace framework
}  // namespace paddle
