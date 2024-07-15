// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

namespace paddle {
namespace framework {
class Scope;
class Value;
class PirInterpreter;
class ValueExecutionInfo;

class PyLayerInstruction : public InstructionBase {
 public:
  PyLayerInstruction(size_t id,
                     const phi::Place& place,
                     ::pir::Operation* op,
                     ValueExecutionInfo* value_exe_info,
                     interpreter::ExecutionConfig execution_config);

  ~PyLayerInstruction();

  void Run() override;

  const std::string& Name() const override { return name_; }

  ::pir::Operation* Operation() const override { return op_; }

  PirInterpreter* ForwardInterpreter() const { return fwd_inter_; }

 private:
  ::pir::Operation* op_;

  std::string name_{"pylayer_instruction"};

  std::vector<Variable*> output_vars_;

  PirInterpreter* fwd_inter_ = nullptr;

  std::vector<std::string> fwd_skip_gc_names_;
};

}  // namespace framework
}  // namespace paddle
