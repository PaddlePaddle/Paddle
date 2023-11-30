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

#include <string>
#include "paddle/fluid/framework/new_executor/instruction/instruction_base.h"
#include "paddle/fluid/framework/new_executor/new_executor_defs.h"
#include "paddle/fluid/framework/tensor_ref_array.h"
#include "paddle/pir/dialect/control_flow/ir/cf_op.h"

namespace pir {
class TuplePushOp;
}  // namespace pir

namespace paddle {
namespace framework {
class ValueExecutionInfo;

class TuplePushInstruction : public InstructionBase {
 public:
  TuplePushInstruction(size_t id,
                       const platform::Place& place,
                       ::pir::Operation* op,
                       ValueExecutionInfo* value_exe_info);

  void Run() override;

  const std::string& Name() const override { return name_; }

  ::pir::Operation* Operation() const override { return op_; }

 private:
  ::pir::Operation* op_;

  ::pir::TuplePushOp tuple_push_op_;

  OpFuncType type_;

  std::string name_{"tuple_push_instruction"};

  VariableRefArray* stack_element_var_array_;  // not owned

  ValueExecutionInfo* value_exe_info_;
};

}  // namespace framework
}  // namespace paddle
