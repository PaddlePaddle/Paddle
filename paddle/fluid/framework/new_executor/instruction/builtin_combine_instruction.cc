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

#include "paddle/fluid/framework/new_executor/instruction/builtin_combine_instruction.h"
#include "paddle/fluid/framework/new_executor/instruction/instruction_util.h"
#include "paddle/fluid/framework/new_executor/new_executor_defs.h"

namespace paddle::framework {

BuiltinCombineInstruction::BuiltinCombineInstruction(
    size_t id,
    const phi::Place& place,
    ::pir::Operation* op,
    ValueExecutionInfo* value_exe_info)
    : InstructionBase(id, place) {
  op_ = op;

  SetKernelType(AnalyseOpFuncType(op, place));

  InitInputsOutputsIds(op, *value_exe_info);

  SetArtificial(true);
}

void BuiltinCombineInstruction::Run() {}

}  // namespace paddle::framework
