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

#include "paddle/fluid/framework/new_executor/instruction/custom_engine_instruction.h"
#include "paddle/fluid/custom_engine/custom_engine_ext.h"
#include "paddle/fluid/custom_engine/custom_engine_manager.h"
#include "paddle/fluid/framework/new_executor/instruction/instruction_util.h"
#include "paddle/fluid/pir/dialect/operator/ir/op_dialect.h"

namespace paddle {
namespace framework {

CustomEngineInstruction::CustomEngineInstruction(
    size_t id,
    const phi::Place &place,
    ::pir::Operation *op,
    const ValueExecutionInfo *value_exec_info)
    : InstructionBase(id, place), value_exec_info_(value_exec_info) {
  PADDLE_ENFORCE_EQ(paddle::dialect::IsCustomEngineOp(op),
                    true,
                    ::common::errors::InvalidArgument(
                        "The Op to construct CustomEngineInstruction must be a "
                        "custom engine op.  "
                        "but got op is %d",
                        op->name()));

  auto op_attributes = op->attributes();
  op_ = op;
  VLOG(6) << "Start Build custom engine";
  interface_ = paddle::custom_engine::CustomEngineManager::Instance()
                   ->GetCustomEngineInterface();
  if (interface_ && interface_->graph_engine_build) {
    interface_->graph_engine_build(
        reinterpret_cast<C_CustomEngineInstruction>(this));
  } else {
    PADDLE_THROW(common::errors::Unimplemented(
        "CustomEngineInstruction's C_CustomEngineInterface->graph_engine_build "
        "not implemented"));
  }

  PADDLE_ENFORCE_NOT_NULL(
      custom_engine_,
      ::common::errors::InvalidArgument(
          "custom_engine_ should not be nullptr after graph_engine_build"));

  VLOG(6) << "Finish build engine for: " << op_name_;

  SetDeviceContext(
      ParseDeviceContext(op,
                         phi::DeviceContextPool::Instance().Get(place),
                         place,
                         GetExecutionStream(),
                         GetStreamPriority()));
  VLOG(6) << "finish process device context";

  InitInputsOutputsIds(op, *value_exec_info_);
  VLOG(6) << "finish process inputs outputs index";
}

void CustomEngineInstruction::Run() {
  if (interface_ && interface_->graph_engine_execute) {
    interface_->graph_engine_execute(
        reinterpret_cast<C_CustomEngineInstruction>(this));
  } else {
    PADDLE_THROW(common::errors::Unimplemented(
        "CustomEngineInstruction's C_CustomEngineInterface->graph_engine_run "
        "not implemented"));
  }
}

}  // namespace framework
}  // namespace paddle
