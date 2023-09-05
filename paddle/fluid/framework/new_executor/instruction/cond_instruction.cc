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

#include "paddle/fluid/framework/new_executor/instruction/cond_instruction.h"

#include "paddle/fluid/framework/new_executor/interpreter/interpreter_util.h"
#include "paddle/fluid/framework/new_executor/interpreter/stream_analyzer.h"
#include "paddle/fluid/framework/new_executor/new_ir_interpreter.h"
#include "paddle/fluid/framework/scope.h"
#include "paddle/fluid/ir/dialect/paddle_dialect/interface/infermeta.h"
#include "paddle/fluid/ir/dialect/paddle_dialect/interface/op_yaml_info.h"
#include "paddle/fluid/ir/dialect/paddle_dialect/ir/pd_dialect.h"
#include "paddle/fluid/ir/dialect/paddle_dialect/utils/op_yaml_info_parser.h"
#include "paddle/fluid/ir/phi_kernel_adaptor/phi_kernel_util.h"
#include "paddle/fluid/platform/collective_helper.h"
#include "paddle/fluid/platform/device_context.h"
#include "paddle/phi/core/infermeta_utils.h"
#include "paddle/phi/core/meta_tensor.h"
#include "paddle/phi/core/type_defs.h"

#include "paddle/ir/core/builtin_attribute.h"
#include "paddle/ir/core/operation.h"
#include "paddle/ir/core/value.h"

#include "paddle/fluid/framework/new_executor/instruction/instruction_util.h"
#include "paddle/fluid/ir/dialect/paddle_dialect/ir/pd_manual_op.h"
namespace paddle {
namespace framework {

CondInstruction::CondInstruction(
    size_t id,
    const platform::Place& place,
    ir::Operation* op,
    Scope* scope,
    Scope* local_scope,
    const std::unordered_map<::ir::Value, std::string>& value_2_var_name,
    const std::map<std::string, int>& var_name_2_id,
    const std::unordered_map<const paddle::framework::Variable*, std::string>&
        variable_2_var_name)
    : InstructionBase(id, place) {
  // Todo: support paddle::dialect::DistAttribute
  //   if (op_attributes.count("dist_attr") != 0) {
  //     if (op_attributes.count("execution_stream") != 0) {
  //         SetExecutionStream(op_attributes.at("execution_stream")
  //                             .dyn_cast<::ir::StrAttribute>()
  //                             .data());
  //     }
  //     if (op_attributes.count("stream_priority") != 0) {
  //         SetStreamPriority(op_attributes.at("stream_priority")
  //                             .dyn_cast<::ir::Int32Attribute>()
  //                             .data());
  //     }
  //     if (op_attributes.count("scheduling_priority") != 0) {
  //         SetSchedulingPriority(op_attributes.at("scheduling_priority")
  //                                 .dyn_cast<::ir::Int64Attribute>()
  //                                 .data());
  //     }
  //   } else {
  //     if (interpreter::IsCommunicationOp(op)) {
  //       // NOTE(Ruibiao): Dispatching computation before communication
  //       improves
  //       // multi-stream overlap when the time cost of communication less than
  //       // that of the calculation (e.g., ResNet50_bs128_pure_fp16 N4C32
  //       // training).
  //       op_func_node.scheduling_priority_ = 1;
  //     }
  //   }
  VLOG(6) << "finish process dist attributes";

  SetKernelType(AnalyseOpFuncType(op, place));
  VLOG(6) << "finish process analyse kernel type";

  // TODO(phlrain): is nupptr ok?
  SetDeviceContext(ParseDeviceContext(
      op, nullptr, place, GetExecutionStream(), GetStreamPriority()));
  VLOG(6) << "finish process device context";

  Scope* inner_scope = local_scope == nullptr ? scope : local_scope;
  // InitInputsOutputsIds(
  //     op, inner_scope, value_2_var_name, var_name_2_id, variable_2_var_name);
  VLOG(6) << "finish process inputs outputs index";

  //   auto& no_need_buffer_ids = yaml_info_parser.NoNeedBufferIds();
  //   std::unordered_set<::ir::Value> no_need_buffer_values;
  //   for (size_t id = 0; id < no_need_buffer_ids.size(); id++) {
  //     no_need_buffer_values.insert(op->operand_source(no_need_buffer_ids[id]));
  //   }
  //   SetNoNeedBuffer(no_need_buffer_values);
  //   VLOG(6) << "finish process no need buffer";

  if (op->isa<paddle::dialect::IfOp>()) {
    auto if_op = op->dyn_cast<paddle::dialect::IfOp>();

    auto true_branch_block = if_op.true_block();
    auto false_branch_block = if_op.false_block();

    true_branch_inter =
        new NewIRInterpreter(place, {}, true_branch_block, inner_scope, {});

    false_branch_inter =
        new NewIRInterpreter(place, {}, false_branch_block, inner_scope, {});
  }
}

void CondInstruction::Run() {
  if (cond_var->Get<phi::DenseTensor>().data<bool>()[0]) {
    true_branch_inter->Run({}, false);
  } else {
    false_branch_inter->Run({}, false);
  }
}

}  // namespace framework
}  // namespace paddle
