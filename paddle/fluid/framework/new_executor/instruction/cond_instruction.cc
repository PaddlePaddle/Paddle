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
#include "paddle/fluid/pir/dialect/operator/interface/infermeta.h"
#include "paddle/fluid/pir/dialect/operator/interface/op_yaml_info.h"
#include "paddle/fluid/pir/dialect/operator/ir/op_dialect.h"
#include "paddle/fluid/pir/dialect/operator/utils/op_yaml_info_parser.h"
#include "paddle/fluid/pir/phi_kernel_adaptor/phi_kernel_util.h"
#include "paddle/fluid/platform/collective_helper.h"
#include "paddle/fluid/platform/device_context.h"
#include "paddle/phi/core/infermeta_utils.h"
#include "paddle/phi/core/meta_tensor.h"
#include "paddle/phi/core/type_defs.h"

#include "paddle/pir/core/builtin_attribute.h"
#include "paddle/pir/core/operation.h"
#include "paddle/pir/core/value.h"

#include "paddle/fluid/framework/new_executor/instruction/instruction_util.h"
#include "paddle/fluid/pir/dialect/operator/ir/manual_op.h"
namespace paddle {
namespace framework {

std::vector<pir::Value> GetYiedOpInputs(pir::Block* block) {
  std::vector<pir::Value> vec_res;
  for (auto op : (*block)) {
    if (op->name() == "cf.yield") {
      for (size_t i = 0; i < op->num_operands(); ++i) {
        vec_res.push_back(op->operand_source(i));
      }
    }
  }

  return vec_res;
}

CondInstruction::CondInstruction(
    size_t id,
    const platform::Place& place,
    pir::Operation* op,
    Scope* scope,
    Scope* local_scope,
    const std::unordered_map<::pir::Value, std::string>& value_2_var_name,
    const std::map<std::string, int>& var_name_2_id,
    const std::unordered_map<const paddle::framework::Variable*, std::string>&
        variable_2_var_name,
    const std::map<pir::Block*, paddle::framework::Scope*>& sub_blocks)
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
  // SetDeviceContext(ParseDeviceContext(
  //     op, nullptr, place, GetExecutionStream(), GetStreamPriority()));
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

    for (size_t i = 0; i < if_op.num_results(); ++i) {
      if_op_outputs_.push_back(
          inner_scope->GetVar(value_2_var_name.at(if_op.result(i))));
    }

    auto cond_value = if_op.operand_source(0);
    auto var_name = value_2_var_name.at(cond_value);
    std::cerr << "var name " << var_name << std::endl;
    cond_var = inner_scope->FindVar(var_name);

    auto true_branch_block = if_op.true_block();
    auto false_branch_block = if_op.false_block();

    auto true_branch_yied_inputs = GetYiedOpInputs(true_branch_block);
    auto false_branch_yied_inputs = GetYiedOpInputs(false_branch_block);

    auto true_scope = sub_blocks.at(true_branch_block);
    true_branch_inter =
        new NewIRInterpreter(place, {}, true_branch_block, true_scope, {});

    std::cerr << "11" << std::endl;
    std::set<std::string> true_skip_gc_names_set;
    for (auto value : true_branch_yied_inputs) {
      true_skip_gc_names_.push_back(true_branch_inter->GetNameByValue(value));
      true_skip_gc_names_set.insert(true_branch_inter->GetNameByValue(value));
    }
    true_branch_inter->SetSkipGcVars(true_skip_gc_names_set);
    // true_branch_iter->SetSkipGcVars( )
    std::cerr << "12" << std::endl;
    auto false_scope = sub_blocks.at(false_branch_block);
    false_branch_inter =
        new NewIRInterpreter(place, {}, false_branch_block, false_scope, {});

    std::set<std::string> false_skip_gc_names_set;
    for (auto value : false_branch_yied_inputs) {
      false_skip_gc_names_.push_back(false_branch_inter->GetNameByValue(value));
      false_skip_gc_names_set.insert(false_branch_inter->GetNameByValue(value));
    }
    false_branch_inter->SetSkipGcVars(false_skip_gc_names_set);
  }

  std::cerr << "13" << std::endl;

  // InitInputsOutputsIds(
  //     op, inner_scope, value_2_var_name, var_name_2_id, variable_2_var_name);
  std::unordered_map<pir::Value, std::vector<int>> inputs;
  for (size_t i = 0; i < op->num_operands(); i++) {
    pir::Value value = op->operand_source(i);
    if (value) {
      PADDLE_ENFORCE_NE(
          value_2_var_name.find(value),
          value_2_var_name.end(),
          phi::errors::PreconditionNotMet(
              "input should in name map, [%d] 'th input of [%s] op",
              i,
              "if op"));
      std::vector<int> inputs_id = GetValueIds(value,
                                               inner_scope,
                                               value_2_var_name,
                                               var_name_2_id,
                                               variable_2_var_name);
      inputs.emplace(value, inputs_id);
    }
  }
  SetInputs(inputs);

  std::unordered_map<pir::Value, std::vector<int>> outputs;
  for (size_t i = 0; i < op->num_results(); i++) {
    pir::Value value = op->result(i);
    if (value && value.type()) {
      PADDLE_ENFORCE_NE(
          value_2_var_name.find(value),
          value_2_var_name.end(),
          phi::errors::PreconditionNotMet(
              "input should in name map, [%d] 'th input of [%s] op",
              i,
              "if op"));
      std::vector<int> outputs_id = GetValueIds(value,
                                                inner_scope,
                                                value_2_var_name,
                                                var_name_2_id,
                                                variable_2_var_name);
      outputs.emplace(value, outputs_id);
    }
  }
  SetOutputs(outputs);
}

void CondInstruction::CopyBranchOutput(
    const std::vector<std::string>& var_names, const NewIRInterpreter* inter) {
  for (size_t i = 0; i < var_names.size(); ++i) {
    auto* inner_var = inter->local_scope()->GetVar(var_names[i]);

    if_op_outputs_[i]->GetMutable<phi::DenseTensor>()->ShareDataWith(
        inner_var->Get<phi::DenseTensor>());
  }
}

void CondInstruction::Run() {
  std::cerr << "cond run" << std::endl;
  if (cond_var->Get<phi::DenseTensor>().data<bool>()[0]) {
    std::cerr << "true  " << std::endl;
    true_branch_inter->Run({}, false);
    std::cerr << "true  fin" << std::endl;
    CopyBranchOutput(true_skip_gc_names_, true_branch_inter);
  } else {
    std::cerr << "false  " << std::endl;
    false_branch_inter->Run({}, false);
    std::cerr << "false fin  " << std::endl;

    CopyBranchOutput(false_skip_gc_names_, false_branch_inter);
  }

  // copy ouptut
}

}  // namespace framework
}  // namespace paddle
