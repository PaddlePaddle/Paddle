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
#include "paddle/fluid/framework/new_executor/pir_adaptor/pir_adaptor_util.h"
#include "paddle/fluid/framework/scope.h"
#include "paddle/fluid/pir/dialect/operator/interface/infermeta.h"
#include "paddle/fluid/pir/dialect/operator/interface/op_yaml_info.h"
#include "paddle/fluid/pir/dialect/operator/ir/op_dialect.h"
#include "paddle/fluid/pir/dialect/operator/utils/op_yaml_info_parser.h"
#include "paddle/fluid/platform/collective_helper.h"
#include "paddle/fluid/platform/device_context.h"
#include "paddle/phi/core/infermeta_utils.h"
#include "paddle/phi/core/meta_tensor.h"
#include "paddle/phi/core/type_defs.h"

#include "paddle/pir/core/builtin_attribute.h"
#include "paddle/pir/core/operation.h"
#include "paddle/pir/core/value.h"

#include "paddle/fluid/framework/new_executor/instruction/instruction_util.h"
#include "paddle/fluid/pir/dialect/operator/ir/control_flow_op.h"
#include "paddle/fluid/pir/dialect/operator/ir/manual_op.h"
namespace paddle {
namespace framework {

CondInstruction::CondInstruction(size_t id,
                                 const platform::Place& place,
                                 pir::Operation* op,
                                 ValueExecutionInfo* value_exec_info)
    : InstructionBase(id, place) {
  PADDLE_ENFORCE(
      op->isa<paddle::dialect::IfOp>(),
      phi::errors::PreconditionNotMet("Cond instruction only support if op"));
  auto if_op = op->dyn_cast<paddle::dialect::IfOp>();
  op_ = op;

  SetKernelType(AnalyseOpFuncType(op, place));
  VLOG(6) << "finish process analyse kernel type";

  auto cond_value = if_op.operand_source(0);
  cond_var_ = value_exec_info->GetScope()->FindVar(
      value_exec_info->GetValue2VarName().at(cond_value));
  for (size_t i = 0; i < if_op.num_results(); ++i) {
    output_vars_.push_back(value_exec_info->GetScope()->GetVar(
        value_exec_info->GetValue2VarName().at(if_op.result(i))));
  }
  VLOG(6) << "finish process cond_var and output_vars";

  // NOTE(zhangbo): IfOp sub_block's inputs include two kind of value: one is
  // OpOperand of IfOp, and the other is external Values used in true_block or
  // false_block.
  auto true_branch_block = if_op.true_block();
  auto false_branch_block = if_op.false_block();
  std::unordered_map<pir::Value, std::vector<int>> inputs;
  GetInputIds(op, *value_exec_info, &inputs);
  auto true_outside_inputs =
      GetOutsideOpInputs(true_branch_block, *value_exec_info, &inputs);
  auto false_outside_inputs =
      GetOutsideOpInputs(false_branch_block, *value_exec_info, &inputs);
  SetInputs(inputs);

  std::unordered_map<pir::Value, std::vector<int>> outputs;
  for (size_t i = 0; i < op->num_results(); i++) {
    pir::Value value = op->result(i);
    if (value && value.type()) {
      PADDLE_ENFORCE_EQ(
          value_exec_info->HasValue(value),
          true,
          phi::errors::PreconditionNotMet(
              "input should in name map, [%d] 'th input of [%s] op",
              i,
              "if op"));
      outputs.emplace(value, GetValueIds(value, *value_exec_info));
    }
  }
  SetOutputs(outputs);
  VLOG(6) << "finish process inputs outputs index";

  Scope* true_scope = &(value_exec_info->GetScope()->NewScope());
  true_branch_inter_ =
      new NewIRInterpreter(place,
                           {},
                           true_branch_block,
                           true_scope,
                           value_exec_info->NewChild(true_scope),
                           {});

  std::set<std::string> true_skip_gc_names_set;
  for (auto value : GetYiedOpInputs(true_branch_block)) {
    true_branch_outputs_.push_back(true_branch_inter_->GetNameByValue(value));
    true_skip_gc_names_.push_back(true_branch_inter_->GetNameByValue(value));
    true_skip_gc_names_set.insert(true_branch_inter_->GetNameByValue(value));
  }
  // NOTE(zhangbo): According to the concept of control flow, child scopes
  // should not control the lifecycle of parent scope variables.
  for (auto value : true_outside_inputs) {
    true_skip_gc_names_.push_back(true_branch_inter_->GetNameByValue(value));
    true_skip_gc_names_set.insert(true_branch_inter_->GetNameByValue(value));
  }
  true_branch_inter_->SetSkipGcVars(true_skip_gc_names_set);
  VLOG(6) << "finish process true branch interpreter";

  Scope* false_scope = &(value_exec_info->GetScope()->NewScope());
  false_branch_inter_ =
      new NewIRInterpreter(place,
                           {},
                           false_branch_block,
                           false_scope,
                           value_exec_info->NewChild(false_scope),
                           {});

  std::set<std::string> false_skip_gc_names_set;
  for (auto value : GetYiedOpInputs(false_branch_block)) {
    false_branch_outputs_.push_back(false_branch_inter_->GetNameByValue(value));
    false_skip_gc_names_.push_back(false_branch_inter_->GetNameByValue(value));
    false_skip_gc_names_set.insert(false_branch_inter_->GetNameByValue(value));
  }
  for (auto value : false_outside_inputs) {
    false_skip_gc_names_.push_back(false_branch_inter_->GetNameByValue(value));
    false_skip_gc_names_set.insert(false_branch_inter_->GetNameByValue(value));
  }
  false_branch_inter_->SetSkipGcVars(false_skip_gc_names_set);
  VLOG(6) << "finish process false branch interpreter";
}

CondInstruction::~CondInstruction() {
  if (true_branch_inter_ != nullptr) {
    delete true_branch_inter_;
  }
  if (false_branch_inter_ != nullptr) {
    delete false_branch_inter_;
  }
}

void CondInstruction::CopyBranchOutput(
    const std::vector<std::string>& var_names, const NewIRInterpreter* inter) {
  for (size_t i = 0; i < var_names.size(); ++i) {
    auto* inner_var = inter->InnerScope()->GetVar(var_names[i]);

    output_vars_[i]->GetMutable<phi::DenseTensor>()->ShareDataWith(
        inner_var->Get<phi::DenseTensor>());
  }
}

void CondInstruction::Run() {
  DeviceContext().Wait();
  if (cond_var_->Get<phi::DenseTensor>().data<bool>()[0]) {
    true_branch_inter_->Run({}, false);
    CopyBranchOutput(true_branch_outputs_, true_branch_inter_);
  } else {
    false_branch_inter_->Run({}, false);
    CopyBranchOutput(false_branch_outputs_, false_branch_inter_);
  }

  // copy ouptut
}

}  // namespace framework
}  // namespace paddle
