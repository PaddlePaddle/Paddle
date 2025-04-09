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

#include "paddle/fluid/framework/new_executor/instruction/control_flow/if_instruction.h"

#include "paddle/fluid/framework/new_executor/interpreter/interpreter_util.h"
#include "paddle/fluid/framework/new_executor/interpreter/stream_analyzer.h"
#include "paddle/fluid/framework/new_executor/pir_adaptor/pir_adaptor_util.h"
#include "paddle/fluid/framework/new_executor/pir_interpreter.h"
#include "paddle/fluid/framework/scope.h"
#include "paddle/fluid/pir/dialect/operator/interface/infermeta.h"
#include "paddle/fluid/pir/dialect/operator/interface/op_yaml_info.h"
#include "paddle/fluid/pir/dialect/operator/ir/op_dialect.h"
#include "paddle/fluid/pir/dialect/operator/utils/op_yaml_info_parser.h"
#include "paddle/phi/core/infermeta_utils.h"
#include "paddle/phi/core/meta_tensor.h"
#include "paddle/phi/core/platform/collective_helper.h"
#include "paddle/phi/core/platform/device_context.h"
#include "paddle/phi/core/type_defs.h"

#include "paddle/pir/include/core/builtin_attribute.h"
#include "paddle/pir/include/core/operation.h"
#include "paddle/pir/include/core/value.h"

#include "paddle/fluid/framework/new_executor/instruction/instruction_util.h"
#include "paddle/fluid/pir/dialect/operator/ir/control_flow_op.h"
#include "paddle/fluid/pir/dialect/operator/ir/manual_op.h"

#ifdef PADDLE_WITH_DNNL
#include "paddle/fluid/platform/onednn_helper.h"
#endif

namespace paddle::framework {

IfInstruction::IfInstruction(size_t id,
                             const phi::Place& place,
                             pir::Operation* op,
                             ValueExecutionInfo* value_exec_info,
                             interpreter::ExecutionConfig execution_config)
    : InstructionBase(id, place),
      op_(op),
      cond_name_("if_instruction"),
      cond_var_(nullptr),
      output_vars_(),
      true_branch_inter_(nullptr),
      false_branch_inter_(nullptr),
      true_skip_gc_names_(),
      false_skip_gc_names_() {
  PADDLE_ENFORCE(op->isa<paddle::dialect::IfOp>(),
                 common::errors::PreconditionNotMet(
                     "Cond instruction only support if op"));
  auto if_op = op->dyn_cast<paddle::dialect::IfOp>();
  op_ = op;

  SetKernelType(AnalyseOpFuncType(op, place));
  VLOG(6) << "finish process analyse kernel type";

  cond_var_ = value_exec_info->GetVarByValue(if_op.cond());
  for (size_t i = 0; i < if_op.num_results(); ++i) {
    output_vars_.push_back(value_exec_info->GetScope()->GetVar(
        value_exec_info->GetValue2VarName().at(if_op.result(i))));
  }
  VLOG(6) << "finish process cond_var and output_vars";

  // NOTE(zhangbo): IfOp sub_block's inputs include two kind of value: one is
  // OpOperand of IfOp, and the other is external Values used in true_block or
  // false_block.
  auto& true_branch_block = if_op.true_block();

  std::unordered_map<pir::Value, std::vector<int>> inputs;
  GetInputIds(op, *value_exec_info, &inputs);
  auto true_outside_inputs =
      GetExternalInputs(&true_branch_block, *value_exec_info, &inputs);
  auto& false_branch_block = if_op.false_block();
  auto false_outside_inputs =
      GetExternalInputs(&false_branch_block, *value_exec_info, &inputs);
  // NOTE(chenxi67): the variable corresponding to container value if a
  // <VariableRefArray> Type. It will recursively get the ID of internal
  // variables when use GetValueId() method. However, the copy_var pushed into
  // the tuple does not have a corresponding ID, and will insert a -1. Here we
  // remove the value of -1.
  for (auto& item : inputs) {
    auto& var_vec = item.second;
    for (auto it = var_vec.begin(); it != var_vec.end();) {
      if (*it == -1) {
        it = var_vec.erase(it);
      } else {
        ++it;
      }
    }
  }
  SetInputs(inputs);

  std::unordered_map<pir::Value, std::vector<int>> outputs;
  bool is_last_op = true;
  for (size_t i = 0; i < op->num_results(); i++) {
    pir::Value value = op->result(i);
    if (value && value.type()) {
      PADDLE_ENFORCE_EQ(
          value_exec_info->HasValue(value),
          true,
          common::errors::PreconditionNotMet(
              "input should in name map, [%d] 'th input of [%s] op",
              i,
              "if op"));
      outputs.emplace(value, GetValueIds(value, *value_exec_info));
    }
    if (value.use_count() > 0) {
      VLOG(6) << "value " << i << " use conutn != 0";
      is_last_op = false;
    }
  }
  InsertTuplePushContainerToOuts(
      &true_branch_block, *value_exec_info, &outputs);
  InsertTuplePushContainerToOuts(
      &if_op.false_block(), *value_exec_info, &outputs);

  InsertInplacedExternalInputsToOuts(
      &true_branch_block, true_outside_inputs, *value_exec_info, &outputs);
  InsertInplacedExternalInputsToOuts(
      &false_branch_block, false_outside_inputs, *value_exec_info, &outputs);

  for (auto& item : outputs) {
    auto& var_vec = item.second;
    for (auto it = var_vec.begin(); it != var_vec.end();) {
      if (*it == -1) {
        it = var_vec.erase(it);
      } else {
        ++it;
      }
    }
  }
  SetOutputs(outputs);
  VLOG(6) << "finish process inputs outputs index";

  Scope* true_scope = &(value_exec_info->GetScope()->NewScope());
  auto skip_gc_vars = execution_config.skip_gc_vars;
  execution_config.skip_gc_vars.clear();
  execution_config.create_local_scope = true;
  true_branch_inter_ = new PirInterpreter(place,
                                          {},
                                          &true_branch_block,
                                          true_scope,
                                          value_exec_info->NewChild(true_scope),
                                          execution_config);

  std::set<std::string> true_skip_gc_names_set;
  // NOTE(zhangbo): According to the concept of control flow, child scopes
  // should not control the lifecycle of parent scope variables.
  for (auto value : true_outside_inputs) {
    true_skip_gc_names_.push_back(true_branch_inter_->GetNameByValue(value));
    true_skip_gc_names_set.insert(true_branch_inter_->GetNameByValue(value));
  }
  for (const auto& var_name : skip_gc_vars) {
    true_skip_gc_names_.push_back(var_name);
    true_skip_gc_names_set.insert(var_name);
  }
  true_branch_inter_->SetSkipGcVars(true_skip_gc_names_set);
  VLOG(6) << "finish process true branch interpreter";

  Scope* false_scope = &(value_exec_info->GetScope()->NewScope());
  false_branch_inter_ =
      new PirInterpreter(place,
                         {},
                         &if_op.false_block(),
                         false_scope,
                         value_exec_info->NewChild(false_scope),
                         execution_config);
  std::set<std::string> false_skip_gc_names_set;
  for (auto value : false_outside_inputs) {
    false_skip_gc_names_.push_back(false_branch_inter_->GetNameByValue(value));
    false_skip_gc_names_set.insert(false_branch_inter_->GetNameByValue(value));
  }
  for (const auto& var_name : skip_gc_vars) {
    false_skip_gc_names_.push_back(var_name);
    false_skip_gc_names_set.insert(var_name);
  }
  false_branch_inter_->SetSkipGcVars(false_skip_gc_names_set);

  VLOG(6) << "finish process false branch interpreter";

  if (op_->HasAttribute("fake_false_branch") &&
      op_->attributes()
          .at("fake_false_branch")
          .dyn_cast<pir::BoolAttribute>()
          .data() &&
      is_last_op) {
    has_fake_false_branch_ = true;
  }
}

IfInstruction::~IfInstruction() {
  delete true_branch_inter_;
  delete false_branch_inter_;
}

void IfInstruction::SetOutputHooks(const std::vector<PirHookFunc>& hookfuncs) {
  true_branch_inter_->SetOutputHooks(hookfuncs);
  false_branch_inter_->SetOutputHooks(hookfuncs);
}

void IfInstruction::SetInputHooks(const std::vector<PirHookFunc>& hookfuncs) {
  true_branch_inter_->SetInputHooks(hookfuncs);
  false_branch_inter_->SetInputHooks(hookfuncs);
}

void IfInstruction::Run() {
  bool cond = true;
  if (cond_var_->IsType<phi::DenseTensor>()) {
    auto& cond_tensor = cond_var_->Get<phi::DenseTensor>();
    if (phi::is_cpu_place(cond_tensor.place())) {
      cond = cond_tensor.data<bool>()[0];
    } else {
      // when phi::is_gpu_place(cond.place()) or
      // phi::is_xpu_place(cond.place()) is true
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP) || \
    defined(PADDLE_WITH_XPU) || defined(PADDLE_WITH_CUSTOM_DEVICE)
      phi::DenseTensor cpu_cond;
      paddle::framework::TensorCopySync(
          cond_tensor, phi::CPUPlace(), &cpu_cond);
      cond = cpu_cond.data<bool>()[0];
#else
      PADDLE_THROW(common::errors::PreconditionNotMet(
          "This version of PaddlePaddle does NOT support GPU/XPU but got "
          "GPU/XPU tensor Cond in WhileOp. Please compile WITH_GPU or "
          "WITH_XPU option."));
#endif
    }
  } else if (cond_var_->IsType<VariableRefArray>()) {
    auto& cond_array = cond_var_->Get<VariableRefArray>();
    cond = std::all_of(
        cond_array.begin(), cond_array.end(), [](const Variable* t) {
          return t->Get<phi::DenseTensor>().numel() != 0;
        });
  }
  if (cond) {
#ifdef PADDLE_WITH_DNNL
    // Executor on being destroyed clears oneDNN cache and resets
    // registered model data layout. This is unwanted for nested
    // Executors (executors declared inside control ops)
    paddle::platform::DontClearMKLDNNCache(true_branch_inter_->GetPlace());
#endif
    true_branch_inter_->Run({}, false);
  } else {
    if (has_fake_false_branch_) return;
#ifdef PADDLE_WITH_DNNL
    // Executor on being destroyed clears oneDNN cache and resets
    // registered model data layout. This is unwanted for nested
    // Executors (executors declared inside control ops)
    paddle::platform::DontClearMKLDNNCache(false_branch_inter_->GetPlace());
#endif
    false_branch_inter_->Run({}, false);
  }
  // copy output
}

}  // namespace paddle::framework
