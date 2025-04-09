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

#include "paddle/fluid/framework/new_executor/instruction/control_flow/pylayer_instruction.h"

#include "paddle/fluid/framework/new_executor/interpreter/interpreter_util.h"
#include "paddle/fluid/framework/new_executor/interpreter/stream_analyzer.h"
#include "paddle/fluid/framework/new_executor/pir_adaptor/pir_adaptor_util.h"
#include "paddle/fluid/framework/new_executor/pir_interpreter.h"
#include "paddle/fluid/framework/scope.h"
#include "paddle/fluid/pir/dialect/operator/interface/infermeta.h"
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
#include "paddle/fluid/pir/dialect/operator/ir/manual_op.h"
#include "paddle/fluid/pir/dialect/operator/ir/manual_pylayer_op.h"

#ifdef PADDLE_WITH_DNNL
#include "paddle/fluid/platform/onednn_helper.h"
#endif

namespace paddle::framework {

PyLayerInstruction::PyLayerInstruction(
    size_t id,
    const phi::Place& place,
    pir::Operation* op,
    ValueExecutionInfo* value_exec_info,
    interpreter::ExecutionConfig execution_config)
    : InstructionBase(id, place), output_vars_(), fwd_skip_gc_names_() {
  PADDLE_ENFORCE(op->isa<paddle::dialect::PyLayerOp>(),
                 common::errors::PreconditionNotMet(
                     "Cond instruction only support pylayer op"));
  auto pylayer_op = op->dyn_cast<paddle::dialect::PyLayerOp>();
  op_ = op;

  SetKernelType(AnalyseOpFuncType(op, place));
  VLOG(6) << "finish process analyse kernel type";
  for (size_t i = 0; i < pylayer_op.num_results(); ++i) {
    if (pylayer_op.result(i) && pylayer_op.result(i).type()) {
      output_vars_.push_back(value_exec_info->GetScope()->GetVar(
          value_exec_info->GetValue2VarName().at(pylayer_op.result(i))));
    }
  }
  VLOG(6) << "finish process output_vars";

  auto& fwd_block = pylayer_op.forward_block();
  std::unordered_map<pir::Value, std::vector<int>> inputs;
  GetInputIds(op, *value_exec_info, &inputs);
  const auto fwd_outside_inputs =
      GetExternalInputs(&fwd_block, *value_exec_info, &inputs);

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
  for (size_t i = 0; i < op->num_results(); i++) {
    pir::Value value = op->result(i);
    if (value && value.type()) {
      PADDLE_ENFORCE_EQ(
          value_exec_info->HasValue(value),
          true,
          common::errors::PreconditionNotMet(
              "output should in name map, [%d] 'th output of [%s] op",
              i,
              "pylayer op"));
      outputs.emplace(value, GetValueIds(value, *value_exec_info));
    }
  }

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

  Scope* fwd_scope = &(value_exec_info->GetScope()->NewScope());
  auto skip_gc_vars = execution_config.skip_gc_vars;
  execution_config.skip_gc_vars.clear();
  execution_config.create_local_scope = true;
  fwd_inter_ = new PirInterpreter(place,
                                  {},
                                  &fwd_block,
                                  fwd_scope,
                                  value_exec_info->NewChild(fwd_scope),
                                  execution_config);

  std::set<std::string> fwd_skip_gc_names_set;

  // NOTE(zhangbo): According to the concept of control flow, child scopes
  // should not control the lifecycle of parent scope variables.
  for (auto value : fwd_outside_inputs) {
    fwd_skip_gc_names_.push_back(fwd_inter_->GetNameByValue(value));
    fwd_skip_gc_names_set.insert(fwd_inter_->GetNameByValue(value));
  }
  for (const auto& var_name : skip_gc_vars) {
    fwd_skip_gc_names_.push_back(var_name);
    fwd_skip_gc_names_set.insert(var_name);
  }

  fwd_inter_->SetSkipGcVars(fwd_skip_gc_names_set);
  VLOG(6) << "finish process forward block interpreter";
}

PyLayerInstruction::~PyLayerInstruction() { delete fwd_inter_; }

void PyLayerInstruction::Run() {
  VLOG(6) << "start pylayer forward block interpreter";

#ifdef PADDLE_WITH_DNNL
  // Executor on being destroyed clears oneDNN cache and resets
  // registered model data layout. This is unwanted for nested
  // Executors (executors declared inside control ops)
  paddle::platform::DontClearMKLDNNCache(fwd_inter_->GetPlace());
#endif
  fwd_inter_->Run({}, false);
}

}  // namespace paddle::framework
