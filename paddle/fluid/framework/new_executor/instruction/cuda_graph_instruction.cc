// Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/framework/new_executor/instruction/cuda_graph_instruction.h"

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
#include "paddle/phi/core/platform/cuda_graph_with_memory_pool.h"
#include "paddle/phi/core/platform/device_context.h"
#include "paddle/phi/core/type_defs.h"

#include "paddle/pir/include/core/builtin_attribute.h"
#include "paddle/pir/include/core/operation.h"
#include "paddle/pir/include/core/value.h"

#include "paddle/fluid/framework/new_executor/instruction/instruction_util.h"
#include "paddle/fluid/pir/dialect/operator/ir/manual_op.h"

#ifdef PADDLE_WITH_CUDA

namespace paddle::framework {

CudaGraphInstruction::CudaGraphInstruction(
    size_t id,
    const phi::Place& place,
    pir::Operation* op,
    ValueExecutionInfo* value_exec_info,
    interpreter::ExecutionConfig execution_config)
    : InstructionBase(id, place),
      op_(op),
      place_(place),
      name_("cuda_graph_instruction"),
      output_vars_(),
      interpreter_(nullptr),
      skip_gc_names_() {
  PADDLE_ENFORCE(op->isa<paddle::dialect::CudaGraphOp>(),
                 common::errors::PreconditionNotMet(
                     "CudaGraph instruction only support cuda_graph op"));
  auto cuda_graph_op = op->dyn_cast<paddle::dialect::CudaGraphOp>();
  op_ = op;

  SetKernelType(OpFuncType::kGpuAsync);
  VLOG(6) << "finish process analyse kernel type";

  for (size_t i = 0; i < cuda_graph_op.num_results(); ++i) {
    output_vars_.push_back(value_exec_info->GetScope()->GetVar(
        value_exec_info->GetValue2VarName().at(cuda_graph_op.result(i))));
  }
  VLOG(6) << "finish process output_vars";

  std::unordered_map<pir::Value, std::vector<int>> inputs;
  GetInputIds(op, *value_exec_info, &inputs);
  auto outside_inputs =
      GetExternalInputs(cuda_graph_op.block(), *value_exec_info, &inputs);

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

  InsertInplacedExternalInputsToOuts(
      cuda_graph_op.block(), outside_inputs, *value_exec_info, &outputs);

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

  Scope* scope = &(value_exec_info->GetScope()->NewScope());
  auto skip_gc_vars = execution_config.skip_gc_vars;
  execution_config.skip_gc_vars.clear();
  execution_config.create_local_scope = true;
  interpreter_ = new PirInterpreter(place,
                                    {},
                                    cuda_graph_op.block(),
                                    scope,
                                    value_exec_info->NewChild(scope),
                                    execution_config);

  std::set<std::string> skip_gc_names_set;
  for (auto value : outside_inputs) {
    skip_gc_names_.push_back(interpreter_->GetNameByValue(value));
    skip_gc_names_set.insert(interpreter_->GetNameByValue(value));
  }
  for (const auto& var_name : skip_gc_vars) {
    skip_gc_names_.push_back(var_name);
    skip_gc_names_set.insert(var_name);
  }
  interpreter_->SetSkipGcVars(skip_gc_names_set);
  VLOG(6) << "finish process interpreter";
}

CudaGraphInstruction::~CudaGraphInstruction() { delete interpreter_; }

void CudaGraphInstruction::SetOutputHooks(
    const std::vector<PirHookFunc>& hookfuncs) {
  interpreter_->SetOutputHooks(hookfuncs);
}

void CudaGraphInstruction::SetInputHooks(
    const std::vector<PirHookFunc>& hookfuncs) {
  interpreter_->SetInputHooks(hookfuncs);
}

void CudaGraphInstruction::Run() {
  if (cuda_graph_) {
    cuda_graph_->Replay();
    return;
  }
  if (false) {
    platform::BeginCUDAGraphCapture(place_, cudaStreamCaptureModeRelaxed);
    interpreter_->Run({}, false);
    cuda_graph_ = platform::EndCUDAGraphCapture();
  } else {
    interpreter_->Run({}, false);
  }
}

}  // namespace paddle::framework

#endif  // PADDLE_WITH_CUDA
