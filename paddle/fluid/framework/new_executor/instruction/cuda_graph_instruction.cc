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
#include "paddle/fluid/pir/dialect/operator/ir/op_dialect.h"
#include "paddle/phi/core/platform/collective_helper.h"
#include "paddle/phi/core/platform/cuda_graph_with_memory_pool.h"
#include "paddle/phi/core/platform/device_context.h"
#include "paddle/phi/core/tensor_utils.h"
#include "paddle/phi/core/type_defs.h"

#include "paddle/pir/include/core/builtin_attribute.h"
#include "paddle/pir/include/core/operation.h"
#include "paddle/pir/include/core/value.h"

#include "paddle/fluid/framework/new_executor/instruction/instruction_util.h"
#include "paddle/fluid/pir/dialect/operator/ir/manual_op.h"

COMMON_DECLARE_bool(check_cuda_error);

#ifdef PADDLE_WITH_CUDA
namespace paddle::framework {

CudaGraphInstruction::CudaGraphInstruction(
    size_t id,
    const phi::Place& place,
    pir::Operation* op,
    uint8_t* cuda_graph_state_ref,
    int64_t cuda_graph_capture_pool_id,
    ValueExecutionInfo* value_exec_info,
    interpreter::ExecutionConfig execution_config)
    : InstructionBase(id, place),
      op_(op),
      place_(place),
      cuda_graph_state_ref_(cuda_graph_state_ref),
      cuda_graph_capture_pool_id_(cuda_graph_capture_pool_id),
      name_("cuda_graph_instruction"),
      input_vars_(),
      output_vars_(),
      interpreter_(nullptr),
      skip_gc_names_() {
  PADDLE_ENFORCE(op->isa<paddle::dialect::CudaGraphOp>(),
                 common::errors::PreconditionNotMet(
                     "CudaGraph instruction only support cuda_graph op"));
  op_ = op;

  SetKernelType(OpFuncType::kGpuAsync);
  VLOG(6) << "finish process analyse kernel type";

  auto cuda_graph_op = op->dyn_cast<paddle::dialect::CudaGraphOp>();

  std::unordered_map<pir::Value, std::vector<int>> inputs;
  GetInputIds(op, *value_exec_info, &inputs);
  const auto outside_inputs =
      GetExternalInputs(cuda_graph_op.block(), *value_exec_info, &inputs);
  for (size_t i = 0; i < outside_inputs.size(); ++i) {
    input_vars_.push_back(value_exec_info->GetScope()->GetVar(
        value_exec_info->GetValue2VarName().at(outside_inputs.at(i))));
  }
  VLOG(6) << "finish process input_vars";

  for (size_t i = 0; i < cuda_graph_op.num_results(); ++i) {
    output_vars_.push_back(value_exec_info->GetScope()->GetVar(
        value_exec_info->GetValue2VarName().at(cuda_graph_op.result(i))));
  }
  VLOG(6) << "finish process output_vars";

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
  if (FLAGS_check_cuda_error) [[unlikely]] {
    CUDAErrorCheck("CudaGraphInstruction begin");
  }

  if (cuda_graph_ != nullptr && *cuda_graph_state_ref_ == 3) {
    VLOG(4) << "Start replaying cuda graph @" << cuda_graph_.get();
    for (size_t i = 0; i < input_vars_.size(); ++i) {
      if (input_vars_[i]->IsType<phi::DenseTensor>()) {
        auto* tensor = input_vars_[i]->GetMutable<phi::DenseTensor>();
        if (tensor->data() != input_tensors_.at(i).data()) {
          LOG(WARNING) << "The input [" << i << "] tensor addr for "
                       << "cuda graph is changed. Pay attention to this!";
          if (phi::is_gpu_place(tensor->place())) {
            const auto* dev_ctx =
                phi::DeviceContextPool::Instance().Get(place_);
            phi::Copy(*dev_ctx, *tensor, place_, false, &input_tensors_.at(i));
          }
        }
      }
    }

    cuda_graph_->Replay();

    // set the output tensors into scope
    for (size_t i = 0; i < output_vars_.size(); ++i) {
      *(output_vars_[i]->GetMutable<phi::DenseTensor>()) =
          output_tensors_.at(i);
    }
    VLOG(4) << "Finish replaying cuda graph";
    return;
  }
  if (*cuda_graph_state_ref_ == 2 && cuda_graph_ == nullptr) {
    VLOG(4) << "Warmup before capturing";
    interpreter_->Run({}, false);
    VLOG(4) << "Start capturing cuda graph ...";
    platform::BeginCUDAGraphCapture(
        place_, cudaStreamCaptureModeRelaxed, cuda_graph_capture_pool_id_);

    auto RecordTensorsForReplay = [&](const std::vector<Variable*>& vars) {
      std::vector<phi::DenseTensor> record_tensors;
      record_tensors.reserve(vars.size());
      for (auto& var : vars) {
        auto& tensor = var->Get<phi::DenseTensor>();
        const auto& holder = tensor.Holder();
        // Note: new_holder only record the memory address of the tensor for
        // cuda graph, original tensor memory will be freed to allocator after
        // graph capture.
        auto new_holder = std::make_shared<phi::Allocation>(
            holder->ptr(), holder->size(), holder->place());
        record_tensors.emplace_back(new_holder, tensor.meta());
      }
      return record_tensors;
    };

    // record the input tensors for replay
    input_tensors_ = RecordTensorsForReplay(input_vars_);

    interpreter_->Run({}, false);

    // record the output tensors for replay
    output_tensors_ = RecordTensorsForReplay(output_vars_);

    cuda_graph_ = platform::EndCUDAGraphCapture();
    VLOG(4) << "Finish capturing cuda graph @" << cuda_graph_.get();

    // compute the right result
    cuda_graph_->Replay();
  } else {
    VLOG(4) << "Run interpreter without cuda graph";
    interpreter_->Run({}, false);
  }

  if (FLAGS_check_cuda_error) [[unlikely]] {
    CUDAErrorCheck("CudaGraphInstruction finish");
  }
}

}  // namespace paddle::framework

#endif  // PADDLE_WITH_CUDA
