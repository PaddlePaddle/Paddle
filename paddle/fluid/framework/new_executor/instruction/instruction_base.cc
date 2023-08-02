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

#include "paddle/fluid/framework/new_executor/instruction/instruction_base.h"
#include "paddle/fluid/framework/new_executor/interpreter/interpreter_util.h"
#include "paddle/fluid/platform/profiler/event_tracing.h"

#include "paddle/fluid/framework/new_executor/interpreter/stream_analyzer.h"
#include "paddle/fluid/platform/collective_helper.h"
#include "paddle/ir/core/builtin_attribute.h"

namespace paddle {
namespace framework {

InstructionBase::InstructionBase(size_t id, const platform::Place& place) {
  id_ = id;

  is_artificial_ = false;

  if (platform::is_cpu_place(place)) {
    type_ = OpFuncType::kCpuSync;
  } else {
    PADDLE_ENFORCE_EQ(
        interpreter::IsSupportedHeterPlace(place),
        true,
        phi::errors::Fatal("Unsupported current place %s", place));
    type_ = OpFuncType::kGpuAsync;
  }

  dev_ctx_ = platform::DeviceContextPool::Instance().Get(place);
}

OpFuncType InstructionBase::KernelType() const { return type_; }

const platform::DeviceContext& InstructionBase::DeviceContext() const {
  return *dev_ctx_;
}

void InstructionBase::RecordEvent(const Place& place) const {
  platform::RecordEvent record(
      "RecordStreamEvent", platform::TracerEventType::UserDefined, 10);
  if (event_to_record_) {
    VLOG(6) << "Record event at instruction: " << id_;
    event_to_record_->event_->Record(dev_ctx_);
  }
}

void InstructionBase::WaitEvent(const Place& place) const {
  // If InterpreterCore in on CPUPlace, do nothing.
  if (platform::is_cpu_place(place)) {
    return;
  }
  for (const EventInter& event_iter : events_to_wait_) {
    platform::RecordEvent record(
        "WaitStreamEvent", platform::TracerEventType::UserDefined, 10);
    VLOG(6) << "Wait instruction: " << event_iter.instr_id_
            << " 's event with waiter_type: " << event_iter.waiter_type_;
    event_iter.event_->Wait(event_iter.waiter_type_, dev_ctx_);
  }
}

void InstructionBase::AddGCCheckVar(size_t id) { gc_check_vars_.push_back(id); }

const std::vector<size_t>& InstructionBase::GCCheckVars() const {
  return gc_check_vars_;
}

const std::vector<std::pair<Variable*, Variable*>>&
InstructionBase::InplaceInfo() const {
  return vec_inplace_in_to_out_;
}

void InstructionBase::AddInplace(Variable* in, Variable* out) {
  vec_inplace_in_to_out_.emplace_back(in, out);
}

void InstructionBase::ClearInplace() { vec_inplace_in_to_out_.clear(); }

void InstructionBase::SetInputs(
    const std::unordered_map<ir::Value, std::vector<int>>& inputs) {
  input_index_ = inputs;
}

void InstructionBase::SetOutputs(
    const std::unordered_map<ir::Value, std::vector<int>>& outputs) {
  output_index_ = outputs;
}

std::vector<int> InstructionBase::GetValueIds(
    ir::Value value,
    Scope* inner_scope,
    const std::unordered_map<::ir::Value, std::string>& value_2_var_name,
    const std::map<std::string, int>& var_name_2_id,
    const std::unordered_map<const paddle::framework::Variable*, std::string>&
        variable_2_var_name) {
  std::vector<int> ids;
  std::string var_name = value_2_var_name.at(value);
  ids.push_back(var_name_2_id.at(var_name));
  // NOTE(zhangbo): Value maybe a VariableRefArray
  auto var = inner_scope->FindVar(var_name);
  if (var->IsType<paddle::framework::VariableRefArray>()) {
    auto& var_array = var->Get<paddle::framework::VariableRefArray>();
    for (size_t i = 0; i < var_array.size(); ++i) {
      ids.push_back(var_name_2_id.at(variable_2_var_name.at(var_array[i])));
    }
  }
  return ids;
}

void InstructionBase::InitInputsOutputsIds(
    ::ir::Operation* op,
    Scope* inner_scope,
    const std::unordered_map<::ir::Value, std::string>& value_2_var_name,
    const std::map<std::string, int>& var_name_2_id,
    const std::unordered_map<const paddle::framework::Variable*, std::string>&
        variable_2_var_name) {
  std::unordered_map<ir::Value, std::vector<int>> inputs;
  for (size_t i = 0; i < op->num_operands(); i++) {
    ir::Value value = op->operand_source(i);
    if (value) {
      PADDLE_ENFORCE_NE(
          value_2_var_name.find(value),
          value_2_var_name.end(),
          phi::errors::PreconditionNotMet(
              "input should in name map, [%d] 'th input of [%s] op",
              i,
              phi_op_name_));
      std::vector<int> inputs_id = GetValueIds(value,
                                               inner_scope,
                                               value_2_var_name,
                                               var_name_2_id,
                                               variable_2_var_name);
      inputs.emplace(value, inputs_id);
    }
  }
  SetInputs(inputs);
  VLOG(8) << "finish process inputs_index";
  std::unordered_map<ir::Value, std::vector<int>> outputs;
  for (size_t i = 0; i < op->num_results(); i++) {
    ir::Value value = op->result(i);
    if (value) {
      std::cerr << "check value " << value.impl() << std::endl;
      PADDLE_ENFORCE_NE(
          value_2_var_name.find(value),
          value_2_var_name.end(),
          phi::errors::PreconditionNotMet(
              "input should in name map, [%d] 'th input of [%s] op",
              i,
              phi_op_name_));
      std::vector<int> outputs_id = GetValueIds(value,
                                                inner_scope,
                                                value_2_var_name,
                                                var_name_2_id,
                                                variable_2_var_name);
      outputs.emplace(value, outputs_id);
    }
  }
  SetOutputs(outputs);
  VLOG(8) << "finish process outputs_index";
}

platform::DeviceContext* InstructionBase::ParseDeviceContext(
    ir::Operation* op,
    platform::DeviceContext* origin_dev_ctx,
    const platform::Place& place,
    const std::string& execution_stream,
    const int stream_priority) {
  auto op_attributes = op->attributes();
  auto op_name =
      op_attributes.at("op_name").dyn_cast<::ir::StrAttribute>().AsString();
  interpreter::ContextManager& ctx_manager =
      interpreter::ContextManager::Instance();

  platform::DeviceContext* dev_ctx = nullptr;

  // only gpu need update. xpu not need, because xpu memcpy op kernel is
  // synchronous.
  if (platform::is_gpu_place(place) || platform::is_custom_place(place)) {
    VLOG(6) << "Parse DeviceContext for " << op_name
            << ", execution stream = " << execution_stream;
    if (execution_stream != kDefaultStream) {
      dev_ctx = ctx_manager
                    .Get(std::string(kCustomStream) + "-" + execution_stream,
                         place,
                         stream_priority)
                    .get()
                    .get();
      interpreter::SetDeviceCommContext(op, dev_ctx);
      return dev_ctx;
    }

    if (op_name == interpreter::kMemcpyD2H) {
      dev_ctx = ctx_manager.Get(std::string(kD2HStream), place, stream_priority)
                    .get()
                    .get();
      interpreter::SetDeviceCommContext(op, dev_ctx);
      return dev_ctx;
    } else if (op_name == interpreter::kMemcpyH2D) {
      dev_ctx = ctx_manager.Get(std::string(kH2DStream), place, stream_priority)
                    .get()
                    .get();
      interpreter::SetDeviceCommContext(op, dev_ctx);
      return dev_ctx;
    }

#if defined(PADDLE_WITH_NCCL) || defined(PADDLE_WITH_RCCL)
    // NOTE(Ruibiao): Here supports multi-stream overlap for c_allreduce_sum
    // with use_cal_stream==false by returning a device context getting from the
    // global NCCLCommContext instance. Because when use_calc_stream==false, in
    // OP kernel, the NCCL communication will be launched to the stream directly
    // getting from the global NCCLCommContext instance rather than the
    // DeviceContext passed from executor (see CAllReduceOpCUDAKernel in
    // c_allreduce_op.h). Now it is just a temporary solution for ONLY
    // c_allreduce_sum which is used in ResNet50 distributed training.
    if (op_name == "c_allreduce_sum" && op_attributes.at("use_calc_stream")
                                                .dyn_cast<::ir::BoolAttribute>()
                                                .data() == false) {
      int ring_id =
          op_attributes.at("ring_id").dyn_cast<::ir::Int32Attribute>().data();
      return platform::NCCLCommContext::Instance()
          .Get(ring_id, place)
          ->dev_context();
    }
#endif
  }

  if (origin_dev_ctx != nullptr) {
    interpreter::SetDeviceCommContext(op, origin_dev_ctx);
  }
  return origin_dev_ctx;
}

OpFuncType InstructionBase::AnalyseOpFuncType(ir::Operation* op,
                                              const platform::Place& place) {
  if (platform::is_cpu_place(place)) {
    return OpFuncType::kCpuSync;
  }

  PADDLE_ENFORCE_EQ(interpreter::IsSupportedHeterPlace(place),
                    true,
                    phi::errors::Fatal("Unsupported current place %s", place));

  // Some GPU OPs do not launch CUDA Kernel, but spend a lot of time on CPU
  // computing. They execute serially in device thread and block CUDA kernel
  // launching in other GPU OPs. To improve performance, set them as kGpuSync
  // and so that they would be dispatched to host thread.
  auto op_attributes = op->attributes();
  auto op_name =
      op_attributes.at("op_name").dyn_cast<::ir::StrAttribute>().AsString();
  if (op_name == kCoalesceTensor &&
      (!platform::is_xpu_place(place) ||
       op->attribute<ir::BoolAttribute>("persist_output").data() == false) &&
      op->attribute<ir::BoolAttribute>("set_constant").data() == false &&
      op->attribute<ir::BoolAttribute>("copy_data").data() == false) {
    return OpFuncType::kGpuSync;
  }

  // for memcpy explicitly called by user
  if (platform::is_gpu_place(place) && op_name == interpreter::kMemcpyD2H) {
    return OpFuncType::kGpuSync;
  }

  if (op_name == "shape") {
    return OpFuncType::kGpuSync;
  }
  return OpFuncType::kGpuAsync;
}

}  // namespace framework
}  // namespace paddle
