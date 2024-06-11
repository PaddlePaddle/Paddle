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

#include "paddle/fluid/framework/new_executor/instruction/legacy_kernel_instruction.h"

#include "paddle/fluid/framework/new_executor/instruction/instruction_util.h"
#include "paddle/fluid/framework/new_executor/interpreter/interpreter_util.h"
#include "paddle/fluid/framework/new_executor/interpreter/stream_analyzer.h"
#include "paddle/fluid/framework/new_executor/pir_adaptor/pir_adaptor_util.h"
#include "paddle/fluid/framework/scope.h"
#include "paddle/fluid/pir/dialect/operator/interface/infermeta.h"
#include "paddle/fluid/pir/dialect/operator/interface/op_yaml_info.h"
#include "paddle/fluid/pir/dialect/operator/ir/op_dialect.h"
#include "paddle/fluid/pir/dialect/operator/utils/op_yaml_info_parser.h"

#include "paddle/fluid/platform/device_context.h"
#include "paddle/phi/core/infermeta_utils.h"
#include "paddle/phi/core/meta_tensor.h"
#include "paddle/phi/core/type_defs.h"

namespace paddle::framework {

LegacyKernelInstruction::LegacyKernelInstruction(
    size_t id,
    const platform::Place& place,
    pir::Operation* op,
    const ValueExecutionInfo* value_exec_info)
    : InstructionBase(id, place), value_exec_info_(value_exec_info) {
  auto& op_attributes = op->attributes();
  auto op_name =
      op_attributes.at("op_name").dyn_cast<pir::StrAttribute>().AsString();
  pir::OpInfo op_info =
      pir::IrContext::Instance()->GetRegisteredOpInfo(op_name);
  op_ = op;
  legacy_op_name_ = op_name;
  VLOG(6) << "construct phi kernel instruction for: " << legacy_op_name_;

  if (op_attributes.count("execution_stream") != 0) {
    SetExecutionStream(op_attributes.at("execution_stream")
                           .dyn_cast<pir::StrAttribute>()
                           .AsString());
  }
  if (op_attributes.count("stream_priority") != 0) {
    SetStreamPriority(op_attributes.at("stream_priority")
                          .dyn_cast<pir::Int32Attribute>()
                          .data());
  }
  if (op_attributes.count("scheduling_priority") != 0) {
    SetSchedulingPriority(op_attributes.at("scheduling_priority")
                              .dyn_cast<pir::Int64Attribute>()
                              .data());
  } else {
    if (interpreter::IsCommunicationOp(op_)) {
      // NOTE(Ruibiao): Dispatching computation before communication improves
      // multi-stream overlap when the time cost of communication less than
      // that of the calculation (e.g., ResNet50_bs128_pure_fp16 N4C32
      // training).
      SetSchedulingPriority(1);
    }
  }
  if (op_attributes.count("force_record_event") != 0) {
    SetForceRecordEvent(op_attributes.at("force_record_event")
                            .dyn_cast<pir::BoolAttribute>()
                            .data());
  }
  if (op_attributes.count("event_to_record") != 0) {
    SetEventToRecordInfo(op_attributes.at("event_to_record")
                             .dyn_cast<pir::StrAttribute>()
                             .AsString());
  }
  if (op_attributes.count("events_to_wait") != 0) {
    std::vector<std::string> events_to_wait;
    auto array_attr = op_attributes.at("events_to_wait")
                          .dyn_cast<pir::ArrayAttribute>()
                          .AsVector();
    for (auto& attr : array_attr) {
      events_to_wait.push_back(attr.dyn_cast<pir::StrAttribute>().AsString());
    }
    SetEventsToWaitInfo(events_to_wait);
  }
  VLOG(6) << "finish process dist attributes for " << op_name
          << " : [execution_stream, stream_priority, scheduling_priority] = ["
          << GetExecutionStream() << ", " << GetStreamPriority() << ", "
          << GetSchedulingPriority() << "]";

  SetKernelType(AnalyseOpFuncType(op, place));
  VLOG(6) << "finish process analyse kernel type";

  infer_meta_interface_ =
      op_info.GetInterfaceImpl<paddle::dialect::InferMetaInterface>();
  VLOG(6) << "finish process infer_meta_interface_";

  auto yaml_interface =
      op_info.GetInterfaceImpl<paddle::dialect::OpYamlInfoInterface>();
  PADDLE_ENFORCE_NOT_NULL(
      yaml_interface,
      phi::errors::PreconditionNotMet(
          "can not find OpYamlInfoInterface from [%s]", legacy_op_name_));
  paddle::dialect::OpYamlInfoParser yaml_info_parser(
      yaml_interface->get_op_info_(op_name),
      paddle::dialect::IsLegacyOp(op_name));
  VLOG(6) << "finish process yaml_info_parser";

  if (infer_meta_interface_) {
    BuildPhiContext<
        phi::InferMetaContext,
        phi::MetaTensor,
        phi::MetaTensor,
        paddle::small_vector<phi::MetaTensor, phi::kInputSmallVectorSize>,
        paddle::small_vector<phi::MetaTensor, phi::kInputSmallVectorSize>,
        false>(op, *value_exec_info_, yaml_info_parser, &infer_meta_context_);
  }
  VLOG(6) << "finish process infer meta context";

  auto kernel_name =
      op_attributes.at("kernel_name").dyn_cast<pir::StrAttribute>().AsString();
  auto kernel_key = op_attributes.at("kernel_key")
                        .dyn_cast<paddle::dialect::KernelAttribute>()
                        .data();
  auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
      kernel_name, kernel_key);
  phi_kernel_ = new phi::Kernel(kernel_result.kernel);
  PADDLE_ENFORCE_EQ(
      phi_kernel_->IsValid(), true, "not found kernel for [%s]", kernel_name);
  VLOG(6) << "finish process select kernel: " << kernel_name;

  const Scope* inner_scope = value_exec_info_->GetScope();

  operator_base_ = BuildOperatorBase(op, *value_exec_info_, yaml_info_parser);

  SetDeviceContext(
      ParseDeviceContext(op,
                         phi::DeviceContextPool::Instance().Get(
                             phi::TransToPhiPlace(kernel_key.backend())),
                         place,
                         GetExecutionStream(),
                         GetStreamPriority()));
  VLOG(6) << "finish process device context";

  paddle::framework::VariableValueMap in_map;
  paddle::framework::VariableValueMap out_map;
  runtime_context_ = std::make_shared<paddle::framework::RuntimeContext>(
      paddle::framework::RuntimeContext(in_map, out_map));
  BuildRuntimeContext(
      op, *value_exec_info, yaml_info_parser, runtime_context_.get());

  kernel_context_ =
      new paddle::framework::ExecutionContext(*operator_base_,
                                              *inner_scope,
                                              DeviceContext(),
                                              *(runtime_context_.get()));

  VLOG(6) << "finish process kernel context";

  InitInputsOutputsIds(op, *value_exec_info);
  VLOG(6) << "finish process inputs outputs index";

  auto& no_need_buffer_ids = yaml_info_parser.NoNeedBufferIds();
  std::unordered_set<pir::Value> no_need_buffer_values;
  for (size_t id = 0; id < no_need_buffer_ids.size(); id++) {
    no_need_buffer_values.insert(op->operand_source(no_need_buffer_ids[id]));
  }
  SetNoNeedBuffer(no_need_buffer_values);
  VLOG(6) << "finish process no need buffer";
}

LegacyKernelInstruction::~LegacyKernelInstruction() {
  delete kernel_context_;
  delete phi_kernel_;
}

void LegacyKernelInstruction::Run() {
  VLOG(6) << "Run op " << legacy_op_name_ << " infer meta.";
  if (infer_meta_interface_) {
    infer_meta_interface_->infer_meta_(&(infer_meta_context_));
  }
  VLOG(6) << "Run op " << legacy_op_name_ << " kernel.";
  (*(phi_kernel_))((kernel_context_));
}
}  // namespace paddle::framework
