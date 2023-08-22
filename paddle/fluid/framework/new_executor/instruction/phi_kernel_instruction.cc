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

#include "paddle/fluid/framework/new_executor/instruction/phi_kernel_instruction.h"

#include "paddle/fluid/framework/new_executor/interpreter/interpreter_util.h"
#include "paddle/fluid/framework/new_executor/interpreter/stream_analyzer.h"
#include "paddle/fluid/framework/scope.h"
#include "paddle/fluid/ir/dialect/pd_dialect.h"
#include "paddle/fluid/ir/interface/infermeta.h"
#include "paddle/fluid/ir/interface/op_yaml_info.h"
#include "paddle/fluid/ir/interface/op_yaml_info_parser.h"
#include "paddle/fluid/ir/phi_kernel_adaptor/phi_kernel_util.h"
#include "paddle/fluid/platform/collective_helper.h"
#include "paddle/fluid/platform/device_context.h"
#include "paddle/phi/core/infermeta_utils.h"
#include "paddle/phi/core/meta_tensor.h"
#include "paddle/phi/core/type_defs.h"

#include "paddle/ir/core/builtin_attribute.h"
#include "paddle/ir/core/operation.h"
#include "paddle/ir/core/value.h"

#include "paddle/phi/core/kernel_context.h"

#include "paddle/fluid/framework/new_executor/instruction/instruction_util.h"
namespace paddle {
namespace framework {

PhiKernelInstruction::PhiKernelInstruction(
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
  auto op_attributes = op->attributes();
  auto op_name =
      op_attributes.at("op_name").dyn_cast<::ir::StrAttribute>().AsString();
  ir::OpInfo op_info = ir::IrContext::Instance()->GetRegisteredOpInfo(op_name);

  phi_op_name_ = op_name;
  VLOG(6) << "construct phi kernel instruction for: " << phi_op_name_;

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

  infer_meta_interface_ =
      op_info.GetInterfaceImpl<paddle::dialect::InferMetaInterface>();
  VLOG(6) << "finish process infer_meta_interface_";

  auto yaml_interface =
      op_info.GetInterfaceImpl<paddle::dialect::OpYamlInfoInterface>();
  PADDLE_ENFORCE_NOT_NULL(
      yaml_interface,
      phi::errors::PreconditionNotMet(
          "can not find OpYamlInfoInterface from [%s]", phi_op_name_));
  paddle::dialect::OpYamlInfoParser yaml_info_parser(
      yaml_interface->get_op_info_());
  VLOG(6) << "finish process yaml_info_parser";

  if (infer_meta_interface_) {
    ::ir::BuildPhiContext<
        phi::InferMetaContext,
        phi::MetaTensor,
        phi::MetaTensor,
        paddle::small_vector<phi::MetaTensor, phi::kInputSmallVectorSize>,
        paddle::small_vector<phi::MetaTensor, phi::kInputSmallVectorSize>,
        false>(op,
               value_2_var_name,
               scope,
               local_scope,
               yaml_info_parser,
               &infer_meta_context_);
  }
  VLOG(6) << "finish process infer meta context";

  auto kernel_name =
      op_attributes.at("kernel_name").dyn_cast<ir::StrAttribute>().AsString();
  auto kernel_key = op_attributes.at("kernel_key")
                        .dyn_cast<paddle::dialect::KernelAttribute>()
                        .data();
  auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
      kernel_name, kernel_key);
  phi_kernel_ = new phi::Kernel(kernel_result.kernel);
  PADDLE_ENFORCE_EQ(
      phi_kernel_->IsValid(), true, "not found kernel for [%s]", kernel_name);
  VLOG(6) << "finish process select kernel";

  ::ir::BuildPhiContext<phi::KernelContext,
                        const phi::TensorBase*,
                        phi::TensorBase*,
                        paddle::small_vector<const phi::TensorBase*>,
                        paddle::small_vector<phi::TensorBase*>,
                        true>(op,
                              value_2_var_name,
                              scope,
                              local_scope,
                              yaml_info_parser,
                              &kernel_context_);
  kernel_context_.SetDeviceContext(phi::DeviceContextPool::Instance().Get(
      phi::TransToPhiPlace(kernel_key.backend())));
  VLOG(6) << "finish process kernel context";

  SetDeviceContext(
      ParseDeviceContext(op,
                         phi::DeviceContextPool::Instance().Get(
                             phi::TransToPhiPlace(kernel_key.backend())),
                         place,
                         GetExecutionStream(),
                         GetStreamPriority()));
  VLOG(6) << "finish process device context";

  Scope* inner_scope = local_scope == nullptr ? scope : local_scope;
  InitInputsOutputsIds(
      op, inner_scope, value_2_var_name, var_name_2_id, variable_2_var_name);
  VLOG(6) << "finish process inputs outputs index";

  auto& no_need_buffer_ids = yaml_info_parser.NoNeedBufferIds();
  std::unordered_set<::ir::Value> no_need_buffer_values;
  for (size_t id = 0; id < no_need_buffer_ids.size(); id++) {
    no_need_buffer_values.insert(op->operand_source(no_need_buffer_ids[id]));
  }
  SetNoNeedBuffer(no_need_buffer_values);
  VLOG(6) << "finish process no need buffer";
}

void PrintKernelContext(const phi::KernelContext& kernel_context) {
  VLOG(6) << "------------kernel_contxt start-----------";
  VLOG(6) << "inputs: " << kernel_context.InputsSize();
  VLOG(6) << "output: " << kernel_context.OutputsSize();
  VLOG(6) << "attrs: " << kernel_context.AttrsSize();
  VLOG(6) << "------------kernel_contxt end-------------";
}

void PhiKernelInstruction::Run() {
  if (infer_meta_interface_) {
    infer_meta_interface_->infer_meta_(&(infer_meta_context_));
  }
  VLOG(6) << "Run op " << phi_op_name_ << " infer meta.";

  PrintKernelContext(kernel_context_);

  (*(phi_kernel_))(&(kernel_context_));
  VLOG(6) << "Run op " << phi_op_name_ << " kernel.";
}

}  // namespace framework
}  // namespace paddle
