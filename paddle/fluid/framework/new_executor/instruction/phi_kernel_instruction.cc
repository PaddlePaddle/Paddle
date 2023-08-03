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

namespace paddle {
namespace framework {

platform::DeviceContext* ParseDeviceContext(
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

OpFuncType AnalyseOpFuncType(ir::Operation* op, const platform::Place& place) {
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

std::vector<int> GetValueIds(
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

void PhiKernelInstruction::InitInputsOutputsIds(
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

void PhiKernelInstruction::Run() {
  if (infer_meta_interface_) {
    infer_meta_interface_->infer_meta_(&(infer_meta_context_));
  }
  VLOG(6) << "Run op " << phi_op_name_ << " infer meta.";
  (*(phi_kernel_))(&(kernel_context_));
  VLOG(6) << "Run op " << phi_op_name_ << " kernel.";
}

}  // namespace framework
}  // namespace paddle
