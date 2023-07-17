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
#include "paddle/fluid/framework/scope.h"
#include "paddle/fluid/ir/dialect/pd_dialect.h"
#include "paddle/fluid/ir/interface/infermeta.h"
#include "paddle/fluid/ir/interface/op_yaml_info.h"
#include "paddle/fluid/ir/interface/op_yaml_info_parser.h"
#include "paddle/fluid/ir/phi_kernel_adaptor/phi_kernel_util.h"
#include "paddle/phi/core/infermeta_utils.h"
#include "paddle/phi/core/meta_tensor.h"
#include "paddle/phi/core/type_defs.h"

#include "paddle/ir/core/builtin_attribute.h"
#include "paddle/ir/core/operation.h"
#include "paddle/ir/core/value.h"

namespace paddle {
namespace framework {

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
      op_attributes.at("op_name").dyn_cast<::ir::StrAttribute>().data();
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
    const std::unordered_map<::ir::Value, std::string>& value_2_name_map)
    : InstructionBase(id, place) {
  auto op_attributes = op->attributes();
  auto op_name =
      op_attributes.at("op_name").dyn_cast<::ir::StrAttribute>().data();
  ir::OpInfo op_info = ir::IrContext::Instance()->GetRegisteredOpInfo(op_name);

  phi_op_name_ = op_name;

  if (op_name == "builtin.combine" || op_name == "pd.feed" ||
      op_name == "builtin.set_parameter" ||
      op_name == "builtin.get_parameter") {
    VLOG(6) << "skip process " << op_name;
    SetArtificial(true);
    return;
  }

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

  SetKernelType(AnalyseOpFuncType(op, place));

  infer_meta_interface_ =
      op_info.GetInterfaceImpl<paddle::dialect::InferMetaInterface>();
  auto yaml_interface =
      op_info.GetInterfaceImpl<paddle::dialect::OpYamlInfoInterface>();
  paddle::dialect::OpYamlInfoParser yaml_info_parser(
      yaml_interface->get_op_info_());

  ::ir::BuildPhiContext<
      phi::InferMetaContext,
      phi::MetaTensor,
      phi::MetaTensor,
      paddle::small_vector<phi::MetaTensor, phi::kInputSmallVectorSize>,
      paddle::small_vector<phi::MetaTensor, phi::kInputSmallVectorSize>,
      false>(op,
             value_2_name_map,
             scope,
             local_scope,
             yaml_info_parser,
             &infer_meta_context_);
  VLOG(6) << "finish process infer meta context";

  auto kernel_name =
      op_attributes.at("kernel_name").dyn_cast<ir::StrAttribute>().data();
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
                              value_2_name_map,
                              scope,
                              local_scope,
                              yaml_info_parser,
                              &kernel_context_,
                              &(GetMutableInputs()),
                              &(GetMutableOutputs()));
  kernel_context_.SetDeviceContext(phi::DeviceContextPool::Instance().Get(
      phi::TransToPhiPlace(kernel_key.backend())));
  VLOG(6) << "finish process kernel context";

  SetDeviceContext(phi::DeviceContextPool::Instance().Get(
      phi::TransToPhiPlace(kernel_key.backend())));
  VLOG(6) << "finish process device context";
}

void PhiKernelInstruction::Run() {
  VLOG(5) << "Run op " << phi_op_name_ << " infer meta.";
  infer_meta_interface_->infer_meta_(&(infer_meta_context_));
  VLOG(5) << "Run op " << phi_op_name_ << " kernel.";
  (*(phi_kernel_))(&(kernel_context_));
}

}  // namespace framework
}  // namespace paddle
