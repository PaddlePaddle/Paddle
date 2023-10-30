// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/pir/transforms/data_transfer_utils.h"
#include "paddle/fluid/framework/data_transform.h"
#include "paddle/pir/core/builtin_attribute.h"
#include "paddle/pir/core/op_info.h"
#include "paddle/pir/core/operation.h"
#include "paddle/pir/core/operation_utils.h"
#include "paddle/pir/core/type.h"

namespace pir {
phi::Kernel* GetKernel(pir::Operation* op, const phi::KernelKey& kernel_key) {
  auto& op_attributes = op->attributes();
  auto kernel_name =
      op_attributes.at("kernel_name").dyn_cast<pir::StrAttribute>().AsString();
  auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
      kernel_name, kernel_key);
  auto phi_kernel = new phi::Kernel(kernel_result.kernel);
  return phi_kernel;
}

const phi::KernelKey GetKernelKeyforVar(pir::Operation* op,
                                        const std::string& var_name,
                                        const phi::DenseTensor& tensor,
                                        phi::KernelKey* expected_kernel_key) {
  auto phi_kernel = GetKernel(op, *expected_kernel_key);
  // auto pd_op = op->dyn_cast<ConcreteOp>;
  // auto kernel_type_for_var = pd_op.GetKernelKeyforVar(var_name, tensor,
  // *kernel_key);
  bool has_infer_varkernel_fn =
      phi_kernel && phi_kernel->get_kerneltype_forvar_fn_ != nullptr;

  pir::IrContext* ctx = pir::IrContext::Instance();
  pir::OpInfo op_info = ctx->GetRegisteredOpInfo(op->name());

  auto get_kernel_type_for_var_interface =
      op_info.GetInterfaceImpl<paddle::dialect::GetKernelTypeForVarInterface>();
  PADDLE_ENFORCE_NOT_NULL(
      get_kernel_type_for_var_interface,
      phi::errors::PreconditionNotMet(
          "can not find GetKernelTypeForVarInterface from [%s]", op->name()));

  phi::KernelKey kernel_type_for_var =
      get_kernel_type_for_var_interface->get_kernel_type_for_var_(
          var_name, tensor, expected_kernel_key);
  return kernel_type_for_var;

  //   phi::GetKernelTypeForVarContext infer_varkernel_context =
  //       BuildGetKernelTypeForVarContext(expected_kernel_key,
  //                                       fluid_attrs,
  //                                       &infer_attrs,
  //                                       has_infer_varkernel_fn);
  //   if (has_infer_varkernel_fn) {
  //     VLOG(2) << "use infer_varkernel_fn to get kernel key for var";
  //     infer_varkernel_context.SetVarName(const_cast<std::string*>(&var_name));
  //     infer_varkernel_context.SetDenseTensor(const_cast<phi::DenseTensor*>(tensor_in));
  //     kernel_type_for_var =
  //         phi_kernel_->get_kerneltype_forvar_fn_(&infer_varkernel_context);
  //   }

  //   auto op = TransferDtype(
  //         *src_var_name,
  //         new_var_name,
  //         framework::TransToProtoVarType(kernel_type_for_var.dtype()),
  //         framework::TransToProtoVarType(expected_kernel_key.dtype()),
  //         var_scope_,
  //         scope_);
  //     if (op) {
  //       RunAndConstructOpFuncNode(
  //           op, *src_var_name, *new_var_name, op_func_nodes, static_build);
  //     }
}
}  // namespace pir
