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
#include "paddle/phi/core/compat/get_kerneltype_forvar_utils.h"
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
      kernel_name, *kernel_key);
  auto phi_kernel = new phi::Kernel(kernel_result.kernel);
  return phi_kernel;
}

const phi::KernelKey GetKernelKeyforVar(pir::Operation* op,
                                        phi::KernelKey* kernel_key) {
  auto phi_kernel = GetKernel(op, *kernel_key);
  bool has_infer_varkernel_fn =
      phi_kernel && phi_kernel->get_kerneltype_forvar_fn_ != nullptr;
  if (phi_kernel && phi_kernel->IsValid() &&
      phi_kernel->GetKernelRegisteredType() ==
          phi::KernelRegisteredType::FUNCTION) {
    for (size_t i = 0; i < op->num_operands(); ++i) {
      break;
    }
  }
}
}  // namespace pir
