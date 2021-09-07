//   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/tcmpt/core/kernel_factory.h"

// See Note [ Why still include the fluid headers? ]
#include "paddle/fluid/platform/enforce.h"

namespace pt {

OpKernelFactory& OpKernelFactory::Instance() {
  static OpKernelFactory g_op_kernel_factory;
  return g_op_kernel_factory;
}

bool OpKernelFactory::ContainsOperation(const char* op_type) const {
  auto iter = kernels_.find(OperationName(op_type));
  return (iter != kernels_.end());
}

const OpKernel& OpKernelFactory::SelectKernel(
    const OperationName& op_name, const OpKernelKey& kernel_key) const {
  auto iter = kernels_.find(op_name);
  PADDLE_ENFORCE_NE(iter,
                    kernels_.end(),
                    paddle::platform::errors::NotFound(
                        "The operation `%s` is not registered.", op_name));

  auto kernel_iter = iter->second.find(kernel_key);
  PADDLE_ENFORCE_NE(
      kernel_iter,
      iter->second.end(),
      paddle::platform::errors::NotFound(
          "The kernel with key %s of operation `%s` is not registered.",
          kernel_key,
          op_name));

  return kernel_iter->second;
}

const OpKernel& OpKernelFactory::SelectKernel(const OperationName& op_name,
                                              Backend backend,
                                              DataLayout layout,
                                              DataType dtype) const {
  return SelectKernel(op_name, OpKernelKey(backend, layout, dtype));
}

std::ostream& operator<<(std::ostream& os, OpKernelFactory& kernel_factory) {
  for (const auto& op_kernel_pair : kernel_factory.kernels()) {
    os << "- op: " << op_kernel_pair.first << "\n";
    for (const auto& kernel_pair : op_kernel_pair.second) {
      os << "\t- kernel: " << kernel_pair.first << "\n";
    }
  }
  return os;
}

}  // namespace pt
