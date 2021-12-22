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

#include "paddle/pten/core/kernel_factory.h"

// See Note [ Why still include the fluid headers? ]
#include "paddle/fluid/platform/enforce.h"

namespace pten {

uint32_t KernelKey::Hash::operator()(const KernelKey& key) const {
  uint32_t hash_value = 0;
  // |----31-20------|---19-12---|---11-8----|---7-0---|
  // | For extension | DataType | DataLayout | Backend |
  hash_value |= static_cast<uint8_t>(key.backend());
  hash_value |=
      (static_cast<uint8_t>(key.layout()) << KernelKey::kBackendBitLength);
  hash_value |=
      (static_cast<uint16_t>(key.dtype())
       << (KernelKey::kBackendBitLength + KernelKey::kDataLayoutBitLength));
  return hash_value;
}

KernelFactory& KernelFactory::Instance() {
  static KernelFactory g_op_kernel_factory;
  return g_op_kernel_factory;
}

Kernel KernelFactory::SelectKernel(const std::string& kernel_name,
                                   const KernelKey& kernel_key) const {
  auto iter = kernels_.find(kernel_name);
  if (iter == kernels_.end()) {
    return Kernel();
  }
  auto kernel_iter = iter->second.find(kernel_key);
  if (kernel_iter == iter->second.end()) {
    return Kernel();
  }
  return kernel_iter->second;
}

const Kernel& KernelFactory::SelectKernelOrThrowError(
    const std::string& kernel_name, const KernelKey& kernel_key) const {
  auto iter = kernels_.find(kernel_name);
  PADDLE_ENFORCE_NE(iter,
                    kernels_.end(),
                    paddle::platform::errors::NotFound(
                        "The kernel `%s` is not registered.", kernel_name));

  auto kernel_iter = iter->second.find(kernel_key);
  // TODO(chenweihang): polish refind impl here
  if (kernel_iter == iter->second.end() &&
      kernel_key.layout() != pten::DataLayout::ANY) {
    pten::KernelKey any_layout_kernel_key(
        kernel_key.backend(), pten::DataLayout::ANY, kernel_key.dtype());
    kernel_iter = iter->second.find(any_layout_kernel_key);
  }
  PADDLE_ENFORCE_NE(
      kernel_iter,
      iter->second.end(),
      paddle::platform::errors::NotFound(
          "The kernel with key %s of kernel `%s` is not registered.",
          kernel_key,
          kernel_name));

  return kernel_iter->second;
}

const Kernel& KernelFactory::SelectKernelOrThrowError(
    const std::string& kernel_name,
    Backend backend,
    DataLayout layout,
    DataType dtype) const {
  return SelectKernelOrThrowError(kernel_name,
                                  KernelKey(backend, layout, dtype));
}

std::ostream& operator<<(std::ostream& os, const Kernel& kernel) {
  os << "InputNum(" << kernel.args_def().input_defs().size() << "): [";
  for (auto& in_def : kernel.args_def().input_defs()) {
    os << "<" << in_def.backend << ", " << in_def.layout << ", " << in_def.dtype
       << ">";
  }
  os << "]), AttributeNum(" << kernel.args_def().attribute_defs().size()
     << "), OutputNum(" << kernel.args_def().output_defs().size() << ")";
  return os;
}

std::ostream& operator<<(std::ostream& os, KernelFactory& kernel_factory) {
  for (const auto& op_kernel_pair : kernel_factory.kernels()) {
    os << "- kernel name: " << op_kernel_pair.first << "\n";
    for (const auto& kernel_pair : op_kernel_pair.second) {
      os << "\t- kernel key: " << kernel_pair.first << " | "
         << "kernel: " << kernel_pair.second << "\n";
    }
  }
  return os;
}

}  // namespace pten
