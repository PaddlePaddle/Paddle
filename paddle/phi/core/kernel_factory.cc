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

#include "paddle/phi/core/kernel_factory.h"

// See Note [ Why still include the fluid headers? ]
#include "paddle/phi/core/enforce.h"

namespace phi {

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

KernelKeyMap KernelFactory::SelectKernelMap(
    const std::string& kernel_name) const {
  auto iter = kernels_.find(kernel_name);
  if (iter == kernels_.end()) {
    return KernelKeyMap();
  }
  return iter->second;
}

const Kernel& KernelFactory::SelectKernelOrThrowError(
    const std::string& kernel_name, const KernelKey& kernel_key) const {
  auto iter = kernels_.find(kernel_name);
  PADDLE_ENFORCE_NE(
      iter,
      kernels_.end(),
      phi::errors::NotFound("The kernel `%s` is not registered.", kernel_name));

  auto kernel_iter = iter->second.find(kernel_key);
  // TODO(chenweihang): polish refind impl here
  if (kernel_iter == iter->second.end() &&
      kernel_key.layout() != phi::DataLayout::ALL_LAYOUT) {
    phi::KernelKey any_layout_kernel_key(
        kernel_key.backend(), phi::DataLayout::ALL_LAYOUT, kernel_key.dtype());
    kernel_iter = iter->second.find(any_layout_kernel_key);
  }
  PADDLE_ENFORCE_NE(
      kernel_iter,
      iter->second.end(),
      phi::errors::NotFound(
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

// print kernel info with json format:
// {
//   "(CPU, Undefined(AnyLayout), complex64)": {
//   "input": ["CPU, NCHW, complex64", "CPU, NCHW, complex64"],
//   "output": ["CPU, NCHW, complex64"],
//   "attribute": ["i"]
// }
std::ostream& operator<<(std::ostream& os, const Kernel& kernel) {
  // input
  os << "{\"input\":[";
  bool need_comma = false;
  for (auto& in_def : kernel.args_def().input_defs()) {
    if (need_comma) os << ",";
    os << "\"" << in_def.backend << ", " << in_def.layout << ", "
       << in_def.dtype << "\"";
    need_comma = true;
  }
  os << "],";

  // output
  os << "\"output\":[";
  need_comma = false;
  for (auto& out_def : kernel.args_def().output_defs()) {
    if (need_comma) os << ",";
    os << "\"" << out_def.backend << ", " << out_def.layout << ", "
       << out_def.dtype << "\"";
    need_comma = true;
  }
  os << "],";

  // attr
  os << "\"attribute\":[";
  need_comma = false;
  for (auto& arg_def : kernel.args_def().attribute_defs()) {
    if (need_comma) os << ",";
    os << "\"" << arg_def.type_index.name() << "\"";
    need_comma = true;
  }
  os << "]}";

  return os;
}

// print all kernels info with json format:
// {
//  "kernel_name1":
//      [
//        {
//          "(CPU, Undefined(AnyLayout), complex64)": {
//          "input": ["CPU, NCHW, complex64", "CPU, NCHW, complex64"],
//          "output": ["CPU, NCHW, complex64"],
//          "attribute": ["i"]
//        },
//        ...
//      ],
//    "kernel_name2": []
//    ...
// }
std::ostream& operator<<(std::ostream& os, KernelFactory& kernel_factory) {
  os << "{";
  bool need_comma_kernels = false;
  for (const auto& op_kernel_pair : kernel_factory.kernels()) {
    if (need_comma_kernels) os << ",";
    os << "\"" << op_kernel_pair.first << "\":[";
    bool need_comma_per_kernel = false;
    for (const auto& kernel_pair : op_kernel_pair.second) {
      if (need_comma_per_kernel) os << ",";
      os << "{\"" << kernel_pair.first << "\":" << kernel_pair.second << "}";
      need_comma_per_kernel = true;
    }
    os << "]";
    need_comma_kernels = true;
  }
  os << "}";

  return os;
}

}  // namespace phi
