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

#include "paddle/infrt/kernel/basic_kernels.h"

#include <iostream>
#include <string>

#include "llvm/Support/raw_ostream.h"
#include "paddle/infrt/host_context/kernel_registry.h"
#include "paddle/infrt/host_context/kernel_utils.h"

using infrt::host_context::Attribute;

namespace infrt {
namespace kernel {

template <typename T>
T add(T a, T b) {
  return a + b;
}

template <typename T>
T sub(T a, T b) {
  return a - b;
}

template <typename T>
T mul(T a, T b) {
  return a * b;
}

template <typename T>
T div(T a, T b) {
  return a / b;
}

template <typename T>
void print(T a) {
  std::cout << a << std::endl;
}

static std::string GetString(Attribute<std::string> value) {
  return value.get();
}

static void PrintString(const std::string &str) {
  llvm::outs() << "string = " << str << '\n';
  llvm::outs().flush();
}

void RegisterBasicKernels(host_context::KernelRegistry *registry) {
  RegisterIntBasicKernels(registry);
  RegisterFloatBasicKernels(registry);
  registry->AddKernel("infrt.get_string", INFRT_KERNEL(GetString));
  registry->AddKernel("infrt.print_string", INFRT_KERNEL(PrintString));
}

void RegisterIntBasicKernels(host_context::KernelRegistry *registry) {
  registry->AddKernel("infrt.add.i32", INFRT_KERNEL(add<int32_t>));
  registry->AddKernel("infrt.sub.i32", INFRT_KERNEL(sub<int32_t>));
  registry->AddKernel("infrt.mul.i32", INFRT_KERNEL(mul<int32_t>));
  registry->AddKernel("infrt.div.i32", INFRT_KERNEL(div<int32_t>));
  registry->AddKernel("infrt.print.i32", INFRT_KERNEL(print<int32_t>));
}

void RegisterFloatBasicKernels(host_context::KernelRegistry *registry) {
  registry->AddKernel("infrt.add.f32", INFRT_KERNEL(add<float>));
  registry->AddKernel("infrt.sub.f32", INFRT_KERNEL(sub<float>));
  registry->AddKernel("infrt.mul.f32", INFRT_KERNEL(mul<float>));
  registry->AddKernel("infrt.div.f32", INFRT_KERNEL(div<float>));
  registry->AddKernel("infrt.print.f32", INFRT_KERNEL(print<float>));
}

}  // namespace kernel
}  // namespace infrt
