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

#include <iostream>

#include "paddle/infrt/host_context/kernel_registry.h"
#include "paddle/infrt/host_context/kernel_utils.h"

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

void RegisterKernels(infrt::host_context::KernelRegistry *registry) {
  // int32
  registry->AddKernel("external.add.i32", INFRT_KERNEL(add<int32_t>));
  registry->AddKernel("external.sub.i32", INFRT_KERNEL(sub<int32_t>));
  registry->AddKernel("external.mul.i32", INFRT_KERNEL(mul<int32_t>));
  registry->AddKernel("external.div.i32", INFRT_KERNEL(div<int32_t>));
  registry->AddKernel("external.print.i32", INFRT_KERNEL(print<int32_t>));

  // float
  registry->AddKernel("external.add.f32", INFRT_KERNEL(add<float>));
  registry->AddKernel("external.sub.f32", INFRT_KERNEL(sub<float>));
  registry->AddKernel("external.mul.f32", INFRT_KERNEL(mul<float>));
  registry->AddKernel("external.div.f32", INFRT_KERNEL(div<float>));
  registry->AddKernel("external.print.f32", INFRT_KERNEL(print<float>));
}
