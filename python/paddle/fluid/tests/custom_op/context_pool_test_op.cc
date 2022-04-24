// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/extension.h"
#include "paddle/phi/backends/all_context.h"

#define CHECK_INPUT(x) \
  PD_CHECK(x.place() == paddle::PlaceType::kCPU, #x " must be a CPU Tensor.")

std::vector<paddle::Tensor> ContextPoolTest(const paddle::Tensor& x) {
  // 1. test cpu context
  paddle::Place cpu_place(paddle::experimental::AllocationType::CPU);
  auto* cpu_ctx =
      paddle::experimental::DeviceContextPool::Instance()
          .Get<paddle::experimental::AllocationType::CPU>(cpu_place);
  PD_CHECK(cpu_ctx->GetPlace() == cpu_place);
  // if want to use the eigen_device here, need to include eigen headers
  auto* cpu_eigen_device = cpu_ctx->eigen_device();
  PD_CHECK(cpu_eigen_device != nullptr);

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
  // 2. test gpu context
  paddle::Place gpu_place(paddle::experimental::AllocationType::GPU);
  auto* gpu_ctx =
      paddle::experimental::DeviceContextPool::Instance()
          .Get<paddle::experimental::AllocationType::GPU>(gpu_place);
  PD_CHECK(gpu_ctx->GetPlace() == gpu_place);
  // if want to use the eigen_device here, need to include eigen headers
  auto* gpu_eigen_device = gpu_ctx->eigen_device();
  PD_CHECK(gpu_eigen_device != nullptr);
#endif

  return {x};
}

PD_BUILD_OP(context_pool_test)
    .Inputs({"X"})
    .Outputs({"Out"})
    .SetKernelFn(PD_KERNEL(ContextPoolTest));
