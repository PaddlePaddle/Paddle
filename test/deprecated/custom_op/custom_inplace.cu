// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WIdata_tHOUdata_t WARRANdata_tIES OR CONDIdata_tIONS OF ANY KIND, either
// express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <iostream>
#include <vector>

#include "paddle/extension.h"

#define CHECK_GPU_INPUT(x) \
  PADDLE_ENFORCE_EQ(       \
      x.is_gpu(), true, common::errors::Fatal(#x " must be a GPU Tensor."))

template <typename data_t>
__global__ void relu_cuda_forward_kernel(data_t* x, int64_t num) {
  int64_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  for (int64_t i = gid; i < num; i += blockDim.x * gridDim.x) {
    x[i] = x[i] > static_cast<data_t>(0.) ? x[i] : static_cast<data_t>(0.);
  }
}

void ReluForwardInplace(paddle::Tensor& x) {  // NOLINT
  CHECK_GPU_INPUT(x);

  PADDLE_ENFORCE_EQ(
      x.place() == paddle::DefaultGPUPlace(),
      true,
      common::errors::InvalidArgument("Input tensor `x` should be on GPU"));

  int64_t numel = x.numel();
  int64_t block = 512;
  int64_t grid = (numel + block - 1) / block;
  PD_DISPATCH_FLOATING_AND_HALF_TYPES(
      x.type(), "relu_cuda_forward_kernel", ([&] {
        relu_cuda_forward_kernel<data_t>
            <<<grid, block, 0, x.stream()>>>(x.data<data_t>(), numel);
      }));
}

PD_BUILD_OP(custom_relu_inplace)
    .Inputs({"X"})
    .Outputs({"Out"})
    .SetInplaceMap({{"X", "Out"}})
    .SetKernelFn(PD_KERNEL(ReluForwardInplace));
