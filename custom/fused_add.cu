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

#include <iostream>

#include "paddle/extension.h"
#include "paddle/phi/backends/context_pool.h"
#include "paddle/phi/common/data_type.h"
#include "paddle/phi/common/place.h"
#include "paddle/phi/kernels/funcs/aligned_vector.h"

template <int64_t VecSize>
__global__ void fused_add_cuda_forward(const float* x,
                                       const phi::bfloat16* y,
                                       float* out,
                                       int64_t numel) {
  int64_t i =
      (threadIdx.x + static_cast<int64_t>(blockIdx.x) * blockDim.x) * VecSize;
  int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x * VecSize;

  for (; i + VecSize <= numel; i += stride) {
    phi::AlignedVector<float, VecSize> x_vec;
    phi::AlignedVector<phi::bfloat16, VecSize> y_vec;
    phi::AlignedVector<float, VecSize> out_vec;
    phi::Load(x + i, &x_vec);
    phi::Load(y + i, &y_vec);
#pragma unroll
    for (int j = 0; j < VecSize; ++j) {
      out_vec[j] = x_vec[j] + static_cast<float>(y_vec[j]);
    }
    phi::Store(out_vec, out + i);
  }

  for (; i < numel; ++i) {
    out[i] = x[i] + static_cast<float>(y[i]);
  }
  //
  // int64_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  // for (int64_t i = gid; i < numel; i += gridDim.x * blockDim.x) {
  //     out[i] = x[i] + static_cast<float>(y[i]);
  // }
}

__global__ void fused_add_cuda_backward(const float* out_grad,
                                        float* x_grad,
                                        phi::bfloat16* y_grad,
                                        int64_t numel) {
  int64_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  for (int64_t i = gid; i < numel; i += gridDim.x * blockDim.x) {
    x_grad[i] = out_grad[i];
    y_grad[i] = static_cast<phi::bfloat16>(out_grad[i]);
  }
}

// x.dtype is float32 and the y.dtype is bfloat16
std::vector<paddle::Tensor> fused_add_forward(const paddle::Tensor& x,
                                              const paddle::Tensor& y) {
  int64_t numel = x.numel();
  auto out = paddle::empty_like(x);
  auto place = phi::GPUPlace();
  // auto *context = static_cast<phi::GPUContext *>(
  //     phi::DeviceContextPool::Instance().Get(place));
  // int x_vec_size = phi::GetVectorizedSize(x.data<float>());
  // int y_vec_size = phi::GetVectorizedSize(y.data<phi::bfloat16>());
  // int out_vec_size = phi::GetVectorizedSize(out.data<float>());
  // int vec_size = min(x_vec_size, min(y_vec_size, out_vec_size));
  // phi::backends::gpu::::GpuLaunchConfig config =
  // phi::backends::gpu::::GetGpuLaunchConfig1D(*context, numel, vec_size); auto
  // threads = config.GetBlockSize(); auto blocks = config.block_per_grid;
  int vec_size = 4;
  int64_t block = 512;
  int64_t grid = ((numel + vec_size - 1) / vec_size + block - 1) / block;
  // std::cout << "grid:" << grid << ", block:" << block << ",x:" <<
  // x.data<float>() << ", y:" << y.data<phi::bfloat16>() << ", out:" <<
  // out.data<float>() << std::endl;
  fused_add_cuda_forward<4><<<grid, block, 0, x.stream()>>>(
      x.data<float>(), y.data<phi::bfloat16>(), out.data<float>(), numel);

  return {out};
}

std::vector<paddle::Tensor> fused_add_backward(const paddle::Tensor& x,
                                               const paddle::Tensor& y,
                                               const paddle::Tensor& out,
                                               const paddle::Tensor& out_grad) {
  auto x_grad = paddle::empty_like(x);
  auto y_grad = paddle::empty_like(y, paddle::DataType::BFLOAT16);

  int64_t numel = out_grad.numel();
  int64_t block = 512;
  int64_t grid = (numel + block - 1) / block;
  fused_add_cuda_backward<<<grid, block, 0, x.stream()>>>(
      out_grad.data<float>(),
      x_grad.data<float>(),
      y_grad.data<phi::bfloat16>(),
      numel);

  return {x_grad, y_grad};
}
