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

#include <gtest/gtest.h>

#include "glog/logging.h"
#include "paddle/phi/api/lib/utils/allocator.h"
#include "paddle/phi/backends/context_pool.h"
#include "paddle/phi/backends/gpu/gpu_launch_config.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/tensor_meta.h"
#include "paddle/phi/core/tensor_utils.h"
#include "paddle/phi/kernels/autotune/auto_tune_base.h"
#include "paddle/phi/kernels/funcs/aligned_vector.h"

namespace tune = phi::autotune;

template <typename T, int VecSize>
__global__ void VecSumTest(const T* x, T* y, int N) {
#ifdef __HIPCC__
  int idx = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
#else
  int idx = blockDim.x * blockIdx.x + threadIdx.x;
#endif
  using LoadT = phi::AlignedVector<T, VecSize>;
  for (int i = idx * VecSize; i < N; i += blockDim.x * gridDim.x * VecSize) {
    LoadT x_vec;
    LoadT y_vec;
    phi::Load<T, VecSize>(&x[i], &x_vec);
    phi::Load<T, VecSize>(&y[i], &y_vec);
#pragma unroll
    for (int j = 0; j < VecSize; j++) {
      y_vec[j] = x_vec[j] + y_vec[j];
    }
    phi::Store<T, VecSize>(y_vec, &y[i]);
  }
}

template <int Vecsize>
float Algo(const phi::GPUContext& ctx,
           const phi::DenseTensor& d_in,
           phi::DenseTensor* d_out,
           size_t N,
           size_t threads,
           size_t blocks) {
  const float* d_in_data = d_in.data<float>();
  float* d_out_data = d_out->data<float>();
#ifdef __HIPCC__
  hipLaunchKernelGGL(HIP_KERNEL_NAME(VecSumTest<float, Vecsize>),
                     dim3(blocks),
                     dim3(threads),
                     0,
                     0,
                     d_in_data,
                     d_out_data,
                     N);
#else
  VLOG(3) << "Vecsize is " << Vecsize;
  VecSumTest<float, Vecsize>
      <<<blocks, threads, 0, ctx.stream()>>>(d_in_data, d_out_data, N);
#endif
  return Vecsize;
}

TEST(AutoTune, sum) {
  int64_t N = 1 << 20;
  size_t blocks = 512;
  size_t threads = 256;
  size_t size = sizeof(float) * N;

  const auto alloc_cpu =
      std::make_unique<paddle::experimental::DefaultAllocator>(phi::CPUPlace());
  auto in1 = std::make_shared<phi::DenseTensor>(
      alloc_cpu.get(),
      phi::DenseTensorMeta(phi::DataType::FLOAT32,
                           common::make_ddim({N}),
                           phi::DataLayout::NCHW));
  auto in2 = std::make_shared<phi::DenseTensor>(
      alloc_cpu.get(),
      phi::DenseTensorMeta(phi::DataType::FLOAT32,
                           common::make_ddim({N}),
                           phi::DataLayout::NCHW));

  float* in1_data = in1->data<float>();
  float* in2_data = in2->data<float>();
  for (size_t i = 0; i < N; i++) {
    in1_data[i] = 1.0f;
    in2_data[i] = 2.0f;
  }
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
  const auto alloc_cuda =
      std::make_unique<paddle::experimental::DefaultAllocator>(phi::GPUPlace());
  phi::DeviceContextPool& pool = phi::DeviceContextPool::Instance();
  auto place = phi::GPUPlace();
  auto* dev_ctx = static_cast<const phi::GPUContext*>(pool.GetByPlace(place));
  auto stream = dev_ctx->stream();

  auto d_in1 = std::make_shared<phi::DenseTensor>(
      alloc_cuda.get(),
      phi::DenseTensorMeta(phi::DataType::FLOAT32,
                           common::make_ddim({N}),
                           phi::DataLayout::NCHW));
  auto d_in2 = std::make_shared<phi::DenseTensor>(
      alloc_cuda.get(),
      phi::DenseTensorMeta(phi::DataType::FLOAT32,
                           common::make_ddim({N}),
                           phi::DataLayout::NCHW));
  phi::Copy(*dev_ctx, *in1.get(), phi::GPUPlace(), false, d_in1.get());
  phi::Copy(*dev_ctx, *in2.get(), phi::GPUPlace(), false, d_in2.get());

  // 1. Test call_back.
  VLOG(3) << ">>> [CallBack]: Test case.";
  auto callback1 = tune::MakeCallback<float>(Algo<4>);
  auto callback2 = tune::MakeCallback<float>(Algo<2>);
  auto callback3 = tune::MakeCallback<float>(Algo<1>);
  std::vector<decltype(callback1)> callbacks{callback1, callback2, callback3};
  for (int i = 0; i < callbacks.size(); ++i) {
    dev_ctx->Wait();
    phi::GpuTimer timer;
    timer.Start(0);
    callbacks[i].Run(*dev_ctx, *d_in1.get(), d_in2.get(), N, threads, blocks);
    timer.Stop(0);
    VLOG(3) << "kernel[" << i << "]: time cost is " << timer.ElapsedTime();
  }
#endif
}
