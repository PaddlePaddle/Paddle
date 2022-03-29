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
#include "paddle/fluid/memory/allocation/allocator_facade.h"
#include "paddle/phi/backends/gpu/gpu_launch_config.h"
#include "paddle/phi/kernels/autotune/auto_tune_base.h"
#include "paddle/phi/kernels/funcs/aligned_vector.h"

namespace tune = phi::autotune;

template <typename T, int VecSize>
__global__ void VecSumTest(T *x, T *y, int N) {
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
void Algo(const phi::GPUContext &ctx,
          float *d_in,
          float *d_out,
          size_t N,
          size_t threads,
          size_t blocks) {
#ifdef __HIPCC__
  hipLaunchKernelGGL(HIP_KERNEL_NAME(VecSumTest<float, Vecsize>),
                     dim3(blocks),
                     dim3(threads),
                     0,
                     0,
                     d_in,
                     d_out,
                     N);
#else
  VLOG(3) << "Vecsize is " << Vecsize;
  VecSumTest<float, Vecsize><<<blocks, threads, 0, ctx.stream()>>>(
      d_in, d_out, N);
#endif
}

TEST(AutoTune, sum) {
  float *in1, *in2, *out;
  float *d_in1, *d_in2;
  size_t N = 1 << 20;
  size_t size = sizeof(float) * N;
  size_t threads = 256;
  size_t blocks = 512;

#ifdef __HIPCC__
  hipMalloc(reinterpret_cast<void **>(&d_in1), size);
  hipMalloc(reinterpret_cast<void **>(&d_in2), size);
#else
  cudaMalloc(reinterpret_cast<void **>(&d_in1), size);
  cudaMalloc(reinterpret_cast<void **>(&d_in2), size);
#endif
  in1 = reinterpret_cast<float *>(malloc(size));
  in2 = reinterpret_cast<float *>(malloc(size));
  out = reinterpret_cast<float *>(malloc(size));
  for (size_t i = 0; i < N; i++) {
    in1[i] = 1.0f;
    in2[i] = 2.0f;
  }

#ifdef __HIPCC__
  hipMemcpy(d_in1, in1, size, hipMemcpyHostToDevice);
  hipMemcpy(d_in2, in2, size, hipMemcpyHostToDevice);
#else
  cudaMemcpy(d_in1, in1, size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_in2, in2, size, cudaMemcpyHostToDevice);
#endif

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
  phi::GPUPlace place;
  phi::GPUContext ctx(place);
  ctx.PartialInitWithoutAllocator();
  ctx.SetAllocator(paddle::memory::allocation::AllocatorFacade::Instance()
                       .GetAllocator(place, ctx.stream())
                       .get());
  ctx.PartialInitWithAllocator();

  // 1. Test call_back.
  VLOG(3) << "[CallBack]: Test case.";
  auto callback1 = tune::MakeCallback(Algo<4>);
  auto callback2 = tune::MakeCallback(Algo<2>);
  auto callback3 = tune::MakeCallback(Algo<1>);
  std::vector<decltype(callback1)> callbacks{callback1, callback2, callback3};
  for (int i = 0; i < callbacks.size(); ++i) {
    ctx.Wait();
    phi::GpuTimer timer;
    timer.Start(0);
    callbacks[i].Call(ctx, d_in1, d_in2, N, threads, blocks);
    timer.Stop(0);
    VLOG(3) << "kernel[" << i << "]: time cost is " << timer.ElapsedTime();
  }

  // 2. Test call_back tune.
  VLOG(3) << "[AutoTune]: Test case.";
  auto tuner = tune::MakeAutoTune(Algo<4>);
  tuner.AddCallBack(tune::MakeCallback(Algo<2>));
  tuner.AddCallBack(tune::MakeCallback(Algo<1>));

  /* The 1st ctx works for ctx.Wait(),
     the 2nd is just the param of call_back function. */
  auto best_call_back =
      tuner.PickBestKernel(ctx, ctx, d_in1, d_in2, N, threads, blocks);
  best_call_back.Call(ctx, d_in1, d_in2, N, threads, blocks);

  ctx.Wait();
  phi::GpuTimer timer;
  timer.Start(0);
  best_call_back.Call(ctx, d_in1, d_in2, N, threads, blocks);
  timer.Stop(0);
  VLOG(3) << "Best CallBackKernel time cost is " << timer.ElapsedTime();
#endif

#ifdef __HIPCC__
  hipMemcpy(out, d_in2, size, hipMemcpyDeviceToHost);
  hipFree(d_in1);
  hipFree(d_in2);
#else
  cudaMemcpy(out, d_in2, size, cudaMemcpyDeviceToHost);
  cudaFree(d_in1);
  cudaFree(d_in2);
#endif

  free(in1);
  free(in2);
  free(out);
}
