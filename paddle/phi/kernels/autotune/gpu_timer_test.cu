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
#include <functional>
#include "glog/logging.h"
#include "paddle/phi/kernels/autotune/gpu_timer.h"
#include "paddle/phi/kernels/funcs/aligned_vector.h"

template <typename T, int VecSize>
__global__ void VecSum(T *x, T *y, int N) {
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

template <int Vecsize, int Threads, size_t Blocks>
void Algo(float *d_in, float *d_out, size_t N) {
#ifdef __HIPCC__
  hipLaunchKernelGGL(HIP_KERNEL_NAME(VecSum<float, Vecsize>),
                     dim3(Blocks),
                     dim3(Threads),
                     0,
                     0,
                     d_in,
                     d_out,
                     N);
#else
  VecSum<float, Vecsize><<<Blocks, Threads>>>(d_in, d_out, N);
#endif
}

TEST(GpuTimer, Sum) {
  float *in1, *in2, *out;
  float *d_in1, *d_in2;
  size_t N = 1 << 20;
  size_t size = sizeof(float) * N;
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

  using Functor = std::function<void(float *, float *, size_t)>;
  Functor alog0 = Algo<4, 256, 1024>;
  Functor algo1 = Algo<1, 256, 1024>;
  Functor alog2 = Algo<1, 256, 8>;

  std::vector<Functor> algos = {alog0, algo1, alog2};

  for (int j = 0; j < algos.size(); ++j) {
    auto algo = algos[j];
    phi::GpuTimer timer;
    timer.Start(0);
    algo(d_in1, d_in2, N);
    timer.Stop(0);
    VLOG(3) << "alog: " << j << " cost: " << timer.ElapsedTime() << "ms";
  }

#ifdef __HIPCC__
  hipMemcpy(out, d_in2, size, hipMemcpyDeviceToHost);
#else
  cudaMemcpy(out, d_in2, size, cudaMemcpyDeviceToHost);
#endif
  free(in1);
  free(in2);
  free(out);
#ifdef __HIPCC__
  hipFree(d_in1);
  hipFree(d_in2);
#else
  cudaFree(d_in1);
  cudaFree(d_in2);
#endif
}
