// Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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
#include <bitset>
#include <iostream>
#include <random>

#define PADDLE_CUDA_FP16
#include "paddle/fluid/platform/cuda_device_function.h"
#include "paddle/fluid/platform/cuda_primitives.h"
#include "paddle/fluid/platform/float16.h"

using paddle::platform::PADDLE_CUDA_NUM_THREADS;
using paddle::platform::float16;

#define CUDA_ATOMIC_KERNEL(op, T)                                      \
  __global__ void op##Kernel(const T* data_a, T* data_b, size_t num) { \
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < num;       \
         i += blockDim.x * gridDim.x) {                                \
      paddle::platform::CudaAtomic##op(&data_b[i], data_a[i]);         \
    }                                                                  \
  }

template <typename T>
struct AddFunctor {
  T operator()(const T& a, const T& b) { return a + b; }
};

template <typename T>
struct SubFunctor {
  T operator()(const T& a, const T& b) { return a - b; }
};

// NOTE(dzhwinter): the float16 add has small underflow/overflow
// so we use EXPECT_NEAR to check the result.
#define ARITHMETIC_KERNEL_LAUNCH(op, T)                                 \
  void Test##T##op(size_t num) {                                        \
    T *in1, *in2, *out;                                                 \
    T *d_in1, *d_in2;                                                   \
    size_t size = sizeof(T) * num;                                      \
    cudaMalloc(reinterpret_cast<void**>(&d_in1), size);                 \
    cudaMalloc(reinterpret_cast<void**>(&d_in2), size);                 \
    in1 = reinterpret_cast<T*>(malloc(size));                           \
    in2 = reinterpret_cast<T*>(malloc(size));                           \
    out = reinterpret_cast<T*>(malloc(size));                           \
    std::minstd_rand engine;                                            \
    std::uniform_real_distribution<double> dist(0.0, 1.0);              \
    for (size_t i = 0; i < num; ++i) {                                  \
      in1[i] = static_cast<T>(dist(engine));                            \
      in2[i] = static_cast<T>(dist(engine));                            \
    }                                                                   \
    cudaMemcpy(d_in1, in1, size, cudaMemcpyHostToDevice);               \
    cudaMemcpy(d_in2, in2, size, cudaMemcpyHostToDevice);               \
    op##Kernel<<<1, PADDLE_CUDA_NUM_THREADS>>>(d_in1, d_in2, num);      \
    cudaDeviceSynchronize();                                            \
    cudaMemcpy(out, d_in2, size, cudaMemcpyDeviceToHost);               \
    cudaDeviceSynchronize();                                            \
    for (size_t i = 0; i < num; ++i) {                                  \
      EXPECT_NEAR(static_cast<float>(out[i]),                           \
                  static_cast<float>(op##Functor<T>()(in1[i], in2[i])), \
                  0.001);                                               \
    }                                                                   \
    free(in1);                                                          \
    free(in2);                                                          \
    free(out);                                                          \
    cudaFree(d_in1);                                                    \
    cudaFree(d_in2);                                                    \
  }
CUDA_ATOMIC_KERNEL(Add, float);
CUDA_ATOMIC_KERNEL(Add, double);
CUDA_ATOMIC_KERNEL(Add, float16);

ARITHMETIC_KERNEL_LAUNCH(Add, float);
ARITHMETIC_KERNEL_LAUNCH(Add, double);
ARITHMETIC_KERNEL_LAUNCH(Add, float16);

namespace paddle {
namespace platform {
USE_CUDA_ATOMIC(Sub, int);
};
};
CUDA_ATOMIC_KERNEL(Sub, int);
ARITHMETIC_KERNEL_LAUNCH(Sub, int);

// cuda primitives
TEST(CudaAtomic, Add) {
  TestfloatAdd(static_cast<size_t>(10));
  TestfloatAdd(static_cast<size_t>(1024 * 1024));
  TestdoubleAdd(static_cast<size_t>(10));
  TestdoubleAdd(static_cast<size_t>(1024 * 1024));
}

TEST(CudaAtomic, Sub) {
  TestintSub(static_cast<size_t>(10));
  TestintSub(static_cast<size_t>(1024 * 1024));
}

TEST(CudaAtomic, float16) {
  using paddle::platform::float16;
  Testfloat16Add(static_cast<size_t>(1));
  Testfloat16Add(static_cast<size_t>(2));
  Testfloat16Add(static_cast<size_t>(3));

  Testfloat16Add(static_cast<size_t>(10));
  Testfloat16Add(static_cast<size_t>(1024 * 1024));
}
