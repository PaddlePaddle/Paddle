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

#include "paddle/phi/kernels/trunc_kernel.h"

#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/backends/gpu/gpu_info.h"
#include "paddle/phi/backends/gpu/gpu_primitives.h"
#include "paddle/phi/core/kernel_registry.h"

namespace phi {

using phi::PADDLE_CUDA_NUM_THREADS;

template <typename T>
__device__ T device_trunc(T x);

template <>
__device__ float device_trunc<float>(float x) {
  return truncf(x);
}

template <>
__device__ double device_trunc<double>(double x) {
  return trunc(x);
}

template <>
__device__ phi::dtype::float16 device_trunc<phi::dtype::float16>(
    phi::dtype::float16 x) {
  return static_cast<phi::dtype::float16>(truncf(static_cast<float>(x)));
}

template <>
__device__ phi::dtype::bfloat16 device_trunc<phi::dtype::bfloat16>(
    phi::dtype::bfloat16 x) {
  return static_cast<phi::dtype::bfloat16>(truncf(static_cast<float>(x)));
}

template <typename T>
class TruncFunctor {
 public:
  __device__ TruncFunctor(T x) : x_(x) {}
  __device__ T operator()() { return device_trunc(x_); }

 public:
  T x_;
};

template <>
class TruncFunctor<int> {
 public:
  __device__ TruncFunctor(const int x) : x_(x) {}
  __device__ int operator()() { return x_; }

 public:
  const int x_;
};

template <>
class TruncFunctor<int64_t> {
 public:
  __device__ TruncFunctor(const int64_t x) : x_(x) {}
  __device__ int64_t operator()() { return x_; }

 public:
  const int64_t x_;
};

template <typename T>
__global__ void Trunc(const T* x, T* out, int64_t N) {
  CUDA_KERNEL_LOOP(index, N) {
    TruncFunctor<T> functor(x[index]);
    out[index] = functor();
  }
}

template <typename T, typename Context>
void TruncKernel(const Context& dev_ctx,
                 const DenseTensor& x,
                 DenseTensor* out) {
  const auto* x_data = x.data<T>();
  auto* out_data = dev_ctx.template Alloc<T>(out);

  int64_t numel = x.numel();

  int theads = PADDLE_CUDA_NUM_THREADS;
  int blocks = (numel + theads - 1) / theads;

  Trunc<<<blocks, theads>>>(x_data, out_data, numel);
}

}  // namespace phi

PD_REGISTER_KERNEL(trunc,
                   GPU,
                   ALL_LAYOUT,
                   phi::TruncKernel,
                   float,
                   double,
                   int,
                   int64_t,
                   phi::dtype::float16,
                   phi::dtype::bfloat16) {}
