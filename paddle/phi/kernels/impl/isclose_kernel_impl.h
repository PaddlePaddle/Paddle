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

#pragma once
#include <cmath>
#include <string>

#include "paddle/fluid/framework/data_type.h"
#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/common/data_type.h"
#include "paddle/phi/common/place.h"
#include "paddle/phi/core/dense_tensor.h"

// TODO(xiongkun): remove the header when decouple the memcpy function in phi.
#include "paddle/fluid/memory/memcpy.h"

namespace phi {
using Tensor = DenseTensor;
template <typename DeviceContext, typename T>
struct GetTensorValue {
  T operator()(const DeviceContext& ctx, const DenseTensor& tensor) const;
};

template <typename DeviceContext, typename T>
struct IscloseFunctor {
  void operator()(const DeviceContext& ctx,
                  const DenseTensor& in,
                  const DenseTensor& other,
                  const float rtol,
                  const float atol,
                  bool equal_nan,
                  DenseTensor* output);
};

template <typename T>
struct GetTensorValue<phi::CPUContext, T> {
  T operator()(const phi::CPUContext& dev_ctx,
               const DenseTensor& tensor) const {
    return *(tensor.data<T>());
  }
};

template <typename T>
struct GetTensorValue<phi::GPUContext, T> {
  T operator()(const phi::GPUContext& dev_ctx,
               const DenseTensor& tensor) const {
    const T* data = tensor.data<T>();
    T value;
    const auto gpu_place = dev_ctx.GetPlace();
    paddle::memory::Copy(
        phi::CPUPlace(), &value, gpu_place, data, sizeof(T), dev_ctx.stream());
    return value;
  }
};

template <typename T>
struct IscloseFunctor<phi::CPUContext, T> {
  void operator()(const phi::CPUContext& ctx,
                  const DenseTensor& in,
                  const DenseTensor& other,
                  const double rtol,
                  const double atol,
                  bool equal_nan,
                  DenseTensor* output) {
    auto* in_a = in.data<T>();
    auto* in_b = other.data<T>();
    auto* out_data = ctx.template Alloc<bool>(output);
    auto num = in.numel();
    // *out_data = true;
    for (int i = 0; i < num; i++) {
      out_data[i] = true;
    }
    for (int i = 0; i < num; i++) {
      const T a = in_a[i], b = in_b[i];
      bool val;
      if (std::isnan(a) || std::isnan(b)) {
        val = equal_nan && std::isnan(a) == std::isnan(b);
      } else {
        T left = (a > b ? a - b : b - a);
        T right = atol + (b > 0 ? rtol * b : (-rtol) * b);
        T diff = (left > right ? left - right : right - left);
        val = a == b || left <= right || diff <= 1e-15;
      }
      // *out_data &= val;
      out_data[i] = val;
    }
  }
};

#if defined(__NVCC__) || defined(__HIPCC__)
template <typename T>
__global__ void IscloseCUDAKernel(const T* in_data,
                                  const T* other_data,
                                  const double rtol,
                                  const double atol,
                                  bool equal_nan,
                                  int num,
                                  bool* out_data) {
  unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
  bool val;
  for (int i = idx; i < num; i += blockDim.x * gridDim.x) {
    const T a = in_data[i], b = other_data[i];
    if (isnan(a) || isnan(b)) {
      val = equal_nan && isnan(a) == isnan(b);
    } else {
      T left = (a > b ? a - b : b - a);
      T right = atol + (b > 0 ? rtol * b : (-rtol) * b);
      T diff = (left > right ? left - right : right - left);
      val = a == b || left <= right || diff <= 1e-15;
    }
    out_data[i] = val;
    // if (!val) *out_data = false;
  }
}

template <typename T>
struct IscloseFunctor<phi::GPUContext, T> {
  void operator()(const phi::GPUContext& dev_ctx,
                  const DenseTensor& in,
                  const DenseTensor& other,
                  const double rtol,
                  const double atol,
                  bool equal_nan,
                  DenseTensor* output) {
    int num = in.numel();
    const T* in_data = in.data<T>();
    const T* other_data = other.data<T>();
    bool* out_data = dev_ctx.template Alloc<bool>(output);
    int block = 1024;
    int grid = (block - 1 + num) / block;
    grid = (grid > block) ? block : grid;
#ifdef PADDLE_WITH_HIP
    hipMemset(out_data, true, num * sizeof(bool));
#else
    cudaMemset(out_data, true, num * sizeof(bool));
#endif
    IscloseCUDAKernel<T><<<grid, block, 0, dev_ctx.stream()>>>(
        in_data, other_data, rtol, atol, equal_nan, num, out_data);
  }
};
#endif

template <typename T, typename Context>
void IscloseKernel(const Context& dev_ctx,
                   const DenseTensor& x,
                   const DenseTensor& y,
                   const Scalar& rtol,
                   const Scalar& atol,
                   bool equal_nan,
                   DenseTensor* out) {
  PADDLE_ENFORCE_EQ(
      atol.dtype(),
      DataType::FLOAT64,
      phi::errors::InvalidArgument("Input(Atol) type must be double"));

  PADDLE_ENFORCE_EQ(
      rtol.dtype(),
      DataType::FLOAT64,
      phi::errors::InvalidArgument("Input(Rtol) type must be double"));

  IscloseFunctor<Context, T>()(
      dev_ctx, x, y, rtol.to<double>(), atol.to<double>(), equal_nan, out);
}
}  // namespace phi
