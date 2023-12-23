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

#include "paddle/phi/core/device_context.h"
#include "paddle/phi/kernels/funcs/eigen/common.h"

#if defined(__NVCC__) || defined(__HIPCC__)
#include "paddle/phi/kernels/funcs/reduce_function.h"
#include "paddle/phi/kernels/primitive/functor_primitives.h"
#include "paddle/phi/kernels/reduce_sum_kernel.h"
#ifdef __NVCC__
#include "cub/cub.cuh"
#else
#include <hipcub/hipcub.hpp>
namespace cub = hipcub;
#endif
#endif

namespace phi {
namespace funcs {

template <typename T>
void RenormFunc(const phi::CPUContext& ctx UNUSED,
                const T* x_data,
                T* out_data,
                float p,
                int dim,
                float max_norm,
                int64_t dimension_each,
                const phi::DDim& input_dims,
                int64_t numel) {
  auto dim_size = input_dims.size();
  int64_t dim_divisor = 1;
  for (int i = dim + 1; i < dim_size; i++) dim_divisor *= input_dims[i];

  std::vector<T> dim_value(dimension_each,
                           0);  // dim_value = (x1^p + x2^p + x3^p....)^(1/p)

  int64_t index = 0, dim_index = 0;
  for (int64_t i = 0; i < numel; i++) {
    dim_value[dim_index] += std::pow(std::abs(x_data[i]), p);
    index++;
    if (index == dim_divisor) {
      dim_index++;
      if (dim_index == dimension_each) {
        dim_index = 0;
      }
      index = 0;
    }
  }
  for (int64_t i = 0; i < dimension_each; i++) {
    dim_value[i] = std::pow(dim_value[i], 1.0 / p);
    if (dim_value[i] > max_norm)
      dim_value[i] = max_norm / dim_value[i];
    else
      dim_value[i] = 1.0;
  }
  index = dim_index = 0;
  for (int64_t i = 0; i < numel; i++) {
    out_data[i] = dim_value[dim_index] < 1.0 ? dim_value[dim_index] * x_data[i]
                                             : x_data[i];
    index++;
    if (index == dim_divisor) {
      dim_index++;
      if (dim_index == dimension_each) {
        dim_index = 0;
      }
      index = 0;
    }
  }
}

template <typename T>
void RenormGradFunc(const phi::CPUContext& ctx UNUSED,
                    const T* x_data,
                    const T* dout_data,
                    T* dx_data,
                    float p,
                    int dim,
                    float max_norm,
                    int64_t dimension_each,
                    const phi::DDim& input_dims,
                    int64_t numel) {
  auto dim_size = input_dims.size();
  int64_t dim_divisor = 1;
  for (int i = dim + 1; i < dim_size; i++) dim_divisor *= input_dims[i];
  std::vector<T> dim_value(dimension_each, 0), dim_power_sum(dimension_each, 0),
      weight_derivative(dimension_each, 0.0);
  int64_t index = 0, dim_index = 0;
  for (int64_t i = 0; i < numel; i++) {
    dim_value[dim_index] += std::pow(std::abs(x_data[i]), p);
    index++;
    if (index == dim_divisor) {
      dim_index++;
      if (dim_index == dimension_each) {
        dim_index = 0;
      }
      index = 0;
    }
  }
  for (int64_t i = 0; i < dimension_each; i++) {
    auto temp = std::pow(dim_value[i], 1.0 / p);
    if (temp > max_norm) {
      dim_power_sum[i] =
          std::pow(dim_value[i], (T)(-1.0 - 1.0 / p)) * -1 * max_norm;
      dim_value[i] = max_norm / temp;
    } else {
      dim_value[i] = 1.0;
    }
  }
  index = dim_index = 0;
  for (int64_t i = 0; i < numel; i++) {
    dx_data[i] = dim_value[dim_index] * dout_data[i];
    weight_derivative[dim_index] += x_data[i] * dout_data[i];
    index++;
    if (index == dim_divisor) {
      dim_index++;
      if (dim_index == dimension_each) {
        dim_index = 0;
      }
      index = 0;
    }
  }
  index = dim_index = 0;
  for (int64_t i = 0; i < numel; i++) {
    dx_data[i] += weight_derivative[dim_index] * dim_power_sum[dim_index] *
                  std::pow(std::abs(x_data[i]), p - 1.0) *
                  (x_data[i] >= 0 ? 1 : -1);
    index++;
    if (index == dim_divisor) {
      dim_index++;
      if (dim_index == dimension_each) {
        dim_index = 0;
      }
      index = 0;
    }
  }
}

#if defined(__NVCC__) || defined(__HIPCC__)
__device__ __forceinline__ float inline_pow(float base, float exponent) {
  return pow(base, exponent);
}

__device__ __forceinline__ double inline_pow(double base, double exponent) {
  return pow(base, exponent);
}

__device__ __forceinline__ float inline_abs(float x) { return abs(x); }
__device__ __forceinline__ double inline_abs(double x) { return abs(x); }

template <typename Tx, typename Ty = Tx>
struct UnsignedPowFunctor {
  HOSTDEVICE explicit inline UnsignedPowFunctor(float porder) {
    this->porder = porder;
  }
  HOSTDEVICE inline Ty operator()(const Tx x) const {
    return static_cast<Ty>(inline_pow(inline_abs(x), static_cast<Tx>(porder)));
  }
  float porder;
};

template <typename T>
__global__ void RenormKernelFunc3(int64_t size,
                                  T* dim_value,
                                  float p,
                                  float max_norm) {
  int64_t i = ((int64_t)blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < size) {
    T temp = pow(dim_value[i], (T)(1.0 / p));
    dim_value[i] = 1.0;
    if (temp > max_norm) dim_value[i] = max_norm / temp;
  }
}

template <typename T>
__global__ void RenormKernelFunc4(const T* x_data,
                                  T* out_data,
                                  int64_t size,
                                  T* dim_value,
                                  int64_t dimension_each,
                                  int64_t dim_divisor) {
  int64_t i = ((int64_t)blockIdx.x) * blockDim.x + threadIdx.x;
  auto dim_index = i / dim_divisor % dimension_each;
  if (i < size) {
    if (dim_value[dim_index] < 1.0)
      out_data[i] = dim_value[dim_index] * x_data[i];
    else
      out_data[i] = x_data[i];
  }
}

template <typename T>
__global__ void RenormElementwisePow(const T* x_data,
                                     T* pow_value,
                                     int64_t size,
                                     float p) {
  int64_t i = ((int64_t)blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < size) {
    pow_value[i] = pow(abs(x_data[i]), (T)p);
  }
}

template <typename T>
__global__ void RenormGradKernelFunc1(const T* x_data,
                                      const T* dout_data,
                                      T* pow_value,
                                      T* mul_value,
                                      int64_t size,
                                      int64_t dimension_each,
                                      float p,
                                      int64_t dim_divisor) {
  int64_t i = ((int64_t)blockIdx.x) * blockDim.x + threadIdx.x;
  auto dim_index = i / dim_divisor % dimension_each;
  if (i < size) {
    pow_value[i] = pow(abs(x_data[i]), (T)p);
    mul_value[i] = x_data[i] * dout_data[i];
  }
}

template <typename T>
__global__ void RenormGradKernelFunc2(const T* x_data,
                                      const T* dout_data,
                                      T* dx_data,
                                      int64_t size,
                                      T* dim_value,
                                      T* dim_power_sum,
                                      T* weight_derivative,
                                      int64_t dimension_each,
                                      float p,
                                      float max_norm,
                                      int64_t dim_divisor) {
  int64_t i = ((int64_t)blockIdx.x) * blockDim.x + threadIdx.x;
  auto dim_index = i / dim_divisor % dimension_each;
  if (i < dimension_each) {
    dim_power_sum[i] = 0;
    auto temp = pow(dim_value[i], (T)(1.0 / p));
    if (temp > max_norm) {
      dim_power_sum[i] = pow(dim_value[i], (T)(-1.0 - 1.0 / p)) * -1 * max_norm;
      dim_value[i] = max_norm / temp;
    } else {
      dim_value[i] = 1.0;
    }
  }
  __syncthreads();
  if (i < size) {
    dx_data[i] = dim_value[dim_index] * dout_data[i];
    dx_data[i] = dx_data[i] + weight_derivative[dim_index] *
                                  dim_power_sum[dim_index] *
                                  pow(abs(x_data[i]), T(p - 1.0)) *
                                  (x_data[i] >= 0 ? 1 : -1);
  }
}

template <typename T>
void RenormFunc(const phi::GPUContext& ctx,
                const T* x_data,
                T* out_data,
                float p,
                int dim,
                float max_norm,
                int64_t dimension_each,
                const phi::DDim& input_dims,
                int64_t numel) {
  auto dim_size = input_dims.size();
  DenseTensor pow_value, dim_value;
  int64_t dim_divisor = 1, pre_mul = 1;
  for (int i = dim + 1; i < dim_size; i++) dim_divisor *= input_dims[i];
  for (int i = 0; i < dim; i++) pre_mul *= input_dims[i];
  pow_value.Resize(common::make_ddim({pre_mul, dimension_each, dim_divisor}));
  dim_value.Resize(common::make_ddim({dimension_each}));
  T* pow_value_data = ctx.template Alloc<T>(&pow_value);
  T* dim_value_data = ctx.template Alloc<T>(&dim_value);
  auto stream = ctx.stream();
  int block = std::min(numel, static_cast<int64_t>(256));
  int grid = (numel + block - 1) / block;
  RenormElementwisePow<T>
      <<<grid, block, 0, stream>>>(x_data, pow_value_data, numel, p);
  int block2 = std::min(dimension_each, static_cast<int64_t>(256));
  int grid2 = (dimension_each + block2 - 1) / block2;
  std::vector<int> reduce_axis = {0, 2};
  phi::SumKernel<T>(
      ctx, pow_value, reduce_axis, pow_value.dtype(), false, &dim_value);

  RenormKernelFunc3<T>
      <<<grid2, block2, 0, stream>>>(numel, dim_value_data, p, max_norm);
  RenormKernelFunc4<T><<<grid, block, 0, stream>>>(
      x_data, out_data, numel, dim_value_data, dimension_each, dim_divisor);
}

template <typename T>
void RenormGradFunc(const phi::GPUContext& ctx,
                    const T* x_data,
                    const T* dout_data,
                    T* dx_data,
                    float p,
                    int dim,
                    float max_norm,
                    int64_t dimension_each,
                    const phi::DDim& input_dims,
                    int64_t numel) {
  auto dim_size = input_dims.size();
  int64_t dim_divisor = 1, pre_mul = 1;
  for (int i = dim + 1; i < dim_size; i++) dim_divisor *= input_dims[i];
  for (int i = 0; i < dim; i++) pre_mul *= input_dims[i];
  DenseTensor pow_value, mul_value, dim_value, dim_power_sum, weight_derivative;
  pow_value.Resize(common::make_ddim({pre_mul, dimension_each, dim_divisor}));
  mul_value.Resize(common::make_ddim({pre_mul, dimension_each, dim_divisor}));
  dim_value.Resize(common::make_ddim({dimension_each}));
  dim_power_sum.Resize(common::make_ddim({dimension_each}));
  weight_derivative.Resize(common::make_ddim({dimension_each}));
  auto stream = ctx.stream();
  int block = std::min(numel, static_cast<int64_t>(256));
  int grid = (numel + block - 1) / block;
  T* pow_value_data = ctx.template Alloc<T>(&pow_value);
  T* mul_value_data = ctx.template Alloc<T>(&mul_value);
  T* dim_value_data = ctx.template Alloc<T>(&dim_value);
  T* dim_power_sum_data = ctx.template Alloc<T>(&dim_power_sum);
  T* weight_derivative_data = ctx.template Alloc<T>(&weight_derivative);
  RenormGradKernelFunc1<T><<<grid, block, 0, stream>>>(x_data,
                                                       dout_data,
                                                       pow_value_data,
                                                       mul_value_data,
                                                       numel,
                                                       dimension_each,
                                                       p,
                                                       dim_divisor);
  std::vector<int> reduce_axis = {0, 2};

  phi::SumKernel<T>(
      ctx, pow_value, reduce_axis, pow_value.dtype(), false, &dim_value);
  phi::SumKernel<T>(ctx,
                    mul_value,
                    reduce_axis,
                    mul_value.dtype(),
                    false,
                    &weight_derivative);

  RenormGradKernelFunc2<T><<<grid, block, 0, stream>>>(x_data,
                                                       dout_data,
                                                       dx_data,
                                                       numel,
                                                       dim_value_data,
                                                       dim_power_sum_data,
                                                       weight_derivative_data,
                                                       dimension_each,
                                                       p,
                                                       max_norm,
                                                       dim_divisor);
}
#endif

}  // namespace funcs
}  // namespace phi
