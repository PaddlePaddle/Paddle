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

#pragma once
#include "paddle/phi/backends/gpu/gpu_launch_config.h"
#include "paddle/phi/kernels/funcs/aligned_vector.h"
#include "paddle/phi/kernels/funcs/complex_functors.h"
#include "paddle/phi/kernels/funcs/function_traits.h"
#include "paddle/phi/kernels/primitive/kernel_primitives.h"

namespace phi {
namespace detail {
template <class F, class Tuple, std::size_t... Index>
// GCC/Clang need the decltype() return type
HOSTDEVICE constexpr decltype(auto) ComputeImpl(F &&f,
                                                Tuple &&t,
                                                std::index_sequence<Index...>) {
  return std::forward<F>(f)(std::get<Index>(std::forward<Tuple>(t))...);
}
}  // namespace detail

template <class F, class Tuple>
HOSTDEVICE constexpr decltype(auto) Compute(F &&f, Tuple &&t) {
  return detail::ComputeImpl(
      std::forward<F>(f),
      std::forward<Tuple>(t),
      std::make_index_sequence<
          std::tuple_size<std::remove_reference_t<Tuple>>::value>{});
}

template <typename outT, typename Functor, int VecSize>
__global__ void VectorizedUnaryKernel(
    const char *x, outT *out, size_t numel, size_t main_offset, Functor func) {
  size_t data_offset = BLOCK_ID_X * BLOCK_NUM_X * VecSize;
  size_t stride = BLOCK_NUM_X * GRID_NUM_X * VecSize;
  using Traits = phi::funcs::FunctionTraits<Functor>;
  using ArgsT = typename Traits::ArgsTuple;
  using Type = std::tuple_element_t<0, ArgsT>;
  ArgsT args[VecSize];
  outT result[VecSize];
  for (; data_offset < main_offset; data_offset += stride) {
    kps::ReadData<Type, VecSize, 1, ArgsT, 0, false>(
        args, reinterpret_cast<const Type *>(x) + data_offset, numel, VecSize);
#pragma unroll
    for (int idx = 0; idx < VecSize; ++idx) {
      result[idx] = static_cast<outT>(Compute(func, args[idx]));
    }
    phi::kps::WriteData<outT, VecSize, 1, false>(
        out + data_offset, &result[0], static_cast<int>(BLOCK_NUM_X * VecSize));
  }
  size_t num = numel - data_offset;
  if (num > 0) {
    kps::ReadData<Type, VecSize, 1, ArgsT, 0, true>(
        args, reinterpret_cast<const Type *>(x) + data_offset, num, VecSize);
#pragma unroll
    for (int idx = 0; idx < VecSize; ++idx) {
      result[idx] = static_cast<outT>(Compute(func, args[idx]));
    }
    phi::kps::WriteData<outT, VecSize, 1, true>(
        out + data_offset, &result[0], num);
  }
}

template <typename outT, typename Context, typename Functor>
void UnaryKernel(const KPDevice &dev_ctx,
                 const DenseTensor *x,
                 DenseTensor *out,
                 Functor func) {
  int numel = x->numel();
  outT *out_data = out->data<outT>();
  if (numel <= 0) return;
  int vec_size = phi::GetVectorizedSize(out_data);
#ifdef PADDLE_WITH_XPU_KP
  int block = 64;
  int grid = 8;
  auto stream = dev_ctx.x_context()->xpu_stream;
#else
  auto config =
      phi::backends::gpu::GetGpuLaunchConfig1D(dev_ctx, numel, vec_size);
  int grid = config.block_per_grid.x;
  int block = config.thread_per_block.x;
  auto stream = dev_ctx.stream();
#endif
  size_t main_offset = (numel / (vec_size * block)) * vec_size * block;
  constexpr bool unvector = sizeof(outT) > sizeof(float);

  auto in_data = (const char *)(x->data());
  if (unvector) {
    VectorizedUnaryKernel<outT, Functor, 1><<<grid, block, 0, stream>>>(
        in_data, out_data, numel, main_offset, func);
  } else {
    switch (vec_size) {
      case 4:
        VectorizedUnaryKernel<outT, Functor, 4><<<grid, block, 0, stream>>>(
            in_data, out_data, numel, main_offset, func);
        break;
      case 2:
        VectorizedUnaryKernel<outT, Functor, 2><<<grid, block, 0, stream>>>(
            in_data, out_data, numel, main_offset, func);
        break;
      case 1:
        VectorizedUnaryKernel<outT, Functor, 1><<<grid, block, 0, stream>>>(
            in_data, out_data, numel, main_offset, func);
        break;
      default: {
        break;
      }
    }
  }
}

}  // namespace phi
