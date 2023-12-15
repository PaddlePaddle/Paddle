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

#include "paddle/common/ddim.h"
#include "paddle/phi/api/include/tensor.h"
#include "paddle/phi/backends/gpu/gpu_launch_config.h"
#include "paddle/phi/core/enforce.h"
#include "paddle/phi/kernels/funcs/elementwise_base.h"

namespace phi {

template <typename InT, typename OutT>
struct CastFunctor {
  HOSTDEVICE OutT operator()(InT x) const { return static_cast<OutT>(x); }
};
template <typename InT, typename OutT, int VecSize>
static void VecCastKernel(const phi::GPUContext &ctx,
                          const InT *x,
                          OutT *y,
                          size_t n) {
  auto config = phi::backends::gpu::GetGpuLaunchConfig1D(ctx, n, VecSize);
  auto block = config.GetGridSize();
  auto thread = config.GetBlockSize();
  auto main_offset = n / (VecSize * thread) * VecSize * thread;
  auto stream = ctx.stream();
  using FunctorT = CastFunctor<InT, OutT>;
  Array<const _ptr_ char *__restrict__, 1> in_arr;
  in_arr[0] = reinterpret_cast<const _ptr_ char *>(x);
  Array<_ptr_ OutT *, 1> out_arr;
  out_arr[0] = y;
  phi::funcs::VectorizedElementwiseKernel<OutT, FunctorT, 1, 1, VecSize>
      <<<block, thread, 0, stream>>>(
          in_arr, out_arr, n, main_offset, VecSize, FunctorT());
}

template <typename InT, typename OutT>
static void LaunchCastKernel(const phi::GPUContext &ctx,
                             const InT *x,
                             OutT *y,
                             size_t n) {
  if (n == 0) return;
  PADDLE_ENFORCE_NE(
      static_cast<const void *>(x),
      static_cast<void *>(y),
      errors::InvalidArgument("Inplace cast is not supported yet."));
  int vec_size = std::min(phi::GetVectorizedSize(x), phi::GetVectorizedSize(y));
  switch (vec_size) {
    case 4:
      return VecCastKernel<InT, OutT, 4>(ctx, x, y, n);
    case 2:
      return VecCastKernel<InT, OutT, 2>(ctx, x, y, n);
    case 1:
      return VecCastKernel<InT, OutT, 1>(ctx, x, y, n);
    default:
      PADDLE_THROW(
          errors::InvalidArgument("The vectorized size must be 1, 2 or 4."));
  }
}

}  // namespace phi
