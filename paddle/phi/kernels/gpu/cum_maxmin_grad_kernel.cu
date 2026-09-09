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

#include "paddle/phi/kernels/cum_maxmin_grad_kernel.h"

#include <numeric>

#include "paddle/common/flags.h"
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/backends/gpu/gpu_device_function.h"
#include "paddle/phi/common/amp_type_traits.h"
#include "paddle/phi/common/place.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/tensor_utils.h"
#include "paddle/phi/kernels/funcs/gather_scatter_functor.h"
#include "paddle/phi/kernels/funcs/math_function.h"

COMMON_DECLARE_bool(use_accuracy_compatible_kernel);
COMMON_DECLARE_bool(cudnn_deterministic);

namespace phi {

#ifdef PADDLE_WITH_HIP
constexpr int kCumMaxMinWarpSize = 64;
#else
constexpr int kCumMaxMinWarpSize = 32;
#endif
constexpr int kCumMaxMinIndicesPerBlock = 4;

template <typename T, typename IndexT>
__global__ void CumMaxMinScatterAddRunKernel(const IndexT* indices,
                                             const T* out_grad,
                                             T* x_grad,
                                             int64_t numel,
                                             int64_t num_irows,
                                             int64_t row_size) {
  using MT = typename phi::dtype::MPTypeTrait<T>::Type;
  int64_t p = static_cast<int64_t>(blockIdx.x) * blockDim.y + threadIdx.y;
  if (p >= numel) return;

  int64_t col = (p / num_irows) % row_size;
  int64_t target = static_cast<int64_t>(indices[p]);
  target = target < 0 ? 0 : (target >= row_size ? row_size - 1 : target);
  if (col != 0) {
    int64_t prev = static_cast<int64_t>(indices[p - num_irows]);
    prev = prev < 0 ? 0 : (prev >= row_size ? row_size - 1 : prev);
    if (prev == target) return;  // not the head of a run
  }

  int64_t num_duplicates = 1;
  while (col + num_duplicates < row_size) {
    int64_t next =
        static_cast<int64_t>(indices[p + num_duplicates * num_irows]);
    next = next < 0 ? 0 : (next >= row_size ? row_size - 1 : next);
    if (next != target) break;
    num_duplicates++;
  }

  MT gradient = static_cast<MT>(0);
  int lane_idx = threadIdx.x % kCumMaxMinWarpSize;
  int64_t num_warp_passes = num_duplicates / kCumMaxMinWarpSize;
  for (int64_t i = 0; i < num_warp_passes; ++i) {
    int64_t grad_row = p + (i * kCumMaxMinWarpSize + lane_idx) * num_irows;
    gradient += static_cast<MT>(out_grad[grad_row]);
  }
  for (int offset = kCumMaxMinWarpSize / 2; offset > 0; offset /= 2) {
    gradient +=
        backends::gpu::CudaShuffleDownSync(0xffffffff, gradient, offset);
  }

  if (lane_idx == 0) {
    for (int64_t i = num_warp_passes * kCumMaxMinWarpSize; i < num_duplicates;
         ++i) {
      gradient += static_cast<MT>(out_grad[p + i * num_irows]);
    }
    int64_t dst = p + (target - col) * num_irows;
    x_grad[dst] = static_cast<T>(static_cast<MT>(x_grad[dst]) + gradient);
  }
}

// Note: if funcs::gpu_scatter_add_kernel has deternistic implementation,
// this path can be removed.
template <typename T, typename Context>
void ScatterAddDeterministic(const Context& dev_ctx,
                             const DenseTensor& x,
                             const DenseTensor& indices,
                             const DenseTensor& out_grad,
                             int axis,
                             DataType dtype,
                             DenseTensor* x_grad) {
  auto sizes = common::vectorize(x.dims());
  if (sizes.empty()) {  // 0D tensor: gradient is a plain copy
    phi::Copy<Context>(dev_ctx, out_grad, dev_ctx.GetPlace(), false, x_grad);
    return;
  }
  int64_t row_size = sizes[axis];
  int64_t num_orows = std::accumulate(sizes.begin(),
                                      sizes.begin() + axis,
                                      static_cast<int64_t>(1),
                                      std::multiplies<int64_t>());
  int64_t num_irows = std::accumulate(sizes.begin() + axis + 1,
                                      sizes.end(),
                                      static_cast<int64_t>(1),
                                      std::multiplies<int64_t>());
  int64_t numel = num_orows * row_size * num_irows;
  if (numel == 0) return;

  dim3 block(kCumMaxMinWarpSize, kCumMaxMinIndicesPerBlock);
  dim3 grid((numel + kCumMaxMinIndicesPerBlock - 1) /
            kCumMaxMinIndicesPerBlock);
  if (dtype == DataType::INT32) {
    CumMaxMinScatterAddRunKernel<T, int32_t>
        <<<grid, block, 0, dev_ctx.stream()>>>(indices.data<int32_t>(),
                                               out_grad.data<T>(),
                                               x_grad->data<T>(),
                                               numel,
                                               num_irows,
                                               row_size);
  } else if (dtype == DataType::INT64) {
    CumMaxMinScatterAddRunKernel<T, int64_t>
        <<<grid, block, 0, dev_ctx.stream()>>>(indices.data<int64_t>(),
                                               out_grad.data<T>(),
                                               x_grad->data<T>(),
                                               numel,
                                               num_irows,
                                               row_size);
  }
}

template <typename T, typename Context>
void CummaxGradKernel(const Context& dev_ctx,
                      const DenseTensor& x,
                      const DenseTensor& indices,
                      const DenseTensor& out_grad,
                      int axis,
                      DataType dtype,
                      DenseTensor* x_grad) {
  if (x_grad && x_grad->numel() == 0) {
    dev_ctx.template Alloc<T>(x_grad);
    return;
  }
  dev_ctx.template Alloc<T>(x_grad);
  funcs::SetConstant<Context, T> functor;
  functor(dev_ctx, x_grad, static_cast<T>(0));
  if (axis < 0) {
    axis = axis + x.dims().size();
  }

  if (FLAGS_use_accuracy_compatible_kernel && FLAGS_cudnn_deterministic) {
    ScatterAddDeterministic<T, Context>(
        dev_ctx, x, indices, out_grad, axis, dtype, x_grad);
    return;
  }

  if (dtype == DataType::INT32) {
    funcs::gpu_scatter_add_kernel<T, int32_t>(
        *x_grad, axis, indices, out_grad, true, dev_ctx);
  } else if (dtype == DataType::INT64) {
    funcs::gpu_scatter_add_kernel<T, int64_t>(
        *x_grad, axis, indices, out_grad, true, dev_ctx);
  }
}

template <typename T, typename Context>
void CumminGradKernel(const Context& dev_ctx,
                      const DenseTensor& x,
                      const DenseTensor& indices,
                      const DenseTensor& out_grad,
                      int axis,
                      DataType dtype,
                      DenseTensor* x_grad) {
  if (x_grad && x_grad->numel() == 0) {
    dev_ctx.template Alloc<T>(x_grad);
    return;
  }
  dev_ctx.template Alloc<T>(x_grad);
  funcs::SetConstant<Context, T> functor;
  functor(dev_ctx, x_grad, static_cast<T>(0));
  if (axis < 0) {
    axis = axis + x.dims().size();
  }

  if (FLAGS_use_accuracy_compatible_kernel && FLAGS_cudnn_deterministic) {
    ScatterAddDeterministic<T, Context>(
        dev_ctx, x, indices, out_grad, axis, dtype, x_grad);
    return;
  }

  if (dtype == DataType::INT32) {
    funcs::gpu_scatter_add_kernel<T, int32_t>(
        *x_grad, axis, indices, out_grad, true, dev_ctx);
  } else if (dtype == DataType::INT64) {
    funcs::gpu_scatter_add_kernel<T, int64_t>(
        *x_grad, axis, indices, out_grad, true, dev_ctx);
  }
}

}  // namespace phi

PD_REGISTER_KERNEL(cummax_grad,
                   GPU,
                   ALL_LAYOUT,
                   phi::CummaxGradKernel,
                   float,
                   double,
                   int32_t,
                   int64_t) {}

PD_REGISTER_KERNEL(cummin_grad,
                   GPU,
                   ALL_LAYOUT,
                   phi::CumminGradKernel,
                   float,
                   double,
                   int32_t,
                   int64_t) {}
