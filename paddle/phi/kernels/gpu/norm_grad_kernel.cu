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

#include "paddle/phi/kernels/norm_grad_kernel.h"

#include <algorithm>
#ifdef __NVCC__
#include "cub/cub.cuh"
#endif
#ifdef __HIPCC__
#include <hipcub/hipcub.hpp>
namespace cub = hipcub;
#endif
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/common/amp_type_traits.h"
#include "paddle/phi/common/bfloat16.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/common_shape.h"

namespace phi {

template <typename T, int BlockDim>
__global__ void NormalizeGradient(const T* x,
                                  const T* x_norm,
                                  const T* y_grad,
                                  const int pre,
                                  const int axis_n,
                                  const int post,
                                  T* x_grad) {
  using MT = typename phi::dtype::MPTypeTrait<T>::Type;
  typedef cub::BlockReduce<MT, BlockDim> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_storage_sum;
  int num = pre * post;
  for (int i = blockIdx.x; i < num; i += gridDim.x) {
    MT sum = 0.0;
    __shared__ MT row_sum;
    __shared__ MT row_sqrt_norm;
    __shared__ MT row_norm;

    auto base = (i / post) * post * axis_n + (i % post);

    for (int j = threadIdx.x; j < axis_n; j += blockDim.x) {
      int index = base + j * post;
      sum += static_cast<MT>(x[index]) * static_cast<MT>(y_grad[index]);
    }
    MT reduce_result = BlockReduce(temp_storage_sum).Sum(sum);

    if (threadIdx.x == 0) {
      row_sum = reduce_result;
      row_sqrt_norm = static_cast<MT>(x_norm[i]);
      row_norm = row_sqrt_norm * row_sqrt_norm;
    }
    __syncthreads();
    for (int j = threadIdx.x; j < axis_n; j += blockDim.x) {
      int index = base + j * post;
      const MT x_ij = static_cast<MT>(x[index]);
      const MT dy_ij = static_cast<MT>(y_grad[index]);
      x_grad[index] =
          static_cast<T>((dy_ij - x_ij * row_sum / row_norm) / row_sqrt_norm);
    }
  }
}

template <typename T, typename Context>
void NormGradKernel(const Context& ctx,
                    const DenseTensor& x,
                    const DenseTensor& norm,
                    const DenseTensor& out_grad,
                    int axis,
                    float epsilon,
                    bool is_test,
                    DenseTensor* x_grad) {
  auto* in_x = &x;
  auto* in_norm = &norm;
  auto* in_dy = &out_grad;
  auto* out_dx = x_grad;
  ctx.template Alloc<T>(out_dx);
  T* dx = out_dx->data<T>();
  const T* x_data = in_x->data<T>();
  const T* x_norm = in_norm->data<T>();
  const T* dy = in_dy->data<T>();

  auto xdim = in_x->dims();
  if (axis < 0) axis = xdim.size() + axis;
  int pre, n, post;
  funcs::GetPrePostNumel(xdim, axis, &pre, &n, &post);

  const int block = 512;
  int max_threads = ctx.GetMaxPhysicalThreadCount();
  const int max_blocks = std::max(max_threads / block, 1);
  int grid = std::min(max_blocks, pre * post);
  NormalizeGradient<T, block>
      <<<grid, block, 0, ctx.stream()>>>(x_data, x_norm, dy, pre, n, post, dx);
}

}  // namespace phi

PD_REGISTER_KERNEL(norm_grad,
                   GPU,
                   ALL_LAYOUT,
                   phi::NormGradKernel,
                   float,
                   double,
                   phi::dtype::float16,
                   phi::dtype::bfloat16) {}
