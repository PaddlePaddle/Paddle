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

#include <algorithm>
#include "paddle/pten/kernels/norm_kernel.h"
#ifdef __NVCC__
#include "cub/cub.cuh"
#endif
#ifdef __HIPCC__
#include <hipcub/hipcub.hpp>
namespace cub = hipcub;
#endif
#include "paddle/fluid/operators/amp/fp16_type_traits.h"
#include "paddle/pten/common/float16.h"

#include "paddle/pten/backends/gpu/gpu_context.h"
#include "paddle/pten/core/kernel_registry.h"

#include "paddle/pten/kernels/funcs/common_shape.h"

namespace pten {

__device__ __forceinline__ dtype::float16 square_root(dtype::float16 x) {
  return static_cast<dtype::float16>(sqrtf(static_cast<float>(x)));
}

__device__ __forceinline__ float square_root(float x) { return sqrtf(x); }

__device__ __forceinline__ double square_root(double x) { return sqrt(x); }

template <typename T, int BlockDim>
__global__ void Normalize(const T* x,
                          const int pre,
                          const int axis_n,  // dim in axis
                          const int post,
                          const T eps,
                          T* y,
                          T* out_norm) {
  using MT = typename paddle::operators::details::MPTypeTrait<T>::Type;
  typedef cub::BlockReduce<MT, BlockDim> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  int num = pre * post;
  for (int i = blockIdx.x; i < num; i += gridDim.x) {
    int base = (i / post) * post * axis_n + (i % post);

    MT sum = 0.0;
    __shared__ MT norm;
    for (int j = threadIdx.x; j < axis_n; j += blockDim.x) {
      const MT x_ij = static_cast<MT>(x[base + j * post]);
      sum += x_ij * x_ij;
    }
    MT reduce_result = BlockReduce(temp_storage).Sum(sum);

    if (threadIdx.x == 0) {
      norm = square_root(reduce_result + static_cast<MT>(eps));
      out_norm[i] = static_cast<T>(norm);
    }
    __syncthreads();
    for (int j = threadIdx.x; j < axis_n; j += blockDim.x) {
      const int index = base + j * post;
      y[index] = static_cast<T>((static_cast<MT>(x[index]) / norm));
    }
  }
}

template <typename T, typename Context>
void NormKernel(const Context& ctx,
                const DenseTensor& x,
                int axis,
                float epsilon,
                bool is_test,
                DenseTensor* out,
                DenseTensor* norm) {
  auto* in_x = &x;
  auto* out_y = out;

  auto xdim = in_x->dims();
  if (axis < 0) axis = xdim.size() + axis;
  T eps = static_cast<T>(epsilon);

  DenseTensor* out_norm;
  DenseTensor out_norm_tmp;
  if (is_test) {
    auto out_dim = in_x->dims();
    out_dim[axis] = 1;
    out_norm = &out_norm_tmp;
    out_norm->Resize(out_dim);
  } else {
    out_norm = norm;
  }

  const T* x_ptr = in_x->data<T>();
  ctx.template Alloc<T>(out_y);
  ctx.template Alloc<T>(out_norm);

  T* y = out_y->data<T>();
  T* norm_ptr = out_norm->data<T>();

  int pre, n, post;
  funcs::GetPrePostNumel(xdim, axis, &pre, &n, &post);

#ifdef __HIPCC__
  const int block = 256;
#else
  const int block = 512;
#endif
  int max_threads = ctx.GetMaxPhysicalThreadCount();
  const int max_blocks = std::max(max_threads / block, 1);
  int grid = std::min(max_blocks, pre * post);
  Normalize<T, block><<<grid, block, 0, ctx.stream()>>>(
      x_ptr, pre, n, post, eps, y, norm_ptr);
}

}  // namespace pten

PT_REGISTER_KERNEL(norm,
                   GPU,
                   ALL_LAYOUT,
                   pten::NormKernel,
                   float,
                   double,
                   paddle::platform::float16) {}
