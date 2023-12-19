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

#include "paddle/phi/kernels/fill_diagonal_tensor_grad_kernel.h"

#include <algorithm>
#include <vector>

#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/common/memory_utils.h"
#include "paddle/phi/core/kernel_registry.h"

namespace phi {

template <typename T>
__global__ void fill_grad_kernel(int64_t size,
                                 T *out_data,
                                 int64_t *strides,
                                 int64_t *matdim,
                                 int64_t offset,
                                 int64_t fill_dims0,
                                 int64_t fill_dims1) {
  int64_t i = blockIdx.x;
  auto sumoff = matdim[i] + offset;
  for (int64_t j = threadIdx.x; j < fill_dims1; j += blockDim.x) {
    auto fill_index = j * (strides[1] + strides[0]) + sumoff;
    if (fill_index < size) {
      out_data[fill_index] = T(0);
    }
  }
}

template <typename T, typename Context>
void FillDiagonalTensorGradKernel(const Context &ctx,
                                  const DenseTensor &out_grad,
                                  int64_t offset,
                                  int dim1,
                                  int dim2,
                                  DenseTensor *x_grad) {
  const int64_t kMaxBlockDim = 512;
  auto matrows = 1;

  if (x_grad) {
    auto *data = ctx.template Alloc<T>(x_grad);
    auto dx_dims = x_grad->dims();
    phi::Copy(ctx, out_grad, ctx.GetPlace(), false, x_grad);

    for (int i = 0; i < dx_dims.size(); i++) {
      if (i != dim1 && i != dim2) {
        matrows *= dx_dims[i];
      }
    }

    int64_t new_dims[2];
    std::vector<int64_t> memory_block;
    memory_block.resize(2 + matrows);
    int64_t *strides = &memory_block[0];
    int64_t *matdim = &memory_block[2];
    CalMatDims(dx_dims, dim1, dim2, &offset, new_dims, strides, matdim);

    auto size = x_grad->numel();

    auto stream = ctx.stream();
    DenseTensor tensor_tmp;
    tensor_tmp.Resize(common::make_ddim({2 + matrows}));
    int64_t *memory_block_cu = ctx.template Alloc<int64_t>(&tensor_tmp);
    const auto gpu_place = ctx.GetPlace();
    memory_utils::Copy(gpu_place,
                       memory_block_cu,
                       CPUPlace(),
                       memory_block.data(),
                       sizeof(int64_t) * (2 + matrows),
                       stream);

    int64_t *strides_cu = &memory_block_cu[0], *matdim_cu = &memory_block_cu[2];

    auto kGridDim = new_dims[0];
    auto kBlockDim = std::min(int64_t(new_dims[1]), kMaxBlockDim);
    fill_grad_kernel<T><<<kGridDim, kBlockDim, 0, stream>>>(
        size, data, strides_cu, matdim_cu, offset, new_dims[0], new_dims[1]);
  }
}

}  // namespace phi

PD_REGISTER_KERNEL(fill_diagonal_tensor_grad,
                   GPU,
                   ALL_LAYOUT,
                   phi::FillDiagonalTensorGradKernel,
                   float,
                   double,
                   int64_t,
                   int,
                   int16_t,
                   int8_t,
                   uint8_t,
                   phi::dtype::float16,
                   phi::dtype::bfloat16,
                   phi::dtype::complex<float>,
                   phi::dtype::complex<double>,
                   bool) {}
