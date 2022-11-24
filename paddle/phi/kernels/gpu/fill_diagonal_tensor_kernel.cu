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

#include "paddle/phi/kernels/fill_diagonal_tensor_kernel.h"

#include <algorithm>
#include <vector>

#include "paddle/fluid/memory/memcpy.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/tensor_utils.h"

namespace phi {

template <typename T>
__global__ void fill_diagonal_tensor_kernel(int64_t size,
                                            T *out_data,
                                            const T *fill_data,
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
      out_data[fill_index] = fill_data[i * fill_dims1 + j];
    }
  }
}

template <typename T, typename Context>
void FillDiagonalTensorKernel(const Context &ctx,
                              const DenseTensor &x,
                              const DenseTensor &y,
                              int64_t offset,
                              int dim1,
                              int dim2,
                              DenseTensor *out) {
#ifdef __HIPCC__
  const int64_t kMaxBlockDim = 256;
#else
  const int64_t kMaxBlockDim = 512;
#endif
  phi::Copy(ctx, x, ctx.GetPlace(), false, out);

  T *out_data = ctx.template Alloc<T>(out);
  const T *fill_data = y.data<T>();

  auto out_dims = out->dims();
  auto matdims = y.dims();
  auto fill_dims = phi::flatten_to_2d(matdims, matdims.size() - 1);

  int64_t new_dims[2];
  std::vector<int64_t> memory_block;
  memory_block.resize(2 + fill_dims[0]);
  int64_t *strides = &(memory_block[0]);
  int64_t *matdim = &(memory_block[2]);
  CalMatDims(out_dims, dim1, dim2, &offset, new_dims, strides, matdim);
  PADDLE_ENFORCE_EQ(
      new_dims[0],
      fill_dims[0],
      errors::InvalidArgument("The dims should be %d x %d, but get "
                              "%d x %d in fill tensor Y",
                              new_dims[0],
                              new_dims[1],
                              fill_dims[0],
                              fill_dims[1]));
  PADDLE_ENFORCE_EQ(
      new_dims[1],
      fill_dims[1],
      errors::InvalidArgument("The dims should be %d x %d, but get "
                              "%d x %d in fill tensor Y",
                              new_dims[0],
                              new_dims[1],
                              fill_dims[0],
                              fill_dims[1]));

  auto size = out->numel();

  auto stream = ctx.stream();
  DenseTensor tensor_tmp;
  tensor_tmp.Resize(phi::make_ddim({2 + fill_dims[0]}));
  int64_t *memory_block_cu = ctx.template Alloc<int64_t>(&tensor_tmp);
  const auto gpu_place = ctx.GetPlace();
  paddle::memory::Copy(gpu_place,
                       memory_block_cu,
                       CPUPlace(),
                       memory_block.data(),
                       sizeof(int64_t) * (2 + fill_dims[0]),
                       stream);

  int64_t *strides_cu = &memory_block_cu[0], *matdim_cu = &memory_block_cu[2];

  auto kGridDim = new_dims[0];
  auto kBlockDim = std::min(int64_t(new_dims[1]), kMaxBlockDim);
  fill_diagonal_tensor_kernel<T>
      <<<kGridDim, kBlockDim, 0, stream>>>(size,
                                           out_data,
                                           fill_data,
                                           strides_cu,
                                           matdim_cu,
                                           offset,
                                           fill_dims[0],
                                           fill_dims[1]);
}

}  // namespace phi

PD_REGISTER_KERNEL(fill_diagonal_tensor,
                   GPU,
                   ALL_LAYOUT,
                   phi::FillDiagonalTensorKernel,
                   float,
                   double,
                   int64_t,
                   int,
                   int8_t,
                   uint8_t,
                   phi::dtype::float16,
                   phi::dtype::bfloat16,
                   phi::dtype::complex<float>,
                   phi::dtype::complex<double>,
                   bool) {}
