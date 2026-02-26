// Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/phi/kernels/index_fill_grad_kernel.h"
#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/tensor_utils.h"

namespace phi {

template <typename T>
void index_fill_grad_kernel(const int64_t N,
                            const int64_t* index_data,
                            const int64_t index_size,
                            const int64_t dim_size,
                            const int64_t outer_size,
                            const int64_t inner_size,
                            T* x_grad) {
#ifdef PADDLE_WITH_MKLML
#pragma omp parallel for
#endif
  for (int64_t idx = 0; idx < N; ++idx) {
    int64_t inner_idx = idx % inner_size;
    int64_t temp = idx / inner_size;
    int64_t index_idx = temp % index_size;
    int64_t outer_idx = temp / index_size;

    int64_t dim_idx = index_data[index_idx];
    if (dim_idx < 0) {
      dim_idx += dim_size;
    }

    if (dim_idx < 0 || dim_idx >= dim_size) {
      continue;
    }

    int64_t offset =
        outer_idx * dim_size * inner_size + dim_idx * inner_size + inner_idx;

    *(x_grad + offset) = static_cast<T>(0);
  }
}

template <typename T, typename Context>
void LaunchIndexFillGradKernel(const Context& dev_ctx,
                               const DenseTensor& index,
                               const DenseTensor& out_grad,
                               const int dim,
                               DenseTensor* x_grad) {
  T* x_grad_data = dev_ctx.template Alloc<T>(x_grad);
  Copy(dev_ctx, out_grad, dev_ctx.GetPlace(), false, x_grad);

  auto out_grad_dims = out_grad.dims();
  const int rank = out_grad_dims.size();

  DenseTensor index_int64;
  const DenseTensor* ptr_index = nullptr;

  if (index.dtype() == phi::DataType::INT32) {
    index_int64.Resize(index.dims());
    int64_t* index_int64_data = dev_ctx.template Alloc<int64_t>(&index_int64);
    const int32_t* index_int32_data = index.data<int32_t>();

    int64_t index_numel = index.numel();
    for (int64_t i = 0; i < index_numel; ++i) {
      index_int64_data[i] = static_cast<int64_t>(index_int32_data[i]);
    }

    ptr_index = &index_int64;
  } else if (index.dtype() == phi::DataType::INT64) {
    ptr_index = &index;
  } else {
    PADDLE_THROW(common::errors::InvalidArgument(
        "The dtype of index must be int32 or int64, but received %s.",
        phi::DataTypeToString(index.dtype())));
  }

  const int64_t* index_data = ptr_index->data<int64_t>();
  int64_t index_size = ptr_index->numel();

  if (index_size == 0) {
    return;
  }

  int64_t outer_size = 1;
  int64_t inner_size = 1;
  int64_t dim_size = out_grad_dims[dim];

  for (int i = 0; i < dim; ++i) {
    outer_size *= out_grad_dims[i];
  }
  for (int i = dim + 1; i < rank; ++i) {
    inner_size *= out_grad_dims[i];
  }

  int64_t numel = outer_size * index_size * inner_size;

  index_fill_grad_kernel<T>(numel,
                            index_data,
                            index_size,
                            dim_size,
                            outer_size,
                            inner_size,
                            x_grad_data);
}

template <typename T, typename Context>
void IndexFillGradKernel(const Context& dev_ctx,
                         const DenseTensor& index,
                         const DenseTensor& out_grad,
                         int dim,
                         DenseTensor* x_grad) {
  if (x_grad && x_grad->numel() == 0) {
    dev_ctx.template Alloc<T>(x_grad);
    return;
  }

  if (out_grad.numel() == 0) {
    dev_ctx.template Alloc<T>(x_grad);
    return;
  }

  auto out_grad_dims = out_grad.dims();
  const int rank = out_grad_dims.size();

  if (dim < 0) {
    dim += rank;
  }

  PADDLE_ENFORCE_GE(
      dim,
      0,
      common::errors::InvalidArgument("The dimension index should be greater "
                                      "than or equal to 0, but got %d.",
                                      dim));
  PADDLE_ENFORCE_LT(
      dim,
      rank,
      common::errors::InvalidArgument(
          "The dimension index should be less than rank %d, but got %d.",
          rank,
          dim));

  PADDLE_ENFORCE_LE(
      rank,
      DDim::kMaxRank,
      common::errors::InvalidArgument("Dims of input tensor should be less "
                                      "than or equal to %d, but got %d.",
                                      DDim::kMaxRank,
                                      rank));

  if (index.numel() == 0) {
    Copy(dev_ctx, out_grad, dev_ctx.GetPlace(), false, x_grad);
    return;
  }

  LaunchIndexFillGradKernel<T, Context>(dev_ctx, index, out_grad, dim, x_grad);
}

}  // namespace phi

PD_REGISTER_KERNEL(index_fill_grad,
                   CPU,
                   ALL_LAYOUT,
                   phi::IndexFillGradKernel,
                   float,
                   double,
                   int,
                   int64_t,
                   bool,
                   int16_t,
                   uint8_t,
                   int8_t,
                   phi::float16,
                   phi::bfloat16,
                   phi::complex64,
                   phi::complex128) {}
