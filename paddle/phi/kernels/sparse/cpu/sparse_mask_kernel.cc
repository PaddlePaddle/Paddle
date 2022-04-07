/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/phi/kernels/sparse/sparse_mask_kernel.h"
#include "paddle/phi/core/ddim.h"
#include "paddle/phi/core/enforce.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/copy_kernel.h"
#include "paddle/phi/kernels/empty_kernel.h"
#include "paddle/phi/kernels/funcs/math_function.h"

#include "paddle/phi/api/ext/dispatch.h"

namespace phi {
namespace sparse {

template <typename T, typename IntT>
void SparseMaskCPUKernel(const CPUContext& dev_ctx,
                         const DenseTensor& x,
                         const SparseCooTensor& mask,
                         SparseCooTensor* out) {
  const DDim& dims = x.dims();
  PADDLE_ENFORCE_EQ(
      x.dims(),
      mask.dims(),
      phi::errors::InvalidArgument("the input x and mask must have the shape"));
  const DenseTensor& indices = mask.non_zero_indices();
  const DenseTensor& values = mask.non_zero_elements();
  int sparse_dim = indices.dims().size();
  std::vector<int64_t> sparse_offsets(sparse_dim);
  int64_t offset = 1;
  for (int i = sparse_dim - 1; i >= 0; i--) {
    sparse_offsets[i] = offset;
    offset *= dims[i];
  }

  DenseTensor out_indices = phi::EmptyLike<T>(dev_ctx, indices);
  DenseTensor out_values = phi::EmptyLike<T>(dev_ctx, values);

  // the out_indices is same as indices of mask
  phi::Copy(dev_ctx, indices, dev_ctx.GetPlace(), false, &out_indices);

  const IntT* indices_ptr = indices.data<IntT>();
  T* out_values_ptr = out_values.data<T>();
  const T* x_ptr = x.data<T>();

  const int64_t non_zero_num = mask.nnz();
  auto dims_2d = flatten_to_2d(dims, sparse_dim);
  const int cols = dims_2d[1];

  for (int64_t i = 0; i < non_zero_num; i++) {
    int64_t index = 0;
    for (int j = 0; j < sparse_dim; j++) {
      index += indices_ptr[j * non_zero_num + i] * sparse_offsets[j];
    }
    memcpy(out_values_ptr + i * cols, x_ptr + index * cols, cols * sizeof(T));
  }
  out->SetMember(out_indices, out_values, dims, true);
}

/**
 * @brief Filter the DenseTensor x by the
 * mask.non_zero_indices() and output a SparseCooTensor
 * x and mask must have the same shape.
**/
template <typename T, typename Context>
void SparseMaskKernel(const Context& dev_ctx,
                      const DenseTensor& x,
                      const SparseCooTensor& mask,
                      SparseCooTensor* out) {
  PD_DISPATCH_INTEGRAL_TYPES(
      mask.non_zero_indices().dtype(), "SparseMaskCPUKernel", ([&] {
        SparseMaskCPUKernel<T, data_t>(dev_ctx, x, mask, out);
      }));
}

}  // namespace sparse
}  // namespace phi

PD_REGISTER_KERNEL(sparse_mask,
                   CPU,
                   ALL_LAYOUT,
                   phi::sparse::SparseMaskKernel,
                   float,
                   double,
                   uint8_t,
                   int8_t,
                   int16_t,
                   int,
                   int64_t) {
  kernel->InputAt(1).SetDataLayout(phi::DataLayout::SPARSE_COO);
}
