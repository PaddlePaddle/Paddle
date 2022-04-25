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
#include "paddle/phi/core/visit_type.h"
#include "paddle/phi/kernels/copy_kernel.h"
#include "paddle/phi/kernels/empty_kernel.h"
#include "paddle/phi/kernels/funcs/math_function.h"
#include "paddle/phi/kernels/funcs/sparse/flatten_indices.h"

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

  DenseTensor out_indices = phi::EmptyLike<T>(dev_ctx, indices);
  DenseTensor out_values = phi::EmptyLike<T>(dev_ctx, values);

  // the out_indices is same as indices of mask
  phi::Copy(dev_ctx, indices, dev_ctx.GetPlace(), false, &out_indices);

  T* out_values_ptr = out_values.data<T>();
  const T* x_ptr = x.data<T>();

  const int64_t non_zero_num = mask.nnz();
  auto dims_2d = flatten_to_2d(dims, sparse_dim);
  const int cols = dims_2d[1];
  const IntT* indices_ptr = indices.data<IntT>();

  std::vector<IntT> out_indexs(non_zero_num), sparse_offsets(sparse_dim);

  phi::funcs::sparse::CalcOffsetsPerDim<IntT>(
      dims, sparse_dim, sparse_offsets.data());

  for (int64_t i = 0; i < non_zero_num; i++) {
    int64_t index = phi::funcs::sparse::CoordinateToIndex<IntT>(
        indices_ptr, sparse_offsets.data(), non_zero_num, sparse_dim, i);
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
  PD_VISIT_INTEGRAL_TYPES(
      mask.non_zero_indices().dtype(), "SparseMaskCPUKernel", ([&] {
        SparseMaskCPUKernel<T, data_t>(dev_ctx, x, mask, out);
      }));
}

template <typename T, typename IntT>
void SparseMaskHelperCPUKernel(const CPUContext& dev_ctx,
                               const SparseCooTensor& x,
                               const DenseTensor& mask_indices,
                               DenseTensor* out) {
  PADDLE_ENFORCE_EQ(
      mask_indices.dims().size(),
      2,
      phi::errors::InvalidArgument("the mask_indices must be 2-D tensor"));

  const int64_t sparse_dim = x.non_zero_indices().dims()[0];

  std::vector<IntT> sparse_offsets(sparse_dim), x_indexs(x.nnz()),
      mask_indexs(mask_indices.dims()[1]);
  phi::funcs::sparse::CalcOffsetsPerDim<IntT>(
      x.dims(), sparse_dim, sparse_offsets.data());

  phi::funcs::sparse::FlattenIndices(x.non_zero_indices().data<IntT>(),
                                     sparse_offsets.data(),
                                     x.nnz(),
                                     sparse_dim,
                                     0,
                                     1,
                                     x_indexs.data());
  phi::funcs::sparse::FlattenIndices(mask_indices.data<IntT>(),
                                     sparse_offsets.data(),
                                     x.nnz(),
                                     sparse_dim,
                                     0,
                                     1,
                                     mask_indexs.data());

  std::unordered_map<IntT, uint64_t> x_indexs_map;
  for (uint64_t i = 0; i < x_indexs.size(); i++) {
    x_indexs_map[x_indexs[i]] = i;
  }
  *out = phi::EmptyLike<T>(dev_ctx, x.non_zero_elements());
  T* out_ptr = out->data<T>();
  memset(out_ptr, static_cast<T>(0), out->numel() * sizeof(T));
  const int64_t stride =
      x.dims().size() == sparse_dim ? 1 : x.non_zero_elements().dims()[1];
  const T* in_ptr = x.non_zero_elements().data<T>();
  // TODO(zhangkaihuo): multithreading can be used for acceleration
  for (uint64_t i = 0; i < mask_indexs.size(); i++) {
    auto iter = x_indexs_map.find(mask_indexs[i]);
    if (iter != x_indexs_map.end()) {
      memcpy(out_ptr + i * stride,
             in_ptr + iter->second * stride,
             stride * sizeof(T));
    }
  }
}

/**
 * @brief filter values from x.values() using mask_indices
 */
template <typename T, typename Context>
void SparseMaskHelperKernel(const Context& dev_ctx,
                            const SparseCooTensor& x,
                            const DenseTensor& mask_indices,
                            DenseTensor* out) {
  PD_VISIT_INTEGRAL_TYPES(
      x.non_zero_indices().dtype(), "SparseMaskHelperCPUKernel", ([&] {
        SparseMaskHelperCPUKernel<T, data_t>(dev_ctx, x, mask_indices, out);
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

PD_REGISTER_KERNEL(sparse_mask_helper,
                   CPU,
                   ALL_LAYOUT,
                   phi::sparse::SparseMaskHelperKernel,
                   float,
                   double,
                   uint8_t,
                   int16_t,
                   int,
                   int64_t) {
  kernel->InputAt(0).SetDataLayout(phi::DataLayout::SPARSE_COO);
}
