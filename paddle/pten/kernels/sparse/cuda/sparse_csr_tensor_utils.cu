/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/operators/math/blas.h"
#include "paddle/fluid/operators/math/sparse.h"
#include "paddle/pten/api/lib/utils/allocator.h"
#include "paddle/pten/core/tensor_meta.h"
#include "paddle/pten/kernels/gpu/utils.h"
#include "paddle/pten/kernels/sparse/cuda/sparse_csr_tensor_utils.h"

namespace pten {

template <typename T>
void ToSparseCsr(const CUDAContext& dev_ctx,
                 const DenseTensor& src,
                 SparseCsrTensor* dst) {
  PADDLE_ENFORCE_EQ(src.dims().size(),
                    2,
                    paddle::platform::errors::InvalidArgument(
                        "SparseCsrTensor only support 2-D Tensor."));

  const T* src_data = src.data<T>();
  const auto& src_dims = src.dims();

  const auto cpu_alloc =
      std::make_shared<paddle::experimental::DefaultAllocator>(
          paddle::platform::CPUPlace());
  const auto allocator =
      std::make_shared<paddle::experimental::DefaultAllocator>(src.place());
  auto nnz_dims = paddle::framework::make_ddim({src_dims[0] + 1});
  DenseTensorMeta nnz_meta(DataType::INT32, nnz_dims, DataLayout::NCHW);
  DenseTensor nnz_tensor(allocator, nnz_meta);
  DenseTensor cpu_nnz_tensor(cpu_alloc, nnz_meta);

  auto sparse =
      paddle::operators::math::GetSparse<paddle::platform::CUDADeviceContext,
                                         T>(dev_ctx);
  int* nnz = nnz_tensor.mutable_data<int32_t>();
  const int M = static_cast<int>(src_dims[0]);
  const int N = static_cast<int>(src_dims[1]);
  sparse.nnz(M, N, src_data, nnz, nnz + 1);
  pten::Copy(dev_ctx, nnz_tensor, true, &cpu_nnz_tensor);
  const int64_t non_zero_num = cpu_nnz_tensor.data<int>()[0];

  dst->Resize(src_dims, non_zero_num);

  int64_t* crows_data = dst->mutable_non_zero_crows();
  int64_t* cols_data = dst->mutable_non_zero_cols();
  T* values_data = dst->mutable_non_zero_elements<T>();
  sparse.DenseToSparseCsr(static_cast<int>(src_dims[0]),
                          static_cast<int>(src_dims[1]),
                          src_data,
                          crows_data,
                          cols_data,
                          values_data);
}

template <typename T>
void SparseCsrToDense(const CUDAContext& dev_ctx,
                      const SparseCsrTensor& src,
                      DenseTensor* dst) {
  auto sparse =
      paddle::operators::math::GetSparse<paddle::platform::CUDADeviceContext,
                                         T>(dev_ctx);
  const auto src_dims = src.dims();
  const int M = src_dims[0];
  const int N = src_dims[1];
  const DenseTensor& crows = src.non_zero_crows();
  const DenseTensor& cols = src.non_zero_cols();
  const DenseTensor& values = src.non_zero_elements();
  const int64_t nnz = src.nnz();
  sparse.SparseCsrToDense(M,
                          N,
                          nnz,
                          crows.data<int64_t>(),
                          cols.data<int64_t>(),
                          values.data<T>(),
                          dst->mutable_data<T>());
}

}  // namespace pten

PT_REGISTER_KERNEL(
    to_sparse_csr, GPU, ALL_LAYOUT, pten::ToSparseCsr, float, double) {}
PT_REGISTER_KERNEL(sparse_csr_to_dense,
                   GPU,
                   ALL_LAYOUT,
                   pten::SparseCsrToDense,
                   float,
                   double) {}
