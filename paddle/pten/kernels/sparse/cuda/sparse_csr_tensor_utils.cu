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
#include "paddle/pten/kernels/hybird/sparse/cuda/sparse_utils.h"
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
  DenseTensorMeta nnz_meta(DataType::INT32, nnz_dims, DataLayout::ANY);
  std::unique_ptr<DenseTensor> nnz_ptr(new DenseTensor(allocator, nnz_meta));
  std::unique_ptr<DenseTensor> cpu_nnz_ptr(
      new DenseTensor(cpu_alloc, nnz_meta));

  int* nnz = nnz_ptr->mutable_data<int32_t>();
  get_non_zero_num<T>(dev_ctx, src, 2, nnz, nnz + 1);
  pten::Copy(dev_ctx, *nnz_ptr, true, cpu_nnz_ptr.get());
  const int64_t non_zero_num = cpu_nnz_ptr->data<int>()[0];

  auto non_zero_dims = paddle::framework::make_ddim({non_zero_num});
  auto crows_dims = paddle::framework::make_ddim({src_dims[0] + 1});
  DenseTensorMeta crows_meta(DataType::INT64, crows_dims, DataLayout::ANY);
  std::unique_ptr<DenseTensor> crows_ptr(
      new DenseTensor(allocator, crows_meta));
  DenseTensorMeta cols_meta(DataType::INT64, non_zero_dims, DataLayout::ANY);
  std::unique_ptr<DenseTensor> cols_ptr(new DenseTensor(allocator, cols_meta));
  DenseTensorMeta values_meta(src.dtype(), non_zero_dims, src.layout());
  std::unique_ptr<DenseTensor> values_ptr(
      new DenseTensor(allocator, values_meta));

  int64_t* crows_data = crows_ptr->mutable_data<int64_t>();
  int64_t* cols_data = cols_ptr->mutable_data<int64_t>();
  T* values_data = values_ptr->mutable_data<T>();

  auto sparse =
      paddle::operators::math::GetSparse<paddle::platform::CUDADeviceContext,
                                         T>(dev_ctx);
#if defined(PADDLE_WITH_CUDA)
  sparse.DenseToSparseCsr(static_cast<int>(src_dims[0]),
                          static_cast<int>(src_dims[1]),
                          src_data,
                          crows_data,
                          cols_data,
                          values_data);
#endif

  dst->SetMemberTensor(std::move(crows_ptr),
                       std::move(cols_ptr),
                       std::move(values_ptr),
                       src_dims);
}

}  // namespace pten

PT_REGISTER_KERNEL(to_sparse_csr, GPU, ANY, pten::ToSparseCsr, float, double) {}
