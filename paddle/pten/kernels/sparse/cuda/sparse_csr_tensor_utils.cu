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

#include "paddle/fluid/memory/memcpy.h"
#include "paddle/fluid/operators/math/blas.h"
#include "paddle/fluid/operators/math/sparse.h"
#include "paddle/fluid/platform/device/gpu/gpu_launch_config.h"
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

__global__ void kernel_convert_coo_rows_to_csr_crows(
    const int64_t* rows,
    int64_t* crows,
    const int m,
    const int64_t non_zero_num) {
  const int tid = threadIdx.x + blockIdx.x * blockDim.x;
  for (int i = tid; i < non_zero_num; i += gridDim.x * blockDim.x) {
    if (i == 0) {
      for (int j = 0; j < rows[0]; j++) {
        crows[j] = 0;
      }
      crows[m] = non_zero_num;
    } else {
      for (int j = rows[i - 1]; j < rows[i]; j++) {
        crows[j + 1] = i;
      }
    }
  }
}

template <typename T>
void SparseCooToCsr(const CUDAContext& dev_ctx,
                    const SparseCooTensor& src,
                    SparseCsrTensor* dst) {
  const auto& dense_dims = src.dims();
  PADDLE_ENFORCE_EQ(dense_dims.size(),
                    2,
                    paddle::platform::errors::InvalidArgument(
                        "SparseCsrTensor only support 2-D matrix"));
  const int64_t non_zero_num = src.nnz();

  dst->Resize(src.dims(), non_zero_num);
  int64_t* csr_crows_data = dst->mutable_non_zero_crows();
  int64_t* csr_cols_data = dst->mutable_non_zero_cols();
  T* csr_values_data = dst->mutable_non_zero_elements<T>();

  const auto& src_indices = src.non_zero_indices();
  const auto& src_values = src.non_zero_elements();
  const int64_t* src_rows_data = src_indices.data<int64_t>();
  const int64_t* src_cols_data = src_rows_data + non_zero_num;
  const T* src_values_data = src_values.data<T>();

  paddle::platform::GpuLaunchConfig config =
      paddle::platform::GetGpuLaunchConfig1D(dev_ctx, non_zero_num);
  kernel_convert_coo_rows_to_csr_crows<<<config.block_per_grid,
                                         config.thread_per_block>>>(
      src_rows_data, csr_crows_data, dense_dims[0], non_zero_num);

  // auto place = BOOST_GET_CONST(paddle::platform::CUDAPlace,
  // dev_ctx.GetPlace());
  // paddle::memory::Copy(place,
  //                     csr_cols_data,
  //                     place,
  //                     src_cols_data,
  //                     sizeof(int64_t) * non_zero_num);
  // paddle::memory::Copy(
  //    place, csr_values_data, place, src_values_data, sizeof(T) *
  //    non_zero_num);
}

__global__ void kernel_convert_csr_crows_to_coo_rows(
    const int64_t* crows,
    int64_t* rows,
    const int m,
    const int64_t non_zero_num) {
  const int tid = threadIdx.x + blockIdx.x * blockDim.x;
  for (int i = tid; i < m; i += gridDim.x * blockDim.x) {
    for (int j = crows[i]; j < crows[i + 1]; j++) {
      rows[j] = i;
    }
  }
}

template <typename T>
void SparseCsrToCoo(const CUDAContext& dev_ctx,
                    const SparseCsrTensor& src,
                    SparseCooTensor* dst) {
  const DDim& dense_dim = src.dims();
  const int64_t non_zero_num = src.nnz();
  const auto& csr_crows = src.non_zero_crows();
  const auto& csr_cols = src.non_zero_cols();
  const auto& csr_values = src.non_zero_elements();
  const int64_t* csr_crows_data = csr_crows.data<int64_t>();
  const int64_t* csr_cols_data = csr_cols.data<int64_t>();
  const T* csr_values_data = csr_values.data<T>();

  dst->Resize(src.dims(), 2, non_zero_num);
  int64_t* coo_indices = dst->mutable_non_zero_indices();
  int64_t* coo_rows_data = coo_indices;
  int64_t* coo_cols_data = coo_indices + non_zero_num;
  T* coo_values_data = dst->mutable_non_zero_elements<T>();

  paddle::platform::GpuLaunchConfig config =
      paddle::platform::GetGpuLaunchConfig1D(dev_ctx, non_zero_num);
  kernel_convert_csr_crows_to_coo_rows<<<config.block_per_grid,
                                         config.thread_per_block>>>(
      csr_crows_data, coo_rows_data, dense_dim[0], non_zero_num);
  // auto place = BOOST_GET_CONST(paddle::platform::CUDAPlace,
  // dev_ctx.GetPlace());
  // paddle::memory::Copy(place,
  //                     coo_cols_data,
  //                     place,
  //                     csr_cols_data,
  //                     sizeof(int64_t) * non_zero_num);
  // paddle::memory::Copy(
  //    place, coo_values_data, place, csr_values_data, sizeof(T) *
  //    non_zero_num);
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

PT_REGISTER_KERNEL(
    sparse_coo_to_csr, GPU, ALL_LAYOUT, pten::SparseCooToCsr, float, double) {}
PT_REGISTER_KERNEL(
    sparse_csr_to_coo, GPU, ALL_LAYOUT, pten::SparseCsrToCoo, float, double) {}
