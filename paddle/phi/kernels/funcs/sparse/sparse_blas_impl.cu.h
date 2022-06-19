//   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

#pragma once

#include "paddle/fluid/memory/malloc.h"
#include "paddle/phi/backends/dynload/cusparse.h"
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/common/float16.h"
#include "paddle/phi/core/ddim.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/sparse_coo_tensor.h"
#include "paddle/phi/core/sparse_csr_tensor.h"
#include "paddle/phi/core/visit_type.h"

namespace phi {
namespace funcs {
namespace sparse {

template <typename T>
cudaDataType_t GetGpuDataType() {
  if (std::is_same<T, float>::value) {
    return CUDA_R_32F;
  } else if (std::is_same<T, double>::value) {
    return CUDA_R_64F;
  } else if (std::is_same<T, phi::dtype::float16>::value) {
    return CUDA_R_16F;
  }
}

inline cusparseOperation_t GetTransposeOperation(const bool trans) {
  if (trans) {
    return CUSPARSE_OPERATION_TRANSPOSE;
  } else {
    return CUSPARSE_OPERATION_NON_TRANSPOSE;
  }
}

template <typename T>
class CuSparseSpMatDescriptor {
 public:
  explicit CuSparseSpMatDescriptor(const phi::SparseCsrTensor& x,
                                   const phi::GPUContext& dev_ctx)
      : dev_ctx_(dev_ctx) {
    PD_VISIT_INTEGRAL_TYPES(
        x.non_zero_crows().dtype(), "CuSparseSpMatDescriptor", ([&] {
          const data_t* crows_data = x.non_zero_crows().data<data_t>();
          const data_t* cols_data = x.non_zero_cols().data<data_t>();
          const T* values_data = x.non_zero_elements().data<T>();
          int64_t nnz = x.nnz();

          std::vector<int64_t> xdim_vec = phi::vectorize(x.dims());
          auto x_ndims = xdim_vec.size();
          int64_t M = xdim_vec[x_ndims - 2];
          int64_t N = xdim_vec[x_ndims - 1];
          int batch_size = 1;
          for (int i = 0; i < x_ndims - 2; i++) {
            batch_size *= xdim_vec[i];
          }

          cudaDataType_t gpu_type = GetGpuDataType<T>();
          dev_ctx_.CusparseCall([&](cusparseHandle_t handle) {
            phi::dynload::cusparseCreateCsr(&descriptor_,
                                            M,
                                            N,
                                            nnz,
                                            const_cast<data_t*>(crows_data),
                                            const_cast<data_t*>(cols_data),
                                            const_cast<T*>(values_data),
                                            CUSPARSE_INDEX_64I,
                                            CUSPARSE_INDEX_64I,
                                            CUSPARSE_INDEX_BASE_ZERO,
                                            gpu_type);
          });
          PADDLE_ENFORCE_EQ(x.non_zero_crows().numel(), batch_size * (M + 1));
          PADDLE_ENFORCE_EQ(x.non_zero_cols().numel(), x.nnz());
          if (batch_size > 1) {
            dev_ctx_.CusparseCall([&](cusparseHandle_t handle) {
              phi::dynload::cusparseCsrSetStridedBatch(
                  descriptor_, batch_size, M + 1, nnz);
            });
          }
        }));

    VLOG(6) << "Create cusparseSpMatDescr_t " << &descriptor_;
  }

  ~CuSparseSpMatDescriptor() {
    dev_ctx_.CusparseCall([&](cusparseHandle_t handle) {
      phi::dynload::cusparseDestroySpMat(descriptor_);
    });
    VLOG(6) << "Destroy cusparseSpMatDescr_t " << &descriptor_;
  }

  const cusparseSpMatDescr_t& descriptor() const { return descriptor_; }

 private:
  const phi::GPUContext& dev_ctx_;
  cusparseSpMatDescr_t descriptor_;
};

template <typename T>
class CuSparseDnMatDescriptor {
 public:
  explicit CuSparseDnMatDescriptor(const phi::DenseTensor& x,
                                   const phi::GPUContext& dev_ctx)
      : dev_ctx_(dev_ctx) {
    const T* x_data = x.data<T>();
    std::vector<int64_t> xdim_vec = phi::vectorize(x.dims());
    auto x_ndims = xdim_vec.size();
    int64_t M = xdim_vec[x_ndims - 2];
    int64_t N = xdim_vec[x_ndims - 1];
    int batch_size = 1;
    for (int i = 0; i < x_ndims - 2; i++) {
      batch_size *= xdim_vec[i];
    }

    cudaDataType_t gpu_type = GetGpuDataType<T>();
    dev_ctx_.CusparseCall([&](cusparseHandle_t handle) {
      phi::dynload::cusparseCreateDnMat(&descriptor_,
                                        M,
                                        N,
                                        N,
                                        const_cast<T*>(x_data),
                                        gpu_type,
                                        CUSPARSE_ORDER_ROW);
    });

    PADDLE_ENFORCE_EQ(x.numel(), batch_size * M * N);
    if (batch_size > 1) {
      dev_ctx_.CusparseCall([&](cusparseHandle_t handle) {
        phi::dynload::cusparseDnMatSetStridedBatch(
            descriptor_, batch_size, M * N);
      });
    }
    VLOG(6) << "Create cusparseDnMatDescr_t " << &descriptor_;
  }

  ~CuSparseDnMatDescriptor() {
    dev_ctx_.CusparseCall([&](cusparseHandle_t handle) {
      phi::dynload::cusparseDestroyDnMat(descriptor_);
    });
    VLOG(6) << "Destroy cusparseDnMatDescr_t " << &descriptor_;
  }

  const cusparseDnMatDescr_t& descriptor() const { return descriptor_; }

 private:
  const phi::GPUContext& dev_ctx_;
  cusparseDnMatDescr_t descriptor_;
};

template <>
template <typename T>
void SparseBlas<phi::GPUContext>::DSDMM(bool transa,
                                        bool transb,
                                        T alpha,
                                        const phi::SparseCsrTensor& mat_a,
                                        const phi::DenseTensor& mat_b,
                                        T beta,
                                        phi::DenseTensor* mat_c) const {
  cudaDataType_t gpu_type = GetGpuDataType<T>();

  auto a_descriptor = CuSparseSpMatDescriptor<T>(mat_a, dev_ctx_);
  auto b_descriptor = CuSparseDnMatDescriptor<T>(mat_b, dev_ctx_);
  auto c_descriptor = CuSparseDnMatDescriptor<T>(*mat_c, dev_ctx_);

  size_t buffer_size = 0;
  dev_ctx_.CusparseCall([&](cusparseHandle_t handle) {
    phi::dynload::cusparseSpMM_bufferSize(handle,
                                          GetTransposeOperation(transa),
                                          GetTransposeOperation(transb),
                                          &alpha,
                                          a_descriptor.descriptor(),
                                          b_descriptor.descriptor(),
                                          &beta,
                                          c_descriptor.descriptor(),
                                          gpu_type,
                                          CUSPARSE_SPMM_ALG_DEFAULT,
                                          &buffer_size);
  });

  paddle::memory::allocation::AllocationPtr tmp_buffer =
      paddle::memory::Alloc(dev_ctx_, buffer_size);
  void* tmp_buffer_ptr = tmp_buffer->ptr();
  dev_ctx_.CusparseCall([&](cusparseHandle_t handle) {
    phi::dynload::cusparseSpMM(handle,
                               GetTransposeOperation(transa),
                               GetTransposeOperation(transb),
                               &alpha,
                               a_descriptor.descriptor(),
                               b_descriptor.descriptor(),
                               &beta,
                               c_descriptor.descriptor(),
                               gpu_type,
                               CUSPARSE_SPMM_ALG_DEFAULT,
                               tmp_buffer_ptr);
  });
}

#if CUDA_VERSION >= 11030
template <>
template <typename T>
void SparseBlas<phi::GPUContext>::SDDMM(bool transa,
                                        bool transb,
                                        T alpha,
                                        const phi::DenseTensor& mat_a,
                                        const phi::DenseTensor& mat_b,
                                        T beta,
                                        phi::SparseCsrTensor* mat_c) const {
  cudaDataType_t gpu_type = GetGpuDataType<T>();

  auto a_descriptor = CuSparseDnMatDescriptor<T>(mat_a, dev_ctx_);
  auto b_descriptor = CuSparseDnMatDescriptor<T>(mat_b, dev_ctx_);
  auto c_descriptor = CuSparseSpMatDescriptor<T>(*mat_c, dev_ctx_);

  size_t buffer_size = 0;
  dev_ctx_.CusparseCall([&](cusparseHandle_t handle) {
    phi::dynload::cusparseSDDMM_bufferSize(handle,
                                           GetTransposeOperation(transa),
                                           GetTransposeOperation(transb),
                                           &alpha,
                                           a_descriptor.descriptor(),
                                           b_descriptor.descriptor(),
                                           &beta,
                                           c_descriptor.descriptor(),
                                           gpu_type,
                                           CUSPARSE_SDDMM_ALG_DEFAULT,
                                           &buffer_size);
  });

  paddle::memory::allocation::AllocationPtr tmp_buffer =
      paddle::memory::Alloc(dev_ctx_, buffer_size);
  void* tmp_buffer_ptr = tmp_buffer->ptr();

  dev_ctx_.CusparseCall([&](cusparseHandle_t handle) {
    phi::dynload::cusparseSDDMM_preprocess(handle,
                                           GetTransposeOperation(transa),
                                           GetTransposeOperation(transb),
                                           &alpha,
                                           a_descriptor.descriptor(),
                                           b_descriptor.descriptor(),
                                           &beta,
                                           c_descriptor.descriptor(),
                                           gpu_type,
                                           CUSPARSE_SDDMM_ALG_DEFAULT,
                                           tmp_buffer_ptr);
  });

  dev_ctx_.CusparseCall([&](cusparseHandle_t handle) {
    phi::dynload::cusparseSDDMM(handle,
                                GetTransposeOperation(transa),
                                GetTransposeOperation(transb),
                                &alpha,
                                a_descriptor.descriptor(),
                                b_descriptor.descriptor(),
                                &beta,
                                c_descriptor.descriptor(),
                                gpu_type,
                                CUSPARSE_SDDMM_ALG_DEFAULT,
                                tmp_buffer_ptr);
  });
}
#endif

}  // namespace sparse
}  // namespace funcs
}  // namespace phi
