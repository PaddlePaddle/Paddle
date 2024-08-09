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

#include "glog/logging.h"

#include "paddle/common/ddim.h"
#include "paddle/phi/backends/dynload/cusparse.h"
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/common/float16.h"
#include "paddle/phi/common/memory_utils.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/enforce.h"
#include "paddle/phi/core/sparse_coo_tensor.h"
#include "paddle/phi/core/sparse_csr_tensor.h"
#include "paddle/phi/core/visit_type.h"
#include "paddle/phi/kernels/cast_kernel.h"
#include "paddle/phi/kernels/concat_kernel.h"
#include "paddle/phi/kernels/empty_kernel.h"

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

template <typename T>
cusparseIndexType_t GetCusparseIndexType() {
  if (std::is_same<T, int32_t>::value) {
    return CUSPARSE_INDEX_32I;
  } else if (std::is_same<T, int64_t>::value) {
    return CUSPARSE_INDEX_64I;
  }
}

inline cusparseOperation_t GetTransposeOperation(const bool trans) {
  if (trans) {
    return CUSPARSE_OPERATION_TRANSPOSE;
  } else {
    return CUSPARSE_OPERATION_NON_TRANSPOSE;
  }
}

inline cusparseSpMMAlg_t GetSpMMAlgorithm(const SparseCsrTensor& x) {
  // TODO(zhouwei): will change to 'CUSPARSE_SPMM_CSR_ALG2' when support batch
  return CUSPARSE_SPMM_CSR_ALG2;
}

inline cusparseSpMMAlg_t GetSpMMAlgorithm(const SparseCooTensor& x) {
  return CUSPARSE_SPMM_ALG_DEFAULT;
}

/************* SPARSE MATRIX DESCRIPTOR (COO/CSR) ************/

template <typename T, typename IntT>
inline void CreateCsrDescriptor(const phi::SparseCsrTensor& x,
                                const phi::GPUContext& dev_ctx,
                                cusparseSpMatDescr_t* descriptor) {
  std::vector<int64_t> xdim_vec = common::vectorize(x.dims());
  auto x_ndims = xdim_vec.size();
  PADDLE_ENFORCE_GE(
      x_ndims,
      2,
      common::errors::InvalidArgument("the dim size of SparseCsrTensor must be "
                                      "greater than or equal to 2."));
  int64_t M = xdim_vec[x_ndims - 2];
  int64_t N = xdim_vec[x_ndims - 1];
  int batch_size = 1;
  for (int i = 0; i < x_ndims - 2; i++) {
    batch_size *= xdim_vec[i];
  }
  PADDLE_ENFORCE_EQ(x.non_zero_crows().numel(),
                    batch_size * (M + 1),
                    common::errors::PreconditionNotMet(
                        "the length of SparseCsrTensor crows is not right."));

  const IntT* crows_data = x.non_zero_crows().data<IntT>();
  const IntT* cols_data = x.non_zero_cols().data<IntT>();
  const T* values_data = x.non_zero_elements().data<T>();
  int64_t batch_nnz = x.nnz() / batch_size;
  cudaDataType_t gpu_type = GetGpuDataType<T>();
  cusparseIndexType_t index_type = GetCusparseIndexType<IntT>();
  dev_ctx.CusparseCall([&](cusparseHandle_t handle) {
    phi::dynload::cusparseCreateCsr(descriptor,
                                    M,
                                    N,
                                    batch_nnz,
                                    const_cast<IntT*>(crows_data),
                                    const_cast<IntT*>(cols_data),
                                    const_cast<T*>(values_data),
                                    index_type,
                                    index_type,
                                    CUSPARSE_INDEX_BASE_ZERO,
                                    gpu_type);
  });
  if (batch_size > 1) {
#if CUDA_VERSION >= 11080
    dev_ctx.CusparseCall([&](cusparseHandle_t handle) {
      phi::dynload::cusparseCsrSetStridedBatch(
          *descriptor, batch_size, M + 1, batch_nnz);
    });
#else
    PADDLE_THROW(common::errors::Unimplemented(
        "Batch Sparse matmul use 'cusparseCsrSetStridedBatch', which is "
        "supported from CUDA 11.8"));
#endif
  }
}

template <typename T, typename IntT>
inline void CreateCooDescriptor(const phi::SparseCooTensor& x,
                                const phi::GPUContext& dev_ctx,
                                cusparseSpMatDescr_t* descriptor) {
  std::vector<int64_t> xdim_vec = common::vectorize(x.dims());
  auto x_ndims = xdim_vec.size();
  PADDLE_ENFORCE_GE(
      x_ndims,
      2,
      common::errors::InvalidArgument("the dim size of SparseCsrTensor must be "
                                      "greater than or equal to 2."));

  int64_t M = xdim_vec[x_ndims - 2];
  int64_t N = xdim_vec[x_ndims - 1];
  int batch_size = 1;
  for (int i = 0; i < x_ndims - 2; i++) {
    batch_size *= xdim_vec[i];
  }
  int64_t nnz = x.nnz();

  const IntT* indices_data = x.non_zero_indices().data<IntT>();
  const T* values_data = x.non_zero_elements().data<T>();
  auto rows_data = indices_data + (x_ndims - 2) * nnz;
  auto cols_data = indices_data + (x_ndims - 1) * nnz;

  int64_t batch_nnz = nnz / batch_size;
  cudaDataType_t gpu_type = GetGpuDataType<T>();
  cusparseIndexType_t index_type = GetCusparseIndexType<IntT>();
  dev_ctx.CusparseCall([&](cusparseHandle_t handle) {
    phi::dynload::cusparseCreateCoo(descriptor,
                                    M,
                                    N,
                                    batch_nnz,
                                    const_cast<IntT*>(rows_data),
                                    const_cast<IntT*>(cols_data),
                                    const_cast<T*>(values_data),
                                    index_type,
                                    CUSPARSE_INDEX_BASE_ZERO,
                                    gpu_type);
  });

  if (batch_size > 1) {
#if CUDA_VERSION >= 11080
    dev_ctx.CusparseCall([&](cusparseHandle_t handle) {
      phi::dynload::cusparseCooSetStridedBatch(
          *descriptor, batch_size, batch_nnz);
    });
#else
    PADDLE_THROW(common::errors::Unimplemented(
        "Batch Sparse matmul use 'cusparseCooSetStridedBatch', which is "
        "supported from CUDA 11.8"));
#endif
  }
}

template <typename T>
class CuSparseSpMatDescriptor {
 public:
  explicit CuSparseSpMatDescriptor(const phi::SparseCsrTensor& x,
                                   const phi::GPUContext& dev_ctx)
      : dev_ctx_(dev_ctx) {
    PD_VISIT_BASE_INTEGRAL_TYPES(
        x.non_zero_crows().dtype(), "Csr CuSparseSpMatDescriptor", ([&] {
          CreateCsrDescriptor<T, data_t>(x, dev_ctx_, &descriptor_);
        }));
    VLOG(6) << "Create csr cusparseSpMatDescr_t " << &descriptor_;
  }

  explicit CuSparseSpMatDescriptor(const phi::SparseCooTensor& x,
                                   const phi::GPUContext& dev_ctx)
      : dev_ctx_(dev_ctx) {
    PD_VISIT_BASE_INTEGRAL_TYPES(
        x.non_zero_indices().dtype(), "Coo CuSparseSpMatDescriptor", ([&] {
          CreateCooDescriptor<T, data_t>(x, dev_ctx_, &descriptor_);
        }));
    VLOG(6) << "Create coo cusparseSpMatDescr_t " << &descriptor_;
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

/************* DENSE MATRIX DESCRIPTOR ************/
template <typename T>
class CuSparseDnMatDescriptor {
 public:
  explicit CuSparseDnMatDescriptor(const phi::DenseTensor& x,
                                   const phi::GPUContext& dev_ctx)
      : dev_ctx_(dev_ctx) {
    std::vector<int64_t> xdim_vec = common::vectorize(x.dims());
    auto x_ndims = xdim_vec.size();
    PADDLE_ENFORCE_GE(
        x_ndims,
        2,
        common::errors::InvalidArgument("the dim size of DenseTensor must be "
                                        "greater than or equal to 2."));

    int64_t M = xdim_vec[x_ndims - 2];
    int64_t N = xdim_vec[x_ndims - 1];
    int batch_size = 1;
    for (int i = 0; i < x_ndims - 2; i++) {
      batch_size *= xdim_vec[i];
    }

    const T* x_data = x.data<T>();
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
#if CUDA_VERSION >= 11080
      dev_ctx_.CusparseCall([&](cusparseHandle_t handle) {
        phi::dynload::cusparseDnMatSetStridedBatch(
            descriptor_, batch_size, M * N);
      });
#else
      PADDLE_THROW(common::errors::Unimplemented(
          "Batch Sparse matmul use 'cusparseDnMatSetStridedBatch', which is "
          "supported from CUDA 11.8"));
#endif
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

/************* DENSE VECTOR DESCRIPTOR ************/
template <typename T>
class CuSparseDnVecDescriptor {
 public:
  explicit CuSparseDnVecDescriptor(const phi::DenseTensor& x,
                                   const phi::GPUContext& dev_ctx)
      : dev_ctx_(dev_ctx) {
    std::vector<int64_t> xdim_vec = common::vectorize(x.dims());
    auto x_ndims = xdim_vec.size();
    PADDLE_ENFORCE_GE(x_ndims,
                      1,
                      common::errors::InvalidArgument(
                          "the dim size of Vec must be equal to 1."));

    const T* x_data = x.data<T>();
    cudaDataType_t gpu_type = GetGpuDataType<T>();
    dev_ctx_.CusparseCall([&](cusparseHandle_t handle) {
      phi::dynload::cusparseCreateDnVec(
          &descriptor_, x.numel(), const_cast<T*>(x_data), gpu_type);
    });

    VLOG(6) << "Create cusparseDnVecDescr_t " << &descriptor_;
  }

  ~CuSparseDnVecDescriptor() {
    dev_ctx_.CusparseCall([&](cusparseHandle_t handle) {
      phi::dynload::cusparseDestroyDnVec(descriptor_);
    });
    VLOG(6) << "Destroy cusparseDnVecDescr_t " << &descriptor_;
  }

  const cusparseDnVecDescr_t& descriptor() const { return descriptor_; }

 private:
  const phi::GPUContext& dev_ctx_;
  cusparseDnVecDescr_t descriptor_;
};

/************* SPARSE*DENSE->DENSE MATMUL ************/
template <>
template <typename T, typename TensorType>
void SparseBlas<phi::GPUContext>::SPMM(bool transa,
                                       bool transb,
                                       T alpha,
                                       const TensorType& mat_a,
                                       const phi::DenseTensor& mat_b,
                                       T beta,
                                       phi::DenseTensor* mat_out) const {
  auto a_descriptor = CuSparseSpMatDescriptor<T>(mat_a, dev_ctx_);
  auto b_descriptor = CuSparseDnMatDescriptor<T>(mat_b, dev_ctx_);
  auto out_descriptor = CuSparseDnMatDescriptor<T>(*mat_out, dev_ctx_);

  cudaDataType_t gpu_type = GetGpuDataType<T>();
  size_t buffer_size = 0;
  dev_ctx_.CusparseCall([&](cusparseHandle_t handle) {
    phi::dynload::cusparseSpMM_bufferSize(handle,
                                          GetTransposeOperation(transa),
                                          GetTransposeOperation(transb),
                                          &alpha,
                                          a_descriptor.descriptor(),
                                          b_descriptor.descriptor(),
                                          &beta,
                                          out_descriptor.descriptor(),
                                          gpu_type,
                                          GetSpMMAlgorithm(mat_a),
                                          &buffer_size);
  });

  phi::Allocator::AllocationPtr tmp_buffer = phi::memory_utils::Alloc(
      dev_ctx_.GetPlace(),
      buffer_size,
      phi::Stream(reinterpret_cast<phi::StreamId>(dev_ctx_.stream())));
  void* tmp_buffer_ptr = tmp_buffer->ptr();
  dev_ctx_.CusparseCall([&](cusparseHandle_t handle) {
    phi::dynload::cusparseSpMM(handle,
                               GetTransposeOperation(transa),
                               GetTransposeOperation(transb),
                               &alpha,
                               a_descriptor.descriptor(),
                               b_descriptor.descriptor(),
                               &beta,
                               out_descriptor.descriptor(),
                               gpu_type,
                               GetSpMMAlgorithm(mat_a),
                               tmp_buffer_ptr);
  });
}

/************* SPARSE*DENSE->DENSE MV ************/
template <>
template <typename T, typename TensorType>
void SparseBlas<phi::GPUContext>::SPMV(bool transa,
                                       T alpha,
                                       const TensorType& mat_a,
                                       const phi::DenseTensor& vec_x,
                                       T beta,
                                       phi::DenseTensor* vec_out) const {
  auto a_descriptor = CuSparseSpMatDescriptor<T>(mat_a, dev_ctx_);
  auto x_descriptor = CuSparseDnVecDescriptor<T>(vec_x, dev_ctx_);
  auto out_descriptor = CuSparseDnVecDescriptor<T>(*vec_out, dev_ctx_);

  cudaDataType_t gpu_type = GetGpuDataType<T>();
  size_t buffer_size = 0;
  dev_ctx_.CusparseCall([&](cusparseHandle_t handle) {
    phi::dynload::cusparseSpMV_bufferSize(handle,
                                          GetTransposeOperation(transa),
                                          &alpha,
                                          a_descriptor.descriptor(),
                                          x_descriptor.descriptor(),
                                          &beta,
                                          out_descriptor.descriptor(),
                                          gpu_type,
#if CUDA_VERSION >= 11040
                                          CUSPARSE_SPMV_ALG_DEFAULT,
#else
                                          CUSPARSE_MV_ALG_DEFAULT,
#endif
                                          &buffer_size);
  });

  phi::Allocator::AllocationPtr tmp_buffer = phi::memory_utils::Alloc(
      dev_ctx_.GetPlace(),
      buffer_size,
      phi::Stream(reinterpret_cast<phi::StreamId>(dev_ctx_.stream())));
  void* tmp_buffer_ptr = tmp_buffer->ptr();
  dev_ctx_.CusparseCall([&](cusparseHandle_t handle) {
    phi::dynload::cusparseSpMV(handle,
                               GetTransposeOperation(transa),
                               &alpha,
                               a_descriptor.descriptor(),
                               x_descriptor.descriptor(),
                               &beta,
                               out_descriptor.descriptor(),
                               gpu_type,
#if CUDA_VERSION >= 11040
                               CUSPARSE_SPMV_ALG_DEFAULT,
#else
                               CUSPARSE_MV_ALG_DEFAULT,
#endif
                               tmp_buffer_ptr);
  });
}

/************* DENSE*DENSE->SPARSE MATMUL ************/
#if CUDA_VERSION >= 11030
template <>
template <typename T, typename TensorType>
void SparseBlas<phi::GPUContext>::SDDMM(bool transa,
                                        bool transb,
                                        T alpha,
                                        const phi::DenseTensor& mat_a,
                                        const phi::DenseTensor& mat_b,
                                        T beta,
                                        TensorType* mat_out) const {
  auto a_descriptor = CuSparseDnMatDescriptor<T>(mat_a, dev_ctx_);
  auto b_descriptor = CuSparseDnMatDescriptor<T>(mat_b, dev_ctx_);
  auto out_descriptor = CuSparseSpMatDescriptor<T>(*mat_out, dev_ctx_);

  cudaDataType_t gpu_type = GetGpuDataType<T>();
  size_t buffer_size = 0;
  dev_ctx_.CusparseCall([&](cusparseHandle_t handle) {
    phi::dynload::cusparseSDDMM_bufferSize(handle,
                                           GetTransposeOperation(transa),
                                           GetTransposeOperation(transb),
                                           &alpha,
                                           a_descriptor.descriptor(),
                                           b_descriptor.descriptor(),
                                           &beta,
                                           out_descriptor.descriptor(),
                                           gpu_type,
                                           CUSPARSE_SDDMM_ALG_DEFAULT,
                                           &buffer_size);
  });

  phi::Allocator::AllocationPtr tmp_buffer = phi::memory_utils::Alloc(
      dev_ctx_.GetPlace(),
      buffer_size,
      phi::Stream(reinterpret_cast<phi::StreamId>(dev_ctx_.stream())));
  void* tmp_buffer_ptr = tmp_buffer->ptr();

  dev_ctx_.CusparseCall([&](cusparseHandle_t handle) {
    phi::dynload::cusparseSDDMM_preprocess(handle,
                                           GetTransposeOperation(transa),
                                           GetTransposeOperation(transb),
                                           &alpha,
                                           a_descriptor.descriptor(),
                                           b_descriptor.descriptor(),
                                           &beta,
                                           out_descriptor.descriptor(),
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
                                out_descriptor.descriptor(),
                                gpu_type,
                                CUSPARSE_SDDMM_ALG_DEFAULT,
                                tmp_buffer_ptr);
  });
}
#endif

/************* SPARSE*SPARSE->SPARSE MATMUL ************/
template <typename T>
__global__ void GetCsrBatchNnz(const int32_t* crow_data,
                               int64_t rows,
                               int32_t* batch_nnz) {
  int64_t i = static_cast<int64_t>(threadIdx.x);
  batch_nnz[i] = crow_data[(i + 1) * (rows + 1) - 1];
}

template <>
template <typename T>
void SparseBlas<phi::GPUContext>::SPGEMM(bool transa,
                                         bool transb,
                                         T alpha,
                                         const SparseCsrTensor& mat_a,
                                         const SparseCsrTensor& mat_b,
                                         T beta,
                                         SparseCsrTensor* mat_out) const {
  DenseTensor* mat_out_crows = mat_out->mutable_crows();
  DenseTensor* mat_out_cols = mat_out->mutable_cols();
  DenseTensor* mat_out_values = mat_out->mutable_values();

  MetaTensor out_crows_meta(mat_out_crows);
  out_crows_meta.set_dtype(phi::DataType::INT32);
  out_crows_meta.set_dims(mat_a.crows().dims());
  dev_ctx_.template Alloc<int32_t>(mat_out_crows);

  std::vector<int64_t> a_dim_vec = common::vectorize(mat_a.dims());
  auto a_ndims = a_dim_vec.size();
  const int64_t a_rows = a_dim_vec[a_ndims - 2];
  const int64_t a_cols = a_dim_vec[a_ndims - 1];
  int a_batch_size = 1;
  for (int i = 0; i < a_ndims - 2; i++) {
    a_batch_size *= a_dim_vec[i];
  }

  std::vector<int64_t> b_dim_vec = common::vectorize(mat_b.dims());
  auto b_ndims = b_dim_vec.size();
  const int64_t b_rows = b_dim_vec[b_ndims - 2];
  const int64_t b_cols = b_dim_vec[b_ndims - 1];

  // cusparseSpGEMM only support 32-bit indices.
  const int32_t *a_crows_data = nullptr, *a_cols_data = nullptr,
                *b_crows_data = nullptr, *b_cols_data = nullptr;
  std::shared_ptr<DenseTensor> a_crows_int = nullptr, a_cols_int = nullptr,
                               b_crows_int = nullptr, b_cols_int = nullptr;

  if (mat_a.crows().dtype() == phi::DataType::INT32) {
    a_crows_data = mat_a.crows().data<int32_t>();
    a_cols_data = mat_a.cols().data<int32_t>();
  } else {
    a_crows_int = std::make_shared<DenseTensor>();
    a_cols_int = std::make_shared<DenseTensor>();
    phi::MetaTensor crows_meta(a_crows_int.get());
    crows_meta.set_dims(mat_a.crows().dims());
    phi::MetaTensor cols_meta(a_cols_int.get());
    cols_meta.set_dims(mat_a.cols().dims());

    phi::CastKernel<int64_t>(
        dev_ctx_, mat_a.crows(), phi::DataType::INT32, a_crows_int.get());
    phi::CastKernel<int64_t>(
        dev_ctx_, mat_a.cols(), phi::DataType::INT32, a_cols_int.get());

    a_crows_data = a_crows_int->data<int32_t>();
    a_cols_data = a_cols_int->data<int32_t>();
  }

  if (mat_b.crows().dtype() == phi::DataType::INT32) {
    b_crows_data = mat_b.crows().data<int32_t>();
    b_cols_data = mat_b.cols().data<int32_t>();
  } else {
    b_crows_int = std::make_shared<DenseTensor>();
    b_cols_int = std::make_shared<DenseTensor>();
    phi::MetaTensor crows_meta(b_crows_int.get());
    crows_meta.set_dims(mat_b.crows().dims());
    phi::MetaTensor cols_meta(b_cols_int.get());
    cols_meta.set_dims(mat_b.cols().dims());

    phi::CastKernel<int64_t>(
        dev_ctx_, mat_b.crows(), phi::DataType::INT32, b_crows_int.get());
    phi::CastKernel<int64_t>(
        dev_ctx_, mat_b.cols(), phi::DataType::INT32, b_cols_int.get());

    b_crows_data = b_crows_int->data<int32_t>();
    b_cols_data = b_cols_int->data<int32_t>();
  }

  const T* a_values_data = mat_a.values().data<T>();
  const T* b_values_data = mat_b.values().data<T>();
  const int32_t* out_crows_data = mat_out->crows().data<int32_t>();

  const int batch_size = a_batch_size;
  std::vector<int32_t> a_batch_nnz_vec(batch_size);
  std::vector<int32_t> b_batch_nnz_vec(batch_size);

  if (batch_size == 1) {
    a_batch_nnz_vec[0] = mat_a.nnz();
    b_batch_nnz_vec[0] = mat_b.nnz();
  } else {
    phi::Allocator::AllocationPtr tmp_buffer = phi::memory_utils::Alloc(
        dev_ctx_.GetPlace(),
        batch_size * sizeof(int32_t),
        phi::Stream(reinterpret_cast<phi::StreamId>(dev_ctx_.stream())));
    void* tmp_buffer_ptr = tmp_buffer->ptr();

    GetCsrBatchNnz<T><<<1, batch_size, 0, dev_ctx_.stream()>>>(
        a_crows_data, a_rows, static_cast<int32_t*>(tmp_buffer_ptr));
    phi::backends::gpu::GpuMemcpyAsync(a_batch_nnz_vec.data(),
                                       tmp_buffer_ptr,
                                       batch_size * sizeof(int32_t),
                                       gpuMemcpyDeviceToHost,
                                       dev_ctx_.stream());

    GetCsrBatchNnz<T><<<1, batch_size, 0, dev_ctx_.stream()>>>(
        b_crows_data, b_rows, static_cast<int32_t*>(tmp_buffer_ptr));
    phi::backends::gpu::GpuMemcpyAsync(b_batch_nnz_vec.data(),
                                       tmp_buffer_ptr,
                                       batch_size * sizeof(int32_t),
                                       gpuMemcpyDeviceToHost,
                                       dev_ctx_.stream());
  }

  std::vector<DenseTensor> out_batch_cols_vec(batch_size);
  std::vector<DenseTensor> out_batch_values_vec(batch_size);
  cudaDataType_t gpu_type = GetGpuDataType<T>();

  const int32_t* a_batch_crows_data = a_crows_data;
  const int32_t* a_batch_cols_data = a_cols_data;
  const T* a_batch_values_data = a_values_data;

  const int32_t* b_batch_crows_data = b_crows_data;
  const int32_t* b_batch_cols_data = b_cols_data;
  const T* b_batch_values_data = b_values_data;

  const int32_t* out_batch_crows_data = out_crows_data;

  for (int i = 0; i < batch_size; ++i) {
    int32_t a_batch_nnz = a_batch_nnz_vec[i];
    int32_t b_batch_nnz = b_batch_nnz_vec[i];

    cusparseSpMatDescr_t a_batch_desc, b_batch_desc, out_batch_desc;
    dev_ctx_.CusparseCall([&](cusparseHandle_t handle) {
      phi::dynload::cusparseCreateCsr(&a_batch_desc,
                                      a_rows,
                                      a_cols,
                                      a_batch_nnz,
                                      const_cast<int32_t*>(a_batch_crows_data),
                                      const_cast<int32_t*>(a_batch_cols_data),
                                      const_cast<T*>(a_batch_values_data),
                                      CUSPARSE_INDEX_32I,
                                      CUSPARSE_INDEX_32I,
                                      CUSPARSE_INDEX_BASE_ZERO,
                                      gpu_type);
    });

    dev_ctx_.CusparseCall([&](cusparseHandle_t handle) {
      phi::dynload::cusparseCreateCsr(&b_batch_desc,
                                      b_rows,
                                      b_cols,
                                      b_batch_nnz,
                                      const_cast<int32_t*>(b_batch_crows_data),
                                      const_cast<int32_t*>(b_batch_cols_data),
                                      const_cast<T*>(b_batch_values_data),
                                      CUSPARSE_INDEX_32I,
                                      CUSPARSE_INDEX_32I,
                                      CUSPARSE_INDEX_BASE_ZERO,
                                      gpu_type);
    });

    dev_ctx_.CusparseCall([&](cusparseHandle_t handle) {
      phi::dynload::cusparseCreateCsr(&out_batch_desc,
                                      a_rows,
                                      b_cols,
                                      0,
                                      nullptr,
                                      nullptr,
                                      nullptr,
                                      CUSPARSE_INDEX_32I,
                                      CUSPARSE_INDEX_32I,
                                      CUSPARSE_INDEX_BASE_ZERO,
                                      gpu_type);
    });

    size_t buffer_a_size = 0, buffer_b_size = 0;
    cusparseSpGEMMDescr_t spgemm_desc;
    dev_ctx_.CusparseCall([&](cusparseHandle_t handle) {
      phi::dynload::cusparseSpGEMM_createDescr(&spgemm_desc);
    });

    dev_ctx_.CusparseCall([&](cusparseHandle_t handle) {
      phi::dynload::cusparseSpGEMM_workEstimation(handle,
                                                  GetTransposeOperation(transa),
                                                  GetTransposeOperation(transb),
                                                  &alpha,
                                                  a_batch_desc,
                                                  b_batch_desc,
                                                  &beta,
                                                  out_batch_desc,
                                                  gpu_type,
                                                  CUSPARSE_SPGEMM_DEFAULT,
                                                  spgemm_desc,
                                                  &buffer_a_size,
                                                  nullptr);
    });

    phi::Allocator::AllocationPtr tmp_buffer_a = phi::memory_utils::Alloc(
        dev_ctx_.GetPlace(),
        buffer_a_size,
        phi::Stream(reinterpret_cast<phi::StreamId>(dev_ctx_.stream())));
    void* tmp_buffer_a_ptr = tmp_buffer_a->ptr();

    dev_ctx_.CusparseCall([&](cusparseHandle_t handle) {
      phi::dynload::cusparseSpGEMM_workEstimation(handle,
                                                  GetTransposeOperation(transa),
                                                  GetTransposeOperation(transb),
                                                  &alpha,
                                                  a_batch_desc,
                                                  b_batch_desc,
                                                  &beta,
                                                  out_batch_desc,
                                                  gpu_type,
                                                  CUSPARSE_SPGEMM_DEFAULT,
                                                  spgemm_desc,
                                                  &buffer_a_size,
                                                  tmp_buffer_a_ptr);
    });

    dev_ctx_.CusparseCall([&](cusparseHandle_t handle) {
      phi::dynload::cusparseSpGEMM_compute(handle,
                                           GetTransposeOperation(transa),
                                           GetTransposeOperation(transb),
                                           &alpha,
                                           a_batch_desc,
                                           b_batch_desc,
                                           &beta,
                                           out_batch_desc,
                                           gpu_type,
                                           CUSPARSE_SPGEMM_DEFAULT,
                                           spgemm_desc,
                                           &buffer_b_size,
                                           nullptr);
    });

    phi::Allocator::AllocationPtr tmp_buffer_b = phi::memory_utils::Alloc(
        dev_ctx_.GetPlace(),
        buffer_b_size,
        phi::Stream(reinterpret_cast<phi::StreamId>(dev_ctx_.stream())));
    void* tmp_buffer_b_ptr = tmp_buffer_b->ptr();

    dev_ctx_.CusparseCall([&](cusparseHandle_t handle) {
      phi::dynload::cusparseSpGEMM_compute(handle,
                                           GetTransposeOperation(transa),
                                           GetTransposeOperation(transb),
                                           &alpha,
                                           a_batch_desc,
                                           b_batch_desc,
                                           &beta,
                                           out_batch_desc,
                                           gpu_type,
                                           CUSPARSE_SPGEMM_DEFAULT,
                                           spgemm_desc,
                                           &buffer_b_size,
                                           tmp_buffer_b_ptr);
    });

    int64_t out_num_crows, out_num_cols, out_num_values;

    dev_ctx_.CusparseCall([&](cusparseHandle_t handle) {
      phi::dynload::cusparseSpMatGetSize(
          out_batch_desc, &out_num_crows, &out_num_cols, &out_num_values);
    });

    out_batch_cols_vec[i].Resize(common::make_dim(out_num_values));
    dev_ctx_.template Alloc<int32_t>(&out_batch_cols_vec[i]);
    out_batch_values_vec[i].Resize(common::make_dim(out_num_values));
    dev_ctx_.template Alloc<T>(&out_batch_values_vec[i]);

    dev_ctx_.CusparseCall([&](cusparseHandle_t handle) {
      phi::dynload::cusparseCsrSetPointers(
          out_batch_desc,
          const_cast<int32_t*>(out_batch_crows_data),
          const_cast<int32_t*>(out_batch_cols_vec[i].data<int32_t>()),
          const_cast<T*>(out_batch_values_vec[i].data<T>()));
    });

    dev_ctx_.CusparseCall([&](cusparseHandle_t handle) {
      phi::dynload::cusparseSpGEMM_copy(handle,
                                        GetTransposeOperation(transa),
                                        GetTransposeOperation(transb),
                                        &alpha,
                                        a_batch_desc,
                                        b_batch_desc,
                                        &beta,
                                        out_batch_desc,
                                        gpu_type,
                                        CUSPARSE_SPGEMM_DEFAULT,
                                        spgemm_desc);
    });

    dev_ctx_.CusparseCall([&](cusparseHandle_t handle) {
      phi::dynload::cusparseSpGEMM_destroyDescr(spgemm_desc);
    });

    a_batch_crows_data += a_rows + 1;
    a_batch_cols_data += a_batch_nnz;
    a_batch_values_data += a_batch_nnz;

    b_batch_crows_data += b_rows + 1;
    b_batch_cols_data += b_batch_nnz;
    b_batch_values_data += b_batch_nnz;

    out_batch_crows_data += a_rows + 1;
  }

  if (batch_size == 1) {
    *(mat_out->mutable_cols()) = std::move(out_batch_cols_vec[0]);
    *(mat_out->mutable_values()) = std::move(out_batch_values_vec[0]);

  } else {
    std::vector<const DenseTensor*> cols_vec, values_vec;

    for (int i = 0; i < batch_size; ++i) {
      cols_vec.push_back(&out_batch_cols_vec[i]);
      values_vec.push_back(&out_batch_values_vec[i]);
    }

    phi::ConcatKernel<int32_t>(dev_ctx_, cols_vec, 0, mat_out->mutable_cols());
    phi::ConcatKernel<T>(dev_ctx_, values_vec, 0, mat_out->mutable_values());
  }

  if (mat_a.crows().dtype() == phi::DataType::INT64 ||
      mat_b.crows().dtype() == phi::DataType::INT64) {
    phi::CastKernel<int32_t>(
        dev_ctx_, *mat_out_crows, phi::DataType::INT64, mat_out_crows);
    phi::CastKernel<int32_t>(
        dev_ctx_, *mat_out_cols, phi::DataType::INT64, mat_out_cols);
  }
}
}  // namespace sparse
}  // namespace funcs
}  // namespace phi
