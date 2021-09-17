// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#include "Eigen/Core"
#include "paddle/fluid/memory/memory.h"
#include "paddle/fluid/operators/math/complex_functors.h"
#include "paddle/fluid/operators/svd_helper.h"
#ifdef PADDLE_WITH_CUDA
#include "paddle/fluid/platform/dynload/cusolver.h"
#endif  // PADDLE_WITH_CUDA

namespace paddle {
namespace operators {
namespace math {

template <typename T, size_t D, int MajorType = Eigen::RowMajor,
          typename IndexType = Eigen::DenseIndex>
using EigenTensor = framework::EigenTensor<T, D, MajorType, IndexType>;

template <typename T, int MajorType = Eigen::RowMajor,
          typename IndexType = Eigen::DenseIndex>
using InputMatrixMap = Eigen::Map<
    const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>;

template <typename T, int MajorType = Eigen::RowMajor,
          typename IndexType = Eigen::DenseIndex>
using OutputMatrixMap = Eigen::Map<
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>;

template <typename ValueType>
inline void ComputeFloatEigenvaluesAndVectors(ValueType *x_data,
                                              ValueType *eigenvalues_data,
                                              ValueType *eigenvectors_data,
                                              int batches, int rows, int cols,
                                              bool has_vectors) {
  int stride = rows * cols;
  for (int i = 0; i < batches; i++) {
    auto m = InputMatrixMap<ValueType>(x_data + i * stride, rows, cols);
    auto eigenvalues =
        OutputMatrixMap<ValueType>(eigenvalues_data + i * rows, 1, rows);
    auto eigenvectors =
        OutputMatrixMap<ValueType>(eigenvectors_data + i * stride, rows, cols);

    Eigen::SelfAdjointEigenSolver<Eigen::Matrix<
        ValueType, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
        eigen_solver(m, has_vectors ? Eigen::ComputeEigenvectors
                                    : Eigen::EigenvaluesOnly);
    PADDLE_ENFORCE_EQ(
        eigen_solver.info(), Eigen::Success,
        platform::errors::InvalidArgument(
            "Self Adjoint Eigen decomposition is not successful. "
            "The %d-th input matrice might not be not be positive definite.",
            i));

    eigenvalues = eigen_solver.eigenvalues().transpose();
    if (has_vectors) {
      eigenvectors = eigen_solver.eigenvectors().transpose();
    }
  }
}

template <typename T, typename ValueType>
inline void ComputeComplexEigenvaluesAndVectors(T *x_data,
                                                ValueType *eigenvalues_data,
                                                T *eigenvectors_data,
                                                int batches, int rows, int cols,
                                                bool has_vectors) {
  using Complex = std::complex<ValueType>;
  Complex *input = reinterpret_cast<Complex *>(x_data);
  Complex *eigenvectors_data_ = reinterpret_cast<Complex *>(eigenvectors_data);

  int stride = rows * cols;
  for (int i = 0; i < batches; i++) {
    auto m = InputMatrixMap<Complex>(input + i * stride, rows, cols);
    auto eigenvalues =
        OutputMatrixMap<ValueType>(eigenvalues_data + i * rows, 1, rows);
    auto eigenvectors =
        OutputMatrixMap<Complex>(eigenvectors_data_ + i * stride, rows, cols);

    Eigen::SelfAdjointEigenSolver<
        Eigen::Matrix<Complex, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
        eigen_solver(m, has_vectors ? Eigen::ComputeEigenvectors
                                    : Eigen::EigenvaluesOnly);
    PADDLE_ENFORCE_EQ(
        eigen_solver.info(), Eigen::Success,
        platform::errors::InvalidArgument(
            "Self Adjoint Eigen decomposition is not successful. "
            "The %d-th input matrice might not be not be positive definite.",
            i));

    eigenvalues = eigen_solver.eigenvalues().transpose();
    if (has_vectors) {
      eigenvectors = eigen_solver.eigenvectors().transpose();
    }
  }
}

inline int64_t GetBatchSize(framework::DDim dims) {
  int64_t batch_size = 1;
  auto dim_size = dims.size();
  for (int i = 0; i < dim_size - 2; i++) {
    batch_size *= dims[i];
  }
  return batch_size;
}

// Calculates the eigenvalues ​​and eigenvectors of Hermitian or real
// symmetric matrices, and uses the variable has_vectors to
// control whether to return the eigenvectors.
template <typename DeviceContext, typename ValueType, typename T>
struct MatrixEighFunctorCPU {
 public:
  void operator()(const framework::ExecutionContext &ctx, const Tensor &input,
                  Tensor *eigen_values, Tensor *eigen_vectors, bool is_lower,
                  bool has_vectors) {
    auto dims = input.dims();
    auto output_value_dim = eigen_values->dims();

    int64_t batch_size = 1;
    int dim_size = dims.size();
    for (int64_t i = 0; i < dim_size - 2; i++) {
      batch_size *= dims[i];
    }
    auto dito = DeviceIndependenceTensorOperations<DeviceContext, T>(ctx);
    Tensor input_tensor;
    TensorCopy(input, ctx.GetPlace(), &input_tensor);
    if (!is_lower) {
      input_tensor = dito.Transpose(input);
    }
    int rows = dims[dims.size() - 2];

    auto *value_data =
        eigen_values->mutable_data<ValueType>(output_value_dim, ctx.GetPlace());

    if (framework::IsComplexType(input_tensor.type())) {
      auto *x_data = input_tensor.data<T>();
      auto *vector_data = eigen_vectors->mutable_data<T>(dims, ctx.GetPlace());
      ComputeComplexEigenvaluesAndVectors<T, ValueType>(
          x_data, value_data, vector_data, batch_size, rows, rows, has_vectors);
    } else {
      auto *x_data = input_tensor.data<ValueType>();
      auto *vector_data =
          eigen_vectors->mutable_data<ValueType>(dims, ctx.GetPlace());
      ComputeFloatEigenvaluesAndVectors<ValueType>(
          x_data, value_data, vector_data, batch_size, rows, rows, has_vectors);
    }
    if (has_vectors) {
      *eigen_vectors = dito.Transpose(*eigen_vectors);
    }
  }
};

#ifdef PADDLE_WITH_CUDA

// Calculates the eigenvalues ​​and eigenvectors of Hermitian or real
// symmetric matrices on GPU, and uses the variable has_vectors
// to control whether to return the eigenvectors.
template <typename ValueType, typename T>
struct MatrixEighFunctor {
 public:
  void operator()(const framework::ExecutionContext &ctx, const Tensor &input,
                  Tensor *eigen_values, Tensor *eigen_vectors, bool is_lower,
                  bool has_vectors) {
    auto *out_value = eigen_values->mutable_data<ValueType>(ctx.GetPlace());
    auto *out_vector = eigen_vectors->mutable_data<T>(ctx.GetPlace());

    auto &dims = input.dims();
    int dim_size = dims.size();
    int64_t batch_size = GetBatchSize(dims);

    cublasFillMode_t uplo =
        is_lower ? CUBLAS_FILL_MODE_LOWER : CUBLAS_FILL_MODE_UPPER;
    cusolverEigMode_t jobz =
        has_vectors ? CUSOLVER_EIG_MODE_VECTOR : CUSOLVER_EIG_MODE_NOVECTOR;

    int n = dims[dim_size - 1];
    int lda = std::max<int>(1, n);
    auto vector_stride = dims[dim_size - 1] * dims[dim_size - 2];
    auto values_stride = dims[dim_size - 1];

    auto &dev_ctx = ctx.template device_context<platform::CUDADeviceContext>();
    auto dito =
        math::DeviceIndependenceTensorOperations<platform::CUDADeviceContext,
                                                 T>(ctx);
    Tensor output_v_var_trans = dito.Transpose(input);
    TensorCopy(output_v_var_trans, ctx.GetPlace(), eigen_vectors);

    int lwork = 0;
    auto info = memory::Alloc(dev_ctx, sizeof(int) * batch_size);
    auto *info_ptr = reinterpret_cast<int *>(info->ptr());

    // When the input type is float32, and the feature value input dimension is
    // greater than or equal to [*,32,32]  and less than or equal to
    // [*,512,512], Syevj has better performance.
    bool use_syevj =
        (eigen_vectors->type() == framework::proto::VarType::FP32 &&
         values_stride >= 32 && values_stride <= 512);

    syevjInfo_t syevj_params;
    if (use_syevj) {
      PADDLE_ENFORCE_CUDA_SUCCESS(
          platform::dynload::cusolverDnCreateSyevjInfo(&syevj_params));
      PADDLE_ENFORCE_CUDA_SUCCESS(
          platform::dynload::cusolverDnSsyevj_bufferSize(
              dev_ctx.cusolver_dn_handle(), jobz, uplo, n,
              reinterpret_cast<const float *>(out_vector), lda,
              reinterpret_cast<const float *>(out_value), &lwork,
              syevj_params));
    } else {
      EvdBuffer(dev_ctx.cusolver_dn_handle(), jobz, uplo, n, out_vector, lda,
                out_value, &lwork);
    }

    auto work = memory::Alloc(dev_ctx, sizeof(T) * lwork);
    auto *work_ptr = reinterpret_cast<T *>(work->ptr());

    for (auto i = 0; i < batch_size; i++) {
      auto vector_data = out_vector + i * vector_stride;
      auto value_data = out_value + i * values_stride;
      auto handle = dev_ctx.cusolver_dn_handle();
      if (use_syevj) {
        PADDLE_ENFORCE_CUDA_SUCCESS(platform::dynload::cusolverDnSsyevj(
            handle, jobz, uplo, n, reinterpret_cast<float *>(vector_data), lda,
            reinterpret_cast<float *>(value_data),
            reinterpret_cast<float *>(work_ptr), lwork, info_ptr,
            syevj_params));
      } else {
        Evd(handle, jobz, uplo, n, vector_data, lda, value_data, work_ptr,
            lwork, info_ptr);
      }
      int error_info;
      memory::Copy(platform::CPUPlace(), &error_info,
                   BOOST_GET_CONST(platform::CUDAPlace, dev_ctx.GetPlace()),
                   info_ptr, sizeof(int), dev_ctx.stream());
      PADDLE_ENFORCE_EQ(
          error_info, 0,
          platform::errors::PreconditionNotMet(
              "For batch [%d]: the [%d] argument had an illegal value", i,
              error_info));
    }

    if (use_syevj) {
      PADDLE_ENFORCE_CUDA_SUCCESS(
          platform::dynload::cusolverDnDestroySyevjInfo(syevj_params));
    }

    if (has_vectors) {
      *eigen_vectors = dito.Transpose(*eigen_vectors);
    }
  }

  inline void EvdBuffer(cusolverDnHandle_t handle, cusolverEigMode_t jobz,
                        cublasFillMode_t uplo, int n, const T *A, int lda,
                        const ValueType *W, int *lwork) const;

  inline void Evd(cusolverDnHandle_t handle, cusolverEigMode_t jobz,
                  cublasFillMode_t uplo, int n, T *A, int lda, ValueType *W,
                  T *work, int lwork, int *devInfo) const;
};

#define FUNC_WITH_TYPES(m)                                       \
  m(float, float, Ssy, float) m(double, double, Dsy, double)     \
      m(float, paddle::platform::complex<float>, Che, cuComplex) \
          m(double, paddle::platform::complex<double>, Zhe, cuDoubleComplex)

#define EVDBUFFER_INSTANCE(ValueType, T, C, CastType)                          \
  template <>                                                                  \
  inline void MatrixEighFunctor<ValueType, T>::EvdBuffer(                      \
      cusolverDnHandle_t handle, cusolverEigMode_t jobz,                       \
      cublasFillMode_t uplo, int n, const T *A, int lda, const ValueType *W,   \
      int *lwork) const {                                                      \
    PADDLE_ENFORCE_CUDA_SUCCESS(                                               \
        platform::dynload::cusolverDn##C##evd_bufferSize(                      \
            handle, jobz, uplo, n, reinterpret_cast<const CastType *>(A), lda, \
            W, lwork));                                                        \
  }

FUNC_WITH_TYPES(EVDBUFFER_INSTANCE);

#define EVD_INSTANCE(ValueType, T, C, CastType)                           \
  template <>                                                             \
  inline void MatrixEighFunctor<ValueType, T>::Evd(                       \
      cusolverDnHandle_t handle, cusolverEigMode_t jobz,                  \
      cublasFillMode_t uplo, int n, T *A, int lda, ValueType *W, T *work, \
      int lwork, int *devInfo) const {                                    \
    PADDLE_ENFORCE_CUDA_SUCCESS(platform::dynload::cusolverDn##C##evd(    \
        handle, jobz, uplo, n, reinterpret_cast<CastType *>(A), lda, W,   \
        reinterpret_cast<CastType *>(work), lwork, devInfo));             \
  }

FUNC_WITH_TYPES(EVD_INSTANCE);

#undef FUNC_WITH_TYPES
#undef EVDBUFFER_INSTANCE
#undef EVD_INSTANCE

#endif  // PADDLE_WITH_CUDA

}  // namespace math
}  // namespace operators
}  // namespace paddle
