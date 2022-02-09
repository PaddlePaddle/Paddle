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

#include "paddle/fluid/operators/math/matrix_solve.h"
#include "paddle/fluid/framework/tensor_util.h"
#include "paddle/fluid/operators/math/blas.h"
#include "paddle/fluid/operators/math/math_function.h"
#include "paddle/fluid/operators/solve_op.h"
#include "paddle/fluid/platform/device_context.h"

namespace paddle {
namespace platform {
class CUDADeviceContext;
}  // namespace platform
}  // namespace paddle

namespace paddle {
namespace operators {
namespace math {

template <typename DeviceContext, typename T>
class MatrixSolveFunctor;

template <typename T>
class MatrixSolveFunctor<platform::CUDADeviceContext, T> {
 public:
  void operator()(const platform::CUDADeviceContext& context,
                  const framework::Tensor& a, const framework::Tensor& b,
                  framework::Tensor* out) {
#ifndef PADDLE_WITH_HIP

    // solve the equation: Ax = B,
    // use cuBlas cublas<S/D>getrfBatched funcion to performs the LU
    // factorization of each matrix A,
    // and then use cuBlas cublas<S/D>getriBatched function to solve the
    // equation after LU factorization.
    // ref:
    // https://docs.nvidia.com/cuda/cublas/index.html#cublas-lt-t-gt-getrfbatched
    const auto& a_dims = a.dims();
    const int a_rank = a_dims.size();
    int n = a_dims[a_rank - 1];
    int lda = n;
    int batch_size = a_rank > 2 ? a.numel() / (n * n) : 1;

    const auto& b_dims = b.dims();
    const int b_rank = b_dims.size();
    int nrhs = b_dims[b_rank - 1];
    int ldb = b_dims[b_rank - 2];

    // make sure the out dims is right
    out->Resize(b_dims);
    out->mutable_data<T>(context.GetPlace());

    // copy input A to a temporary tensor tmp_a,
    // LU factorization, written back to original matrix A, so in the beginning,
    // it's necessary to create a temporary tensor tmp_a.
    Tensor tmp_a(a.type());
    tmp_a.Resize(a.dims());
    tmp_a.mutable_data<T>(context.GetPlace());
    framework::TensorCopy(a, context.GetPlace(), &tmp_a);

    // copy input B to a temporary tensor tmp_b, and transpose tmp_b,
    // because cuBlas assumes column-major while Paddle uses row-majar.
    Tensor tmp_b(b.type());
    const auto& new_dims_vec = getNewDimsVec(b_dims);
    tmp_b.Resize(framework::make_ddim(new_dims_vec));
    tmp_b.mutable_data<T>(context.GetPlace());
    math::TransposeNormal<platform::CUDADeviceContext, T> trans;
    std::vector<int> new_axis = getNewAxis(b_rank);
    trans(context, b, &tmp_b, new_axis);

    const T* a_data_in_gpu = tmp_a.data<T>();
    const T* b_data_in_gpu = tmp_b.data<T>();

    std::vector<const T*> cpu_ptrs(batch_size * 2);
    for (int i = 0; i < batch_size; ++i) {
      cpu_ptrs[i] = a_data_in_gpu + i * n * n;
      cpu_ptrs[i + batch_size] = b_data_in_gpu + i * n * nrhs;
    }

    // Copy the addresses of A and tmp_b from host to device.
    memory::allocation::AllocationPtr tmp_gpu_ptrs_data =
        memory::Alloc(context, cpu_ptrs.size() * sizeof(T*));
    memory::Copy(context.GetPlace(), tmp_gpu_ptrs_data->ptr(),
                 platform::CPUPlace(), static_cast<void*>(cpu_ptrs.data()),
                 cpu_ptrs.size() * sizeof(T*), context.stream());

    T** gpu_tmp_b_ptrs =
        reinterpret_cast<T**>(tmp_gpu_ptrs_data->ptr()) + batch_size;

    // Allocate device memory for BatchedGETRF's info and pivots.
    int num_ints = n < 32 ? batch_size : batch_size * (n + 1);
    memory::allocation::AllocationPtr tmp_gpu_info_data =
        memory::Alloc(context, num_ints * sizeof(int));
    int* gpu_info_ptr = reinterpret_cast<int*>(tmp_gpu_info_data->ptr());

    auto blas = math::GetBlas<platform::CUDADeviceContext, T>(context);

    // only for singular checking
    std::vector<int> info;
    info.resize(batch_size);

    int* gpu_pivot_ptr =
        reinterpret_cast<int*>(tmp_gpu_info_data->ptr()) + batch_size;

    // This function performs the LU factorization of each matrix A by the
    // equation A = L * U. L and U are written back to original matrix A,
    // and diagonal elements of L are discarded.
    blas.BatchedGETRF(n, reinterpret_cast<T**>(tmp_gpu_ptrs_data->ptr()),
                      gpu_pivot_ptr, gpu_info_ptr, batch_size);

    // check whether BatchedGETRF is executed successfully or not
    memory::Copy(platform::CPUPlace(), info.data(), context.GetPlace(),
                 gpu_info_ptr, sizeof(int) * batch_size, context.stream());
    for (int i = 0; i < batch_size; ++i) {
      PADDLE_ENFORCE_EQ(info[i], 0,
                        platform::errors::PreconditionNotMet(
                            "For batch [%d]: U(%d, %d) is zero, singular U. "
                            "Please check the matrix value and change it to a "
                            "non-singular matrix",
                            i, info[i], info[i]));
    }

    // hold the result code from BatchedGETRS
    int host_info = 0;

    // to solve the equation after LU factorization
    CBLAS_TRANSPOSE transA = CblasTrans;
    blas.BatchedGETRS(
        transA, n, nrhs, reinterpret_cast<const T**>(tmp_gpu_ptrs_data->ptr()),
        lda, gpu_pivot_ptr, gpu_tmp_b_ptrs, ldb, &host_info, batch_size);

    // check whether BatchedGETRS is executed successfully or not
    PADDLE_ENFORCE_EQ(host_info, 0,
                      platform::errors::InvalidArgument(
                          "The [%d]'th argument to cublas*getrsBatched had "
                          "an illegal value.",
                          -host_info));

    // transpose tmp_b to get the final result in row-major form.
    math::TransposeNormal<platform::CUDADeviceContext, T> trans2;
    trans2(context, tmp_b, out, new_axis);

#else
    compute_solve_eigen<platform::CUDADeviceContext, T>(context, a, b, out);
#endif
  }
};

template class MatrixSolveFunctor<platform::CUDADeviceContext, float>;
template class MatrixSolveFunctor<platform::CUDADeviceContext, double>;

template <typename T>
class TriangularSolveFunctor<platform::CUDADeviceContext, T> {
 public:
  void operator()(const platform::CUDADeviceContext& context, const Tensor* a,
                  Tensor* b, bool left, bool upper, bool transpose,
                  bool unitriangular) {
    CBLAS_SIDE side = left ? CblasLeft : CblasRight;
    CBLAS_UPLO uplo = upper ? CblasUpper : CblasLower;
    CBLAS_TRANSPOSE transA = transpose ? CblasTrans : CblasNoTrans;
    CBLAS_DIAG diag = unitriangular ? CblasUnit : CblasNonUnit;

    const T* a_data = a->data<T>();
    T* b_data = b->mutable_data<T>(context.GetPlace());

    int a_dim_size = a->dims().size();
    int b_dim_size = b->dims().size();

    int M = static_cast<int>(b->dims()[b_dim_size - 2]);
    int N = static_cast<int>(b->dims()[b_dim_size - 1]);
    auto lda = left ? std::max(1, M) : std::max(1, N);
    auto ldb = std::max(1, N);

    int batch_size = 1;
    auto& a_dim = a->dims();
    for (int i = 0; i < a_dim_size - 2; i++) {
      batch_size *= a_dim[i];
    }

    auto blas = math::GetBlas<platform::CUDADeviceContext, T>(context);
    if (batch_size <= 8 && M >= 64) {
      for (auto i = 0; i < batch_size; i++) {
        blas.TRSM(side, uplo, transA, diag, M, N, static_cast<T>(1.0),
                  a_data + i * M * M, lda, b_data + i * N * M, ldb);
      }
    } else {
      std::vector<const T*> cpu_ptrs(batch_size * 2);
      for (int i = 0; i < batch_size; ++i) {
        cpu_ptrs[i] = a_data + i * M * M;
        cpu_ptrs[i + batch_size] = b_data + i * M * N;
      }

      // Copy the addresses of A and tmp_b from host to device.
      memory::allocation::AllocationPtr tmp_gpu_ptrs_data =
          memory::Alloc(context, cpu_ptrs.size() * sizeof(T*));
      memory::Copy(context.GetPlace(), tmp_gpu_ptrs_data->ptr(),
                   platform::CPUPlace(), static_cast<void*>(cpu_ptrs.data()),
                   cpu_ptrs.size() * sizeof(T*), context.stream());

      const T** gpu_a_ptrs =
          reinterpret_cast<const T**>(tmp_gpu_ptrs_data->ptr());
      T** gpu_b_ptrs =
          reinterpret_cast<T**>(tmp_gpu_ptrs_data->ptr()) + batch_size;
      blas.BatchedTRSM(side, uplo, transA, diag, M, N, static_cast<T>(1.0),
                       gpu_a_ptrs, lda, gpu_b_ptrs, ldb, batch_size);
    }
  }
};

template class TriangularSolveFunctor<platform::CUDADeviceContext, float>;
template class TriangularSolveFunctor<platform::CUDADeviceContext, double>;

}  // namespace math
}  // namespace operators
}  // namespace paddle
