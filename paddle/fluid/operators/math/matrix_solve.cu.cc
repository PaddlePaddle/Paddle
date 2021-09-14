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
#include <type_traits>
#include "paddle/fluid/framework/tensor_util.h"
#include "paddle/fluid/operators/math/blas.h"
#include "paddle/fluid/operators/math/math_function.h"
#include "paddle/fluid/operators/matmul_v2_op.h"
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

// for TransposeNormal, transpose the last two dimmentions
std::vector<int> getNewAxis(const int b_rank) {
  std::vector<int> axis_1 = {0};
  std::vector<int> axis_2 = {1, 0};
  std::vector<int> axis_3 = {0, 2, 1};
  std::vector<int> axis_4 = {0, 1, 3, 2};
  std::vector<int> axis_5 = {0, 1, 2, 4, 3};
  std::vector<int> axis_6 = {0, 1, 2, 3, 5, 4};
  std::vector<int> axis_7 = {0, 1, 2, 3, 4, 6, 5};
  std::vector<int> axis_8 = {0, 1, 2, 3, 4, 5, 7, 6};
  std::vector<int> axis_9 = {0, 1, 2, 3, 4, 5, 6, 8, 7};
  switch (b_rank) {
    case 1:
      return axis_1;
      break;
    case 2:
      return axis_2;
      break;
    case 3:
      return axis_3;
      break;
    case 4:
      return axis_4;
      break;
    case 5:
      return axis_5;
      break;
    case 6:
      return axis_6;
      break;
    case 7:
      return axis_7;
      break;
    case 8:
      return axis_8;
      break;
    default:
      return axis_9;
  }
}

// for Resize
std::vector<int64_t> getNewDimsVec(const DDim& b_dims) {
  std::vector<int64_t> b_dims_vec = paddle::framework::vectorize(b_dims);
  int size = b_dims_vec.size();
  if (b_dims_vec.size() >= 2) {
    // swap the last 2 elements in b_dims_vec
    int64_t temp = b_dims_vec[size - 1];
    b_dims_vec[size - 1] = b_dims_vec[size - 2];
    b_dims_vec[size - 2] = temp;
    return b_dims_vec;
  }
  // if b_dims_vec.size() == 1, just retun original vec
  return b_dims_vec;
}

template <typename T>
class MatrixSolveFunctor<platform::CUDADeviceContext, T> {
 public:
  void operator()(const platform::CUDADeviceContext& context,
                  const framework::Tensor& a, const framework::Tensor& b,
                  framework::Tensor* out) {
#ifndef PADDLE_WITH_HIP

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
    TensorCopy(a, context.GetPlace(), &tmp_a);

    // copy input B to a temporary tensor tmp_b, and transpose tmp_b,
    // because cuBlas assumes column-major while Paddle uses row-majar.
    Tensor tmp_b(b.type());
    const auto& new_dims_vec = getNewDimsVec(b_dims);
    tmp_b.Resize(framework::make_ddim(new_dims_vec));
    tmp_b.mutable_data<T>(context.GetPlace());
    math::TransposeNormal<platform::CUDADeviceContext, T> trans;
    std::vector<int> new_axis = getNewAxis(b_rank);
    trans(context, b, &tmp_b, new_axis);

    memory::allocation::AllocationPtr tmp_a_data_in_gpu;
    const T* a_data_in_gpu = tmp_a.data<T>();

    std::vector<const T*> cpu_ptrs(batch_size * 2);
    for (int i = 0; i < batch_size; ++i) {
      cpu_ptrs[i] = a_data_in_gpu + i * n * n;
      cpu_ptrs[i + batch_size] = tmp_b.data<T>() + i * n * nrhs;
    }

    // Copy the addresses of A and tmp_b from host to device.
    memory::allocation::AllocationPtr tmp_gpu_ptrs_data =
        memory::Alloc(context, cpu_ptrs.size() * sizeof(T*));
    memory::Copy(boost::get<platform::CUDAPlace>(context.GetPlace()),
                 tmp_gpu_ptrs_data->ptr(), platform::CPUPlace(),
                 static_cast<void*>(cpu_ptrs.data()),
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
    memory::Copy(platform::CPUPlace(), info.data(),
                 BOOST_GET_CONST(platform::CUDAPlace, context.GetPlace()),
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

}  // namespace math
}  // namespace operators
}  // namespace paddle
