/* Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/operators/math/matrix_inverse.h"
#include "paddle/fluid/memory/malloc.h"
#include "paddle/fluid/operators/math/blas.h"

namespace paddle {
namespace operators {
namespace math {

template <typename T>
class MatrixInverseFunctor<platform::CUDADeviceContext, T> {
 public:
  void operator()(const platform::CUDADeviceContext& context,
                  const framework::Tensor& a, framework::Tensor* a_inv) {
    const auto& mat_dims = a.dims();
    const int rank = mat_dims.size();
    int n = mat_dims[rank - 1];
    int batch_size = rank > 2 ? a.numel() / (n * n) : 1;

    memory::allocation::AllocationPtr tmp_gpu_mat_data;
    const T* gpu_mat = a.data<T>();
    if (n >= 32) {
      // Copy all elements of input matrix A to a temporary memory space to
      // avoid being overriden by getrf.
      tmp_gpu_mat_data = memory::Alloc(context, a.numel() * sizeof(T));
      memory::Copy(boost::get<platform::CUDAPlace>(context.GetPlace()),
                   tmp_gpu_mat_data->ptr(),
                   boost::get<platform::CUDAPlace>(context.GetPlace()),
                   a.data<void>(), a.numel() * sizeof(T), context.stream());
      gpu_mat = reinterpret_cast<const T*>(tmp_gpu_mat_data->ptr());
    }

    std::vector<const T*> cpu_ptrs(batch_size * 2);
    for (int i = 0; i < batch_size; ++i) {
      cpu_ptrs[i] = gpu_mat + i * n * n;
      cpu_ptrs[i + batch_size] = a_inv->data<T>() + i * n * n;
    }

    // Copy the addresses of A and A_inv from host to device.
    memory::allocation::AllocationPtr tmp_gpu_ptrs_data =
        memory::Alloc(context, cpu_ptrs.size() * sizeof(T*));
    memory::Copy(boost::get<platform::CUDAPlace>(context.GetPlace()),
                 tmp_gpu_ptrs_data->ptr(), platform::CPUPlace(),
                 static_cast<void*>(cpu_ptrs.data()),
                 cpu_ptrs.size() * sizeof(T*), context.stream());
    T** gpu_inv_ptrs =
        reinterpret_cast<T**>(tmp_gpu_ptrs_data->ptr()) + batch_size;

    // Allocate device memory for info and pivots.
    int num_ints = n < 32 ? batch_size : batch_size * (n + 1);
    memory::allocation::AllocationPtr tmp_gpu_info_data =
        memory::Alloc(context, num_ints * sizeof(int));
    int* gpu_info_ptr = reinterpret_cast<int*>(tmp_gpu_info_data->ptr());

    auto blas = math::GetBlas<platform::CUDADeviceContext, T>(context);

    // This functions in cuBLAS is intended to be used for matrices of small
    // sizes where the launch overhead is a significant factor.
    // TODO(Xreki): call function in cusolver for large matrices.
    if (n < 32) {
      // cublas<S/D>matinvBatched is a short cut of cublas<S/D>getrfBatched
      // plus cublas<S/D>getriBatched.
      // However it only works if N is less than 32. If not, we need to
      // go through cublas<S/D>getrfBatched and cublas<S/D>getriBatched.
      blas.BatchedMatInv(n,
                         reinterpret_cast<const T**>(tmp_gpu_ptrs_data->ptr()),
                         gpu_inv_ptrs, gpu_info_ptr, batch_size);
    } else {
      // This function performs the LU factorization of each matrix A by the
      // equation P * A = L * U. L and U are written back to original matrix A,
      // and diagonal elements of L are discarded.
      int* gpu_pivot_ptr =
          reinterpret_cast<int*>(tmp_gpu_info_data->ptr()) + batch_size;
      blas.BatchedGETRF(n, reinterpret_cast<T**>(tmp_gpu_ptrs_data->ptr()),
                        gpu_pivot_ptr, gpu_info_ptr, batch_size);

      blas.BatchedGETRI(n,
                        reinterpret_cast<const T**>(tmp_gpu_ptrs_data->ptr()),
                        gpu_pivot_ptr, gpu_inv_ptrs, gpu_info_ptr, batch_size);
    }
  }
};

template class MatrixInverseFunctor<platform::CUDADeviceContext, float>;
template class MatrixInverseFunctor<platform::CUDADeviceContext, double>;

}  // namespace math
}  // namespace operators
}  // namespace paddle
