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
                  const framework::Tensor& A, framework::Tensor* A_inv) {
    const auto& mat_dims = A.dims();
    const int rank = mat_dims.size();
    int N = mat_dims[rank - 1];
    int batch_size = rank > 2 ? A.numel() / (N * N) : 1;

    std::vector<T*> cpu_ptrs(batch_size * 2);
    for (int i = 0; i < batch_size; ++i) {
      cpu_ptrs[i] = const_cast<T*>(A.data<T>()) + i * N * N;
      cpu_ptrs[i + batch_size] = A_inv->data<T>() + i * N * N;
      LOG(INFO) << "A_i: " << cpu_ptrs[i]
                << ", A_inv_i: " << cpu_ptrs[i + batch_size];
    }

    memory::allocation::AllocationPtr tmp_gpu_ptrs_data =
        memory::Alloc(context, cpu_ptrs.size() * sizeof(T*));
    memory::Copy(boost::get<platform::CUDAPlace>(context.GetPlace()),
                 tmp_gpu_ptrs_data->ptr(), platform::CPUPlace(),
                 static_cast<void*>(cpu_ptrs.data()),
                 cpu_ptrs.size() * sizeof(T*), context.stream());
    T** gpu_ptrs = reinterpret_cast<T**>(tmp_gpu_ptrs_data->ptr());
    T** gpu_mat_ptrs = gpu_ptrs;
    T** gpu_inv_ptrs = gpu_ptrs + batch_size;

    memory::allocation::AllocationPtr tmp_gpu_info_data =
        memory::Alloc(context, batch_size * sizeof(int));
    int* gpu_info_ptr = reinterpret_cast<int*>(tmp_gpu_info_data->ptr());

    context.CublasCall([&](cublasHandle_t handle) {
      CUBlas<T>::MATINV_BATCH(handle, N, gpu_mat_ptrs, N, gpu_inv_ptrs, N,
                              gpu_info_ptr, batch_size);
    });
  }
};

template class MatrixInverseFunctor<platform::CUDADeviceContext, float>;
template class MatrixInverseFunctor<platform::CUDADeviceContext, double>;

}  // namespace math
}  // namespace operators
}  // namespace paddle
