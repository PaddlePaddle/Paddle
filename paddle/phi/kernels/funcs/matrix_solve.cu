/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/phi/kernels/funcs/matrix_solve.h"
#include "paddle/phi/common/memory_utils.h"
#include "paddle/phi/core/tensor_utils.h"
#include "paddle/phi/kernels/funcs/blas/blas.h"
#include "paddle/phi/kernels/funcs/math_function.h"

namespace phi {
namespace funcs {

template <typename Context, typename T>
void MatrixSolveFunctor<Context, T>::operator()(const Context& context,
                                                const DenseTensor& a,
                                                const DenseTensor& b,
                                                DenseTensor* out) {
#ifndef PADDLE_WITH_HIP

  // solve the equation: Ax = B,
  // use cuBlas cublas<S/D>getrfBatched function to performs the LU
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

  context.template Alloc<T>(out);

  // copy input A to a temporary tensor tmp_a,
  // LU factorization, written back to original matrix A, so in the beginning,
  // it's necessary to create a temporary tensor tmp_a.
  DenseTensor tmp_a(a.dtype());
  tmp_a.Resize(a.dims());

  context.template Alloc<T>(&tmp_a);
  phi::Copy(context, a, context.GetPlace(), false, &tmp_a);

  // copy input B to a temporary tensor tmp_b, and transpose tmp_b,
  // because cuBlas assumes column-major while Paddle uses row-majar.
  DenseTensor tmp_b(b.type());
  const auto& new_dims_vec = getNewDimsVec(b_dims);
  tmp_b.Resize(common::make_ddim(new_dims_vec));
  context.template Alloc<T>(&tmp_b);
  phi::funcs::TransposeNormal<Context, T> trans;
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
  phi::Allocator::AllocationPtr tmp_gpu_ptrs_data = phi::memory_utils::Alloc(
      context.GetPlace(),
      cpu_ptrs.size() * sizeof(T*),
      phi::Stream(reinterpret_cast<phi::StreamId>(context.stream())));
  memory_utils::Copy(context.GetPlace(),
                     tmp_gpu_ptrs_data->ptr(),
                     phi::CPUPlace(),
                     static_cast<void*>(cpu_ptrs.data()),
                     cpu_ptrs.size() * sizeof(T*),
                     context.stream());

  T** gpu_tmp_b_ptrs =
      reinterpret_cast<T**>(tmp_gpu_ptrs_data->ptr()) + batch_size;

  // Allocate device memory for BatchedGETRF's info and pivots.
  int num_ints = n < 32 ? batch_size : batch_size * (n + 1);
  phi::Allocator::AllocationPtr tmp_gpu_info_data = phi::memory_utils::Alloc(
      context.GetPlace(),
      num_ints * sizeof(int),
      phi::Stream(reinterpret_cast<phi::StreamId>(context.stream())));
  int* gpu_info_ptr = reinterpret_cast<int*>(tmp_gpu_info_data->ptr());

  auto blas = phi::funcs::GetBlas<Context, T>(context);

  // only for singular checking
  std::vector<int> info;
  info.resize(batch_size);

  int* gpu_pivot_ptr =
      reinterpret_cast<int*>(tmp_gpu_info_data->ptr()) + batch_size;

  // This function performs the LU factorization of each matrix A by the
  // equation A = L * U. L and U are written back to original matrix A,
  // and diagonal elements of L are discarded.
  blas.BatchedGETRF(n,
                    reinterpret_cast<T**>(tmp_gpu_ptrs_data->ptr()),
                    gpu_pivot_ptr,
                    gpu_info_ptr,
                    batch_size);

  // check whether BatchedGETRF is executed successfully or not
  memory_utils::Copy(phi::CPUPlace(),
                     info.data(),
                     context.GetPlace(),
                     gpu_info_ptr,
                     sizeof(int) * batch_size,
                     context.stream());
  for (int i = 0; i < batch_size; ++i) {
    PADDLE_ENFORCE_EQ(info[i],
                      0,
                      common::errors::PreconditionNotMet(
                          "For batch [%d]: U(%d, %d) is zero, singular U. "
                          "Please check the matrix value and change it to a "
                          "non-singular matrix",
                          i,
                          info[i],
                          info[i]));
  }

  // hold the result code from BatchedGETRS
  int host_info = 0;

  // to solve the equation after LU factorization
  CBLAS_TRANSPOSE transA = CblasTrans;
  blas.BatchedGETRS(transA,
                    n,
                    nrhs,
                    reinterpret_cast<const T**>(tmp_gpu_ptrs_data->ptr()),
                    lda,
                    gpu_pivot_ptr,
                    gpu_tmp_b_ptrs,
                    ldb,
                    &host_info,
                    batch_size);

  // check whether BatchedGETRS is executed successfully or not
  PADDLE_ENFORCE_EQ(host_info,
                    0,
                    common::errors::InvalidArgument(
                        "The [%d]'th argument to cublas*getrsBatched had "
                        "an illegal value.",
                        -host_info));

  // transpose tmp_b to get the final result in row-major form.
  phi::funcs::TransposeNormal<Context, T> trans2;
  trans2(context, tmp_b, out, new_axis);

#else
  compute_solve_eigen<Context, T>(context, a, b, out);
#endif
}

template class MatrixSolveFunctor<GPUContext, float>;
template class MatrixSolveFunctor<GPUContext, double>;

}  // namespace funcs
}  // namespace phi
