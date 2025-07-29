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
#include "paddle/phi/kernels/funcs/scatter.cu.h"

namespace phi {
namespace funcs {

#ifndef PADDLE_WITH_HIP
/**
 * Transform pivot array to permutation by swapping perm[i] and perm[pivot[i]]
 * from 0 to n-1, where pivot and perm have shape [batch_size, n].
 * Example:
 *    Input pivot = [[6, 7, 4, 5, 5, 7, 8, 8]]
 *    Output perm = [[5, 6, 3, 4, 2, 1, 7, 0]]
 */
__global__ void UnpackPivot(const int* __restrict__ pivot,
                            int* __restrict__ perm,
                            int64_t batch_size,
                            int64_t n) {
  constexpr int warp_size = 32;
  int warps_per_block = blockDim.x / warp_size;
  int warp_id = threadIdx.x / warp_size;
  int warp_offset = threadIdx.x % warp_size;
  int64_t offset = static_cast<int64_t>(blockIdx.x) * warps_per_block + warp_id;
  int64_t stride = static_cast<int64_t>(gridDim.x) * warps_per_block;

  for (; offset < batch_size; offset += stride) {
    // init perm[*, n] with 0...n-1
    for (int64_t i = warp_offset; i < n; i += warp_size) {
      perm[offset * n + i] = offset * n + i;
    }
    __syncwarp();

    // Since the swapping makes entirely discrete access, we only use the first
    // thread in each warp to avoid warp divergence.
    if (warp_offset > 0) continue;

    // Swap perm[i] and perm[pivot[i]] for i in 0...n-1
    for (int64_t i = offset * n; i < offset * n + n; ++i) {
      int64_t j = pivot[i] - 1 + offset * n;  // cublas use 1-index
      int tmp = perm[i];
      perm[i] = perm[j];
      perm[j] = tmp;
    }
  }
}

/**
 * Eliminate the L and U in equation:
 *    (U^T @ L^T @ P) @ X = B  (the U^T @ L^T @ P is stored in A)
 * by solving the inversion of L^T and U^T respectively. The result is:
 *    P @ X = L^T^-1 @ U^T^-1 @ B
 * and is stored in B.
 */
template <typename Context, typename T>
void SolveLU(const phi::funcs::BlasT<Context, T>& blas,
             int m,
             int n,
             const T* A,
             T* B,
             int batch_size) {
  constexpr T alpha = 1.0;
  for (int64_t i = 0; i < batch_size; ++i) {
    // Before: U^T @ L^T @ P @ X = B
    blas.TRSM(CblasRight,
              CblasLower,
              CblasTrans,
              CblasNonUnit,
              m,
              n,
              alpha,
              A + i * n * n,
              n,
              B + i * m * n,
              n);
    // After: L^T @ P @ X = U^T^-1 @ B
    blas.TRSM(CblasRight,
              CblasUpper,
              CblasTrans,
              CblasUnit,
              m,
              n,
              alpha,
              A + i * n * n,
              n,
              B + i * m * n,
              n);
    // After: P @ X = L^T^-1 @ U^T^-1 @ B
  }
}

// Batched version of SolveLU.
template <typename Context, typename T>
void BatchedSolveLU(const phi::funcs::BlasT<Context, T>& blas,
                    int m,
                    int n,
                    const T** A,
                    T** B,
                    int batch_size) {
  constexpr T alpha = 1.0;
  blas.BatchedTRSM(CblasRight,
                   CblasLower,
                   CblasTrans,
                   CblasNonUnit,
                   m,
                   n,
                   alpha,
                   A,
                   n,
                   B,
                   n,
                   batch_size);
  blas.BatchedTRSM(CblasRight,
                   CblasUpper,
                   CblasTrans,
                   CblasUnit,
                   m,
                   n,
                   alpha,
                   A,
                   n,
                   B,
                   n,
                   batch_size);
}
#endif

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
  int64_t batch_size = a_rank > 2 ? a.numel() / (n * n) : 1;

  const auto& b_dims = b.dims();
  const int b_rank = b_dims.size();
  int nrhs = b_dims[b_rank - 1];
  int ldb = n;

  // 1. Copy input A to a temporary tensor tmp_a for LU factorization.
  DenseTensor tmp_a(a.dtype());
  tmp_a.Resize(a.dims());
  context.template Alloc<T>(&tmp_a);
  phi::Copy(context, a, context.GetPlace(), false, &tmp_a);

  // 2. Transpose B and save it in out, because cuBlas assumes column-major
  // while Paddle uses row-majar.
  const auto& new_b_dims = getNewDimsVec(b_dims);
  out->Resize(common::make_ddim(new_b_dims));
  context.template Alloc<T>(out);
  phi::funcs::TransposeNormal<Context, T> trans;
  std::vector<int> new_axis = getNewAxis(b_rank);
  trans(context, b, out, new_axis);

  const T* a_data_in_gpu = tmp_a.data<T>();
  T* b_data_in_gpu = out->data<T>();

  std::vector<const T*> cpu_ptrs(batch_size * 2);
  for (int64_t i = 0; i < batch_size; ++i) {
    cpu_ptrs[i] = a_data_in_gpu + i * n * n;
    cpu_ptrs[i + batch_size] = b_data_in_gpu + i * n * nrhs;
  }

  // 3. Copy the addresses of A and B from host to device.
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

  // 4. Allocate device memory for BatchedGETRF's info and pivots.
  int64_t num_ints = batch_size * (n + 1);
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

  // 5. Performs LU factorization on A.
  blas.BatchedGETRF(n,
                    reinterpret_cast<T**>(tmp_gpu_ptrs_data->ptr()),
                    gpu_pivot_ptr,
                    gpu_info_ptr,
                    batch_size);
  // After: P @ A^T = L @ U

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

  // 6. Solve L and U in equation Ax = B where A = U^T @ L^T @ P.
  // The batched version is advantageous for small shapes, but has error for
  // large shapes. In this case, we call the non-batched version for batch_size
  // times instead.
  // Ref: https://docs.nvidia.com/cuda/cublas/#cublas-t-trsmbatched
  constexpr int max_batch_nrhs = 65535 * 8;  // max(gridDim.y) * 8
  if (batch_size > 1 && nrhs <= max_batch_nrhs) {
    BatchedSolveLU(blas,
                   nrhs,
                   n,
                   reinterpret_cast<const T**>(tmp_gpu_ptrs_data->ptr()),
                   gpu_tmp_b_ptrs,
                   batch_size);
  } else {
    SolveLU(blas, nrhs, n, a_data_in_gpu, b_data_in_gpu, batch_size);
  }

  // 7. Transpose B back to row-major form.
  DenseTensor tmp_b(b.type());
  tmp_b.Resize(b_dims);
  context.template Alloc<T>(&tmp_b);
  phi::funcs::TransposeNormal<Context, T> trans2;
  trans2(context, *out, &tmp_b, new_axis);

  // 8. Permute B according to pivots to get the final result.
  DenseTensor perm;
  perm.Resize({batch_size * n});
  context.template Alloc<int>(&perm);

  auto config =
      phi::backends::gpu::GetGpuLaunchConfig1D(context, batch_size * 32);
  auto stream = context.stream();
  UnpackPivot<<<config.block_per_grid, config.thread_per_block, 0, stream>>>(
      gpu_pivot_ptr, perm.data<int>(), batch_size, n);

  // fuse dims 0...n-2 because scatter only supports one index dim
  tmp_b.Resize({batch_size * n, nrhs});
  out->Resize({batch_size * n, nrhs});
  GPUScatterAssign<T>(context, tmp_b, perm, out);
  out->Resize(b_dims);
  // After: X = P^T @ L^T^-1 @ U^T^-1 @ B

#else
  compute_solve_eigen<Context, T>(context, a, b, out);
#endif
}

template class MatrixSolveFunctor<GPUContext, float>;
template class MatrixSolveFunctor<GPUContext, double>;

}  // namespace funcs
}  // namespace phi
