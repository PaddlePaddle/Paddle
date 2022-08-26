// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/phi/kernels/sparse/fused_attention_grad_kernel.h"

#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/math_cuda_utils.h"
#include "paddle/phi/kernels/funcs/sparse/sparse_blas.h"
#include "paddle/phi/kernels/sparse/empty_kernel.h"
#include "paddle/phi/kernels/sparse/matmul_grad_kernel.h"

namespace phi {
namespace sparse {

template <typename T>
__global__ void AttnSoftmaxGpuGradKernel(const int64_t* out_crows,
                                         const T* out_values,
                                         const T* dout_values,
                                         T* dx_values,
                                         int M,
                                         int total_row_num,
                                         float scale,
                                         int batch_nnz) {
  // dx = (dout - sum(dout * out)) * out
  int row = blockIdx.x * blockDim.y + threadIdx.y;
  if (row >= total_row_num) return;

  int cur_batch = row / M;
  int crow_idx = cur_batch * (M + 1) + (row % M);
  int row_first = cur_batch * batch_nnz + static_cast<int>(out_crows[crow_idx]);
  int row_nnz = static_cast<int>(out_crows[crow_idx + 1] - out_crows[crow_idx]);
  if (row_nnz == 0) return;

  T mul = 0;
  for (int idx = threadIdx.x; idx < row_nnz; idx += blockDim.x) {
    mul += out_values[row_first + idx] * dout_values[row_first + idx];
  }
  T mul_sum = phi::funcs::warpReduceSum<T>(mul, 0xFFFFFFFF);

  for (int idx = threadIdx.x; idx < row_nnz; idx += blockDim.x) {
    dx_values[row_first + idx] = (dout_values[row_first + idx] - mul_sum) *
                                 out_values[row_first + idx] / scale;
  }
}

template <typename T, typename Context>
void FusedAttentionCsrGradKernel(const Context& dev_ctx,
                                 const DenseTensor& query,
                                 const DenseTensor& key,
                                 const DenseTensor& value,
                                 const SparseCsrTensor& softmax,
                                 const DenseTensor& dout,
                                 DenseTensor* dquery,
                                 DenseTensor* dkey,
                                 DenseTensor* dvalue) {
#if CUDA_VERSION >= 11070
  /* Step1: Forward: softmax{CSR} * value{Dense} -> out{Dense}, reuse */
  SparseCsrTensor dsoftmax;
  MatmulCsrDenseGradKernel<T, Context>(
      dev_ctx, softmax, value, dout, &dsoftmax, dvalue);

  /* Step2: Calculate grad of sdd_result, manualy not reuse */
  SparseCsrTensor d_sdd_result;
  EmptyLikeCsrKernel<T, Context>(dev_ctx, dsoftmax, &d_sdd_result);
  auto q_dim = query.dims();
  auto q_rank = q_dim.size();

  int total_row_num = 1;
  int batch_num = 1;
  for (int i = 0; i < q_rank - 1; ++i) {
    total_row_num *= q_dim[i];
    if (i < q_rank - 2) {
      batch_num *= q_dim[i];
    }
  }
  int M = q_dim[q_rank - 2];
  int N = q_dim[q_rank - 1];
  int batch_nnz = softmax.nnz() / batch_num;

  dim3 grid((total_row_num + 7) / 8);
  dim3 block(WARP_SIZE, 8);

  AttnSoftmaxGpuGradKernel<T><<<grid, block, 0, dev_ctx.stream()>>>(
      softmax.crows().data<int64_t>(),
      softmax.values().data<T>(),
      dsoftmax.mutable_values()->data<T>(),
      d_sdd_result.mutable_values()->data<T>(),
      M,
      total_row_num,
      std::sqrt(N),
      batch_nnz);

  /* Step3: Forward: query{Dense} * key'{Dense} -> sdd_result{SparseCsr} */
  auto sparse_blas = phi::funcs::sparse::GetSparseBlas<Context, T>(dev_ctx);
  // dquery{Dense} = d_sdd_result{SparseCsr} * key{Dense} //
  dquery->Resize(query.dims());
  dev_ctx.template Alloc<T>(dquery);
  sparse_blas.SPMM(false,
                   false,
                   static_cast<T>(1.f),
                   d_sdd_result,
                   key,
                   static_cast<T>(0.f),
                   dquery);

  // dkey{Dense} = d_sdd_result'{SparseCsr} * query{Dense} //
  dkey->Resize(key.dims());
  dev_ctx.template Alloc<T>(dkey);
  sparse_blas.SPMM(true,
                   false,
                   static_cast<T>(1.f),
                   d_sdd_result,
                   query,
                   static_cast<T>(0.f),
                   dkey);
#else
  PADDLE_THROW(
      phi::errors::Unimplemented("backward of 'sparse.nn.functional.attention' "
                                 "use 'cusparseCsrSetStridedBatch', which is "
                                 "completed supported from CUDA 11.7"));
#endif
}

}  // namespace sparse
}  // namespace phi

PD_REGISTER_KERNEL(fused_attention_csr_grad,
                   GPU,
                   ALL_LAYOUT,
                   phi::sparse::FusedAttentionCsrGradKernel,
                   float,
                   double) {
  kernel->InputAt(0).SetDataLayout(phi::DataLayout::SPARSE_CSR);
}
