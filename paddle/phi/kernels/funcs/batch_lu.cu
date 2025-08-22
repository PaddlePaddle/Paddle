/// Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/phi/kernels/funcs/batch_lu.h"

#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/common/memory_utils.h"
#include "paddle/phi/core/tensor_utils.h"
#include "paddle/phi/kernels/funcs/blas/blas.h"

namespace phi {
namespace funcs {

template <typename T, typename Context>
void BatchLUFunctor<T, Context>::operator()(const Context& dev_ctx,
                                            const DenseTensor& x,
                                            DenseTensor* lu_out,
                                            DenseTensor* pivots,
                                            DenseTensor* infos) {
  const auto& dims = x.dims();
  const int rank = dims.size();
  const int64_t n = dims[rank - 1];
  const int64_t batch_size = x.numel() / (n * n);

  if (batch_size == 0) {
    return;
  }

  // getrf is an in-place operation. Copy input 'x' to 'lu_out'.
  dev_ctx.template Alloc<T>(lu_out);
  phi::Copy(dev_ctx, x, dev_ctx.GetPlace(), false, lu_out);

  dev_ctx.template Alloc<int>(pivots);
  dev_ctx.template Alloc<int>(infos);

  auto* lu_data = lu_out->data<T>();
  auto* pivots_data = pivots->data<int>();
  auto* infos_data = infos->data<int>();

  std::vector<T*> cpu_ptrs(batch_size);
  for (int64_t i = 0; i < batch_size; ++i) {
    cpu_ptrs[i] = lu_data + i * n * n;
  }

  // Copy pointer arrays from Host to Device
  size_t ptrs_bytes = cpu_ptrs.size() * sizeof(T*);
  auto gpu_ptrs_data = phi::memory_utils::Alloc(
      dev_ctx.GetPlace(),
      ptrs_bytes,
      phi::Stream(reinterpret_cast<phi::StreamId>(dev_ctx.stream())));

  memory_utils::Copy(dev_ctx.GetPlace(),
                     gpu_ptrs_data->ptr(),
                     phi::CPUPlace(),
                     static_cast<void*>(cpu_ptrs.data()),
                     ptrs_bytes,
                     dev_ctx.stream());

  // Get the BLAS handle and call BatchedGETRF
  auto blas = phi::funcs::GetBlas<Context, T>(dev_ctx);
  blas.BatchedGETRF(n,
                    reinterpret_cast<T**>(gpu_ptrs_data->ptr()),
                    pivots_data,
                    infos_data,
                    static_cast<int>(batch_size));
}

template class BatchLUFunctor<float, GPUContext>;
template class BatchLUFunctor<double, GPUContext>;
template class BatchLUFunctor<phi::dtype::complex<float>, GPUContext>;
template class BatchLUFunctor<phi::dtype::complex<double>, GPUContext>;

}  // namespace funcs
}  // namespace phi
