// Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/phi/kernels/funcs/inverse_from_lu.h"

#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/common/memory_utils.h"
#include "paddle/phi/core/tensor_utils.h"
#include "paddle/phi/kernels/funcs/blas/blas.h"

namespace phi {
namespace funcs {

template <typename T, typename Context>
void InverseFromLUFunctor<T, Context>::operator()(const Context& dev_ctx,
                                                  const DenseTensor& lu_data,
                                                  const DenseTensor& pivots,
                                                  DenseTensor* inverse_out) {
  const auto& dims = lu_data.dims();
  const int rank = dims.size();
  const int64_t n = dims[rank - 1];
  const int64_t batch_size = lu_data.numel() / (n * n);

  if (batch_size == 0) {
    return;
  }

  // getri is an in-place operation, copy `lu_data` to `inverse_out`.
  dev_ctx.template Alloc<T>(inverse_out);

  std::vector<const T*> a_ptr_host(batch_size);
  std::vector<T*> c_ptr_host(batch_size);
  for (int64_t i = 0; i < batch_size; ++i) {
    a_ptr_host[i] = lu_data.data<T>() + i * n * n;
    c_ptr_host[i] = inverse_out->data<T>() + i * n * n;
  }

  // Copy pointer arrays from Host to Device
  auto a_ptr_device = memory_utils::Alloc(
      dev_ctx.GetPlace(),
      batch_size * sizeof(T*),
      phi::Stream(reinterpret_cast<phi::StreamId>(dev_ctx.stream())));
  auto c_ptr_device = memory_utils::Alloc(
      dev_ctx.GetPlace(),
      batch_size * sizeof(T*),
      phi::Stream(reinterpret_cast<phi::StreamId>(dev_ctx.stream())));

  memory_utils::Copy(dev_ctx.GetPlace(),
                     a_ptr_device->ptr(),
                     phi::CPUPlace(),
                     a_ptr_host.data(),
                     batch_size * sizeof(T*),
                     dev_ctx.stream());
  memory_utils::Copy(dev_ctx.GetPlace(),
                     c_ptr_device->ptr(),
                     phi::CPUPlace(),
                     c_ptr_host.data(),
                     batch_size * sizeof(T*),
                     dev_ctx.stream());

  // Allocate device memory for info array
  auto info_device = memory_utils::Alloc(
      dev_ctx.GetPlace(),
      batch_size * sizeof(int),
      phi::Stream(reinterpret_cast<phi::StreamId>(dev_ctx.stream())));

  // Get the BLAS handle and call BatchedGETRI
  auto blas = phi::funcs::GetBlas<GPUContext, T>(dev_ctx);
  blas.BatchedGETRI(n,
                    reinterpret_cast<const T**>(a_ptr_device->ptr()),
                    pivots.data<int>(),
                    reinterpret_cast<T**>(c_ptr_device->ptr()),
                    reinterpret_cast<int*>(info_device->ptr()),
                    batch_size);
}

template class InverseFromLUFunctor<float, GPUContext>;
template class InverseFromLUFunctor<double, GPUContext>;
template class InverseFromLUFunctor<phi::dtype::complex<float>, GPUContext>;
template class InverseFromLUFunctor<phi::dtype::complex<double>, GPUContext>;

}  // namespace funcs
}  // namespace phi
