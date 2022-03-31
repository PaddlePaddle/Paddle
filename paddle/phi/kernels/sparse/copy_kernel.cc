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

#include "paddle/phi/kernels/sparse/copy_kernel.h"

#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/sparse_coo_tensor.h"
#include "paddle/phi/core/sparse_csr_tensor.h"
#include "paddle/phi/kernels/copy_kernel.h"

namespace phi {
namespace sparse {

template <typename Context>
void CopyCoo(const Context& dev_ctx,
             const SparseCooTensor& src,
             Place dst_place,
             bool blocking,
             SparseCooTensor* dst) {
  phi::Copy<Context>(dev_ctx,
                     src.non_zero_indices(),
                     dst_place,
                     blocking,
                     dst->mutable_non_zero_indices());

  phi::Copy<Context>(dev_ctx,
                     src.non_zero_elements(),
                     dst_place,
                     blocking,
                     dst->mutable_non_zero_elements());
  dst->set_dims(src.dims());
}

template <typename Context>
void CopyCsr(const Context& dev_ctx,
             const SparseCsrTensor& src,
             Place dst_place,
             bool blocking,
             SparseCsrTensor* dst) {
  phi::Copy<Context>(dev_ctx,
                     src.non_zero_crows(),
                     dst_place,
                     blocking,
                     dst->mutable_non_zero_crows());

  phi::Copy<Context>(dev_ctx,
                     src.non_zero_cols(),
                     dst_place,
                     blocking,
                     dst->mutable_non_zero_cols());

  phi::Copy<Context>(dev_ctx,
                     src.non_zero_elements(),
                     dst_place,
                     blocking,
                     dst->mutable_non_zero_elements());
  dst->set_dims(src.dims());
}

}  // namespace sparse
}  // namespace phi

PD_REGISTER_GENERAL_KERNEL(copy_sparse_coo,
                           CPU,
                           ALL_LAYOUT,
                           phi::sparse::CopyCoo<phi::CPUContext>,
                           ALL_DTYPE) {}

PD_REGISTER_GENERAL_KERNEL(copy_sparse_csr,
                           CPU,
                           ALL_LAYOUT,
                           phi::sparse::CopyCsr<phi::CPUContext>,
                           ALL_DTYPE) {}

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
PD_REGISTER_GENERAL_KERNEL(copy_sparse_coo,
                           GPU,
                           ALL_LAYOUT,
                           phi::sparse::CopyCoo<phi::GPUContext>,
                           ALL_DTYPE) {}
PD_REGISTER_GENERAL_KERNEL(copy_sparse_csr,
                           GPU,
                           ALL_LAYOUT,
                           phi::sparse::CopyCsr<phi::GPUContext>,
                           ALL_DTYPE) {}
#endif
